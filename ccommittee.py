"""Calibrated committee: learn the aggregation instead of guessing it.

The membrane-committee measured a clean negative result: majority,
confidence-weighted, and reliability-weighted voting all failed to
beat the best single member, while the oracle gap proved the
ensemble holds unused information. This package answers it:

  - STACKED aggregation: on CALIBRATION seeds (labeled trials the
    test never sees), fit per-member weights by multiclass logistic
    (softmax cross-entropy) gradient descent over the vote
    indicators: score(c) = sum_m w_m * [vote_m == c]. Freeze the
    weights. Evaluate on FRESH seeds. Calibrate/evaluate separation
    is structural, not a promise.
  - SELECT-BEST routing: per noise level, the single member with
    the best calibration accuracy is used alone at test -- the
    "select, don't average" baseline.

Core member/committee logic is copied (not cross-imported) from
membrane-committee's hcommittee.py, including the normalization
fix: stored patterns are normalized AFTER merging, even for the
empty merge set (full resolution).
"""

import math
import random

# ------------------------------------------------------------------
# Seed discipline: calibration and evaluation ranges must not overlap.
# Structural separation -- the fitter never sees a test seed.
CAL_SEED_BASE = 100000
TEST_SEED_BASE = 700000


# ------------------------------------------------------------------
# core (copied from membrane-committee hcommittee.py)
def normalize(v):
    n = math.sqrt(sum(abs(c) ** 2 for c in v))
    return [c / n for c in v]


def overlap(field, pattern):
    return sum(p.conjugate() * f for p, f in zip(pattern, field))


def noisy(state, sigma, rng):
    return normalize([c + complex(rng.gauss(0, sigma), rng.gauss(0, sigma))
                      for c in state])


def merge_qubit(state, q):
    """Haar low-pass merging qubit q: 2^n -> 2^(n-1) amplitudes."""
    s2 = math.sqrt(2.0)
    out = []
    for i in range(len(state)):
        j = i ^ (1 << q)
        if j > i:
            out.append((state[i] + state[j]) / s2)
    return normalize(out)


def merge_qubits(state, qs):
    """Merge a set of *original* qubit indices (order-independent)."""
    for shift, q in enumerate(sorted(set(qs))):
        state = merge_qubit(state, q - shift)
    return state


class Member:
    """One committee member: a fixed resolution and the patterns
    stored at that resolution, normalized AFTER merging."""

    def __init__(self, name, merge_set, patterns):
        self.name = name
        self.merge_set = set(merge_set)
        self.patterns = [normalize(merge_qubits(p, self.merge_set))
                         for p in patterns]

    def _view(self, probe):
        return merge_qubits(probe, self.merge_set)

    def vote(self, probe):
        """(winner, weight): winner by max overlap, weight = top prob."""
        pv = self._view(probe)
        ms = [overlap(pv, p) for p in self.patterns]
        tot = sum(abs(m) ** 2 for m in ms)
        w = max(range(len(ms)), key=lambda i: abs(ms[i]))
        return w, (abs(ms[w]) ** 2 / tot if tot else 0.0)


def decide_majority(votes):
    """Majority over (winner, weight) votes; weight breaks ties."""
    tally = {}
    for w, wt in votes:
        tally[w] = tally.get(w, [0, 0.0])
        tally[w][0] += 1
        tally[w][1] = max(tally[w][1], wt)
    return max(tally, key=lambda w: (tally[w][0], tally[w][1]))


def decide_weighted(votes, weights):
    """score(c) = sum_m weights[m] * [vote_m == c]; argmax."""
    score = {}
    for (w, _), wm in zip(votes, weights):
        score[w] = score.get(w, 0.0) + wm
    return max(score, key=lambda w: score[w])


# ------------------------------------------------------------------
# trials
def trial_votes(members, patterns, sigma, seed):
    """One labeled trial -> (label, [winners], [weights])."""
    rng = random.Random(seed)
    label = rng.randrange(len(patterns))
    probe = noisy(patterns[label], sigma, rng)
    votes = [m.vote(probe) for m in members]
    return label, [v for v, _ in votes], [w for _, w in votes]


# ------------------------------------------------------------------
# stacking: logistic fit of per-member weights on calibration trials
def fit_weights(cal_trials, n_classes, lr=0.2, iters=400, l2=1e-3):
    """Multiclass logistic (softmax cross-entropy) fit.

    score_c = sum_m w_m * [vote_m == c]; loss = -mean log softmax_y.
    dL/dw_m = mean over trials of (p_{vote_m} - [vote_m == label])
    plus L2. Deterministic: no randomness anywhere in the fit.
    """
    n_members = len(cal_trials[0][1])
    w = [0.0] * n_members
    n = len(cal_trials)
    for _ in range(iters):
        grad = [0.0] * n_members
        for label, votes, _ in cal_trials:
            scores = [0.0] * n_classes
            for m, v in enumerate(votes):
                scores[v] += w[m]
            mx = max(scores)
            ex = [math.exp(s - mx) for s in scores]
            tot = sum(ex)
            for m, v in enumerate(votes):
                grad[m] += ex[v] / tot - (1.0 if v == label else 0.0)
        for m in range(n_members):
            w[m] -= lr * (grad[m] / n + l2 * w[m])
    return w


class Calibration:
    """Frozen per-sigma calibration: weights, member accuracies,
    best member, and the seed range used (for separation checks)."""

    def __init__(self, sigma, weights, member_acc, seed_base, n_cal):
        self.sigma = sigma
        self.weights = weights
        self.member_acc = member_acc
        self.best_member = max(range(len(member_acc)),
                               key=lambda m: member_acc[m])
        self.seed_base = seed_base
        self.n_cal = n_cal

    def seed_range(self):
        return range(self.seed_base, self.seed_base + self.n_cal)


def calibrate(members, patterns, sigma, n_cal=200,
              seed_base=CAL_SEED_BASE):
    """Fit on calibration seeds only. Returns a frozen Calibration."""
    trials = [trial_votes(members, patterns, sigma, seed_base + s)
              for s in range(n_cal)]
    n_members = len(members)
    acc = [sum(1 for (label, votes, _) in trials if votes[m] == label)
           / n_cal for m in range(n_members)]
    weights = fit_weights(trials, len(patterns))
    return Calibration(sigma, weights, acc, seed_base, n_cal)


class CalibratedCommittee:
    """Members + one frozen Calibration per noise level."""

    def __init__(self, members, patterns, sigmas, n_cal=200):
        self.members = members
        self.patterns = patterns
        self.cal = {s: calibrate(members, patterns, s, n_cal)
                    for s in sigmas}

    def votes(self, probe):
        return [m.vote(probe) for m in self.members]

    def decide_stacked(self, probe, sigma):
        return decide_weighted(self.votes(probe),
                               self.cal[sigma].weights)

    def decide_majority(self, probe):
        return decide_majority(self.votes(probe))

    def decide_select(self, probe, sigma):
        m = self.cal[sigma].best_member
        return self.members[m].vote(probe)[0]

    def oracle(self, probe, label):
        return any(m.vote(probe)[0] == label for m in self.members)


def check_seed_separation(calibration, test_seeds):
    """True iff no test seed falls inside the calibration range."""
    cal = set(calibration.seed_range())
    return not any(s in cal for s in test_seeds)
