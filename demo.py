"""Demo: does LEARNED aggregation close the oracle gap?

Calibrate (200 labeled trials/sigma, seeds 100000+) -> freeze
weights -> evaluate on FRESH seeds (700000+). Per sigma in
{0.25, 0.50}, 200 test trials:

  stacked   per-member logistic weights (frozen)
  majority  unweighted vote (membrane-committee baseline)
  select    single best-calibration member ("select, don't average")
  bestT     best member measured on TEST (reference; peeks)
  stack20   richer per-(member,class) weights (20 params; overfit check)
  oracle    any member right (headroom bound; peeks at label)

Run: python demo.py
"""

import math
import random

from ccommittee import (CalibratedCommittee, Member, trial_votes,
                        noisy, CAL_SEED_BASE, TEST_SEED_BASE,
                        check_seed_separation)

S2 = 2 ** -0.5
S8 = 1.0 / 8 ** 0.5
PATTERNS = [
    [S2, 0j, 0j, 0j, 0j, 0j, 0j, S2],                     # GHZ
    [0j, 3 ** -0.5, 3 ** -0.5, 0j, 3 ** -0.5, 0j, 0j, 0j],  # W
    [1, 0j, 0j, 0j, 0j, 0j, 0j, 0j],                       # |000>
    [S8] * 8,                                               # |+++>
]
MEMBERS = [("full", set()), ("mrg0", {0}), ("mrg1", {1}),
           ("mrg2", {2}), ("mrg01", {0, 1})]
SIGMAS = (0.25, 0.50)
N_CAL, N_TEST = 200, 200


def fit_perclass(cal_trials, n_classes, n_members,
                 lr=0.2, iters=400, l2=1e-3):
    """Richer stacking: score_c = sum_m w[m][c] * [vote_m == c]."""
    w = [[0.0] * n_classes for _ in range(n_members)]
    n = len(cal_trials)
    for _ in range(iters):
        g = [[0.0] * n_classes for _ in range(n_members)]
        for label, votes, _ in cal_trials:
            sc = [0.0] * n_classes
            for m, v in enumerate(votes):
                sc[v] += w[m][v]
            mx = max(sc)
            ex = [math.exp(s - mx) for s in sc]
            tot = sum(ex)
            for m, v in enumerate(votes):
                g[m][v] += ex[v] / tot - (1.0 if v == label else 0.0)
        for m in range(n_members):
            for c in range(n_classes):
                w[m][c] -= lr * (g[m][c] / n + l2 * w[m][c])
    return w


def main():
    members = [Member(name, ms, PATTERNS) for name, ms in MEMBERS]
    cc = CalibratedCommittee(members, PATTERNS, SIGMAS, n_cal=N_CAL)

    print("calibration (200 labeled trials/sigma; cost of learning):")
    for s in SIGMAS:
        cal = cc.cal[s]
        print(f"  sigma={s}: acc={[round(a, 3) for a in cal.member_acc]} "
              f"weights={[round(x, 3) for x in cal.weights]} "
              f"-> select={MEMBERS[cal.best_member][0]}")

    test_seeds = [TEST_SEED_BASE + t for t in range(N_TEST)]
    sep = all(check_seed_separation(cc.cal[s], test_seeds)
              for s in SIGMAS)
    print(f"calibrate/evaluate seed separation: "
          f"{'ENFORCED' if sep else 'VIOLATED'}")

    print(f"\n{'sigma':>6} {'stacked':>8} {'majority':>9} {'select':>7} "
          f"{'bestT':>6} {'stack20':>8} {'oracle':>7}")
    for sigma in SIGMAS:
        cal_trials = [trial_votes(members, PATTERNS, sigma,
                               CAL_SEED_BASE + i)
                      for i in range(N_CAL)]
        w20 = fit_perclass(cal_trials, len(PATTERNS), len(members))
        st = mj = se = s20 = orc = 0
        bs = [0] * len(members)
        for t in range(N_TEST):
            rng = random.Random(TEST_SEED_BASE + t)
            label = rng.randrange(len(PATTERNS))
            probe = noisy(PATTERNS[label], sigma, rng)
            st += (cc.decide_stacked(probe, sigma) == label)
            mj += (cc.decide_majority(probe) == label)
            se += (cc.decide_select(probe, sigma) == label)
            votes = [m.vote(probe)[0] for m in members]
            sc = [0.0] * len(PATTERNS)
            for m, v in enumerate(votes):
                sc[v] += w20[m][v]
            s20 += (max(range(len(PATTERNS)),
                        key=lambda c: sc[c]) == label)
            orc += cc.oracle(probe, label)
            for i, m in enumerate(members):
                bs[i] += (m.vote(probe)[0] == label)
        n = N_TEST
        print(f"{sigma:>6.2f} {st/n:>8.3f} {mj/n:>9.3f} {se/n:>7.3f} "
              f"{max(bs)/n:>6.3f} {s20/n:>8.3f} {orc/n:>7.3f}")

    print("\nVERDICT: learned aggregation does not close the oracle gap "
          "-- it learns to ignore the committee (see DESIGN.md).")


if __name__ == "__main__":
    main()
