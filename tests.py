"""Tests for calibrated-committee. Run: python tests.py"""

import math
import random

from ccommittee import (Calibration, Member, calibrate, check_seed_separation,
                        decide_majority, decide_weighted, fit_weights,
                        merge_qubit, normalize, trial_votes,
                        CAL_SEED_BASE, TEST_SEED_BASE)

S2 = 2 ** -0.5
PATS2 = [[1] + [0j] * 7, [0j] * 7 + [1]]


def test_member_normalizes_even_without_merge():
    # regression (from membrane-committee): the full-resolution member
    # once skipped normalization, inflating one pattern's overlaps so
    # it won every vote. Stored patterns must be unit norm, always.
    bad = [0.7071] * 8  # norm 2, not 1
    m = Member("full", set(), [bad, [1] + [0j] * 7])
    for p in m.patterns:
        assert abs(sum(abs(c) ** 2 for c in p) - 1.0) < 1e-12


def test_weights_frozen_across_runs():
    # the fit is deterministic: same calibration seeds -> identical
    # frozen weights, accuracies, best member.
    members = [Member("full", set(), PATS2), Member("mrg0", {0}, PATS2)]
    c1 = calibrate(members, PATS2, 0.25, n_cal=50, seed_base=CAL_SEED_BASE)
    c2 = calibrate(members, PATS2, 0.25, n_cal=50, seed_base=CAL_SEED_BASE)
    assert c1.weights == c2.weights
    assert c1.member_acc == c2.member_acc
    assert c1.best_member == c2.best_member


def test_calibrate_evaluate_separation_enforced():
    members = [Member("full", set(), PATS2)]
    cal = calibrate(members, PATS2, 0.25, n_cal=50, seed_base=CAL_SEED_BASE)
    test_seeds = [TEST_SEED_BASE + t for t in range(50)]
    assert check_seed_separation(cal, test_seeds)
    # and the checker actually catches an overlap:
    bad_seeds = [CAL_SEED_BASE + 3]
    assert not check_seed_separation(cal, bad_seeds)


def test_decisions_deterministic():
    members = [Member("full", set(), PATS2), Member("mrg0", {0}, PATS2)]
    cal = calibrate(members, PATS2, 0.25, n_cal=50, seed_base=CAL_SEED_BASE)
    label1, votes1, _ = trial_votes(members, PATS2, 0.25, TEST_SEED_BASE)
    label2, votes2, _ = trial_votes(members, PATS2, 0.25, TEST_SEED_BASE)
    assert (label1, votes1) == (label2, votes2)
    tv1 = [(v, 1.0) for v in votes1]
    tv2 = [(v, 1.0) for v in votes2]
    assert decide_weighted(tv1, cal.weights) == \
        decide_weighted(tv2, cal.weights)
    assert decide_majority(tv1) == decide_majority(tv2)


def test_fitter_trusts_reliable_member():
    # synthetic: member 0 always right, others random votes.
    # The logistic fit must crown member 0 (argmax weight).
    rng = random.Random(11)
    trials = []
    for t in range(300):
        label = rng.randrange(4)
        votes = [label] + [rng.randrange(4) for _ in range(4)]
        trials.append((label, votes, [1.0] * 5))
    w = fit_weights(trials, 4)
    assert max(range(5), key=lambda m: w[m]) == 0
    assert w[0] > 0  # and it learned a positive trust weight


def test_select_best_uses_calibration_best():
    members = [Member("full", set(), PATS2), Member("mrg0", {0}, PATS2)]
    cal = calibrate(members, PATS2, 0.25, n_cal=50, seed_base=CAL_SEED_BASE)
    assert cal.best_member == max(range(2), key=lambda m: cal.member_acc[m])
    # decide_select must equal that member's own vote
    from ccommittee import CalibratedCommittee
    cc = CalibratedCommittee.__new__(CalibratedCommittee)
    cc.members = members
    cc.cal = {0.25: cal}
    rng = random.Random(TEST_SEED_BASE)
    from ccommittee import noisy
    probe = noisy(PATS2[0], 0.25, rng)
    assert cc.decide_select(probe, 0.25) == \
        members[cal.best_member].vote(probe)[0]


def test_merge_qubit_known():
    ghz = [S2, 0j, 0j, 0j, 0j, 0j, 0j, S2]
    m = merge_qubit(ghz, 2)
    assert len(m) == 4
    assert abs(m[0] - S2) < 1e-12 and abs(m[3] - S2) < 1e-12
    assert abs(m[1]) < 1e-12 and abs(m[2]) < 1e-12


def test_clean_probe_unanimous_and_correct():
    from ccommittee import CalibratedCommittee
    cc = CalibratedCommittee.__new__(CalibratedCommittee)
    members = [Member("full", set(), PATS2), Member("mrg01", {0, 1}, PATS2)]
    cc.members = members
    cc.cal = {0.0: Calibration(0.0, [1.0, 1.0], [1.0, 1.0], CAL_SEED_BASE, 10)}
    cc.patterns = PATS2
    assert cc.decide_stacked(PATS2[1], 0.0) == 1
    assert cc.decide_majority(PATS2[1]) == 1
    assert cc.decide_select(PATS2[1], 0.0) == 1
    assert cc.oracle(PATS2[1], 1)


if __name__ == "__main__":
    for name, fn in sorted(
            [(k, v) for k, v in globals().items() if k.startswith("test_")]):
        fn()
        print(f"PASS {name}")
    print("all tests passed")
