# Calibrated Committee

**Answering the membrane-committee's negative result by learning
the aggregation — and deepening it.**

Per noise level, per-member vote weights are fit by multiclass
logistic regression on 200 calibration trials, frozen, and
evaluated on fresh seeds (calibrate/evaluate separation is
structural). At σ=0.25 the fit learns a "dictatorship" in the
technical sense used here: the best member's weight exceeds the
sum of all others, so stacked aggregation is provably identical
to just using the best member. At σ=0.50 the best member's weight
(1.049) is just below the sum of the others (1.063), and stacked
is only marginally above select-best (0.670 vs 0.665). A richer
20-parameter stacking does no better. The oracle gap stands.
Stdlib-only Python. AGPL-3.0-only.

Follow-up to [membrane-committee](https://github.com/sparkainlp-x/membrane-committee),
which found that majority, confidence-weighted and
reliability-weighted voting do not beat the best single member.
Synthetic toy simulation: classical software, not physical
holography or quantum hardware.

## Measured in this repo (SYNTHETIC, seeded)

- σ=0.25: stacked 0.895 = select-best 0.895; oracle 0.980 untouched
- σ=0.50: stacked 0.670 ≈ select-best 0.665; oracle 0.865 untouched
- Learned weights at σ=0.25 [2.277, 0.505, 0.384, 0.396, 0.287]:
  the best member's weight alone beats the rest combined
  (2.277 > 1.572), so the committee is effectively ignored
- Learned weights at σ=0.50 [1.049, 0.321, 0.352, 0.269, 0.121]:
  not a strict dictatorship (1.049 < 1.063), but stacking still
  gains only 0.005 over select-best
- Seed separation enforced and tested both directions; weights
  frozen (identical across runs)
- 8/8 tests pass, incl. the normalization regression test

**Uncertainty (arithmetic on the counts above, no new runs).** Each accuracy is one frozen-weight evaluation on 200 test trials from one seed range, so there is no spread across seeds. Wilson 95% intervals, treating trials as independent: σ=0.25 stacked and select-best 179/200 = 0.895 (0.845–0.930), oracle 196/200 = 0.980 (0.950–0.992); σ=0.50 stacked 134/200 = 0.670 (0.602–0.731), select-best 133/200 = 0.665 (0.597–0.727), oracle 173/200 = 0.865 (0.811–0.906). The stacked-minus-select-best differences (0.000 and 0.005, one trial) are far inside this sampling noise, so the supported statement is "no measurable gain", not "stacking is slightly better". Variation across calibration sets and seeds is not measured here and would need new runs.

## Run it

Python 3.10 or newer, standard library only. CI runs both commands on
Python 3.10–3.13 and checks the demo table above.

```bash
python tests.py   # 8 tests
python demo.py    # calibration, frozen weights, full comparison
```

## Honest limits

Toy scale; linear vote aggregation only (probe-conditional or
margin-based aggregation is the named next experiment, untested);
calibration costs 200 labeled trials per σ; the oracle peeks at
the label. Full account in `DESIGN.md`.

## Acknowledgements

Mun Seok Lee is acknowledged for discussion only. We cannot verify any specific contribution to this repository, and no file here is known to derive from a package received under his name. He is not an author of this release, and no endorsement of this code or its conclusions is implied. This repository follows up [membrane-committee](https://github.com/sparkainlp-x/membrane-committee); its helper functions (`normalize`, `overlap`, `noisy`, `merge_qubit`, `Member._view`) match those in that repository's original `hcommittee.py`, which was committed before the received package there was added.
