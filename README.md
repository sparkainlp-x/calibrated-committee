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

## Verified

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
