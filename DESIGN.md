# DESIGN — Answering the committee's negative result

## The question

membrane-committee measured: majority, confidence-weighted, and
reliability-weighted voting all fail to beat the best single
member, while the oracle gap proves the ensemble holds unused
information. So: **does LEARNED aggregation close the oracle gap?**

## The stacking design

Per noise level, on 200 CALIBRATION trials (labeled; seeds
100000+, never seen at test):

- **stacked**: multiclass logistic (softmax cross-entropy)
  gradient descent over per-member weights —
  score(c) = Σ_m w_m · [vote_m == c]. Deterministic fit
  (lr=0.2, 400 iters, L2 1e-3), frozen, then evaluated on 200
  FRESH seeds (700000+). Calibrate/evaluate separation is
  structural: disjoint seed ranges, enforced by
  `check_seed_separation` (tested both directions).
- **select**: per-σ routing — the single member with the best
  calibration accuracy, used alone at test ("select, don't
  average").
- **stack20**: richer variant, per-(member, class) weights
  (20 params) — the overfitting check.

Cost of learning: 200 labeled trials per σ.

## Measured (200 test trials, fresh seeds)

| σ    | stacked | majority | select | bestT | stack20 | oracle |
|------|---------|----------|--------|-------|---------|--------|
| 0.25 | 0.895   | 0.885    | 0.895  | 0.895 | 0.895   | 0.980  |
| 0.50 | 0.670   | 0.605    | 0.665  | 0.665 | 0.645   | 0.865  |

Fitted weights at σ=0.25: **[2.277, 0.505, 0.384, 0.396, 0.287]**
(full, mrg0, mrg1, mrg2, mrg01).

## The verdict: the negative result DEEPENS

The logistic fit **learned a dictatorship**: 2.277 > 0.505+0.384+
0.396+0.287 = 1.572, so the full-resolution member's vote wins
every score comparison *provably, on every probe* — stacked is
literally identical to select-best. The optimal linear
aggregation of votes is to ignore the committee. The richer
20-parameter stacking does no better (0.895) and overfits at
σ=0.50 (0.645 < 0.670).

The oracle gap is untouched: 0.980 vs 0.895, 0.865 vs 0.670.
**The oracle information is not in the votes in any linearly
extractable form.** Each member's vote is a hard argmax — a lossy
bottleneck that discards the margins. Closing the gap would need
aggregation that sees *more than the votes*: per-member margins,
the probe itself (a meta-learner), or member-conditional
features. That is the precise shape of what remains untested.

## Honest limits

- **Toy scale** (3 qubits, 4 patterns, 5 members, 2 noise levels).
  The dictatorship result is about *this* ensemble; a committee
  of genuinely complementary members (not coarse views of one
  memory) could behave differently.
- **Linear vote aggregation only.** Probe-conditional or
  margin-based aggregation is the named next experiment —
  untested here, claimed nowhere.
- **Calibration costs labels**: 200 labeled trials per σ. In a
  setting where labels are expensive, select-best needs the same
  budget and stacking buys nothing over it.
- **bestT peeks** (best member chosen on test data) — it is a
  reference, not a method. select-best is the honest baseline,
  and stacking ties it exactly.
- The oracle peeks at the label; it bounds headroom, not
  achievable performance.
