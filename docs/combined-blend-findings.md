# Which model classes belong in the blend

Score on the slot-machine payout probabilities is the sole criterion here. All 32
combinations of five challengers over the frozen four-recipe blend were measured with
the existing four-fold protocol — convex weights fitted on three folds, held-fold
predictions concatenated, organizer scorer — so every number is directly comparable to
the 31.702332 bar. Reproduce with `scripts/combined_evaluate.py`; full output in
`artifacts/combined/merged_evaluation.json`.

All 96 public validation episodes were examined before this work. These are exploratory
development numbers, not an untouched holdout, and no independent evaluation set exists.

## Decision

**Keep: `tabpfn`, `mtd`, `two_head`. Remove: `shared_state`, `dense_cb`.**

Encoded in the `COMPONENTS` registry in `scripts/combined_evaluate.py`. Rejected
components keep their code and artifacts, matching how the partial-pooling patch and
calibration cycle 01 were preserved; `--include-rejected` re-measures them rather than
asking anyone to trust this page.

| Combination | Points / 60 | Gain | Folds | Challenger wt | 95% CI vs bar | Stable | Short hist | Drift | Shared ch |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| frozen blend (bar) | 31.702332 | — | — | — | — | 0.8770 | 0.9549 | 0.9350 | 0.8035 |
| **+ tabpfn + mtd + two_head** | **32.2605** | **+0.5582** | 4/4 | 1.000 | [−0.0511, −0.0150] * | 0.8282 | 0.9222 | 0.9045 | 0.7846 |
| + tabpfn + two_head | 32.1985 | +0.4962 | 4/4 | 0.986 | [−0.0468, −0.0113] * | 0.8297 | 0.9183 | 0.9278 | 0.7781 |
| + tabpfn + mtd | 32.1455 | +0.4432 | 3/4 | 0.791 | [−0.0452, −0.0070] * | 0.8342 | 0.9397 | 0.9017 | 0.7905 |
| + tabpfn | 32.0522 | +0.3499 | 4/4 | 0.610 | [−0.0361, −0.0041] * | 0.8408 | 0.9404 | 0.9244 | 0.7821 |
| + mtd + two_head | 31.9482 | +0.2458 | 4/4 | 0.688 | [−0.0339, +0.0037] | 0.8567 | 0.9345 | 0.9124 | 0.8085 |
| + mtd | 31.8572 | +0.1548 | 3/4 | 0.278 | [−0.0241, +0.0050] | 0.8646 | 0.9520 | 0.9071 | 0.8100 |
| + two_head | 31.7924 | +0.0901 | 3/4 | 0.415 | [−0.0161, +0.0053] | 0.8677 | 0.9393 | 0.9406 | 0.8013 |
| + shared_state | 31.7439 | +0.0416 | 2/4 | 0.802 | [−0.0149, +0.0086] | 0.8800 | 0.9564 | 0.9178 | 0.8064 |
| + dense_cb | 31.7011 | **−0.0012** | 3/4 | 0.217 | [−0.0052, +0.0051] | 0.8826 | 0.9533 | 0.9335 | 0.8014 |

`*` = paired 95% whole-episode interval entirely below zero. Bootstrap settings match
the phase-two comparisons: 2,000 draws stratified by condition, seed 314159, reference
denominators recomputed per draw. Intervals are unadjusted for selection.

## TabPFN is the load-bearing component

Across all 32 combinations the separation is exact: **the 16 combinations containing
TabPFN occupy the top 16 positions and every one has an interval below zero. Not one of
the 16 without it is significant** — the best TabPFN-free result is 31.9482 with
[−0.0339, +0.0037].

TabPFN also never gets dropped. Its fitted weight is 0.45–0.67 in every fold of every
combination it appears in. Removing it does not cost 0.3 points; it costs the only
statistically separable result available.

## Why `shared_state` is removed

It is the second-best single model in the project (31.8314, and the best drift ratio of
any standalone component at 0.9125), and it is still worth nothing in a blend. Its
fitted weight, per fold, in every combination containing it:

| Combination | shared_state weight per fold |
| --- | --- |
| frozen + tabpfn + mtd + two_head + shared_state | 0.000, 0.000, 0.000, 0.000 |
| frozen + mtd + two_head + shared_state | 0.000, 0.000, 0.000, 0.000 |
| frozen + mtd + shared_state + dense_cb | 0.000, 0.000, 0.000, 0.000 |
| frozen + tabpfn + mtd + shared_state | 0.000, 0.000, 0.290, 0.000 |
| frozen + tabpfn + shared_state | 0.245, 0.501, 0.526, 0.448 |
| frozen + shared_state | 0.380, 0.830, 1.000, 1.000 |

It earns weight only when the stronger components are absent. Once `tabpfn`, `mtd` and
`two_head` are present it is zeroed in all four folds, and the five-component blend
scores **identically** to the four-component one (32.2605 both). Its information is a
strict subset of what the kept components already supply. Including it would add
inference cost for a provably zero change in output.

## Why `dense_cb` is removed

It is the only component in the entire search that makes the bar worse on its own:
31.7011 against 31.702332, a gain of **−0.0012**, despite receiving 0.10–0.34 fitted
weight in every fold. A component that is given weight and still loses is adding noise.
It also subtracts from the best combination, 32.2605 → 32.2478.

Note `dense_cb` is not uniformly harmful — paired only with TabPFN it reaches 32.1703,
above `frozen + tabpfn` at 32.0522. That gain disappears entirely once `mtd` and
`two_head` are present, so it is a redundant substitute for them, not an addition.

## Correction: `two_head` is kept, not removed

An earlier reading of an equal-weight test suggested `two_head` lowered the score. That
was an artifact of the weighting scheme, not the model:

| | on top of tabpfn | on top of tabpfn + mtd |
| --- | ---: | ---: |
| fitted simplex weight | **+0.1463** | **+0.1150** |
| forced equal weight at w=0.35 | −0.0297 | −0.0238 |

Forcing a weaker model to the same weight as TabPFN dilutes the blend. Given a fitted
weight it takes 0.24–0.41 and is the largest single contributor to `short_history`
(0.9397 → 0.9222 when added to tabpfn+mtd). Equal-weight fixed mixing is the wrong test
for heterogeneous components and is not used for the retention decision.

## What `mtd` actually contributes

Weakest family standalone — 30.2582 plain, 30.5352 changepoint, 30.5905 for the lag-7
pairwise variant used here — and its standalone `short_history` ratio exceeds 1.0
(1.0778), worse than the frozen pooled reference on that condition. It is kept anyway
because it owns **drift**, the condition the rest of the blend handles worst:

- bar 0.9350 → 0.9045 in the best combination
- removing `mtd` from that combination moves drift back to 0.9278

It earns 0.110–0.247 fitted weight in every fold, and it is NumPy-only, so it costs
essentially nothing at inference. A specialist, not a core model.

## The open risk in the top result

In the best combination the frozen four-recipe blend itself is assigned weight
**0.000 in all four folds**:

| Fold | frozen_blend | tabpfn | mtd | two_head |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.483 | 0.110 | 0.408 |
| 1 | 0.000 | 0.483 | 0.221 | 0.296 |
| 2 | 0.000 | 0.451 | 0.222 | 0.327 |
| 3 | 0.000 | 0.515 | 0.247 | 0.237 |

So 32.2605 is not the verified release plus an improvement; it is a **replacement** of
that release by three newer model families. Two things cut in opposite directions:

- In favour: the weights are fitted out-of-fold, so the held-fold score is an honest
  estimate of that fitting procedure, not an in-sample number. 4/4 folds improve and the
  interval is comfortably below zero. It is also the only configuration measured that
  clears the project's own 0.5-point replacement gate (+0.5582).
- Against: component *selection* used all 96 previously examined episodes, no untouched
  holdout exists to confirm it, and deploying it means the submitted probabilities come
  entirely from code paths that have never been scored on hidden data. The frozen blend
  has at least been packaged, checksummed and replayed byte-identically.

A deliberately conservative alternative keeps the release at 0.65 and adds nothing
fitted on validation: `0.65 x frozen_blend + 0.35 x mean(tabpfn, mtd)` scores 32.0434
(+0.3410), improves 4/4 folds and all four conditions, with interval
[−0.0307, −0.0106]. It gives up 0.217 points and fails the 0.5-point gate.

## What is not established

- No untouched holdout anywhere in this comparison. Every figure may overstate the
  hidden-test gain, and the ranking between the close options is not resolvable with
  the data available.
- The `mtd` column uses the lag-7 pairwise variant. The plain and changepoint variants
  were scored standalone but not re-searched through all 32 combinations.
- Fixed-weight mixing spreads one weight equally over the selected challengers. No
  per-component fixed weight grid was searched, so the best non-fitted configuration is
  probably better than 32.0434.
- Nothing here explains *why* any component helps. The conditions each one improves are
  reported; no ablation isolates a mechanism.
