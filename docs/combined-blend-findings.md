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

The `+ shared_state` row is not a fifth model added to the blend. That column is itself
a composite of two components already in the simplex; see the correction below.

## TabPFN is the load-bearing component

Across all 32 combinations the separation is exact: **the 16 combinations containing
TabPFN occupy the top 16 positions and every one has an interval below zero. Not one of
the 16 without it is significant** — the best TabPFN-free result is 31.9482 with
[−0.0339, +0.0037].

TabPFN also never gets dropped. Its fitted weight is 0.45–0.67 in every fold of every
combination it appears in. Removing it does not cost 0.3 points; it costs the only
statistically separable result available.

## Correction: `shared_state` is not a standalone model

An earlier version of this page described `shared_state` as "the second-best single model
in the project (31.8314, and the best drift ratio of any standalone component at 0.9125)"
and removed it on the grounds that its fitted weight was zero, so "its information is a
strict subset of what the kept components already supply." Both halves were wrong. That
number is not a standalone-model score, and the zero fitted weight does not establish
what the page claimed. **The removal stands**; the corrected reason is below. The error is
recorded here rather than edited away, so a reader can see what was wrong.

### What the column actually is

`artifacts/shared_state_01/validation.csv`, the file registered as the `shared_state`
component in `scripts/combined_evaluate.py:53`, is not one model's output. It is a small
correction applied to a composite that already contains two of the kept components:

- `scripts/evaluate_shared_state.py:40-42` builds it as
  `adjust(artifacts/mtd_pairs_01/combined_additive.csv, state, beta)`;
- `scripts/evaluate_mtd_pairs.py:41-44` builds that input as
  `0.75 * artifacts/blend_release/evaluated_oof_predictions.csv + 0.25 *
  artifacts/mtd_pairs_01/validation_additive.csv`.

Both steps were rechecked numerically rather than read off the source. Recomputing
`0.75 * blend + 0.25 * additive MTD` matches `combined_additive.csv` to 1.11e-16, and
`artifacts/blend_release/evaluated_oof_predictions.csv` is byte-identical to the
simplex's own base column `artifacts/phase2/predictions/oof_ensemble.csv` (both
`sha256:248b55abb69ff6f3...`, the `frozen_blend` hash in `artifacts/audit/recompute.json`).
The additive MTD differs from the registered `mtd` component — the lag-7 pairwise variant
— by a mean absolute 0.001342 per coordinate, maximum 0.044665: the same model family
with a near-identical output.

The two-state filter then moves the result very little. `adjust` clips the win-mass shift
to +/-0.08, the fitted slope is beta = 0.04 and the state is bounded in [-0.5, 0.5], so
the delivered column sits at most 0.019868 from its own composite input in any
coordinate, with a mean absolute 0.014237 on `p_0`.

### Why 31.8314 must not be presented as a model score

31.8314 is the score of `0.75 * frozen blend + 0.25 * additive MTD` with that bounded
nudge applied. The composite input alone scores **31.902299**, so the two-state
correction *lowered* points by 0.070873; `docs/mtd-pairs-investigation.md` reports that
fall correctly. The 0.9125 drift ratio is likewise the composite's ratio, not a
standalone component's.

The fit is mismatched as well. The slope was fitted against
`artifacts/mtd_pairs_01/train_oof_additive.csv`, the **raw MTD train OOF**
(`scripts/evaluate_shared_state.py:12` and `:25`) — a different baseline from the
blend-plus-MTD composite the fitted correction is then applied to. Whatever the state
variable is worth, it was never calibrated against the thing it adjusts.

### Why the old redundancy argument does not hold

The fitted weights themselves are a sound observation. Per fold, in every combination
containing it:

| Combination | shared_state weight per fold |
| --- | --- |
| frozen + tabpfn + mtd + two_head + shared_state | 0.000, 0.000, 0.000, 0.000 |
| frozen + mtd + two_head + shared_state | 0.000, 0.000, 0.000, 0.000 |
| frozen + mtd + shared_state + dense_cb | 0.000, 0.000, 0.000, 0.000 |
| frozen + tabpfn + mtd + shared_state | 0.000, 0.000, 0.290, 0.000 |
| frozen + tabpfn + shared_state | 0.245, 0.501, 0.526, 0.448 |
| frozen + shared_state | 0.380, 0.830, 1.000, 1.000 |

It earns weight only when the stronger components are absent, and the five-component
blend scores identically to the four-component one (32.2605 both).

But that is what a free simplex *must* do with a column equal to 0.75 of its own base
plus 0.25 of a near-twin of one of its components, shifted by at most 0.02. The column is
**redundant by construction**, because two of its own inputs are already separate
components of the same simplex. So the zero weight says nothing about whether a
differently constructed shared-state column — fitted against the baseline it actually
corrects, and supplied as a residual rather than as a composite — would also be
redundant. The old text used the zero weight as if it proved that, and it does not.

### What justifies the removal

- The column is, by construction, two components already in the simplex plus a bounded
  nudge, so it cannot contribute to a simplex that contains them.
- That nudge is measured to *lose* 0.070873 points on the composite it was fitted for,
  and its slope was fitted against a different baseline than the one it adjusts.
- Including it would add inference cost for no change in output.

That is sufficient and it does not rest on the invalid inference. **Shared-state
modelling is not reopened here**: the conclusion is unchanged, only its justification.
`--include-rejected` still re-measures the column rather than asking anyone to trust
this page.

Reproduce the construction checks:

```sh
.venv/bin/python - <<'PY'
import sys; from pathlib import Path
R = Path('.').resolve(); sys.path[:0] = [str(R), str(R / 'organizer')]
import numpy as np, json
from slot_forecasting.io import read_submission
blend = read_submission(R / 'artifacts/blend_release/evaluated_oof_predictions.csv')
add   = read_submission(R / 'artifacts/mtd_pairs_01/validation_additive.csv')
comb  = read_submission(R / 'artifacts/mtd_pairs_01/combined_additive.csv')
ss    = read_submission(R / 'artifacts/shared_state_01/validation.csv')
print('composite residual',
      max(abs(.75*np.array(blend[k]) + .25*np.array(add[k]) - np.array(comb[k])).max()
          for k in comb))
print('shared_state vs its own input',
      max(abs(np.array(ss[k]) - np.array(comb[k])).max() for k in comb))
e = json.loads((R / 'artifacts/shared_state_01/evaluation.json').read_text())
print('input', e['baseline']['technical_points'], 'adjusted', e['candidate']['technical_points'])
PY
shasum -a 256 artifacts/phase2/predictions/oof_ensemble.csv \
              artifacts/blend_release/evaluated_oof_predictions.csv
```

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
- The `shared_state` column's zero fitted weight is explained by its construction, so
  this search establishes nothing about the shared-state variable itself. It was not
  supplied to the simplex in a form that could have shown anything either way, and no
  such form was built or measured.
