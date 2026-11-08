# TabPFN as a third model class

**TabPFN earns its slot.** A free simplex over five components gives it weight 0.48–0.65
in every development fold (mean **0.566**), far above the 0.15 retention bar. Every fixed
weight in {0.15, 0.25, 0.35} improves all four folds over the current CatBoost + Dynamax
blend, with paired whole-episode intervals entirely below zero.

Built with PriorLabs-TabPFN. See [third-party attribution](../THIRD_PARTY_NOTICES.md);
TabPFN is under the Prior Labs License v1.1, not Apache-2.0.

All scores use the organizer calculation on concatenated held-development-fold
predictions, the same 96 previously examined validation episodes and the same four-fold
protocol that produced the 31.702332 bar in [phase-two results](phase2-results.md).
These are exploratory development numbers, not an untouched evaluation. No new
independent labeled data was available.

## What TabPFN is given

TabPFN is a prior-fitted transformer: `fit` stores a training context and
`predict_proba` runs one forward pass with that context in-attention. No gradient step
touches our data, so there is no learning rate, no early stopping and no tuning budget
beyond the seed, the feature variant and the estimator count.

It reuses the **already-validated CatBoost feature table unchanged** — 182 observable
columns from `phase2_features.observable`, optionally plus the 7 portable HMM summary
columns. That is 182 or 189 columns, all numeric and finite, so the ~250-column adapter
the brief allows for was not needed: no categorical encoding and no importance pruning.
The ceiling is enforced in code rather than assumed, and
`scripts/tabpfn_verify.py` checks that a 251-column table is rejected.

Context is 3,134 rows: the 768 actual training queries plus up to eight historical
pseudo-queries each, exactly the rows `scripts/phase2_train.py` builds. That is inside
TabPFN's 10,000-row pre-training limit, so the deterministic episode-grouped
subsampler never fired; it remains in place with a recorded `subsampled: false`.

## Rebuilding the feature table without JAX

`artifacts/phase2/cache/` is not versioned, so the tables had to be reconstructed. The
eight cross-fit HMM population checkpoints *are* versioned, and
`phase2_predict.hmm_summary` is the portable float64 recurrence already checked against
Dynamax, so `slot_model/phase2_tables.py` rebuilds both tables with numpy and the
standard library in about 10 seconds.

This is verified, not asserted. The rebuilt validation table is fed through all eight
committed CatBoost checkpoints and must reproduce the committed
`artifacts/phase2/predictions/cb_*.csv` rows; the worst deviation is **4.44e-16**, pure
float64 rounding. `scripts/tabpfn_train.py` runs that gate before fitting anything and
records it in `artifacts/tabpfn/rebuild_verification.json`.

## Development scores

| Predictor | Points / 60 | Mean ratio | Stable | Short history | Drift | Shared change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current blend (bar) | 31.702332 | 0.892605 | 0.877050 | 0.954858 | 0.935042 | 0.803472 |
| tabpfn alone | 31.914042 | 0.880050 | 0.842299 | 0.951117 | 0.938496 | 0.788289 |
| blend, TabPFN w=0.15 | 31.857008 | 0.883416 | 0.863762 | 0.947078 | 0.928025 | 0.794800 |
| blend, TabPFN w=0.25 | 31.936596 | 0.878722 | 0.856487 | 0.943306 | 0.924825 | 0.790272 |
| blend, TabPFN w=0.35 | 31.997015 | 0.875175 | 0.850478 | 0.940667 | 0.922806 | 0.786748 |
| free five-component simplex | 32.076118 | 0.870551 | 0.843443 | 0.931748 | 0.922713 | 0.784298 |

Lower ratios and higher points are better. The organizer's own
`organizer/check_submission.py` was run on each written OOF CSV and agrees with our
scorer to within 1e-6 on all six rows; every CSV passes its format check.

### Which feature variant, and why it matters

| TabPFN input | Points / 60 | Mean ratio | Stable | Short history | Drift | Shared change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| observable only, 182 cols, seed 20261108 | 31.245054 | 0.920304 | 0.8990 | 0.9451 | 1.0139 | 0.8233 |
| observable only, 182 cols, seed 20261109 | 31.125575 | 0.927675 | 0.8571 | 0.9920 | 1.0361 | 0.8255 |
| observable only, seed average | 31.296775 | 0.917130 | 0.8697 | 0.9602 | 1.0194 | 0.8191 |
| + HMM summary, 189 cols, seed 20261108 | 31.805085 | 0.886491 | 0.8569 | 0.9436 | 0.9545 | 0.7909 |
| + HMM summary, 189 cols, seed 20261109 | 31.891004 | 0.881408 | 0.8365 | 0.9684 | 0.9289 | 0.7917 |
| + HMM summary, seed average | 31.914042 | 0.880050 | 0.8423 | 0.9511 | 0.9385 | 0.7883 |

**TabPFN's advantage is not independent of the Dynamax work.** On the 182 observable
columns alone it scores 31.296775, *below* the 31.702332 bar, and its drift ratio exceeds
1.0 — worse than the frozen pooled reference on that condition. Adding the seven portable
HMM summary columns moves it to 31.914042 and pulls drift to 0.9385. The regime posterior
the HMM supplies is what makes this model class competitive here; a TabPFN-only pipeline
would have been a regression. All four development folds selected the `hmm` variant.

Seed averaging is also doing real work: 31.805 and 31.891 individually, 31.914 averaged.
With `n_estimators=1` that averaging is the only ensembling in play.

### Per fold

| Fold | current blend | tabpfn | w=0.15 | w=0.25 | w=0.35 | free |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 31.6923 | 31.8759 | 31.8549 | 31.9375 | 31.9990 | 31.9478 |
| 1 | 31.6503 | 32.0995 | 31.8402 | 31.9436 | 32.0281 | 32.1814 |
| 2 | 31.6766 | 32.1054 | 31.8600 | 31.9597 | 32.0409 | 32.1798 |
| 3 | 31.7747 | 31.6100 | 31.8627 | 31.8993 | 31.9182 | 31.9958 |

Every blended variant improves 4/4 folds. TabPFN alone improves 3/4 and loses fold 3,
which is the honest shape of a single high-variance model.

### Paired comparisons

| Paired comparison | 95% normalized-error difference interval |
| --- | --- |
| tabpfn minus current blend | [-0.037311, 0.014397] |
| blend w=0.15 minus current blend | [-0.013029, -0.005034] |
| blend w=0.25 minus current blend | [-0.020219, -0.007018] |
| blend w=0.35 minus current blend | [-0.026141, -0.007793] |
| free simplex minus current blend | [-0.035780, -0.007281] |

Bootstrap: 2,000 paired whole-episode draws stratified by condition, reference
denominators recomputed in each draw, seed 314159 — identical settings to the phase-two
comparisons. Intervals are exploratory and unadjusted for selection and multiple
comparisons; they do not restore independence after development exposure.

**TabPFN alone does not separate from the current blend** — its interval spans zero. The
blends do. The useful reading is that TabPFN is complementary, not that it is
individually better, even though its point score is higher.

## Does it clear the project's own gate?

The predeclared replacement gate in `artifacts/phase2/protocol.json` asks for a
**0.5-point** gain, 3/4 folds improved, no condition ratio regression beyond 0.10, and
consistency across seeds. Measured against that gate:

| Check | Required | Free simplex | w=0.35 |
| --- | --- | --- | --- |
| Points gain | >= 0.50 | +0.3738 **fails** | +0.2947 **fails** |
| Folds improved | >= 3 | 4 passes | 4 passes |
| Worst condition regression | <= 0.10 | none, all four improve | none, all four improve |
| Seed consistency | both | both seeds averaged in every variant | same |

So TabPFN clears the task's 0.15-weight bar comfortably and clears three of the four
frozen gate conditions, but **not** the 0.5-point replacement threshold this project set
for itself before seeing these numbers. That threshold was written for replacing the
champion outright; whether a 0.37-point gain justifies adding torch to the inference path
is a deployment decision, recorded here rather than resolved by relaxing the gate.

## Cost

| Step | Time | Notes |
| --- | --- | --- |
| Feature rebuild | 10 s | cached afterwards in `artifacts/tabpfn/cache/` |
| Rebuild verification gate | ~20 s | 8 CatBoost checkpoints re-scored |
| Four TabPFN fits + validation predictions | 973 s | 2 variants x 2 seeds, `n_estimators=1` |
| Full offline test inference, 1,536 queries, w=0.35 | 580 s | 2 threads, `HF_HUB_OFFLINE=1` |
| Same pipeline at `--tabpfn-weight 0` | 15 s | torch never imported |

`n_estimators` is the dominant cost and the reason it is 1. On this table, predicting 768
rows against the 3,134-row context took 118 s at `n_estimators=1`, 305 s at 2 and 950 s
at 4. Four estimators alone would have exceeded the 30-minute two-core budget, so seed
averaging over two seeds buys ensemble diversity far more cheaply.

TabPFN adds roughly 38x to inference wall-clock (15 s to 580 s). That is the real
complexity cost to weigh against +0.29 to +0.37 points.

## Leakage audit

`scripts/tabpfn_audit.py`, recorded in `artifacts/tabpfn/leakage_audit.json`:

- **Post-cutoff invariance.** 2,688 adversarial plays were appended at ticks
  `cutoff .. cutoff+7` for 336 machines across 42 episodes — the target machines *and*
  their episode neighbours — and the whole feature table was rebuilt. Both the 182-column
  observable table and the 7-column HMM summary came back **bitwise identical**, max delta
  0.0. This is the strong form: future rows provably cannot move a feature, not merely
  that they were filtered.
- **Lag arithmetic.** For 64 sampled queries, all 48 `lag{k}_*` values per row were
  recomputed from an independent tick-indexed dictionary using `cutoff - k`, never a row
  offset. 0 failures. A missing tick presents as `lag{k}_missing`, never as 0x.
- **Window containment.** 0 of 768 examples hold a tick at or above their own cutoff.

## Component verification

`scripts/tabpfn_verify.py`, 13/13 passing, recorded in
`artifacts/tabpfn/verification.json`. These are Python test outputs about our wrapper on
synthetic tables, kept separate from the competition scores above, and they check the
contract the blend depends on rather than predictive quality: bitwise determinism across
refits, a different seed genuinely changing the forward pass, five-column simplex output
summing to 1 within 2.2e-16, exact save/load round trip, enforcement of the stored
context hash, whole-episode and order-independent subsampling, a payout absent from
context still receiving a finite column, and the feature ceiling being enforced.

## The shipped component is the measured component

`scripts/tabpfn_inference_check.py` loads `artifacts/blend_release/model/tabpfn` through
`TabPFNEnsemble` — the exact path `predict_blend.py` takes — and requires it to reproduce
the seed-averaged predictions that were scored above. Max deviation **1.11e-16** over 768
rows, offline, recorded in `artifacts/tabpfn/inference_checks.json`. The evaluated model
and the deployable model cannot drift apart unnoticed.

## Reproduce

```sh
DATA=/absolute/path/to/main-participants-60-40/public
.venv/bin/python -m pip install -r requirements-tabpfn.lock
.venv/bin/python scripts/tabpfn_fetch_checkpoint.py          # once, needs network
.venv/bin/python scripts/tabpfn_audit.py   --data "$DATA"
.venv/bin/python scripts/tabpfn_verify.py
.venv/bin/python scripts/tabpfn_train.py   --data "$DATA" --n-estimators 1
.venv/bin/python scripts/tabpfn_evaluate.py --data "$DATA"
.venv/bin/python scripts/tabpfn_inference_check.py --data "$DATA"
```

Offline prediction, with the component at its measured weight:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python predict_blend.py --data "$DATA" \
  --out /tmp/submission.csv --split test --tabpfn-weight 0.35
```

`--tabpfn-weight` defaults to 0, and at 0 `predict_blend.py` reproduces
`artifacts/blend_release/submission.csv` byte-for-byte. The wiring is additive: no frozen
recipe, internal blend weight, preprocessing or calibration was refitted, and the
deployed default in `artifacts/blend_release/model/deployment.json` is unchanged.

## What is not established

- No untouched holdout. All 96 validation episodes were previously examined, so these
  points are development evidence and may overstate test gains.
- The free simplex weight of 0.566 is fitted on validation residuals. Only the fixed
  weights 0.15/0.25/0.35 are free of that fit, and 0.35 is the edge of the requested grid,
  so the optimum may lie beyond it and was not searched.
- TabPFN carries no sample weights, so the equal-episode/equal-machine normalization the
  CatBoost `Pool` applies is **not** reproduced in its context. Recorded in each
  `config.json` as `weight_handling`; its effect was not measured.
- The variant comparison above shows the gain *depends on* the HMM summary columns, but it
  does not isolate which of the seven columns matter, nor which observable features
  TabPFN exploits differently from CatBoost on the identical table. The gain is
  attributable to the model class given those inputs; no finer ablation was run.
- `n_estimators=1` was chosen for cost, not quality. Whether 2 or 4 estimators would
  score better was measured only for time (305 s and 950 s per predict), never for points.
