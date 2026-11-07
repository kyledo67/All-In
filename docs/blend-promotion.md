# Promotion: equal-weight ensemble of four frozen fold recipes

The user authorized test predictions as the arithmetic mean of the four existing
five-category probability vectors: p(q) = (p0(q)+p1(q)+p2(q)+p3(q))/4. The four
**deployment weights are fixed at 0.25 each**; nothing was fitted to select them.
Each recipe's internal blend weights, models, preprocessing, saved population
prior, feature schema and seeds remain unchanged. No new search, training,
calibration or optimization was performed.

## Policy change and existing evidence

The complete saved candidate is compared directly against the deployed shared
HMM. The previous requirement for another 0.25-point gain over the undeployed
CatBoost challenger is removed at the user's request. This is an explicit
post-evaluation policy change; the old protocol, evaluation and decision remain
preserved in artifacts/phase2/ rather than rewritten.

| Existing combined OOF evaluation | Shared HMM | Fold recipes |
| --- | ---: | ---: |
| Official main points | 31.163114 | 31.702332 |
| Mean normalized error | 0.925353 | 0.892605 |
| Stable normalized error | 0.905053 | 0.877050 |
| Short-history normalized error | 1.010810 | 0.954858 |
| Drift normalized error | 0.976225 | 0.935042 |
| Shared-change normalized error | 0.809325 | 0.803472 |

The existing gain is 0.539217 points. All condition errors improve in aggregate;
three of four evaluation folds improve. The saved paired whole-episode,
condition-stratified interval for fold-blend-minus-HMM normalized error is
[-0.05459994, -0.01149251], favoring the blend. It remains exploratory: validation
was already inspected, model/weight selection used development evidence, and the
promotion policy changed after the results were seen.

**Deployment:** equal-weight ensemble of four frozen fold recipes.

**Cross-validation evidence:** 31.7023316677/60 from their held-out, out-of-fold
predictions. The official scorer was applied once to all combined OOF predictions,
including episode averaging and condition-specific reference normalization.
It was not computed by averaging fold points; oof_score_audit.json records both
calculations for verification.

**The deployed four-recipe average itself has no independent validation score.**
It has not been scored on validation in this promotion. Some recipes used those
episodes when selecting weights, so a score of their validation average would
not be independent evidence. No hidden-test score is available. Equal averaging
does not establish superiority over every recipe on the hidden test.

## Immutable recipe identities

| Identifier | Saved fold | Deployment weight |
| --- | ---: | ---: |
| fold0-8f6f88ac207d | 0 | 0.25 |
| fold1-33b73b81eda7 | 1 | 0.25 |
| fold2-a69eceb838eb | 2 | 0.25 |
| fold3-063bbc8e6f94 | 3 | 0.25 |

The full recipe hashes and aggregation specification are in model/deployment.json.
Every original internal weight, including numerical boundary values, is preserved
in model/recipes.json. The six CatBoost checkpoints are reused across recipes
where appropriate. Both original CatBoost seeds (20261108 and 20261109), their
already-selected tree counts, the HMM parameters and the type priors are reused.
There is no additional calibration artifact in the original recipes.

All base model fits used public/train observed outcomes. Cross-fitted training
priors/HMM features excluded the owning episode; the original causal tests and
upstream fitting-episode lists were checked. Test prediction reads only test
queries and history, strictly pre-cutoff inputs, and never uses condition labels.
Public validation probabilities were used for prior model/weight selection as
recorded in the original protocol, not for training the base CatBoost models.

Reconstructing the combined OOF forecasts from the saved models gives the same
31.7023316677 score, with maximum probability difference 1.11e-16 from the archived
OOF CSV. This is numerical verification of existing evidence, not a new experiment.

## Prediction and packaging verification

All component calculations share a unique query-ID index. Serialized per-recipe
CSV files are read back and joined by query ID to check the final arithmetic mean.
Payout order is fixed at 0,1,2,5,20; each recipe and the final average must be finite,
nonnegative and normalized. The organizer's loader, writer, checker and archive
utilities are reused. No test-time recipe is selected by its fold score.

The release is under artifacts/blend_release/. `submission.csv` and `entry.zip`
are the upload pair. `aggregation_verification.json`, `test_check.txt`,
`reproduction.json` and `checksums.json` record final validation and hashes.
The packaging command requires a byte-identical replay from a clean extraction,
with read-only code/data, only test queries/history available, and Python network
connections/DNS blocked. It uses the pinned installed dependencies; organizer
Linux is not executed locally. No upload is performed.

From the extracted ZIP, after installing its pinned requirements:

```sh
python predict.py --data /data --out /output/submission.csv --model model
```

From the project root:

```sh
.venv/bin/python predict_blend.py --data /absolute/path/to/public --out /tmp/submission.csv
```

No REA, JAX, Databricks, MLflow or credentials are required for inference.

## Preserved fallback

The shared HMM CSV and ZIP remain at artifacts/final/submission.csv and
artifacts/final/entry.zip, with unchanged SHA-256 values:

- CSV: `655c228fc0c767e66b58c9f57f7c35452aef0e2f9b520c8bbb89d1e30792f236`
- ZIP: `a35ecc99d4ffe7bc05af1da1a2b5da61ebc5472bf62b712264099802bff75738`
