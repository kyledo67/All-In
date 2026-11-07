# First targeted accuracy cycle: probability sharpness

The previously promoted equal average of four frozen recipes remains the
champion. This cycle tested **one** additional mechanism and did not change the
release CSV, ZIP, recipe weights, models, or HMM fallback.

The original 31.702332/60 is the organizer score on **combined out-of-fold**
predictions from the four frozen recipes. It is not an independent score for the
deployed equal average. All 96 validation episodes were previously inspected;
the new four-fold result below is exploratory development evidence.

## Diagnostic and hypothesis

The champion procedure's normalized error contributions by condition are
0.877050 stable, 0.954858 short history, 0.935042 drift, and 0.803472 shared
change. Stable and short-history errors are concentrated in 0× forecasts:
their 0× terms contribute 0.146675 and 0.160427 respectively to the overall
mean normalized error of 0.892605. Across all four conditions the 0× category
dominates; 20× contributes less than 0.001 to the normalized mean in each
condition. The strongest raw-error machine types are steady and balanced.
The overall 0× bias is positive in all four conditions; this is observed bias,
not proof about hidden simulator behavior.

**One testable hypothesis:** The recipe vectors may be slightly too sharp.
We tested one scalar power transformation, `p' = p**gamma / sum(p**gamma)`,
with gamma constrained to [0.8, 1.2]. Values below one soften all five
probabilities while preserving payout order and normalization. For each held
fold, gamma was fit on the other three folds by minimizing the exact organizer
normalized error using that held fold's saved recipe. The recipe itself had
been selected and weighted without the held fold's probability targets. A
fold on which gamma worsens the official score disproves universal benefit.
This is calibration on allowed development probability targets, not base-model
training. Condition labels enter scoring only and are absent from inference.

| Existing OOF evidence | Frozen recipes | Calibrated recipes |
| --- | ---: | ---: |
| Official points /60 | 31.702332 | 31.716775 |
| Mean normalized error | 0.892605 | 0.891743 |
| Stable ratio | 0.877050 | 0.876293 |
| Short-history ratio | 0.954858 | 0.953310 |
| Drift ratio | 0.935042 | 0.936538 |
| Shared-change ratio | 0.803472 | 0.800833 |

The absolute ratio reduction is 0.000862, or 0.097% of the champion ratio.
Gamma values were 0.9621, 0.9479, 0.9376, and 0.9752. Three folds improved;
fold 2 regressed by 0.0625 points. The paired whole-episode exploratory 95%
interval for candidate-minus-champion normalized error is
[-0.00556, +0.00421]. It includes both improvement and deterioration.

The small and inconsistent gain does not justify an added deployment
calibration artifact. The existing blend remains the release; no test
predictions or package were regenerated for this rejected candidate.

The organizer scorer independently reproduced the saved control result. The
combined OOF predictions, per-episode errors, condition/category diagnostics,
input and source hashes, fixed fold settings, runtime and the full result are in
`artifacts/calibration_cycle_01/`. Four recipe forecasts on validation were
computed in 11.38 seconds total; this is development runtime, not new test
inference runtime. The original clean offline release reproduction remains
5.75 seconds. No new independent dataset was available, and hidden-test
generalization remains unknown. No Databricks connection was available;
the run is logged to the existing local MLflow file store.
