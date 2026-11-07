# Phase-two feature challengers

**The deployed champion is unchanged.** The original verified shared HMM ZIP and CSV remain the submission. The planned search is complete; partial pooling was not reopened.

All scores below use the organizer calculation on concatenated held-development-fold predictions. These are the same 96 previously examined validation episodes, regrouped for configuration and blend selection—not an untouched evaluation. No new independent labeled data was available.

| Predictor | Points / 60 | Mean ratio | Stable | Short history | Drift | Shared change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hmm | 31.163114 | 0.925353 | 0.905053 | 1.010810 | 0.976225 | 0.809325 |
| local | 30.912771 | 0.940945 | 0.969838 | 1.002831 | 0.939423 | 0.851689 |
| cat_observable | 31.398446 | 0.910923 | 0.914529 | 0.964984 | 0.949548 | 0.814629 |
| cat_hmm | 31.551951 | 0.901626 | 0.866389 | 0.965570 | 0.958078 | 0.816465 |
| ensemble | 31.702332 | 0.892605 | 0.877050 | 0.954858 | 0.935042 | 0.803472 |

The original HMM’s earlier 48-episode holdout score remains 30.759921/60; that holdout is now development evidence. The phase-two HMM score 31.163114 uses all public validation and must not be compared to 30.760 as if the difference were a model gain.

## Decision

CatBoost plus HMM features gains 0.388837 points over the champion, improves 3/4 folds and improves under both seeds. It does not meet the frozen 0.5-point replacement gate. Its 95% exploratory normalized-error difference interval includes zero. The simplex blend gains 0.539218 over the champion and 0.150381 over the best individual; it fails the frozen 0.25-point ensemble gate. The blend’s improvement over the champion has an exploratory interval below zero, but its incremental gain over the best individual does not. The predeclared gates are respected rather than relaxed after seeing these results.

| Paired comparison | 95% normalized-error difference interval |
| --- | --- |
| local minus HMM | [-0.019799868750107617, 0.05165722723023707] |
| cat_observable minus HMM | [-0.051165065396536376, 0.020978346090804168] |
| cat_hmm minus HMM | [-0.05462320220558217, 0.006773864936890731] |
| ensemble minus HMM | [-0.05459994256378038, -0.01149250872744677] |
| ensemble minus cat_hmm | [-0.020524643187666294, 0.002923409884205744] |
| CatBoost HMM features minus observable-only | [-0.0305973482914126, 0.011325889360525152] |

Bootstrap: 2,000 paired whole-episode draws stratified by condition; reference denominators are recomputed in each draw. Intervals are exploratory and unadjusted for selection/multiple comparisons. They do not restore independence after development exposure. Per-condition intervals, all fold scores, seed comparisons and selected settings are in evaluation.json. Per-episode absolute errors and paired differences are in per_episode.csv and paired_episode_differences.csv.

## Hypothesis outcomes

1. **Local recency: partial support for complementarity, no replacement.** Local history alone scores 30.912771, below the HMM, but improves the drift condition (ratio 0.9394 versus 0.9762). Every fold selects half-life 16 and prior strength 8. It receives positive blend weight in every fold. This supports some complementary information, not a proven mechanism of HMM overreaction.

2. **Observable feature CatBoost: modest development gain, insufficient practical evidence.** It scores 31.398446; the zero-payout/type/signal diagnostics motivated it, but the improvement is small and its interval overlaps zero. Depth 4/L2=8 is selected in every fold. The aggregate test does not identify which individual feature caused the gain.

3. **HMM-derived features: small, unstable incremental gain.** The matched variant scores 31.551951 versus 31.398446. Depth choice changes across folds; it beats observable-only in three folds, while the incremental paired interval overlaps zero. Neither feature stacking nor a positive ensemble coefficient establishes independent HMM value.

No added features or ensemble components are retained in deployment. The original 15-feature, two-state shared HMM and its standard-library inference remain the champion. All challenger checkpoints and experiments are preserved for inspection.

## Champion diagnostics

| Condition | Absolute error | Reference error | Normalized error |
| --- | ---: | ---: | ---: |
| stable | 0.177617 | 0.196251 | 0.905053 |
| short_history | 0.179259 | 0.177341 | 1.010810 |
| drift | 0.185321 | 0.189834 | 0.976225 |
| shared_change | 0.166367 | 0.205563 | 0.809325 |

| Payout | Squared error contribution | Share of total | Mean prediction minus truth |
| --- | ---: | ---: | ---: |
| 0× | 0.118389 | 66.83% | +0.026255 |
| 1× | 0.033875 | 19.12% | -0.011576 |
| 2× | 0.018966 | 10.71% | -0.010503 |
| 5× | 0.005470 | 3.09% | -0.004065 |
| 20× | 0.000441 | 0.25% | -0.000110 |

Category contributions normalized by each condition reference are in category_contributions.json. The following subgroup summaries average within each represented episode, then across represented episodes; they are descriptive subgroup errors, not standalone official points.

| Group | Queries | Episodes | Error | 0× bias |
| --- | ---: | ---: | ---: | ---: |
| machine_type=balanced | 234 | 92 | 0.204359 | +0.046171 |
| machine_type=jackpot | 265 | 92 | 0.089547 | +0.020013 |
| machine_type=steady | 269 | 90 | 0.239710 | +0.024131 |
| signal=-1 | 247 | 93 | 0.169125 | +0.021899 |
| signal=0 | 259 | 90 | 0.186101 | +0.008730 |
| signal=1 | 262 | 92 | 0.174530 | +0.059801 |
| n_history=short | 192 | 24 | 0.179259 | +0.017449 |
| n_history=long | 576 | 72 | 0.176435 | +0.029190 |
| coverage_40=40 | 768 | 96 | 0.177141 | +0.026255 |

| Observable diagnostic quartile | Error | 0× bias |
| --- | ---: | ---: |
| recent_win/quartile_1 | 0.073090 | +0.003130 |
| recent_win/quartile_2 | 0.196005 | +0.023631 |
| recent_win/quartile_3 | 0.225100 | +0.058621 |
| recent_win/quartile_4 | 0.217818 | +0.038768 |
| win_change/quartile_1 | 0.181727 | +0.061129 |
| win_change/quartile_2 | 0.190647 | +0.031389 |
| win_change/quartile_3 | 0.148576 | +0.007002 |
| win_change/quartile_4 | 0.179772 | +0.019997 |
| distribution_change/quartile_1 | 0.135990 | -0.000763 |
| distribution_change/quartile_2 | 0.205697 | +0.020482 |
| distribution_change/quartile_3 | 0.179119 | +0.019363 |
| distribution_change/quartile_4 | 0.190938 | +0.050027 |
| lag_strength/quartile_1 | 0.245026 | +0.073754 |
| lag_strength/quartile_2 | 0.201638 | +0.015136 |
| lag_strength/quartile_3 | 0.157202 | +0.038964 |
| lag_strength/quartile_4 | 0.110007 | -0.030064 |

All recent 40 windows are fully observed. Short histories contain 64 observed rows while long histories contain their full 192–256 tick prefixes. Recent/older categorical disagreement correlates 0.144 with error, but the signed win-change correlation is only 0.072. A confident diagnosis of slow adaptation or overreaction is not established. Local conditional lag-strength error correlation is negative (-0.306), so it does not support a general claim that strong observed lag responses are missed. Rare 20× probability is slightly underpredicted on average.

## Complementarity and fitting

Query-level squared-error correlations are approximately 0.657 for HMM/local,0.671 for HMM/observable CatBoost, 0.756 for HMM/CatBoost+HMM, and 0.890 between the two CatBoost variants. Weight fitting additionally checks weighted signed residual similarity. Different errors justify the limited blend test, not automatic adoption. Fold weights are global, nonnegative and sum to one; condition labels only stratify/report and normalize the training objective. No prediction-time gating uses conditions.

The feature audit and fitting-data provenance are in phase2-protocol.md. Base models use only training outcomes. There are 768 actual queries and2,366 historical examples, with equal total episode/machine weight. Historical cutoffs and signal missingness are imperfect matches to real queries; there is no fabricated historical signal. Cross-fitted priors and HMMs exclude the entire owning episode. Early stopping uses nested train-only splits. All causal tests passed.

The fixed search is six local settings and two CatBoost configurations × two feature variants × two seeds. Eight early-stop and eight final training fits are complete. Some depth 6 runs hit 400 trees; the cap was not expanded. No additional search was launched after evaluation. A metadata-only resume fix preserves upstream provenance when cached feature matrices are reused; the exact originally executed scripts are retained under artifacts/phase2/source/.

| CatBoost run | Trees | Train + early-stop seconds | Validation prediction seconds |
| --- | ---: | ---: | ---: |
| cb_observable_0_20261108 | 214 | 7.027 | 0.0309 |
| cb_observable_0_20261109 | 307 | 8.464 | 0.0275 |
| cb_observable_1_20261108 | 400 | 29.608 | 0.0407 |
| cb_observable_1_20261109 | 396 | 28.544 | 0.0298 |
| cb_hmm_0_20261108 | 278 | 8.645 | 0.0336 |
| cb_hmm_0_20261109 | 335 | 10.139 | 0.0340 |
| cb_hmm_1_20261108 | 400 | 33.071 | 0.0379 |
| cb_hmm_1_20261109 | 390 | 53.872 | 0.0406 |

Cross-fitted feature preparation took35.24 seconds on this host; tree inference timings exclude feature preparation. Prepared sequences, feature matrices and model predictions remain cached. Training is CPU-only, with CatBoost restricted to two threads.

## Final handoff

Original CSV: `artifacts/final/submission.csv` — SHA256 `655c228fc0c767e66b58c9f57f7c35452aef0e2f9b520c8bbb89d1e30792f236`.
Original ZIP: `artifacts/final/entry.zip` — SHA256 `a35ecc99d4ffe7bc05af1da1a2b5da61ebc5472bf62b712264099802bff75738`.

A fresh extraction of that exact ZIP reproduced the CSV byte-for-byte in **2.215 seconds** with socket creation and third-party packages disabled. The prior organizer checker acceptance remains valid because every original file hash is unchanged. The checker was also rerun in this phase. This local macOS/Python3.13 test does not execute the organizer Linux container.

From the extracted original entry:

```sh
python predict.py --data /absolute/path/to/public --out /output/submission.csv
```

To reproduce the bounded investigation from the project root with its pinned environment:

```sh
.venv/bin/python scripts/phase2_train.py
.venv/bin/python scripts/phase2_evaluate.py
.venv/bin/python scripts/phase2_verify.py
.venv/bin/python scripts/phase2_report.py
.venv/bin/python scripts/phase2_log.py
```

The current scripts default to the supplied kit path. Executed snapshots are preserved separately. The manifest records original source HEAD plus content hashes, data/configuration/fold/dependency hashes, all saved checkpoints and predictions. Nothing was uploaded, and no Databricks infrastructure was provisioned. Local MLflow uses the existing tracking directory and records the parent champion.

## Saved challenger inference check

Observable features exactly match the cached training-run validation matrix. Portable HMM summaries agree with JAX within 6.66e-16. Re-loading both selected CatBoost seed checkpoints reproduces their saved validation forecasts within 1e-12. This validates the retained research code; no challenger is deployed.

| Validation inference component (768 queries) | Seconds |
| --- | ---: |
| Load and index | 0.686 |
| Local expert | 0.031 |
| Observable feature construction | 1.095 |
| HMM summaries | 0.628 |
| CatBoost observable, including required features and loading | 1.862 |
| CatBoost hmm, including required features and loading | 2.472 |
