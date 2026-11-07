# Round-one results

The machine-deviation patch is implemented and independently verified, but **rejected for final deployment**. Its predeclared paired holdout comparison does not establish a benefit over the shared model. The final CSV uses the unmodified shared HMM, exported through an equivalent standard-library forward pass.

The released kit specifies 60 main points and 40 adaptation points. All numbers below use the official episode- and condition-normalized scorer with a 60-point maximum; larger points and smaller ratios are better.

## Frozen comparisons

There are 48 development and 48 holdout episodes, 384 queries per partition, with 12 episodes per condition. The three positive precisions were compared on development only. Lambda=30 was frozen before holdout scoring. The saved rule retains the patch only if its paired 95% interval beats both the shared and unshrunk controls. The fallback (shared HMM) was also fixed in advance. The holdout was used for this confirmatory gate; no settings were changed afterward.

### Development

| Model | Points / 60 | Mean ratio | Stable ratio | Short-history ratio | Drift ratio | Shared-change ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 30.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| upstream | 31.524494 | 0.903282 | 0.888422 | 0.941983 | 0.964353 | 0.818369 |
| unshrunk | 27.784126 | 1.159507 | 1.091679 | 1.268749 | 1.249593 | 1.028005 |
| pooled_0.3 | 28.650598 | 1.094197 | 1.015873 | 1.176171 | 1.192263 | 0.992481 |
| pooled_3 | 30.463374 | 0.969578 | 0.902053 | 1.002808 | 1.067085 | 0.906368 |
| pooled_30 | 31.553202 | 0.901550 | 0.870290 | 0.935920 | 0.969671 | 0.830320 |

### Holdout

| Model | Points / 60 | Mean ratio | Stable ratio | Short-history ratio | Drift ratio | Shared-change ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 30.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| upstream | 30.759921 | 0.950590 | 0.923722 | 1.089176 | 0.989716 | 0.799747 |
| unshrunk | 27.562198 | 1.176895 | 1.185853 | 1.275352 | 1.220049 | 1.026324 |
| pooled_30 | 30.780538 | 0.949284 | 0.942411 | 1.070167 | 0.986568 | 0.797989 |

The JSON reports also retain absolute condition errors, reference errors and episode/query counts. All controls see identical queries and features. No CatBoost benchmark was present in the supplied project.

### Paired whole-episode bootstrap

2,000 resamples, stratified by condition, seed 314159. Each draw uses the same episodes for both models and the reference. Values are differences in normalized loss ratio; negative favors the first model.

| Comparison on holdout | 95% interval |
| --- | --- |
| pooled_30 minus baseline | [-0.118290, 0.018546] |
| pooled_30 minus unshrunk | [-0.297080, -0.164603] |
| pooled_30 minus upstream | [-0.019958, 0.017785] |
| upstream minus baseline | [-0.110972, 0.011337] |

Per-condition intervals are included in the corresponding JSON. These are unadjusted intervals for the frozen comparisons, not simultaneous guarantees. The shared HMM has a measured holdout gain over baseline, but the interval still includes zero. The experiment does **not** establish a statistically resolved competition improvement from the source modification.

The delivered model scores **31.163114/60** across all public validation episodes. This combines development and holdout, so it is descriptive, not a new independent test. The 1,536 test targets are hidden; no test score is claimed.

## Numerical verification

- Tiny clean upstream filtering/smoothing/enumeration maximum error: 2.78e-16.
- Actual inherited optimizer objective gradient versus finite differences: 2.45e-10.
- Frozen population and transition parameter drift during adaptation: 0.0.
- Clean-source population replay maximum parameter difference: 0.
- Post-cutoff outcome changes produce identical forecasts. Missing lags are rejected; sampled older rows are never treated as adjacent.
- Full validation/test probabilities agree between portable inference and Dynamax within 4.5e-16. Every output is finite, nonnegative, normalized and correctly ordered.
- Fresh-package offline reproduction and resource usage are recorded in `artifacts/final/reproduction.json`.

### Synthetic known-probability cases

64 machines per case, fixed seeds, same population and adaptation budgets. These are independently constructed cases, not competition-simulator reconstructions. The population is correctly specified in the identical and changing-regime cases, so shared inference is already optimal under the supplied model.

| Case | Shared error | Unshrunk error | Pooled (30) error | Pooled deviation RMS |
| --- | ---: | ---: | ---: | ---: |
| changing_regimes | 6.66214e-33 | 0.0780187 | 0.00145377 | 0.0491807 |
| identical | 3.85186e-34 | 0.0639933 | 0.0022404 | 0.0543925 |
| opposite_lag | 0.17422 | 0.0764171 | 0.0960562 | 0.0970615 |
| short_history | 3.85186e-34 | 0.410951 | 0.00120191 | 0.0307366 |

Shrinkage suppresses, but does not completely eliminate, spurious adaptation. It greatly reduces short-history overfitting. It helps relative to shared coefficients when the synthetic machines have opposite lag responses, but strong shrinkage underfits those responses and loses to unshrunk fitting there. It adds noise when a known shared model is already correct. These failures are part of the result.

## Approximations, compute and provenance

Training uses 3,768 nonoverlapping 32-tick observed windows, with eight earlier ticks required for lag inputs. Query inference uses the most recent 32 outcomes and their eight lags. Older complete windows can contribute to population training, but older query history is discarded. A known-regime synthetic sensitivity test found mean squared disagreement 0.000405 between 32-tick and full 128-tick inference (8 ticks: 0.002695; 16 ticks: 0.001370). This documents a cost of resetting the state prior; it was not used to tune the competition window. Unknown autoregressive inputs are avoided, not integrated.

Signal is observed only at the query cutoff. The HMM omits it rather than inventing a historical series. Machine type is assumed stable within an episode. Cross-machine evidence is not used. Lag features encode win/loss and four pairs; they do not model every payout-size interaction.

The population run uses 15 generalized EM iterations and 30 Adam steps per emission M-step; adaptation uses eight iterations with the same optimizer. The original population fit took approximately 4.36 seconds; validation shared inference 0.16 seconds; pooled adaptation about 1.53 seconds on this host, including JIT work for those paths. These are local timings, not two-core organizer benchmarks. Logged objectives did not decrease in this run, but the last population trace gain is about 42.08: convergence is not established. Parameter uncertainty is not integrated.

The incoming workspace already contained the source patch, pinned dependency environment, split list and frozen development finalist. They were preserved, audited and replayed; earlier split-creation history cannot be independently reconstructed from this session. No subsequent holdout tuning occurred. The organizer baseline replay prints full-validation baseline summaries after the finalist freeze; those summaries did not change candidate selection.

Dynamax supplied initialization, likelihoods, forward/backward inference, EM orchestration, priors, parameter freezing and optimization. The 38-line emission extension tests the pooling hypothesis without rewriting those components. No source change is credited with a final competition improvement. The rejected extension remains available for reproducible research; it is not in the deployed model.

No Databricks integration or credentials were configured in this workspace. The existing local MLflow fallback retains the experiment and final artifacts. No REA runtime was needed for available Python source. Source paths, exact dependency versions, licenses, data hashes and evidence boundaries are in the source map, ledger and artifact manifests.
