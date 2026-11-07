# Feature-based challenger protocol

The prior champion is frozen in `artifacts/phase2/champion/manifest.json`, including
its ZIP, CSV, parameters, configuration, dependencies and reproduction record.
Repository HEAD is the initial scaffold; the champion's implementation was
uncommitted. The saved ZIP's source manifest, rather than that commit alone,
identifies the exact executable source. The previous partial-pooling investigation
is unchanged and is not reopened.

All 96 public-validation episodes are previously examined development evidence.
No genuinely unopened labeled evaluation set exists in the supplied kit. Four
condition-stratified, SHA256-ordered episode folds are frozen in `folds.json`.
Whole episodes stay together. For each held development fold, configuration
selection and any convex blend fitting use the other three folds. Final reported
points come from concatenated held-fold predictions passed to the organizer
scorer, not an average of fold points. Hidden test generalization remains unknown.

## Three diagnostic hypotheses

1. **Local recency estimates may correct shared-model errors.** The frozen HMM's
   probability error is 0.177141; 0× contributes 0.118389 (66.8%). Error correlates
   0.374 with recent win frequency and 0.144 with recent/older categorical-distribution
   disagreement. Compare six local estimates: half-lives 16, 48, 128 ticks and
   prior strengths 8, 32. This tests useful recency adaptation; these correlations
   alone do not establish overreaction or slow adaptation.
2. **Observable context and local conditional estimates may reduce systematic
   errors.** Episode-averaged 0× bias is +0.0598 for signal +1, versus +0.0087 for
   signal 0; machine types also differ. Test one compact CatBoost family with
   actual categorical lags, local conditional probabilities, counts, change
   summaries, signal-at-query and missingness indicators. Observable lag strength
   correlates *negatively* with error (-0.306); this does not prove delayed
   dependencies are the principal failure. These features are a targeted richer
   context test, not evidence of a recovered simulator mechanism.
3. **Cross-fitted HMM information may complement observable features.** Test a
   matched CatBoost variant adding five HMM probabilities, terminal regime entropy
   and maximum state probability. Compare errors before trying global convex
   blending. Added HMM features do not by themselves demonstrate independent value.

There is no observed aggregate overprediction of the rarest payout: mean 20×
bias is -0.000110 and its squared-error contribution is only 0.000441. No jackpot
oversampling, class balancing, payout-value weighting or dedicated rare-event
model is justified by this diagnostic. The full machine-type, signal, coverage,
history-length and change/lag quartile summaries are in `diagnostics.json`.

## Feature audit and fitting provenance

| Family | Original deployed HMM | New pipeline | Fitting/availability |
| --- | --- | --- | --- |
| Actual lags 1–8 | Binary win indicators and four products | Five-category one-hot plus missing flags | Strictly earlier actual ticks; missing is neither zero nor adjacent sampled row |
| Missing-lag indicators | Rejects incomplete 40-tick suffix | Explicit flags | No learned imputation |
| Recent categorical counts/proportions | Absent as features | 8,16,40,128-tick windows, smoothed by population prior | Own pre-cutoff counts; population excludes own training episode |
| Exponential decay | Absent | Half-lives 16,48,128; decay mass and effective sample size | Actual tick distances, not row distances |
| Recent versus older distributions | Absent | Recent16 minus older40–128 | Both strictly before cutoff; missing old evidence shrinks toward prior |
| Machine conditional distributions | Indirect shared lag regression | Organizer context utility for lags1,2,4,8 and pairs1–2,4–8 | Local past contexts/outcomes only, decayed and shrunk; no global machine identity |
| Earlier related-machine summaries | Absent | Other machines' last16-tick category frequencies | Same episode, excludes own machine and every result at/after target tick |
| Signal | Omitted | Signal plus known/missing flag | Real-query signal only; historical signal value is unavailable, represented with flag0 and neutral numeric0 |
| Population prior | HMM population coefficients | Type-specific categorical prior | Training histories only; equal episode/machine mass, own episode excluded via cross-fit |
| HMM summaries | Model itself | Separate ablation: five probabilities, entropy, max regime probability | Upstream population fitted outside feature owner's episode; local filtering pre-cutoff only |

Existing `FrequencyForecaster.fit` assigns all historical outcomes to the query's
signal context. That organizer behavior is preserved as the baseline, but is not
reused for new priors: it would assume a historical signal association we cannot
verify. The new local expert reuses the organizer loader/writer and its categorical
context utility, with a small type-only population prior and explicit tick decay.
Machine type is assumed stable within an episode, as in the original HMM.

## Fixed budget and leakage controls

Six local settings; CatBoost depths4/6 with L2 values8/16, learning rate0.04,
maximum400 trees, patience50, CPU threads2, two seeds. Observable and HMM variants
have matched search budgets: eight early-stop fits and eight release-training
fits. Early stopping uses weighted multiclass log loss on a separate train-episode
fold, because train true probabilities are unavailable. Configuration selection
uses the official validation probability score, never classification accuracy.

CatBoost sees only train observed outcome labels, not validation probability
vectors. Historical targets are actual observed ticks in192–255, spaced by8,
at most8 per machine; complete preceding40 ticks are required for matched HMM
features. Alternate historical examples retain recent40 plus24 sampled older
ticks to approximate released short histories. Original short machines generally
contribute only their real query. Each machine's total weight is divided among
its examples, then each episode gets equal total mass. This is still an
approximation to the true query distribution, explicitly exposed by the signal
availability feature.

Four train-episode cross-fits generate priors and HMM features. Early-stopping
training uses nested cross-fits within the other three train folds, so its stop
fold never enters upstream fitting either. Full-train population artifacts
produce validation/test features. No validation labels enter base fits.

Cache identities include train/validation data, feature/dependency source,
configuration, split and dependency-lock hashes. Upstream model provenance lists
exact allowed episodes. The full train HMM checkpoint is reused unchanged.

A global nonnegative simplex blend is attempted only when training-fold weighted
residual cosine is below0.98 for some components. It has at most four weights,
no condition inputs and no flexible gate. Weights are fit on other development
folds only. Release weights may then use all development episodes, explicitly
reported as fitted development results. This is permitted model selection, not
training a base predictor on probability targets.

The frozen replacement gate requires at least0.5 points over the original,
improvement in at least3/4 folds, no condition ratio regression above0.10, and
both seeds improving. A blend additionally needs at least0.25 points over the
best individual, improvement in3/4 folds, and an exploratory paired interval
favoring it. Whole-episode stratified bootstrap intervals remain exploratory:
fold grouping does not undo earlier exposure or model selection.

CatBoost1.2.8 uses its existing Python implementation. API references:
[CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier),
[predict_proba](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict_proba),
[upstream repository](https://github.com/catboost/catboost).
No other model repositories are explored. Local MLflow continues the existing
tracking setup; Databricks restoration is not a dependency.
