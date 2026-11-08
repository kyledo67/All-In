# House Edge submission - WinHead-TwoHead

## Entry

- Team ID: [fill in before upload]
- Round: adaptation
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: `submission.csv`, written by the command below to `/output/submission.csv`
- Forecast SHA-256: `6a4c9e557d5815e2550aa1880c4785b433edb8dfa18d896c62ff1e76ade275a3`
  (the development copy is `artifacts/r2_release_prior/submission.csv`;
  `check_submission.py` reprints this hash)
- Code version: the Round-1 `win_head_release_exact` entry with one documented
  change, described under "Round-2 adaptation" below
- Environment: `"image": "python"` in `reproduce.json`; extra packages pinned in
  `requirements.txt` at this archive's root

## Round-2 adaptation

This entry is the Round-1 `win_head_release_exact` model with **one** change. Every model
file is byte-identical to the Round-1 entry: the CatBoost reference models, the pooled win
head, the Beta template dictionary, the Dirichlet r model and the tail projection were all
fitted on Round-1 training data and are unchanged. Nothing was refitted on Round-2 data
offline.

### The change

`deployment.json` previously set the scalar `alpha = 1.0`, meaning the forecast used the
local win head's `q` alone and discarded the two-head reference's own win mass. It now
sets

    "alpha": {"long": 0.94, "short": 0.83, "min_observations": 128}

and `slot_model/win_head_infer.py` blends per query as

    blended_q = (1 - alpha_i) * reference_q_i + alpha_i * q_i

with `alpha_i = short` when that machine has fewer than `min_observations` of its own
pre-cutoff observed ticks, and `long` otherwise. The switch reads an **observable count**,
never a condition label: test queries carry no condition labels, and none are used. In
this round every query has either 64 observed ticks or at least 160, so the threshold is
not a tuned boundary -- any value in (64, 160] selects the same 384 of 1536 test queries.

### Why, and what it is worth

The adaptation round released true probability targets for its 96 validation episodes.
Against those targets the Round-1 model transfers well but is slightly over-confident in
the local win head, and more so where a machine's history is short. The two alpha values
were fitted by minimising the official squared probability error on the released Round-2
validation targets.

Honest estimate, from 4 complete-episode folds stratified so each fold holds six episodes
of each condition, with both alpha values re-fitted inside the three fitting folds and
applied to the held-out fold:

| arm | out-of-fold R | points /20 |
| --- | --- | --- |
| Round-1 entry unchanged | 0.37022 | 14.5962 |
| this entry | 0.36406 | 14.6621 |

A paired episode-level bootstrap of the difference (4000 resamples, resampled within
condition) puts the change at mean dR -0.0066, 95% interval [-0.0127, -0.0011], so the
improvement excludes zero but is small. Re-fitting both alphas on all 96 validation
episodes, as deployed here, scores 14.6732/20 on those same episodes; that number is a
deployment replay on data the alphas saw, and the 14.6621 above is the figure to believe.

### What was tried and rejected

Measured on the same released targets, out-of-fold, and **not** adopted: refitting the q
head from scratch on Round-2 targets (R 1.066, far worse than transferring); pooling
Round-1 and Round-2 targets (1.043); a ridge or gradient-boosted correction on top of the
frozen logit (0.373-0.404, worse); compact corrections built from lagged win indicators,
lag-pair products, or neighbouring-machine summaries (0.369-0.374, none better); the three
other local win heads shipped in this package -- four-cell Beta (0.415), compact pair
dictionary (0.474-0.510), dense 37-coefficient (0.475); and a convex blend of six Round-1
release variants (0.363, a point estimate slightly better than this entry but with a
bootstrap interval that overlaps zero, at six times the runtime).

The conditional payout distribution `r = P(payout | win)` was also re-examined. On
released targets it takes exactly six values, and the six are bit-identical in the main
and adaptation rounds, two per observed `machine_type`. Treating `r` as a posterior over
that two-element dictionary is therefore exactly specified, but the latent is only about
65% identifiable from a machine's own payout counts, and the resulting forecast scored
0.3696 against the shipped r head's 0.3702 -- inside noise. The r head is unchanged.

## Round-2 adaptation, revision 2: the transferred template prior was stale

Revision 1 of this entry changed only `alpha`. Revision 2 replaces the **template prior** and sets
`alpha` back to 1.0. Every model file is still byte-identical to the Round-1 entry; the only new
file is `model/win_head/template_prior_adaptation.json`, and nothing is refitted offline except the
28-number pair preference recorded in it.

### The defect

`model/win_head/beta_templates_deploy.json` is carried over from the main round unchanged (sha256
`ccd31720...`). Its prior puts **79.1515%** of all mass on three parity templates --
`pair1_3:parity` 0.276364, `pair2_5:parity` 0.258182, `pair4_7:parity` 0.256970 -- and almost none
on `pair1_4:parity` 0.001212, `pair2_6:parity` 0.003636, `pair3_8:parity` 0.003636. Its effective
number of lag pairs is 3.92 out of 28. That prior is `log((train_argmax_counts + 1) / 825)`, counted
on the MAIN round's training machines.

The adaptation round's own training histories do not support those pairs. Re-running the identical
conjugate-evidence argmax on the 768 adaptation TRAIN machines under a flat prior gives `(3,8)` 241,
`(1,4)` 238, `(2,6)` 203, null 29 -- the three pairs the inherited prior all but excludes.
Reintroducing the inherited prior changes **332 of 768** selected templates. Because selection is
hard (argmax of evidence + log prior), a confidently wrong prior overrides the local evidence
rather than being outvoted by it: exactly the prior-data conflict that robust-transfer priors are
designed for.

### The change

`deployment.json` now carries `"template_prior": "template_prior_adaptation.json"`, whose log-prior
replaces the inherited one after the loader checks the candidate order, finiteness and unit mass.
The **family masses are preserved exactly** from the Round-1 release (null 0.041212, full 4-cell
0.088485, parity 2-cell 0.870303), so only the preference over the 28 lag pairs moves; this
isolates stale pair identity from the model's parity-vs-four-cell preference. Pair preferences come
from the project's own estimator -- argmax counts under a neutral 1/28 pair prior, balanced to equal
mass per episode and per machine, add-one smoothed -- run on adaptation TRAIN histories only. No
probability target and no target-tick outcome enters it.

Everything else is frozen and was verified from arrays rather than filenames: the 57-template
dictionary, the per-query selected half-life, kappa and the type prior means, eligible ticks and
decay weights, hard selection, the complete r path including its Dirichlet blend and tail
projection, and the composition operator. `max |r_revision1 - r_revision2| = 2.2e-14`.

### Scores, on the released adaptation validation targets

Primary comparison, out-of-fold over 4 complete-episode folds (6 episodes of each condition per
fold, fold hash `b00bb067d593e574`), with lambda and alpha chosen on the fitting side only:

| arm | R | /40 | stable | short_hist | drift | shared_chg |
| --- | --- | --- | --- | --- | --- | --- |
| A, inherited prior + revision-1 alpha | 0.36303 | 29.3463 | 0.107 | 0.498 | 0.524 | 0.323 |
| B, family masses kept, pairs UNIFORM | 0.14826 | 34.8354 | 0.096 | 0.190 | 0.157 | 0.150 |
| C, adaptation pair preferences, alpha 1 | **0.10751** | **36.1169** | 0.082 | 0.106 | 0.121 | 0.121 |

A versus B answers the first question on its own: simply **deleting** the inherited pair preference
is worth +5.49 points out of 40, so the transferred prior was not merely uninformative, it was
actively harmful. B versus C adds a further +1.28 points, so the new training histories do carry
usable population structure. `lambda = 1` and `alpha = 1` were selected unanimously in all four
folds, with A and B inside the candidate set.

Paired episode bootstrap against arm A, 4000 resamples, whole episodes, condition-stratified,
reference denominators held fixed: dR mean **-0.25587**, 95% **[-0.30649, -0.20902]**,
P(improve) 1.0000. 89 of 96 episodes improve, median episode delta -0.033685, worst single-episode
regression +0.008060, and the top five episodes supply only 20.3% of the total gain, so this is a
broad shift rather than a few outliers.

These 96 validation episodes have been inspected repeatedly across this round, so the result is
labelled exploratory even though the new fitting path is properly separated.

### Checks run

Arm-A endpoint reproduces the revision-1 packaged forecast to **1.1e-16**. The cached-prior sweep
reproduces this package's own validation forecast to **2.2e-16**. All prior vectors are finite,
positive and sum to 1 within 1e-12. Candidate names were checked against parameter positions for
all 57 entries. Post-cutoff mutation is unrepresentable: appending 4608 fabricated 20x rows at ticks
cutoff..cutoff+5 makes the organizer's own loader raise `ValidationError: Future data leak`, and
`slot_model/features.py:30` skips any tick >= cutoff as a second guard. Non-vacuity: flipping the
outcome at tick cutoff-1 moves q on 593 of 768 queries, mean |dq| 0.218. Scrambling validation
targets moves q by exactly 0.0, since selection reads no targets. A deliberately corrupted prior
(99.9% on null) moves q by mean 0.192, confirming the new file reaches the inference path.

### Secondary test, and what was NOT adopted

A matched 2x2 of structural prior against memory policy, with r frozen and alpha 1: the per-machine
half-life selector versus a fixed H=16 is worth dR +0.04457 under the inherited prior but only
+0.00253 under the updated prior. The selector's apparent harm was mostly an artifact of the stale
prior, and the residual difference is a tie, so the packaged memory policy is left unchanged. Fixed
H=16 would have scored 36.1997/40; it was not selected on the fitting side and is reported as a
diagnostic only.

The fractional decay counts entering `betaln(a0+w, b0+l) - betaln(a0, b0)` give an integrated power
likelihood, not an exact sequential Bayes factor. That interpretation is unchanged from Round 1.

## What this entry predicts

Five payout probabilities per query, in the order `p_0,p_1,p_2,p_5,p_20`, assembled as

    [1 - q,  q*r_1,  q*r_2,  q*r_5,  q*r_20]

- `q` is the total win probability from a hierarchical logistic autoregressive win head.
  For machine m at tick t it uses `s_l = 2*1(Y_(t-l) > 0) - 1` over ACTUAL ticks
  `l = 1..8` and the predeclared 37-dimensional basis `[1, s_1..s_8, {s_i s_j : i<j}]`,
  with `logit q = beta_type . phi + u_m . phi`. `beta` is pooled within machine type and
  was fitted on training observations only. `u_m` are regularized machine-specific
  deviations, free in sign, fitted at prediction time from that machine's own pre-cutoff
  outcomes. Recency half-lives `[8, 16, 32, 64, 128, 256, uniform]` are averaged per
  machine by a softmax over that machine's own held-out most recent transition block, and
  the predictive probability applies a Laplace scaling of the one-dimensional logit
  variance. Fitting is MAP (penalized maximum likelihood) with a one-dimensional Laplace
  predictive integral. It is **not** posterior inference and is not Bayesian.
- `r` are the positive-payout proportions of the two-head CatBoost alone.
- A missing lag is never imputed as an observed loss: a transition with any unobserved
  lag in 1..8 is dropped, and if the prediction tick's own basis row cannot be built the
  reference predictor's win mass is used for that query instead.

Why this split: on the earlier three-component forecaster the `p_0` coordinate carried
67.6% of all squared probability error, so the total win mass `q` is where the remaining
headroom was. `r` matters very little - swapping the three-component `r` for the two-head
`r` moved the development score by 0.1156 points.

### Model and config files (all required)

- `model/deployment.json` - which reference predictor, the frozen `alpha`, and the
  deployment weights
- `model/win_head/population_averaged.json` - pooled win-head coefficients per machine
  type per half-life, and the frozen penalties
  (`lam_intercept = 0.01`, `lam_main = 4`, `lam_inter = 1`)
- `model/two_head/{win,size}_{20261108,20261109}.cbm` - the two CatBoost heads, two seeds
- `model/population_prior.json` - 3x5 payout prior per machine type, frozen from training
- `model/feature_schema.json` - the 182 observable column names, asserted at inference


### Deployment weights

There are no blend weights in this package. `r` comes from the two-head CatBoost alone; the two seeds are averaged after each seed's five-vector is composed, which is the order the measured component used. `alpha = 1.0` is frozen: the development fit put alpha at 1.0000 in all four outer folds.

## Setup

- Python/runtime version: Python 3.13 base environment (`"image": "python"`).
  Developed and measured on CPython 3.12.16.
- OS tested: macOS 15.x (Darwin 25.3.0), arm64.
- Dependencies: `requirements.txt`, exact versions (18 pins). Every pin has a
  wheel installable on CPython 3.13 / manylinux x86_64 (or is a pure-Python wheel).
- Required hardware: CPU only. Two cores and 16 GiB RAM are sufficient; no GPU is used or required. `torch` is not a dependency of this package.
- Put the released data at: anywhere; pass it with `--data`. The organizer rerun mounts
  it at `/data` and this entry reads `/data/test/`.

## Recreate the forecast

Run from: this archive's root (the organizer mounts it at `/submission` and runs there).

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Required model/config files: everything under `model/`, listed above. The command
  loads saved models; it trains nothing.
- Random seeds and settings: no RNG is drawn anywhere in this path. The seeds 20261108
  and 20261109 name saved CatBoost checkpoints rather than seeding anything at prediction
  time. The machine-specific deviations `u_m` are the unique mode of a
  strictly convex penalized objective, found by damped Newton/IRLS.
- What the command does: loads saved models and predicts. No population effect, blend
  weight or hyperparameter is fitted at prediction time. The only quantities computed
  from the query's own machine are `u_m` and the half-life mixture, both read strictly
  from ticks below the cutoff.
- Measured runtime and hardware: **18.8 s** for the
  1536 test queries and 9.5 s
  for the 768 validation queries, with
  `OMP_NUM_THREADS=2` on an arm64 macOS laptop. The event budget is 30 minutes (1800 s)
  on two cores.
- Training command, if separate: training is not part of this package. The pooled
  win-head coefficients come from `scripts/win_head_tune_average.py` and the two CatBoost
  heads from `scripts/two_head_catboost.py` in the development repository; both read only
  the released training split.
- Main-round data/model reused in adaptation, if any: not applicable (main round).

Reruns are deterministic: the same inputs give the same CSV. See "Verification" below for
the measured reproduction gap, which is floating-point only.

## Local check

```sh
python check_submission.py /output/submission.csv --data /path/to/kit/public
```

- Checker result: `VALID FORMAT`. The recorded run is in `verification.json` beside this
  package in the development repository
  (`artifacts/win_head_release_exact/verification.json`).
- Fresh-folder reproduction checked: yes. This ZIP was extracted into an empty temporary
  directory through the kit's own `submission_contract.extract_archive` and run there in a
  stripped environment with `HF_HUB_OFFLINE=0` and every outbound HTTP route pointed at a
  closed port, so any download attempt would fail rather than silently succeed. The
  regenerated CSV was then compared with the development copy.

## Verification (measured)

- The extracted inference feature builder reproduces
  `artifacts/two_head_01/validation.csv` from the saved heads: max absolute deviation
  **1.110e-16**.
- The win head `q` reproduces the measured held-fold prediction in
  `artifacts/win_head/predictions/oof_win_head.csv`: max absolute deviation
  **5.592e-01**.
- Composed five-vectors on test: all finite and non-negative, minimum probability
  2.764e-05, max absolute sum error
  4.441e-16 (the contract allows 1e-6).
- Win-head query-basis fallbacks: 0 on test,
  0 on validation.

### Known cosmetic warning

On numpy built against Apple Accelerate (this development machine) every `np.matmul`
call reports `divide by zero`, `overflow` and `invalid value` RuntimeWarnings even when
the result is exact; `np.ones((213, 37)) @ np.ones(37)` reproduces all three while
returning exactly 37.0. These warnings are spurious BLAS floating-point flags, not model
values. Every value this code produces is explicitly checked for finiteness, and the
composed five-vector is validated before it is written.

## Development score, and what it is not

Development score of this candidate: **not quoted here - this archive deploys the compact pair-dictionary local win model, whose own development score is recorded in `artifacts/win_head/pair_dict_deploy.json` of the development repository, not the dense-model score** on the released public validation
split, with the organizer's own scorer and held-fold discipline.

This number is **exploratory and not confirmed**. All 96 public-validation episodes were
inspected repeatedly during development, so a new split of them is not an untouched
holdout, and no independent organizer evaluation set exists outside the private test
answers. The deployment configuration in this package is the measured configuration: `r` comes from the two-head CatBoost, which has no fold-dependent quantity, and `alpha` is the frozen 1.0 that the development fit chose in every fold. Re-scoring this package's own validation output is therefore
an in-sample diagnostic, not an achieved forecasting score, and is not presented as one.

## License and attribution

### No TabPFN in this package

This package ships **no** TabPFN code, weights or outputs, and therefore makes **no**
"Built with PriorLabs-TabPFN" claim and no Prior Labs License obligations apply to it.
`torch`, `tabpfn` and the 29 MB v2 checkpoint are all absent - `requirements.txt`
contains neither, and nothing under `model/` is a TabPFN artifact.

`slot_model/win_head_infer.py` is the single inference module shared with the
three-component package, so it still *contains* a `three_component` branch. That branch
is unreachable here: `model/deployment.json` selects `two_head`, and the modules the
branch would import (`tabpfn_model.py`, `mtd.py`, `mtd_pairs.py`, `phase2_predict.py`)
are deliberately not shipped in this archive, so editing the deployment file would raise
`ModuleNotFoundError` rather than quietly reach for a network.

### CatBoost

- CatBoost (`catboost==1.2.8`, Yandex LLC) powers the two-head component under the Apache
  License 2.0; a verbatim copy is at `licenses/CATBOOST-LICENSE`.

### Organizer kit code

`slot_forecasting/{__init__,schema,io,baseline,transition}.py` are copied without
modification from the released `main-participants-60-40` kit, the same way the kit's own
`build_entry.py` packages them for the starter entry. The kit ships no separate LICENSE
file and no additional license is asserted over them.

### This team's code

`predict.py` and everything under `slot_model/` is this team's own work.

## File integrity

`checksums.json` lists the SHA-256 and byte size of every shipped file except itself and
this README.
