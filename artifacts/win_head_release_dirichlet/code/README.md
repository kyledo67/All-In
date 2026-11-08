# House Edge submission - WinHead-TwoHead

## Entry

- Team ID: [fill in before upload]
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: `submission.csv`, written by the command below to `/output/submission.csv`
- Forecast SHA-256: `2e1ccc789882327cd686c3a7f6d162c34f9bbcdde18d87bba35158b04fc2fab6`
  (the development copy is `artifacts/win_head_release_dirichlet/test_submission.csv`;
  `check_submission.py` reprints this hash)
- Code version: repository commit `ad4aaff` plus the uncommitted win-head work
- Environment: `"image": "python"` in `reproduce.json`; extra packages pinned in
  `requirements.txt` at this archive's root

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
- Measured runtime and hardware: **18.6 s** for the
  1536 test queries and 9.6 s
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
  (`artifacts/win_head_release_dirichlet/verification.json`).
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
  6.660e-05, max absolute sum error
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
