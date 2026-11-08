# House Edge submission - TabPFN-WinHead-ThreeComponent

## Entry

- Team ID: [fill in before upload]
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: `submission.csv` (written by the command below to `/output/submission.csv`)
- Forecast SHA-256: `ed5caacb79b8517071dab368ec1652c7e4fe99d9f79c957f3594b0482e78adb6`
  (of the copy produced in development at `artifacts/win_head_release/test_submission.csv`;
  re-print it with `python check_submission.py <csv> --data <kit>/public`)
- Code version: repository commit `f002c77` plus the uncommitted win-head work
- Environment: `"image": "python"` in `reproduce.json`; extra packages in `requirements.txt`

## What this entry predicts

Five payout probabilities per query, in the order `p_0,p_1,p_2,p_5,p_20`, assembled as

    [1 - q,  q*r_1,  q*r_2,  q*r_5,  q*r_20]

- `q` is the total win probability from a hierarchical logistic autoregressive win head.
  For machine m at tick t it uses `s_l = 2*1(Y_(t-l) > 0) - 1` over ACTUAL ticks
  `l = 1..8` and the predeclared 37-dimensional basis
  `[1, s_1..s_8, {s_i s_j : i<j}]`, with `logit q = beta_type . phi + u_m . phi`.
  `beta` is pooled within machine type and was fitted on training observations only;
  `u_m` are regularized machine-specific deviations fitted from that machine's own
  pre-cutoff outcomes at prediction time. Recency half-lives
  `[8, 16, 32, 64, 128, 256, uniform]` are averaged per machine by a softmax over that
  machine's own held-out most recent transition block, and the predictive probability
  applies a Laplace scaling of the one-dimensional logit variance. Fitting is MAP
  (penalized maximum likelihood) with a one-dimensional Laplace predictive integral.
  It is **not** posterior inference and is not described as Bayesian.
- `r` are the positive-payout proportions of a frozen convex blend of three components.

Why this split: on the earlier three-component forecaster the `p_0` coordinate carried
67.6% of all squared probability error, so the win mass `q` is where the remaining
headroom was. `r` matters very little - swapping the three-component `r` for the
two-head `r` moved the development score by 0.1156 points.

### Model and config files (all required)

- `model/deployment.json` - which reference predictor, the frozen `alpha`, and the
  deployment weights
- `model/win_head/population_averaged.json` - pooled win-head coefficients per machine
  type per half-life, plus the frozen penalties
  (`lam_intercept = 0.01`, `lam_main = 4`, `lam_inter = 1`)
- `model/two_head/{win,size}_{20261108,20261109}.cbm` - the two CatBoost heads, two seeds
- `model/population_prior.json` - 3x5 payout prior per machine type, frozen from training
- `model/feature_schema.json` - the 182 observable column names, asserted at inference
- `model/population.json` - frozen HMM whose 7 summary columns extend the observable table for TabPFN
- `model/mtd/change.json` - saved changepoint MTD, the base of the lag-7 pair-correction model
- `model/tabpfn/seed_*/{context.npz,config.json}` - saved TabPFN context tables, seeds and context hashes
- `model/tabpfn_checkpoints/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` - TabPFN v2 classifier weights

### Deployment weights

The three-component `r` is a FIXED convex combination, recorded here explicitly because test episodes carry no fold assignment:\n\n  - `mtd_lag7` = `0.19998414057692182`
  - `tabpfn_raw` = `0.4829584454464017`
  - `two_head` = `0.31705741397667647`\n\nThese are the **arithmetic mean of the four outer-fold simplex fits** recorded in `artifacts/audit/recompute.json` of the development repository, and they were **NOT tuned on anything further** - no search, no validation re-fit, no per-condition adjustment. The fourth component of that fit, the earlier frozen four-recipe blend, had mean weight 5.049528681541388e-17 and no test-time analogue (its measured column was a held-fold prediction), so it is dropped; the remaining three already sum to exactly 1.0 in float64, so dropping it changes each surviving weight by exactly 0.0.\n\n`alpha = 1.0` is also frozen: the development fit put alpha at 1.0000 in all four outer folds, i.e. the win head replaces rather than blends with the reference win mass.

## Setup

- Python/runtime version: Python 3.13 base environment (`"image": "python"`).
  Developed and measured on CPython 3.12.16, macOS 15 (Darwin 25.3.0), arm64.
- OS tested: macOS 15.x (Darwin 25.3.0), arm64
- Dependencies: `requirements.txt`, exact versions, union of the repository's frozen blend and TabPFN locks
  - resolved pin conflict: packaging 26.3 kept over 25.0 (from blend)
  - resolved pin conflict: pandas 2.3.3 kept over 3.1.1 (from blend)

- Required hardware: CPU only. Two cores and 16 GiB RAM are sufficient; no GPU is used
  or required (`device: cpu` in every saved TabPFN config).
- Put the released data at: anything; pass it with `--data`. The organizer rerun mounts
  it at `/data` and this entry reads `/data/test/`.

## Recreate the forecast

Run from: the ZIP root (the organizer mounts it at `/submission` and runs from there).

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Required model/config files: everything under `model/` (listed above). The command
  loads saved models; it trains nothing.
- Random seeds and settings: no RNG is drawn. The CatBoost seeds 20261108 and 20261109
  name saved checkpoints rather than seeding anything at prediction time; each saved TabPFN component carries its own `random_state` (20261108, 20261109) and runs single-threaded-deterministic on CPU with `n_estimators = 1`.
  The machine-specific deviations `u_m` are the unique mode of a strictly convex
  penalized objective, found by damped Newton/IRLS.
- What the command does: loads saved models and predicts. No fitting of population
  effects, blend weights or hyperparameters happens at prediction time. The only
  quantities computed from the query's own machine are `u_m` and the half-life mixture,
  both read strictly from ticks below the cutoff.
- Measured runtime and hardware: 356.1 s for the
  1536 test queries, and 205.4 s
  for the 768 validation queries, with
  `OMP_NUM_THREADS=2` on an arm64 macOS laptop. The event budget is 30 minutes
  (1800 s) on two cores.
- Training command, if separate: training is not part of this package. The pooled
  win-head coefficients come from `scripts/win_head_tune_average.py` and the two
  CatBoost heads from `scripts/two_head_catboost.py` in the development repository;
  both read only the released training split.
- Main-round data/model reused in adaptation, if any: not applicable (main round).

Reruns are deterministic: the same inputs give the same CSV. The one measured
reproduction gap is floating-point only; see "Verification" below.

## Local check

```sh
python check_submission.py <kit>/public/test/../../submission.csv --data <kit>/public
```

- Checker result: see `verification.json` beside this package in the development
  repository (artifacts/win_head_release/verification.json).
- Fresh-folder reproduction checked: yes. The ZIP was extracted into an empty temporary
  directory and run there with outbound network access forced to fail; the regenerated
  CSV was compared against the in-repo copy.

## Verification (measured, with the commands)

- Extracted inference feature builder reproduces `artifacts/two_head_01/validation.csv`
  from the saved heads: max absolute deviation **1.110e-16**.
- Win head `q` reproduces the measured held-fold prediction
  `artifacts/win_head/predictions/oof_win_head.csv`: max absolute deviation
  **8.882e-16**.
- Window of the composed five-vector: all finite, non-negative, max absolute sum error
  4.441e-16 on test
  (the submission contract allows 1e-6).

## Development score, and what it is not

Development score of this candidate: **40.705473 / 60 (R = 0.474003)** on the released public
validation split, scored by the organizer's own scorer with held-fold discipline.

This number is **exploratory and not confirmed**. All 96 public-validation episodes were
inspected repeatedly during development, so a new split of them is not an untouched
holdout, and no independent organizer evaluation set exists outside the private test
answers. The deployment configuration in this package differs from the measured
configuration in exactly one way: the three-component `r` uses the mean-of-folds weight
vector above instead of each fold's own fitted weights, because test episodes have no
fold assignment. Re-scoring this package's own validation output would therefore be an
in-sample diagnostic, not a forecasting score, and is not reported as one.

## License and attribution

## TabPFN attribution (required)

**Built with PriorLabs-TabPFN**

This package distributes the TabPFN v2 classifier model weights
(`model/tabpfn_checkpoints/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`). TabPFN is **not** Apache-2.0: it is licensed
under the **Prior Labs License Version 1.1 (May 2025)**, whose Paragraph 10 requires
that a distributor of the source or model weights (A) provide a copy of the license and
(B) prominently display "Built with PriorLabs-TabPFN". A verbatim copy of that license
is shipped at `licenses/PRIORLABS-TABPFN-LICENSE`, and the line above satisfies (B).
Paragraph 10 also asks that a distributed AI model built from TabPFN outputs carry
"TabPFN" at the start of its name, so this candidate is named
**TabPFN-WinHead-ThreeComponent**.

TabPFN upstream: https://github.com/priorlabs/tabpfn, pinned to `tabpfn==2.2.1`.

- CatBoost (`catboost==1.2.8`, Yandex LLC) is used for the two-head component under the
  Apache License 2.0; a verbatim copy is at `licenses/CATBOOST-LICENSE`.
- Organizer kit modules `slot_forecasting/{__init__,schema,io,baseline,transition}.py`
  are copied without modification from the released `main-participants-60-40` kit, the
  same way `build_entry.py` packages them for the starter entry. The kit ships no
  separate LICENSE file and no additional license is asserted over them.
- All other code in this archive (`predict.py`, `slot_model/*`) is this team's own work.

## File integrity

`checksums.json` lists the SHA-256 and byte size of every shipped file except itself
and this README.
