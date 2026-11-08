# The hierarchical logistic win head

A separate model for one coordinate: the total win probability `q = 1 - p_0`. It
replaces the win mass of the existing three-component forecaster and keeps that
forecaster's split of the win mass across the positive payouts.

Headline, held-fold public validation, official scorer: **40.705473/60** against the
three-component reference's 32.260531. **The 50/60 target is not met; the gap is
9.294527 points.** Nothing is promoted. `artifacts/blend_release/`,
`artifacts/active_champion.json` and `artifacts/final/` are untouched.

All 96 public-validation episodes have been inspected repeatedly across this project,
so these are exploratory development numbers. No untouched organizer evaluation set
exists and nothing on this page is independently confirmed.

## Why the win mass, and nothing else

The work started from an error decomposition of the existing three-component candidate,
recorded in `artifacts/headroom/diagnostics.json`.

Squared-error share by payout coordinate, three-component reference:

| coordinate | p_0 | p_1 | p_2 | p_5 | p_20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| share of total squared error | 0.675804 | 0.179001 | 0.108780 | 0.033596 | 0.002818 |

The zero-payout coordinate carries 67.6% of all squared probability error, and its RMSE
as a win-mass estimate is 0.333689.

Five reference points, in points out of 60:

| quantity | points / 60 | what it is |
| --- | ---: | --- |
| current three-component candidate | 32.260531 | a measured forecast |
| best global convex blend of the four saved columns | 32.339891 | **in-sample**: one weight vector fitted on the same 96 episodes it is scored on |
| best per-query convex blend of the same columns | 38.415739 | **oracle**: weights chosen per query using the true probability vector |
| true win mass, candidate's positive-payout proportions | 57.292704 | **oracle**: q replaced by the truth |
| optimal win mass given those proportions | 57.386205 | **oracle**: q chosen per query to minimise the official loss |

Read these as bounds on where error lives, not as achievable scores. Only the first row
is a forecast. The second is in-sample and the last three use the held-out truth, so
none of them is attainable by any predictor. What they establish is the size of the
remaining room: reweighting the existing columns buys 0.08 points in-sample, while
getting `q` exactly right is worth about 25 points. Substituting the true win mass while
keeping the candidate's proportions `r` recovers 57.292704 of the 57.386205 available
from a perfect `q`, which is why this work changes only `q` and leaves `r` alone.

## The model

For machine `m` at tick `t`, with `s_l = 2*1(Y_{t-l} > 0) - 1` over **actual** ticks
`l = 1..8`, the predeclared local basis is

```
phi_t = [1, s_1..s_8, {s_i s_j : 1 <= i < j <= 8}]        1 + 8 + 28 = 37 dims
logit q_{m,t} = beta_{type(m)}^T phi_{m,t} + u_m^T phi_{m,t}
```

`beta` is pooled within machine type and fitted on permitted training observations.
`u_m` are regularized machine-specific deviations, free in sign, fitted using only that
machine's own pre-cutoff outcomes.

The pair products can represent XOR-type responses between two lags, which neither
additive MTD nor a heavily shrunk pair-cell correction can. The implementation check
`xor` in `scripts/win_head_verify.py` measures that separation directly: on a synthetic
machine with a true `s1*s2` effect, the interaction model separates agreeing from
disagreeing lag pairs by +0.6200 and the main-effects-only model by +0.0054.

### Exact objective

`slot_model/win_head.py:191` is the objective the solver minimises, over `theta`, with
`eta_i = offset_i + phi_i^T theta`:

```
0.5 * sum_j lam_j theta_j^2  -  sum_i w_i [ y_i eta_i - log(1 + exp(eta_i)) ]
```

- `y_i` is 1 when tick `i` paid out and 0 otherwise.
- `w_i = 2^(-(cutoff - 1 - t_i)/H)` is the recency weight; `H = None` means uniform.
- `offset` is zero for the population fit and `phi @ beta` for a local fit, so `u_m` is
  estimated as a deviation from the pooled prediction.
- `lam` is block-diagonal by coefficient group. Deployed values: `lam_intercept = 0.01`,
  `lam_main = 4`, `lam_inter = 1`.
- A coefficient group that a control is not allowed to adapt gets `lam = 1e12` instead
  of being removed, so every control in the ladder solves the same objective with the
  same solver.

This is convex in `theta` for positive penalties, so the mode is unique. It is solved by
damped Newton/IRLS: at most 60 iterations, step tolerance 1e-10, `1e-10` Hessian jitter
with an `lstsq` fallback, step-norm cap 10 and `eta` clipped to +/-30. The analytic
gradient agrees with central finite differences to a maximum relative deviation of
9.35e-10 over all 37 coordinates, and at the returned mode the gradient norm is 5.36e-15
with a minimum Hessian eigenvalue of 4.839.

### Fitting is MAP, not Bayesian inference

The point estimates are MAP (penalized maximum likelihood). The predictive probability
applies a Laplace approximation at the local mode and integrates the sigmoid over the
**one-dimensional** Gaussian uncertainty of the logit only, by the standard probit
scaling:

```
var    = phi_star^T H^{-1} phi_star        H = Hessian at the mode
q      = sigmoid( logit / sqrt(1 + pi*var/8) )
```

That is a single scalar correction that widens short-history predictions toward 0.5. It
is **not posterior inference**: no posterior over `beta`, no hyperparameter marginal, no
joint predictive. Nothing in this project should describe the win head as Bayesian. The
same statement is recorded in the `fit` field of every saved population file.

### Data rules

A transition at tick `t` is usable only when `t` and all of `t-1..t-8` are observed. A
missing lag never becomes an observed loss; the transition is dropped instead. All
arithmetic is on actual tick numbers, not array positions. The `missing` check measures
this: removing one tick dropped the 9 transitions that needed it as a target or as a
lag, and the removed tick never reappeared as an observed outcome.

## Matched control ladder

Four controls share feature availability, recency schedule, training data, tuning budget
and evaluation procedure; they differ only in which coefficient groups may deviate per
machine. Each candidate `q` is composed with the reference predictor's positive-payout
proportions as `[1-q, q*r_1, q*r_2, q*r_5, q*r_20]` and the complete five-vector is
scored by the organizer's scorer.

`alpha`, the blend weight between the reference win mass and the new one, is the only
validation-fitted quantity. For held fold `k` it is fitted on the fitting episodes of
fold `k` alone, in closed form, against the exact official full-vector loss with
reference normalization computed from those fitting episodes only. Held-fold predictions
are concatenated and the scorer runs once.

| control | points / 60 | R | stable | short_history | drift | shared_change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| existing three-component reference | 32.260531 | 0.859858 | 0.828155 | 0.922192 | 0.904481 | 0.784603 |
| population only, no deviations | 32.258644 | 0.859967 | 0.825959 | 0.923440 | 0.906187 | 0.784280 |
| local main effects only | 32.133579 | 0.867206 | 0.830668 | 0.937326 | 0.907263 | 0.793566 |
| local + 28 interactions, single H=16 | 38.845500 | 0.544580 | 0.467583 | 0.636295 | 0.585229 | 0.489215 |
| **local + interactions, per-machine H** | **40.705473** | **0.474003** | **0.211056** | **0.595597** | **0.590484** | **0.498876** |

`R` is the mean normalized loss ratio; lower is better for `R` and every condition ratio.

Fitted `alpha` per fold:

| control | fold 0 | fold 1 | fold 2 | fold 3 | points at alpha=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| population only | 0.1156 | 0.1957 | 0.1196 | 0.0595 | 31.209769 |
| local main effects only | 0.0343 | 0.3393 | 0.1729 | 0.0814 | 31.058532 |
| local + interactions, single H=16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 38.845500 |
| local + interactions, per-machine H | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40.705473 |

Two things follow from that table. The interactions are what matters: without them the
two shrunk controls are *worse* than the reference at `alpha=1` (31.209769 and 31.058532
against 32.260531) and only reach parity because `alpha` is fitted small. With them,
`alpha` goes to the boundary 1.0000 in all four folds, so the headline candidate uses
the reference only for `r`, not for `q`.

Paired bootstrap against the reference, whole episode as the unit, 2,000 draws
stratified by condition, seed 314159, reference denominators recomputed per draw,
negative favouring the win head:

| interval | 95% CI on the loss-ratio difference |
| --- | --- |
| overall | [-0.423135, -0.346498] |
| stable | [-0.673390, -0.560070] |
| short_history | [-0.412534, -0.237752] |
| drift | [-0.377001, -0.251330] |
| shared_change | [-0.360432, -0.200979] |

All four conditions and the overall interval are entirely below zero. The intervals are
unadjusted for selection, and the episodes are the same ones used throughout the
project.

Where the gain comes from, win mass only:

| quantity | reference | best control |
| --- | ---: | ---: |
| q RMSE | 0.333689 | 0.241759 |
| p_0 MSE | 0.111348 | 0.058447 |
| p_0 share of total squared error | 0.675804 | 0.646385 |
| p_0 signed bias | +0.042105 | -0.010329 |

q RMSE for the other controls: population 0.346208, local main effects 0.347159, single
`H=16` 0.260677. Both shrunk controls are worse than the reference on the very
coordinate they model.

By the number of complete-lag transitions available to the local fit:

| transitions | queries | full-vector loss, candidate | reference | q RMSE, candidate | reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-49 | 192 | 0.105624 | 0.163543 | 0.263290 | 0.332440 |
| 120-199 | 112 | 0.081990 | 0.162547 | 0.229858 | 0.329887 |
| 200+ | 464 | 0.086167 | 0.165804 | 0.235165 | 0.335113 |

No validation machine fell in the 0 or 50-119 buckets.

## Per-machine half-life averaging

A single frozen recency half-life is a compromise. A short memory tracks a drifting
machine but starves a 37-parameter fit; a long memory recovers a stationary machine
precisely and is stale once its coefficients move. Which regime a machine is in is
visible in that machine's own pre-cutoff history, so the half-life can be chosen per
machine without a condition label (which does not exist at test time) and without a
change detector.

For each candidate half-life `H`, `u_m` is fitted on all of the machine's transitions
except the most recent block, the unweighted predictive log likelihood is scored on that
held-out recent block, and `H` is weighted by a softmax over those scores. The
prediction for each `H` is its full-history fit. The held-out block is the transitions
closest to the cutoff, which is the regime the forecast has to match. The held-out block
is `min(40, max(8, n//3))` transitions and the mixture scoring is used only when at
least 16 transitions remain for fitting; otherwise every half-life scores 0 and the
mixture is uniform. Everything in this step reads only ticks strictly below
the cutoff.

Justification, on the TRAIN sampled-outcome surrogate (episode-equal win-indicator Brier
on train out-of-fold predictions):

| setting | train OOF episode win Brier |
| --- | ---: |
| single frozen `H=16`, `lam_inter=2` | 0.158248 |
| per-machine mixture over 7 half-lives, `lam_inter=1` | **0.147693** |

**Grid disclosure.** The first selection, over `H` in {16, 32, 64, 128, 256} x
`lam_main` in {0.5, 1, 2, 4} x `lam_inter` in {2, 4, 8, 16, 32}, landed on `H=16` and
`lam_inter=2` — both at the **edge** of the predeclared grid. The grid was then extended
downward, to half-lives {8, 16, 32, 64, 128, 256, uniform} and `lam_inter` in
{0.5, 1, 2}, *after* seeing that boundary. The extension and the re-selection used the
train surrogate exclusively; no public-validation target, condition label or reference
forecast is read in `scripts/win_head_train.py` or
`scripts/win_head_tune_average.py`. The disclosure is also carried in the
`grid_extension_disclosure` field of each saved population file rather than being
folded silently into the original grid. The selected `lam_inter = 1` is interior to the
extended grid; the chosen mixture still includes `H=8` at the extended boundary.
`lam_main = 4` was also at the edge of its original grid and was **not** extended - the
second stage fixes it at 4 - so that one hyperparameter remains unresolved at a boundary.

**A second, weaker disclosure.** The *idea* of averaging over half-lives was prompted by
a validation observation, quoted in `scripts/win_head_tune_average.py:5-7`: `H=128` reached
a 0.2177 stable ratio while pushing drift to 1.0972, and `H=16` reached a 0.5852 drift
ratio while giving up stable to 0.4676. The settings were then selected on train only,
but the decision to pursue this family at all was informed by validation. 40.705473 is
therefore not free of validation-guided search, and should not be read as if the whole
procedure had been specified in advance.

Which half-life each machine actually ends up favouring (argmax of the softmax mixture):

| half-life | 8 | 16 | 32 | 64 | 128 | 256 | uniform |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train machines | 288 | 208 | 72 | 58 | 23 | 19 | 100 |
| validation machines | 282 | 209 | 68 | 59 | 31 | 11 | 108 |

Median mixture entropy is 1.148809 on train and 1.193218 on validation, so most machines
spread weight over several half-lives rather than committing to one.

## Negative results

Both extra basis blocks were rejected on the TRAIN surrogate and were **never scored on
validation**. Their code paths and flags remain in `slot_model/win_head.py`, switched
off.

| basis | extra dims | train OOF episode win Brier | verdict |
| --- | ---: | ---: | --- |
| deployed basis | 37 | 0.147693 | kept |
| + payout magnitude (`big_l`, large-win indicator per lag) | 8 | 0.151515 | worse, rejected |
| + neighbour win sign at lags 1..3 | 3 | 0.147549 | 0.000144 better, treated as noise, not used |

Each row is the best of its own `lam_inter` grid {0.5, 1, 2}, so this is a best-against-best
comparison and not a single-setting ablation: the magnitude run selected `lam_inter = 0.5`
and the other two selected 1.

The magnitude block distinguishes a 5x/20x payout at lag `l` from a 1x/2x win and from a
loss. It made the surrogate clearly worse. The neighbour block is the mean win sign of
the other machines in the same episode at the same earlier tick; its improvement is
about one part in a thousand on a single surrogate with no repeated-fit spread measured,
which is not evidence of anything.

### Stated basis limitations

The deployed basis covers sign-coded win/loss main effects and pairwise products only.
It therefore cannot represent:

- interactions of three or more lags;
- any dependence on payout magnitude — 2x, 5x and 20x are all simply "win" (the block
  that would fix this is the rejected magnitude block above);
- anything beyond lag 8;
- any cross-machine structure (the rejected neighbour block);
- the split of the win mass across positive payouts, which still comes entirely from the
  reference predictor.

## Why it stops where it does: the model is variance-limited

The local fit has 37 free parameters and very little effective data.

| control | median Laplace logit variance | p90 |
| --- | ---: | ---: |
| local main effects only | 1.199136 | 1.989910 |
| local + interactions, single H=16 | 7.637792 | 13.402296 |
| local + interactions, per-machine H | **11.211033** | 23.869097 |

A median logit variance of 11.21 means the probit-scaled shrinkage divides the logit by
roughly `sqrt(1 + pi*11.21/8) = 2.3`, so the predictions are being pulled hard toward
0.5 because the model knows it cannot resolve 37 coefficients from what it has.

The arithmetic of the recency weights makes the cause concrete. 282 of 768 validation
machines put most of their mixture weight on `H = 8`. Under `2^(-(cutoff-1-t)/8)` the
recency weights sum to 12.049 no matter how long the history is, with a Kish effective
sample size of 23.098; at `H = 16` those figures are 23.587 and 46.173. Against 37 local
parameters, that is the binding constraint. Averaging over half-lives raised the median
variance (7.64 to 11.21) and still improved the score, which is consistent with the
mixture buying its gain by widening uncertain predictions rather than by sharpening them.

So the remaining gap is not obviously a feature-engineering gap. More basis functions
make the variance problem worse, and the two blocks tried did not pay for themselves on
train. Closing 9.3 points would need either more effective data per machine or a
genuinely different estimator for the same basis.

## Does `r` matter? Measured, four ways

The five-vector is `[1-q, q*r_1, q*r_2, q*r_5, q*r_20]`. Only `q` comes from the win
head; `r` comes from whatever reference predictor is supplied. Swapping `r` and
rescoring:

| `r` source | points / 60 |
| --- | ---: |
| three-component blend (deployed) | 40.705473 |
| frozen blend held-fold OOF | 40.597238 |
| `two_head` CatBoost alone | 40.589853 |
| global train marginal, one constant `r` for every query | 38.652830 |

In all four runs the fitted `alpha` was 1.0000 in all four folds, so the headline depends
only on `r` and not on the changed reference baseline. Note that swapping `--reference`
also changes the row the script prints as `existing 3-component`: in those runs that row
is the substituted predictor's own score (31.589327 for `two_head`, 31.702332 for the
frozen blend, 26.752042 for the constant `r`), not the three-component reference's
32.260531. Only the win-head row is comparable across the four runs.

The practical consequence: a package that drops TabPFN, MTD and the frozen blend and
keeps only the two CatBoost heads for `r` costs **0.115620 points** and needs no torch
and no TabPFN wheel. Reproduction runs on Python 3.13, where torch and tabpfn wheels are
a real availability risk, so that is a cheap way to remove the risk.

Even one constant `r` taken from the train marginal keeps 38.652830, which is 6.392299
above the three-component reference and a further measure of how little `r` carries
here. It is still below the 38.845500 that the single-half-life win head reaches with
the deployed `r`, so `r` is not free - only cheap.

## Target status

| | |
| --- | ---: |
| best measured held-fold score | 40.705473 / 60 |
| target | 50 / 60 |
| **gap** | **9.294527 points** |

The target is **not met**. No release, submission or champion file was changed, and
nothing was submitted.

## Implementation checks

`scripts/win_head_verify.py` is a set of Python checks about the implementation, kept
separate from the competition scores. Synthetic success establishes implementation
capability, not competition improvement. All 10 pass: `gradient`, `stationary_mode`,
`opposite`, `xor`, `shrinkage`, `post_cutoff`, `missing`, `outer_fold`, `simplex`,
`signal`.

The two that bear on leakage: `post_cutoff` appended 6,144 adversarial plays at ticks
`cutoff..cutoff+7` across all 768 machines and 0 of 768 win estimates changed;
`outer_fold` replaced the held-fold target vectors with random simplex points in all
four folds and the fitted `alpha` changed in 0 of 4 folds.

## Exposure and what is not established

- Every number on this page except the train surrogate Brier values was computed on the
  96 public-validation episodes, which have been inspected repeatedly across this
  project. New splits of them do not create untouched data. Nothing here is
  independently confirmed, and no untouched organizer evaluation set exists.
- The oracle rows in the headroom table are oracle or in-sample diagnostics. They are
  not achieved forecasting scores and must not be quoted as such.
- The half-life grid was extended after the first selection hit a grid boundary, and the
  decision to pursue half-life averaging was prompted by a validation observation. Both
  are disclosed above.
- `artifacts/headroom/diagnostics.json` was produced by an ad-hoc diagnostic that was
  **not committed as a script**, so those five numbers are recorded but not reproducible
  from a command in this repository. Every other number here is reproducible from the
  commands in the last section.
- The candidate is not packaged, not promoted and not scored against any hidden data.
  Its win mass comes entirely from code paths that have never been scored outside these
  96 episodes.
- Nothing here explains *why* pairwise lag interactions help on these machines. The
  conditions that improve are reported; no ablation isolates a mechanism.

## Artefacts

| path | contents |
| --- | --- |
| `slot_model/win_head.py` | the model, the objective, the solver, the Laplace integral |
| `artifacts/win_head/population.json` | first selection, single frozen `H=16`, `lam_main=4`, `lam_inter=2` |
| `artifacts/win_head/population_averaged.json` | **deployed**: `lam_intercept=0.01`, `lam_main=4`, `lam_inter=1`, half-lives [8, 16, 32, 64, 128, 256, uniform], `beta_by_half_life` per machine type |
| `artifacts/win_head/population_magnitude.json` | rejected magnitude basis, 45 dims |
| `artifacts/win_head/population_neighbour.json` | rejected neighbour basis, 40 dims |
| `artifacts/win_head/evaluation.json` | the full matched-control report |
| `artifacts/win_head/predictions/oof_win_head.csv` | the 40.705473 held-fold five-vectors |
| `artifacts/audit/oof_candidate_raw.csv` | the three-component reference, 32.260531 |
| `artifacts/audit/recompute.json` | that reference's per-fold simplex weights and component hashes |
| `artifacts/headroom/diagnostics.json` | the headroom numbers |

## Reproduce

```sh
DATA=/absolute/path/to/main-participants-60-40/public
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_train.py --data "$DATA"
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_tune_average.py --data "$DATA"
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_evaluate.py --data "$DATA"
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_verify.py --data "$DATA" \
  --out /tmp/win_head_verification.json
```

The rejected basis blocks:

```sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_tune_average.py --data "$DATA" \
  --magnitude --name population_magnitude.json
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_tune_average.py --data "$DATA" \
  --neighbour-lags 3 --name population_neighbour.json
```

The `r` swaps, each of which writes its own evaluation JSON, so send `--out` somewhere
scratch rather than over `artifacts/win_head/`:

```sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_evaluate.py --data "$DATA" \
  --reference artifacts/blend_release/evaluated_oof_predictions.csv --out /tmp/r_frozen
OMP_NUM_THREADS=2 .venv/bin/python scripts/win_head_evaluate.py --data "$DATA" \
  --reference artifacts/two_head_01/validation.csv --out /tmp/r_two_head
```

The train-marginal `r` row needs a constant reference CSV built first: `q = 0.438802`
and `r = (0.477745, 0.341246, 0.145401, 0.035608)`, the train win rate and the train
proportions among positive payouts, repeated for every validation query. Scoring that
file as `--reference` reproduces 38.652830.

The whole four-control evaluation takes 7.1 seconds on the local CPU and loads no
TabPFN, no torch and no CatBoost; it reads the reference predictions from CSV.
