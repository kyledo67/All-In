# Round 2 (adaptation) — findings, candidate, and evidence

Dataset: `/Users/ayan/Downloads/adaptation-participants`, `public/metadata.json` reports
`"round": "adaptation"`, season `5fffdc4d0c700bbf61c3f911`, suite `experimentation-suite-v3`.
Confirmed to be the adaptation kit. All Round-1 models, artifacts and receipts are untouched.

## 0. Two things to resolve before reading the numbers

**The deadline.** The RULES.md shipped with the Round-1 kit we were issued states: *"Round
two closes Sunday, November 8 at 11 AM Central."* That instant passed while this work was
running. The adaptation kit's own RULES.md gives no date, only "90 minutes from the common
release time". Whether anything can still be uploaded is an organizer question, not one the
files can answer.

**The point scale.** The two kits disagree, and neither is something we chose:

| | main kit we were issued | adaptation kit on disk |
| --- | --- | --- |
| `slot_forecasting/rounds.py` | `ROUND_POINTS = {"main": 60, "adaptation": 40}` | `{"main": 80, "adaptation": 20}` |
| version string | `temporal-adaptation-v3-60-40` | `temporal-adaptation-v2` |
| `metadata.json` max_points | 60 | 20 |

The adaptation kit is the older v2 80/20 build; the main kit's validator explicitly rejects
it ("old 80/20 kits must be updated by organizers"). So `check_submission.py` in the
adaptation folder prints points out of 20, while the brief and the main kit's RULES.md say
40. R is identical either way and so is any ranking; only the multiplier differs. Both are
reported below.

## 1. What changed in the simulator between rounds

Measured on the two rounds' validation splits, the only places both rounds expose true
probability vectors (768 queries each).

**Unchanged — the conditional payout distribution.** `r = P(payout | win)` takes exactly
**six** values on released targets, and the six are **bit-identical across the two rounds**:

| r (1x, 2x, 5x, 20x) | machine_type | R1 n | R2 n |
| --- | --- | --- | --- |
| 0.455122, 0.435902, 0.106796, 0.002180 | steady | 132 | 92 |
| 0.753989, 0.196808, 0.048218, 0.000984 | steady | 137 | 115 |
| 0.214066, 0.546737, 0.205026, 0.034171 | balanced | 103 | 129 |
| 0.499853, 0.347928, 0.130473, 0.021746 | balanced | 131 | 166 |
| 0.120787, 0.323921, 0.416469, 0.138823 | jackpot | 118 | 123 |
| 0.335146, 0.244946, 0.314931, 0.104977 | jackpot | 147 | 143 |

So `machine_type` — which test queries *do* carry — narrows r to exactly two candidates, and
the whole r problem is one binary latent per query. The previous round's "two-endpoint
structure and fixed tail ratios" is not merely still valid, it is exact and it is the same
dictionary. p20/p5 median 0.16667 and p2/p1 median 0.73086 agree to five decimals across
rounds.

**Unchanged — observation density and gap structure.** Pre-cutoff observed ticks: median
210 (R2) vs 216 (R1); mean gap 1.212 vs 1.202; P(gap>1) 0.0285 vs 0.0284. Every split in
both rounds has exactly 25% of queries at 64 observed ticks and the rest at 160+, so the
short-history condition is identifiable at test time from an observable count.

**Changed — the overall win rate falls.** Mean true q 0.4384 -> 0.4098; train-history win
rate 0.4166 -> 0.4041; validation 0.4073 -> 0.3870. The marginal payout mix shifts
accordingly: p_0 0.5616 -> 0.5902.

**Changed — the lag-7 spike is gone.** Win-indicator autocovariance from train history:

| lag | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | .0034 | .0053 | .0060 | .0077 | .0033 | .0054 | **.0336** | .0067 |
| R2 | .0032 | .0060 | .0052 | .0069 | .0073 | .0077 | .0049 | .0075 |

Round 1's lag-7 structure — the thing the lag-7 pair MTD family was built for — does not
survive. Round 2's dependence is flat across lags 1-8.

**Changed — the coefficients on true logit q move.** Ridge on true logit q: `win_lag1`
-0.033 -> **-0.248**, `rate_all` +0.276 -> +0.061, `rate20` +0.753 -> **+0.972**,
`is_steady` +0.243 -> +0.387, `is_jackpot` -0.315 -> -0.419. More weight on the recent
window, a real negative lag-1 term, stronger type separation. R^2 on logit q is unchanged
(0.357 -> 0.358), so the *amount* of linearly extractable signal is the same; its location
moved.

**Unchanged — condition composition.** 24 episodes x 8 machines per condition per split in
both rounds; short_history is exactly 64 observed ticks in both.

## 2. How well the strongest Round 1 model transferred

The Round-1 submitted entry is `artifacts/win_head_release_exact` (the 53.532918/60 audited
candidate; its `entry.zip` is `b5b1a83c…`, the same file in the Round-1 upload bundle).
Pointed at the adaptation kit's `public/` with no change whatsoever:

| | R | /20 | /40 |
| --- | --- | --- | --- |
| frozen pooled reference | 1.00000 | 10.0000 | 20.0000 |
| Round-1 entry, unchanged | **0.37022** | **14.5962** | **29.1924** |

Per-condition loss ratios 0.1046 (stable), 0.5206 (short history), 0.5330 (drift), 0.3226
(shared change). It transfers well. Note what "frozen" means here: the package's local win
head fits its per-machine deviations at prediction time from the round's own data, so the
architecture and all pooled parameters are Round-1 but the machine-level fit is already
Round-2. That is why transfer is this good.

## 3. Where the remaining error actually is

Oracle decomposition on R2 validation, substituting one head at a time with the truth:

| forecast | R | /20 |
| --- | --- | --- |
| frozen q, frozen r (as submitted) | 0.37022 | 14.5962 |
| frozen q, **true r** | 0.33355 | 14.9976 |
| **true q**, frozen r | 0.04336 | 19.1688 |
| true q, true r | 0.00000 | 20.0000 |

**q carries about 90% of the remaining error.** A perfect r head is worth +0.40 points/20
(+0.80/40); a perfect q head is worth +4.57/20 (+9.15/40). Everything below is scaled by
that fact.

## 4. The four controlled arms, and everything that failed

All out-of-fold on 4 complete-episode folds, stratified so each fold holds six episodes of
each condition, with every hyperparameter chosen inside the three fitting folds.

| arm | out-of-fold R | /20 |
| --- | --- | --- |
| 1. frozen Round-1 model | 0.37022 | 14.5962 |
| 2. Round-2-only refit (134 causal features, ridge) | 1.06626 | 9.6793 |
| 2b. Round-2-only refit (gradient boosting) | 1.09304 | 9.5555 |
| 3. combined R1+R2 targets, pooled | 1.04327 | 9.7882 |
| 4. hierarchical: correction on the frozen logit | 0.38586 | 14.4314 |
| 4b. same, gradient boosted | 0.40418 | 14.2431 |
| **4c. hierarchical: reference/local blend re-tuned by history length** | **0.36406** | **14.6621** |

Pooling did not help and refitting from scratch was catastrophic: the released targets are
768 queries, against which the Round-1 win head brings ~140k history rows of pooled
structure. The evidence says transfer almost everything.

Arm 4 is instructive. Fitting the correction by least squares on **logit** q lowered logit
MSE (2.5248 -> 2.4459) and **raised** R. The metric is squared error on probabilities, so a
logit fit spends capacity on near-0/near-1 machines where probability error is already
negligible. Every later candidate is fitted to the probability-space residual and selected
on the exact official objective.

Rejected after measurement, out-of-fold, none adopted:

- monotone recalibration of the frozen q: 0.36929 (+0.010/20, inside noise)
- probability-space ridge correction, 134 features: 0.37314 (worse)
- compact lag-win correction (8 features): 0.37217 (worse)
- + lag-pair products: 0.37366 (worse); + type interactions: 0.37409 (worse)
- neighbouring-machine summaries (7 features): 0.36874 (+0.016/20)
- lag + neighbour + horizon + signal, 38 features: 0.37162 (worse)
- the three other local win heads shipped in the package: four-cell Beta 0.41523, compact
  pair dictionary 0.50994 hard / 0.47382 soft, dense 37-coefficient 0.47488. A nested
  selection that could pick any of the four chose `beta_templates` in all four folds.
- convex blend of six Round-1 release variants, simplex weights fitted per fold: 0.36301.
  Best point estimate of anything tried, but its bootstrap interval overlaps zero
  ([-0.0184, +0.0026]) and it needs six packages and six times the runtime.

The lag-correction failures are consistent with the rules' own statement that *"machines of
the same type can respond differently or oppositely"*: a population-level lag coefficient
averages opposing machine-level responses toward zero. The inner selection repeatedly chose
blend weight 0, i.e. it declined the correction on its own training data.

**The r head, despite the exact dictionary.** Treating r as a posterior over its two
candidates is exactly specified, but the latent is only ~65% identifiable from a machine's
own recency-weighted payout counts (R2 0.667, R1 0.691 at the best half-life — and the two
rounds agree, which is itself a cross-round check of the method). Results with q frozen:
hard argmax 0.39900 (worse than doing nothing), type-mean 0.37992, count posterior 0.38253,
frozen r projected onto the dictionary segment 0.37016, in-fold-selected posterior/frozen
blend 0.36960 against the shipped head's 0.37022. The r head is left unchanged. `signal`
does not predict the latent (P(idx=1) = 0.51-0.58 across its three values in both rounds).

## 5. The deployed candidate

One change to the Round-1 entry. `deployment.json` had the scalar `alpha = 1.0`, using the
local win head's q alone and discarding the two-head reference's own win mass. It now holds

    "alpha": {"long": 0.94, "short": 0.83, "min_observations": 128}

and `win_head_infer.py` blends per query, `blended_q = (1-a_i)*reference_q + a_i*q_i`, with
the short value when that machine has fewer than 128 of its own pre-cutoff observed ticks.
The switch reads an observable count; no condition label is used, and none exists for test
queries. Every model file is byte-identical to the Round-1 entry; nothing was refitted
offline. The threshold is not a tuned boundary: in every split of both rounds each query has
either 64 observed ticks or at least 160, so any value in (64, 160] selects the same 384 of
1536 test queries.

### Official score

| | R | /20 | /40 | basis |
| --- | --- | --- | --- | --- |
| Round-1 entry unchanged | 0.37022 | 14.5962 | 29.1924 | no Round-2 fitting |
| **candidate, out-of-fold** | **0.36406** | **14.6621** | **29.3243** | **the figure to believe** |
| candidate, as deployed | 0.36303 | 14.6732 | 29.3463 | replay; the alphas saw these episodes |

Out-of-fold condition ratios: stable 0.1064, short history 0.5004, drift 0.5260, shared
change 0.3234. `check_submission.py` on the shipped validation CSV prints 14.673162/20,
matching the deployed row.

### Is it credible

- Paired episode-level bootstrap of dR, 4000 resamples within condition: mean **-0.00619**,
  95% **[-0.01290, -0.00032]**, P(improve) 0.9795. Excludes zero, but barely.
- Fold stability: dR -0.0079, **+0.0027**, -0.0108, -0.0109. Three of four folds improve;
  fold 1 regresses.
- Paired episodes: better on 45 of 96, worse on 51. Mean dE -0.00112, **median +0.00014**.
  The mean gain comes from large improvements on a minority of episodes, not from a broad
  shift; worst single-episode regression +0.0096.
- The gain lands where the mechanism predicts: short histories -4.58% mean squared
  probability error, drift -1.38%. It costs a little elsewhere: stable +3.11%, shared change
  +0.33%.
- Both alphas were re-fitted independently inside each fold and came out stable:
  long 0.94/0.94/0.94/0.94, short 0.83/0.83/0.84/0.83.

Verdict: **statistically credible on the metric the rules define, and mechanistically
motivated, but small and not uniform.** A reasonable person could ship the unchanged
Round-1 entry instead; the difference is ~0.13 points out of 40. The alternative is staged
alongside.

### Calibration of the full payout distribution (out-of-fold)

| category | mean forecast | mean true | bias | rmse |
| --- | --- | --- | --- | --- |
| p_0 | 0.59700 | 0.59017 | +0.00683 | 0.21001 |
| p_1 | 0.18354 | 0.17988 | +0.00367 | 0.11541 |
| p_2 | 0.13880 | 0.14588 | -0.00709 | 0.09299 |
| p_5 | 0.06644 | 0.06933 | -0.00289 | 0.05457 |
| p_20 | 0.01421 | 0.01473 | -0.00052 | 0.01611 |

Win-probability reliability is good except in the 0.3-0.5 band, where the forecast reads
0.4003 against a true 0.4492 (n=93, bias -0.049). That band is the residual
over-confidence the alpha change only partly addresses.

## 6. What remains, and the next experiment

Remaining error is q, and it is large: out-of-fold R 0.364 against 0.043 with a perfect q.
The local win head's logit correlation with the truth is 0.843.

The highest-value next experiment is a **joint latent-regime model that couples the two
heads**. Three measurements point the same way:

1. r is a binary latent per machine within its observed type, exactly two candidates.
2. That latent is only ~65% recoverable from payout amounts alone.
3. Within an episode and within a machine type, the latent agrees across machines 0.73 of
   the time, against 0.50 for independence — there is a shared episode-level component.

Today the q head reads only the win/loss sequence and the r head reads only the payout
amounts; neither sees the other's evidence, and neither uses the episode-level shared
component. A state-space model inferring one regime per machine from **both** sequences,
with an episode-level prior over regimes, would let the amount sequence sharpen q and the
win sequence sharpen r. The ~35% of r-latent uncertainty and the 0.3-0.5 q band are the two
places that would show up first.

Second: short histories hold a 0.50 loss ratio on 64 observations, and the local head pools
to a *population* prior. Pooling to an *episode* prior over the eight co-observed machines
is a small, well-posed change the shared_change condition suggests should carry information.

## 7. Artifacts and hashes

Upload pair, `/Users/ayan/Downloads/All-In-round2-submission/`:

| file | bytes | sha256 |
| --- | --- | --- |
| `submission.csv` | 209,870 | `0f00c63a8a29869a5351c02d82bf36fdb5f514e5db7092a8a6be0bf59ae98bea` |
| `entry.zip` | 484,034 | `5e4f19a6f0de5d280d06d0c00fb5c1fc2216d8ac998dc0010de8070d7d5eadbc` |

Verified fallback, the unchanged Round-1 entry applied to Round-2 data,
`fallback-round1-unchanged/`:

| file | bytes | sha256 |
| --- | --- | --- |
| `submission.csv` | 210,351 | `b78a3aafbfd0cfca40ec37b2be860eaf2c454ab1f5c40abc3379014cbfcb7ea7` |
| `entry.zip` | 481,969 | `b5b1a83c44d01f182ecdeea188d4355a810bbeb7a5c52d4101d559002e93d522` (identical to the Round-1 upload) |

Checks run, all passing: `check_submission.py submission.csv` -> VALID FORMAT, 1536
queries, adaptation/test, hash as above. `check_submission.py validation_predictions.csv
--split validation` -> 14.673162/20. Reproduction from a clean extraction of `entry.zip`
into an empty directory, `python predict.py --data <kit>/public --out <out>/submission.csv`
-> **byte-identical**, max per-value difference **0.000e+00** against the 1e-8 tolerance,
52 s wall clock against the 30-minute limit, 0 win-head fallbacks, max row-sum error
3.3e-16. ZIP opens directly to `README.md` and `reproduce.json`, 31 entries, 484,034 bytes
compressed, 761,612 bytes unpacked, no bytecode caches; `checksums.json` verifies 29/29
shipped files.

Still unverified, as in Round 1: Linux and Python 3.13. Reproduction here ran on macOS
(darwin 25.3.0) under Python 3.12.16, and the timing was measured on a 10-core machine, not
the organizers' two cores. The 52-second runtime leaves a wide margin, but the platform
claim itself is untested.

Repository artifacts: `artifacts/r2_release/` (code tree, entry.zip, submission.csv,
validation_predictions.csv, evaluation.json, this report);
`artifacts/r2_candidates/frozen_r1/` (the unchanged-transfer arm);
`artifacts/r2_adaptation/arms.json` (the four controlled arms). Scripts:
`scripts/r2_common.py` (official-equivalent scorer, cross-checked against
`check_submission.py` to all printed digits), `scripts/r2_features.py`,
`scripts/r2_diagnose.py`, `scripts/r2_qmodel.py`, `scripts/r2_qcorrect.py`,
`scripts/r2_rhead.py`, `scripts/r2_lagcorrect.py`, `scripts/r2_blend.py`,
`scripts/r2_evaluate.py`.

Nothing was uploaded or pushed. All Round-1 candidates are preserved unmodified.
