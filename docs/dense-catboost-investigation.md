# Dense causal CatBoost training experiment

This experiment changes one axis: historical cutoff density. Observable feature
schema, CatBoost depth 4, L2 regularization 8, learning rate 0.04, both saved
seeds, their saved 214/307 tree counts, five-class order, and episode/machine
weights are fixed. The sparse control uses at most eight historical targets per
machine spaced eight ticks apart. Dense uses every observed eligible historical
tick from 192 to the query cutoff, with the same complete recent 40-tick
requirement and alternating sampled-older construction. Each machine keeps
the sparse control's total weight and its original real-query versus
historical-example mass. The dense arm spreads only that historical mass
over its additional cutoffs. Actual tick arithmetic and an explicit unknown
historical signal flag are preserved.

For each held train-episode fold, type priors for historical examples are fitted
on other training episodes in the fitting partition. The held episodes are
excluded from model fitting and from those priors. Train query outcomes support
an episode-equal one-hot Brier comparison. The public-validation probability
score is a separate exploratory audit; all validation episodes were already
inspected. The experiment does not tune ensemble weights or train on validation
true probabilities. Reused input matrices are cached under
`artifacts/dense_catboost_01/features_*.npz` with source, train-history,
density, and allowed-episode identity; those caches are local-only. The frozen
schema and protocol versions are recorded separately in the run artifacts.

From the repository root, with the pinned CatBoost/NumPy environment:

```
python scripts/dense_catboost.py
python scripts/report_dense_catboost.py
```

The active blend and MTD candidate release files are preserved. Measurements
and any failures will be recorded after the run.

## Completed density comparison

Train-only episode OOF one-hot Brier improved from 0.556885 sparse to
0.555512 dense. Three of four train folds improved; the other regressed.
All CatBoost settings, features, and category order were fixed. The sparse
full-train replay matched the saved observable CatBoost forecasts within
2.22e-16 maximum probability difference.

| Full-train model on previously inspected validation | Points /60 | Stable ratio | Short ratio | Drift ratio | Shared ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse control | 31.398446 | 0.91453 | 0.96498 | 0.94955 | 0.81463 |
| Dense histories | 31.391743 | 0.91281 | 0.97679 | 0.94191 | 0.81381 |

The official normalized error worsened from 0.910923 to 0.911331, despite
three of four condition ratios improving. The short-history regression
outweighed them. A paired whole-episode exploratory interval for dense-minus-
sparse normalized error is [-0.01044,+0.01121]. The aggregate normalized 0x
contribution improved slightly from 0.614627 to 0.613567, but total five-
category error did not. The bounded run took 269.57 seconds and used at most
two CatBoost CPU threads. No dense model enters the release or existing stack.

An initial partial run was stopped because its weighting had allowed the added
historical rows to reduce the mass of real signal-known query examples. The
completed run corrected this, keeping original real/historical mass fixed.
The stopped run produced no scored result and is not used as evidence.

The next planned architecture check uses the same sparse inputs with a binary
win/loss head and a separate four-class positive-payout head; it does not
blend that model into the current stack before standalone evaluation.
