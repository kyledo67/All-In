# Targeted pairwise MTD investigation

The verified four-recipe blend and the separate 0.25 MTD candidate are preserved.
Their CSV and ZIP files are unchanged. This investigation uses the fixed 0.25 MTD
contribution only; it does not optimize global mixture weights.

`slot_model/mtd_pairs.py` adds sparse interactions between actual-tick payout
lags. For each pair it estimates a categorical cell distribution and subtracts
its two marginal effects, leaving a double-centered interaction residual.
The residual is smoothed toward the additive MTD probability with strength 40,
then scaled again by cell support. A short history usually has little evidence
for a specific cell, so its correction is small. Unknown lag symbols have no
interaction correction. All outcomes used by the correction are strictly before
the target cutoff. This is a point-estimate correction, not a posterior model.

Predefined groups are four lag-7 pairs `(7,1),(7,2),(7,4),(7,8)` and control
pairs `(1,2),(4,8)`, alongside the unchanged additive model. Train episode
folds from `artifacts/phase2/folds.json` fit upstream MTD models outside their
held episodes. Episode-equal one-hot Brier on training query outcomes selects
one group, since training true probability vectors are unavailable. The group
selection is frozen in `artifacts/mtd_pairs_01/selection.json` before public
validation scoring. Validation targets were previously examined; results are
exploratory. Every public-validation comparison uses identical queries and the
organizer's full five-probability score.

Run from the repository root, with NumPy installed:

```
python scripts/evaluate_mtd_pairs.py
python scripts/verify_mtd_pairs.py
python scripts/report_mtd_pairs.py
```

The port and evaluation use only NumPy and standard library modules beyond the
bundled organizer code. A local macOS file-loading stall in the existing virtual
environment required running from a temporary Git archive with a fresh bytecode
cache and NumPy 2.2.6 in a temporary folder. This did not change data or model
settings. Final measurements and any failures are recorded below after the run.

## Pairwise result and decision

Train-only episode OOF one-hot Brier: additive 0.5602461, lag-7 pairs
0.5599486, control pairs 0.5600043. The predeclared rule selected lag-7,
but this is a small train-label gain.

| Combined OOF procedure | Points /60 | Stable ratio | Short ratio | Drift ratio | Shared ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen four-recipe blend | 31.7023 | 0.8770 | 0.9549 | 0.9350 | 0.8035 |
| Blend + unchanged additive MTD | 31.9023 | 0.8628 | 0.9448 | 0.9087 | 0.8066 |
| Blend + lag-7 pair MTD | 31.9048 | 0.8621 | 0.9447 | 0.9089 | 0.8067 |
| Blend + control-pair MTD | 31.9002 | 0.8630 | 0.9450 | 0.9091 | 0.8064 |

The selected lag-7 correction gains only 0.00248 points over unchanged
additive MTD, while drift and shared-change conditions worsen. The 2,000-draw
paired whole-episode interval for its change in normalized error is
[-0.000503,+0.000245]. This does not support a meaningful structural gain or
a causal interpretation of lag 7. The correction is rejected for deployment.
The 0x payout category contributes 0.594101 to the additive mean normalized
error and 0.594013 after lag-7 pairs, a negligible reduction.
The control-pair ablation also fails. `evaluation.json` records per-episode
errors and 0x contributions by condition. The held train-fold runtimes were
25.46, 21.42, 22.70, 21.81 seconds; validation prediction of all three pair
sets took 102.52 seconds on the local CPU. No release files were changed.

The targeted next check is the shared episode switching-state model in
`slot_model/shared_state.py`. It filters synchronized earlier neighbor wins
through a two-state transition model, then fits a single win-probability
correction using only training-episode OOF predictions and observed outcomes.
Its public-validation audit is exploratory.

## Shared episode-state result

`slot_model/shared_state.py` filters synchronized other-machine wins strictly
before each cutoff through a two-state low/high win-rate model. The state
transition self-probability is fixed at 0.97; low/high emission rates are the
training mean +/-0.08. A single bounded adjustment to win probability was fit
with whole training-episode folds using OOF additive-MTD forecasts and observed
query outcomes. Its full-train selected slope was 0.04. One of four training
folds worsened on the held episode labels; training OOF Brier changed from
0.560246 to 0.559873. The state is only a compact observable proxy, not a
verified simulator regime.

On the previously inspected validation episodes, official points **fell** from
31.9023 to 31.8314. Condition ratios changed from
stable 0.862798 to 0.875335, short 0.944829 to 0.946338,
drift 0.908724 to 0.912485, and shared 0.806618 to 0.805560.
Only shared change improved slightly. The exploratory paired whole-episode
candidate-minus-additive normalized-error interval is
[-0.00184,+0.01040]. The full per-episode comparisons and 0x probability-error
contributions are in `artifacts/shared_state_01/evaluation.json`.
The aggregate normalized 0x contribution worsened from 0.594101 to 0.596969.
The shared-state correction is rejected. It uses other-machine observations
strictly before the cutoff and passed cutoff-invariance and normalization
checks. No release files or original prediction CSVs were modified.

Reproduce the second check:

```
python scripts/evaluate_shared_state.py
python scripts/verify_shared_state.py
python scripts/report_shared_state.py
```

The first check plus this targeted follow-up exhaust the planned cycle. A new
cycle would need a specific new diagnostic reason and fresh training-only
selection protocol. Public-validation generalization remains uncertain.
