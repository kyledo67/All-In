# Round-1 campaign: five families, and what survives

Workflow `wf_4c85bcd7-8f1`, 16 agents, 88 minutes, 2.63M subagent tokens, 781 tool calls, 0 errors.
Five modelling families run against the shared harness, then adversarially verified.

**Round 1 is closed.** Main submissions closed before the adaptation kit opened and "main files
cannot be replaced once adaptation opens." The submitted Round-1 entry remains
`artifacts/win_head_release_exact` at the audited 53.532918/60. Nothing below changes that score.
These are research results and a correction to the record, not points.

## Headline by family

| family | hypothesis answered | best arm | points /60 | determinism |
| --- | --- | --- | --- | --- |
| T | **Yes, decisively** | TALL / T3B7 / T3U | **54.919878** | reproducible |
| X | Yes, but not its own hypothesis | X5 block-general | 53.969893 | reproducible |
| E | Yes, but thin | E3_q_combination | 53.880491 | reproducible |
| N | **No** | A5 boost_q_only_enriched | 53.794794 | reproducible |
| D | **No** | D4 support-gated | 53.557153 | reproducible |

Families N and D answer their own questions in the negative and said so. N's best arm is "one of 16
with a form chosen after seeing the outer-fold decomposition, so it fails multiplicity correction
and I do not claim it." D confirmed its mechanism — the complete-eight-lag requirement really does
discard strictly causal transitions, +1.87% of rows and +7.90% of recency-weighted support in
short_history — but the gain is +0.020 points with an interval containing zero, dropping to
53.537992 once the choice among family members is paid for inside the folds. D's negative control
(row-offset lags instead of actual tick differences) correctly scored 52.939820, below the control.

Family X's own report is the model of an honest negative: four arms beat 53.532918 with intervals
excluding zero, yet "every one of them is statistically indistinguishable from the family's already
established clean reference separable_rescaled 53.892202, while the family's actual question — do
general convex weights beat the rank-one restriction — is answered NO with a precise null
(dR -0.000017, interval [-0.00019, +0.00015]) against an in-sample ceiling of only 0.008 points."

## Family T is the real result

Replacing the protected release's hard-selected lag template plus borrowed decay half-life with a
joint posterior over (lag template, run length) — a bounded changepoint / run-length recursion —
scores 54.919878 (R = 0.0925006) under the full nested protocol against the protected 53.532918
(R = 0.1208057), with a condition-stratified paired interval of [-0.042491, -0.015075] entirely
below zero. That is +1.387 points, an order of magnitude larger than anything else in the campaign.

Two arms were adversarially verified and **not refuted**: T2 (54.615640) and T3 (54.906013). Both
reproduced byte-identically across re-runs and PYTHONHASHSEED changes, and both verifiers wrote
independent from-scratch re-derivations of the recursion that matched the saved forecasts to
2.2e-16. The nested figure sits *below* every component setting's full-validation score
(54.6547 / 54.6445 / 54.6191 for T2's three selected settings) — the expected signature of honest
nesting rather than selection leakage.

The mechanism checks out on the data: the true q is bimodal, with mass at 0.0077–0.195 and
0.74–0.97, i.e. a two-regime process, and T2 cuts q RMS error against the truth from 0.102748 to
0.086733 with r held bit-frozen. All 15 RAW pool members have q RMS >= 0.10179.

## Correction to the record: Family T's leakage statement is false as written

Family T declares that "NO tail constants, endpoint values or normalizers are estimated anywhere in
this family and `harness.tail_constants` is never called." That is **true about estimation and
false about what the forecast carries.** Every arm composes its q with `r_prot` from
`bundle.protected`, which is `artifacts/win_head_release_exact/validation_submission.csv` —
classified `DEPLOY_FIT` by `scripts/stack_manifest.py:117`. The release's own payload says so
outright: `model/win_head/tail_crossfit_deploy.json` carries
`"tail_provenance": "Fitted on ALL released validation probability targets AFTER the method was
frozen."` Its `tail_by_type` equals `harness.tail_constants` on the full validation truth to
<= 2.6e-15.

This is the same channel that forced the withdrawal of 53.948484 (commit `d4edb37`), so it deserved
the alarm it raised. **It is nonetheless immaterial here, and the reason is structural.** Measured
directly:

- Within each `machine_type`, the two r dictionary candidates share the *identical* renormalised
  {2,5,20} tail split — steady, balanced and jackpot all agree to <= 1.2e-08. That is why the
  release's `tail_by_type` holds exactly one value per type.
- Those per-type splits match the release's `tail_by_type` to <= 7.3e-09.
- The same splits are recoverable from **train history alone**, in both rounds, with maximum
  absolute deviation 0.0040–0.0128; from the *test* histories, 0.0003–0.0042. R1 and R2 agree.

So the quantity is a three-number-per-type invariant of the simulator, identical across rounds and
across splits, not information about any held episode. All three Family-T verifiers independently
measured its materiality and agreed: a train-only tail estimate gives T2MAP 54.605797 with paired
[-0.035249, -0.010954] (still significant); substituting the RAW two-head r gives T2 54.266523,
paired [-0.034564, -0.011225]; substituting raw-pool-mean, two-head and TabPFN r heads leaves T3's
gain at +1.368, +1.362 and +1.336 points respectively. The leak is worth ~0.011 points of absolute
score and ~0.000 of the gain, and it is carried bit-identically by the arm, by the family's own
control C1, and by the baseline — so it cancels in every comparison.

**Adjudication.** T2MAP (54.616050) was refuted on this channel while T2 and T3, which share it,
were not. All three verifiers found the same facts and measured the same materiality; they differed
only on whether a shared common-mode `DEPLOY_FIT` input counts as a Check-3 failure. T2MAP's
verifier applied the conjunctive rule literally and said so explicitly ("REFUTED on the letter of
the brief's conjunctive rule — but only on one channel, which I measured and which does not
manufacture the gain"). T2's and T3's verifiers scoped Check 3 to stacking pool members, noted the
family builds no stack, and recorded it as a caveat. The literal reading is correct about the
classification; the substantive conclusion is that **Family T's gain is real, is entirely on the q
side, and survives a fully RAW-admissible route.** The right remedy is not to discard the arm but to
rerun the family composing a RAW r head, which it should have done and did not — a one-line change
worth about 0.011 points.

Two further defects the verifiers surfaced, neither affecting a score:
- `scripts/family_t_joint_structure_memory.py:519,524-526` computes a field named
  `official_objective_argmin_all_fitting_episodes` using `row_weights(np.ones(...))` — all 96
  episodes, every held fold included, not the fitting episodes the name claims. It feeds no
  forecast; the name is simply wrong.
- The (kappa, hazard) grid bounds were informed by an exploratory public-validation sweep, disclosed
  at lines 97–101. Bounded by measurement: dropping the two decades actually selected still gives
  54.9114; a coarse decade grid gives 54.8071; nothing finer than h=1e-2 loses significance
  (+0.4602, interval straddling zero). So the sweep accounts for at most ~0.14 of the 1.08–1.39
  points, and the declared endpoints are catastrophic (h=0 -> 32.11, h=0.1 -> 47.36), with the gate
  correctly falling back to the control there.

Family T also misstates one secondary claim: `artifacts/family_t_joint/evaluation.json` reports
prequential log-score corroboration, but the joint model is *worse* on that diagnostic
(-53163.0 vs -53094.9 for fixed structure, 68.1 nats worse). The official-metric result stands; the
log-score corroboration does not.

## A verification coverage gap, now being closed

The campaign verified roughly the top three significant arms per family. That cap was silent, and it
left the **highest-scoring arm in three families unverified** — precisely the numbers anyone would
quote:

| arm | points /60 | status |
| --- | --- | --- |
| T: TALL / T3B7 / T3U (three-way tie) | 54.919878 | never verified |
| X: X5 block-general weights | 53.969893 | never verified |
| E: E3_q_combination | 53.880491 | never verified |

TALL is the arm Family T leans on to argue its headline is not an outer-table cherry-pick, which
makes it the most load-bearing unverified number in the campaign. X4, one of X5's siblings, was
refuted on nesting plus a `DEPLOY_FIT` archived candidate; E2x, one of E3's siblings, was refuted on
a documented nesting failure in `scripts/endpoint_supervised.py` (lines 550 / 655 / 663 / 726, plus
a 9-rung ladder designed by inspecting fold-0 targets that are held episodes for folds 1–3). Both
refutations are on channels their unverified siblings could share.

Workflow `wf_e38ae5e9-38c` is verifying all three, and separately testing whether Family T's q head
transfers to Round 2 — where the remaining error is ~90% q-side and the two weakest conditions,
drift (0.526) and shared_change (0.323), are exactly where Family T's per-condition intervals
exclude zero.

## Refuted, for the record

| arm | claimed /60 | failed |
| --- | --- | --- |
| X4 general mixture, width and shrinkage by inner folds | 53.887304 | Check 2 nesting + Check 3, a `DEPLOY_FIT` archived candidate |
| E2x_calibrator_no_uprot | 53.638116 | Check 2, twice: the `beta`-weighted `u_prot` channel it claims to remove, and a ladder designed on fold-0 targets |
| T2MAP same recursion, MAP run length | 54.616050 | Check 3 provenance + the tail-constants item of Check 2 — immaterial, see above; clean route 54.605797 |
| AB boost_enriched_nopreq | 53.861959 | Check 2, the feature subset was selected using every fold's held episodes |
