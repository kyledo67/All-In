# Withdrawn results

Nothing in this directory is a qualified comparator. Do not cite these numbers and do not
feed these predictions to any stacking layer.

## separable_glob_pool_112

Reported `separable_projected` 54.034149 and `separable_rescaled` 54.029117 over a pool of
112 forecasts discovered by directory globs.

WITHDRAWN. The glob pool admitted candidates that are fitted against validation targets on
the same 4-fold map used by the outer evaluation, so their fitting-side values leak the
outer held fold into the stacking weights. A per-row out-of-fold label does not make such a
candidate admissible.

An independent, adversarially verified audit later found that 15 of the 30 candidates in the
*narrowed* manifest pool were also target-fitted, mostly through their conditional payout
head `r`, which many candidates inherited from a truth-fitted blend rather than estimating.
The audited RAW pool is 15 members; see `scripts/stack_manifest.py` for the per-file
evidence.

Successor results, both from `scripts/separable_stack.py --manifest`:

| pool | arm | points | status |
|------|-----|--------|--------|
| 30 (half contaminated) | separable_rescaled | 53.949676 | withdrawn |
| 15 (audited RAW)       | separable_rescaled | 53.892202 | current, nested, significant, no test recipe |

The only candidate with a verified offline test CSV and ZIP remains
`artifacts/win_head_release_exact/` at 53.532918.
