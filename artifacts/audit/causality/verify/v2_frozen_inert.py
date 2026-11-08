"""Re-derive the 'frozen blend is causally inert / below float noise' claim.

The prior agent's justification: max weight 1.7251774146612511e-16 is 'below float64
rounding at p~0.5 (2.220446049250313e-16), so its maximum possible contribution to any
probability is below float noise'. 2.22e-16 is the ULP at 1.0, not at 0.5. Check whether
dropping the term actually leaves the candidate bitwise unchanged, and what the points
delta is.
"""
import json, sys
from pathlib import Path
ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from slot_forecasting.io import read_submission, read_rows, read_targets
from slot_forecasting.contest import contest_score

D = ROOT / 'main-participants-60-40/public'
rec = json.loads((ROOT / 'artifacts/audit/recompute.json').read_text())
W = rec['fold_weights']
folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']
targets = read_targets(D / 'validation/targets.csv')
groups = {r['query_id']: r['condition'] for r in read_rows(D / 'validation/conditions.csv', ('query_id','condition'))}
reference = read_submission(D / 'validation/reference.csv')
ids = sorted(reference)
ep = {t.query_id: t.episode_id for t in targets}

def load(rel):
    d = read_submission(ROOT / rel)
    return np.array([d[q] for q in ids])

fb = load('artifacts/phase2/predictions/oof_ensemble.csv')
tp = (load('artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv')
      + load('artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv')) / 2.0
mt = load('artifacts/mtd_pairs_01/validation_lag7.csv')
th = load('artifacts/two_head_01/validation.csv')
saved = load('artifacts/audit/oof_candidate_raw.csv')

def build(drop_fb):
    out = np.zeros_like(fb)
    for f in range(4):
        w = W[f]
        rows = np.array([folds[ep[q]] == f for q in ids])
        p = (0.0 if drop_fb else w['frozen_blend']) * fb[rows] \
            + w['tabpfn_raw'] * tp[rows] + w['mtd_lag7'] * mt[rows] + w['two_head'] * th[rows]
        out[rows] = p / p.sum(1, keepdims=True)
    return out

A = build(False); B = build(True)
def pts(a):
    m = dict(zip(ids, a.tolist()))
    return contest_score(m, targets, groups, reference, max_points=60)

pa, pb = pts(A), pts(B)
ulp_half = float(np.spacing(0.5))
ulp_one = float(np.spacing(1.0))
maxw = max(f['frozen_blend'] for f in W)
rows_diff = int((~np.all(A == B, axis=1)).sum())
out = {
 'max_frozen_blend_weight': maxw,
 'np.spacing(0.5)_ULP_at_half': ulp_half,
 'np.spacing(1.0)_machine_eps': ulp_one,
 'max_weight_in_ULPs_at_p_0.5': maxw / ulp_half,
 'agent_claim_weight_below_float_rounding_at_p_half': maxw < ulp_half,
 'with_vs_without_frozen_blend': {
   'bitwise_identical': bool(np.array_equal(A, B)),
   'rows_differing': rows_diff,
   'max_abs_prob_delta': float(np.max(np.abs(A - B))),
   'points_with': pa['technical_points'], 'points_without': pb['technical_points'],
   'points_delta': pa['technical_points'] - pb['technical_points']},
 'my_rebuild_vs_saved_oof_candidate_raw': {
   'max_abs_delta': float(np.max(np.abs(A - saved))),
   'points_mine': pa['technical_points'], 'R_mine': pa['loss_ratio'] if 'loss_ratio' in pa else None,
   'keys': sorted(pa)},
}
print(json.dumps(out, indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
