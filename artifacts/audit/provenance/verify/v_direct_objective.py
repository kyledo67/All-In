"""Verify the candidate WITHOUT the gram/row-weight trick: minimize the organizer's
own mean_loss_ratio on the fit folds directly by calling contest_score, then score the
concatenated held-fold predictions once. Independent of the auditor's code."""
import json, sys
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from scipy.optimize import minimize
from slot_forecasting.io import read_targets, read_rows, read_submission
from slot_forecasting.contest import contest_score

D = R/'main-participants-60-40/public'
targets = read_targets(D/'validation/targets.csv')
groups = {r['query_id']: r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))}
reference = read_submission(D/'validation/reference.csv')
folds = json.loads((R/'artifacts/phase2/folds.json').read_text())['validation']
ids = sorted(reference)
eo = {t.query_id: t.episode_id for t in targets}
ep = np.array([eo[q] for q in ids])
episodes = sorted(set(ep.tolist()))
truth = np.array([{t.query_id: t.probabilities for t in targets}[q] for q in ids])

def load(rel):
    d = read_submission(R/rel); return np.array([d[q] for q in ids])

comps = {
 'frozen_blend': load('artifacts/phase2/predictions/oof_ensemble.csv'),
 'tabpfn_raw': (load('artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv')
                + load('artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv'))/2.0,
 'mtd_lag7': load('artifacts/mtd_pairs_01/validation_lag7.csv'),
 'two_head': load('artifacts/two_head_01/validation.csv'),
}
NAMES = list(comps)
P = np.array([comps[n] for n in NAMES])

def official(a, es=None):
    es = set(episodes if es is None else es)
    sel = [t for t in targets if t.episode_id in es]
    k = {t.query_id for t in sel}
    m = dict(zip(ids, a.tolist()))
    return contest_score({q: m[q] for q in k}, sel, {q: groups[q] for q in k},
                         {q: reference[q] for q in k}, max_points=60)

def obj(x, es):
    x = np.maximum(x, 0); x = x/x.sum()
    return official(np.einsum('m,mnc->nc', x, P), es)['mean_loss_ratio']

out = {'per_fold': []}
held = np.empty_like(P[0])
for k in range(4):
    h = {e for e in episodes if folds[e]==k}
    fit = sorted(set(episodes)-h)
    best = None
    for init in (np.ones(4)/4, np.array([.7,.1,.1,.1]), np.array([.0,.5,.2,.3]), np.array([.1,.6,.1,.2])):
        r = minimize(obj, init, args=(fit,), method='SLSQP', bounds=[(0,1)]*4,
                     constraints={'type':'eq','fun':lambda z: z.sum()-1},
                     options={'ftol':1e-14,'maxiter':500})
        w = np.maximum(r.x,0); w = w/w.sum()
        v = obj(w, fit)
        if best is None or v < best[0]: best = (v, w)
    v, w = best
    out['per_fold'].append({'fold':k, 'objective_on_fit':v, 'weights':dict(zip(NAMES, w.tolist()))})
    mask = np.array([e in h for e in ep])
    held[mask] = np.einsum('m,mnc->nc', w, P)[mask]

s = official(held)
out['direct_objective_candidate_points'] = s['technical_points']
out['direct_objective_mean_loss_ratio'] = s['mean_loss_ratio']
out['by_condition'] = {c: v['loss_ratio'] for c, v in s['by_condition'].items()}
out['reference_points'] = official(np.array([reference[q] for q in ids]))['technical_points']
out['truth_points'] = official(truth)['technical_points']
# standalone scores + drift ratios for every component mentioned in the audit
STAND = {
 'frozen_blend':'artifacts/phase2/predictions/oof_ensemble.csv',
 'tabpfn_oof_registry':'artifacts/tabpfn/predictions/oof_tabpfn.csv',
 'mtd_lag7':'artifacts/mtd_pairs_01/validation_lag7.csv',
 'mtd_additive':'artifacts/mtd_pairs_01/validation_additive.csv',
 'mtd_control':'artifacts/mtd_pairs_01/validation_control.csv',
 'two_head':'artifacts/two_head_01/validation.csv',
 'dense_cb':'artifacts/dense_catboost_01/validation_dense.csv',
 'sparse_cb':'artifacts/dense_catboost_01/validation_sparse.csv',
 'shared_state':'artifacts/shared_state_01/validation.csv',
 'combined_additive':'artifacts/mtd_pairs_01/combined_additive.csv',
}
out['standalone'] = {}
for n,p in STAND.items():
    ss = official(load(p))
    out['standalone'][n] = {'points': ss['technical_points'],
                            'drift': ss['by_condition']['drift']['loss_ratio']}
ss = official(comps['tabpfn_raw'])
out['standalone']['tabpfn_raw_seedavg'] = {'points': ss['technical_points'],
                                           'drift': ss['by_condition']['drift']['loss_ratio']}
(R/'artifacts/audit/provenance/verify/v_direct_objective.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
