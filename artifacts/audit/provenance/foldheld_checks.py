"""Two operational tests of the fold-held claim in scripts/combined_evaluate.py:
(a) the simplex objective equals the official mean_loss_ratio on the fit folds;
(b) randomizing the HELD fold's true probability vectors leaves the fitted weights bit-identical."""
import json, sys
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from scipy.optimize import minimize
from slot_forecasting.io import read_targets, read_rows, read_submission
from slot_forecasting.contest import contest_score
from slot_forecasting.schema import CONDITIONS
D = R/'main-participants-60-40/public'
targets = read_targets(D/'validation/targets.csv')
groups = {r['query_id']: r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))}
reference = read_submission(D/'validation/reference.csv')
folds = json.loads((R/'artifacts/phase2/folds.json').read_text())['validation']
ids = sorted(reference); eo = {t.query_id: t.episode_id for t in targets}
ep = np.array([eo[q] for q in ids])
TRUTH = np.array([{t.query_id: t.probabilities for t in targets}[q] for q in ids])
ref = np.array([reference[q] for q in ids]); episodes = sorted(set(ep.tolist()))
def load(rel):
    d = read_submission(R/rel); return np.array([d[q] for q in ids])
comps = [load('artifacts/phase2/predictions/oof_ensemble.csv'),
         (load('artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv')+load('artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv'))/2,
         load('artifacts/mtd_pairs_01/validation_lag7.csv'),
         load('artifacts/two_head_01/validation.csv')]
def rw(es, truth):
    es=set(es); w=np.zeros(len(ids))
    for c in sorted(CONDITIONS):
        ces={e for i,e in enumerate(ep) if e in es and groups[ids[i]]==c}
        b=float(np.mean([np.mean(np.sum((ref[ep==e]-truth[ep==e])**2,axis=1)) for e in ces]))
        for e in ces: w[ep==e]=1/(4*len(ces)*int(np.sum(ep==e))*max(b,1e-9))
    return w
def fit(es, truth):
    p=np.array(comps); w=rw(es, truth)
    f=((p-truth)*np.sqrt(w)[None,:,None]).reshape(len(p),-1); g=f@f.T
    r=minimize(lambda x: float(x@g@x), np.ones(len(p))/len(p), jac=lambda x: 2*g@x, method='SLSQP',
               bounds=[(0,1)]*len(p), constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones_like(x)},
               options={'ftol':1e-12,'maxiter':200}); assert r.success
    return r.x/r.x.sum()
def official(a, es):
    es=set(es); sel=[t for t in targets if t.episode_id in es]; k={t.query_id for t in sel}
    m=dict(zip(ids,a.tolist()))
    return contest_score({q:m[q] for q in k}, sel, {q:groups[q] for q in k}, {q:reference[q] for q in k}, max_points=60)
rng = np.random.default_rng(7)
out = {'objective_equals_official_on_fit_folds': [], 'weights_invariant_to_held_fold_truth': []}
for k in range(4):
    h = {e for e in episodes if folds[e]==k}; es = sorted(set(episodes)-h)
    w = rw(es, TRUTH)
    for i,a in enumerate(comps):
        lhs = float(np.sum(w*np.sum((a-TRUTH)**2, axis=1)))
        rhs = official(a, es)['mean_loss_ratio']
        out['objective_equals_official_on_fit_folds'].append({'fold':k,'component':i,'abs_diff':abs(lhs-rhs)})
    base_w = fit(es, TRUTH)
    scrambled = TRUTH.copy()
    mask = np.array([e in h for e in ep])
    r = rng.random((int(mask.sum()),5)); scrambled[mask] = r/r.sum(1,keepdims=True)
    alt_w = fit(es, scrambled)
    out['weights_invariant_to_held_fold_truth'].append(
        {'fold':k,'max_abs_weight_change':float(np.max(np.abs(base_w-alt_w))),
         'weights':base_w.tolist()})
out['worst_objective_gap'] = max(r['abs_diff'] for r in out['objective_equals_official_on_fit_folds'])
out['worst_weight_change_under_held_fold_scramble'] = max(r['max_abs_weight_change'] for r in out['weights_invariant_to_held_fold_truth'])
(R/'artifacts/audit/provenance/foldheld_checks.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('worst |objective - official mean_loss_ratio| on fit folds:', out['worst_objective_gap'])
print('worst fitted-weight change when the HELD fold truth is randomized:', out['worst_weight_change_under_held_fold_scramble'])
