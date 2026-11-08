"""Quantify the one validation-informed bit in mtd_lag7: changepoint=True vs plain base."""
import json, os, sys, time
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from scipy.optimize import minimize
from slot_model.mtd import MTDModel
from slot_model.mtd_pairs import PairMTDModel, LAG7_PAIRS
from slot_forecasting.io import load_inputs, read_submission, read_targets, read_rows
from slot_forecasting.contest import contest_score
from slot_forecasting.schema import CONDITIONS
D = R/'main-participants-60-40/public'
vq, vh = load_inputs(D/'validation')
targets = read_targets(D/'validation/targets.csv')
groups = {r['query_id']: r['condition'] for r in read_rows(D/'validation/conditions.csv', ('query_id','condition'))}
reference = read_submission(D/'validation/reference.csv')
folds = json.loads((R/'artifacts/phase2/folds.json').read_text())['validation']
ids = sorted(reference)
eo = {t.query_id: t.episode_id for t in targets}
ep = np.array([eo[q] for q in ids])
truth = np.array([{t.query_id: t.probabilities for t in targets}[q] for q in ids])
ref = np.array([reference[q] for q in ids]); episodes = sorted(set(ep.tolist()))
def arr(d): return np.array([d[q] for q in ids])
def load(rel): return arr(read_submission(R/rel))
frozen = load('artifacts/phase2/predictions/oof_ensemble.csv')
tab = (load('artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv')+load('artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv'))/2
two = load('artifacts/two_head_01/validation.csv')
mtd_change = load('artifacts/mtd_pairs_01/validation_lag7.csv')
t0=time.perf_counter()
mtd_plain = arr(PairMTDModel(pairs=LAG7_PAIRS, base=MTDModel.load(R/'artifacts/mtd_01/plain.json')).predict(vq, vh))
secs=time.perf_counter()-t0
def official(a): return contest_score(dict(zip(ids,a.tolist())), targets, groups, reference, max_points=60)
def rw(es):
    es=set(es); w=np.zeros(len(ids))
    for c in sorted(CONDITIONS):
        ces={e for i,e in enumerate(ep) if e in es and groups[ids[i]]==c}
        b=float(np.mean([np.mean(np.sum((ref[ep==e]-truth[ep==e])**2,axis=1)) for e in ces]))
        for e in ces: w[ep==e]=1/(4*len(ces)*int(np.sum(ep==e))*max(b,1e-9))
    return w
def fit(comps, es):
    p=np.array(comps); w=rw(es)
    f=((p-truth)*np.sqrt(w)[None,:,None]).reshape(len(p),-1); g=f@f.T
    r=minimize(lambda x: float(x@g@x), np.ones(len(p))/len(p), jac=lambda x: 2*g@x, method='SLSQP',
               bounds=[(0,1)]*len(p), constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones_like(x)},
               options={'ftol':1e-12,'maxiter':200}); assert r.success
    return r.x/r.x.sum()
def build(comps):
    held=np.empty_like(comps[0]); ws=[]
    for k in range(4):
        h={e for e in episodes if folds[e]==k}; w=fit(comps, sorted(set(episodes)-h)); ws.append(w.tolist())
        m=np.array([e in h for e in ep]); held[m]=np.einsum('m,mnc->nc',w,np.array(comps))[m]
    return held, ws
c_change,_=build([frozen,tab,mtd_change,two]); c_plain,wp=build([frozen,tab,mtd_plain,two])
rep={'mtd_lag7_change_standalone_points': official(mtd_change)['technical_points'],
     'mtd_lag7_plain_standalone_points': official(mtd_plain)['technical_points'],
     'candidate_with_change_base': official(c_change)['technical_points'],
     'candidate_with_plain_base': official(c_plain)['technical_points'],
     'delta_points_from_the_validation_informed_changepoint_bit':
        official(c_change)['technical_points']-official(c_plain)['technical_points'],
     'plain_fold_weights': wp, 'predict_seconds': secs,
     'max_abs_change_minus_plain': float(np.max(np.abs(mtd_change-mtd_plain)))}
(R/'artifacts/audit/provenance/mtd_changepoint_sensitivity.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
print(json.dumps(rep,indent=2,sort_keys=True))
