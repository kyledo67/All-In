"""Demonstrate that Family T's T2 arm really has a deterministic test-time recipe:
run it end to end on main-participants-60-40/public/test, which has no labels at all."""
import os, sys, json, time
os.environ.setdefault('MPLCONFIGDIR','/tmp/all-in-mpl')
ROOT='/Users/ayan/Downloads/All-In'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT+'/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
from slot_model import win_head as W
from slot_model import pair_dict as PD
from slot_model import win_head_infer as WI
from slot_model.features import causal_index
from slot_forecasting.io import load_inputs, read_submission
from scripts.win_head_train import build
from scripts.family_t_joint_structure_memory import TEMPLATES, t1_mixture, half_lives
from scripts.verify_family_t2_independent import t2_q

start=time.perf_counter()
data=Path(ROOT+'/main-participants-60-40/public')
pkg=Path(ROOT+'/artifacts/win_head_release_exact')
model=pkg/'code/model'
kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(model/'win_head/beta_templates_deploy.json')
beta, lam, _ = WI.load_win_head(model/'win_head/population_averaged.json')
full=list(range(len(TEMPLATES))); lp=np.asarray(log_prior,dtype=np.float64)

tq, th = load_inputs(data/'test')
index = causal_index(tq, th)
ids = sorted(q.query_id for q in tq)
entries = build(tq, index)
recs=[]
for qid in ids:
    e=entries[qid]
    order=np.argsort(e['used'],kind='stable')
    assert e['star'] is not None and len(e['y'])>0, qid
    recs.append({'qid':qid,'ticks':e['used'][order].astype(float),'y':e['y'][order].astype(float),
                 'signs':PD.signs_from(e['phi'][order]),'star_full':e['star'],
                 'star_signs':e['star'][1:1+W.HORIZON][None,:],'phi':e['phi'][order],
                 'cutoff':float(e['cutoff']),'type':e['type']})
# causality assertion on the test split
worst=max(max(index[(q.episode_id,q.machine_id)])-q.cutoff for q in tq)
gaps=sorted({float(g) for r in recs for g in np.diff(r['ticks'])} |
            {float(r['cutoff']-r['ticks'][-1]) for r in recs})

class Shim:  pass
sh=Shim(); sh.index=index
hl = half_lives(sh, recs, beta, lam)
q_c1, pick, _ = t1_mixture(recs, hl, full, lp, prior_mean, kappa0, None)
tpl=[TEMPLATES[int(p)] for p in pick]
K,Hz=1.0,1e-4        # the recipe's stated global setting k1_h0.0001
q2=np.array([t2_q(recs[i],tpl[i][0],tpl[i][1],K,float(prior_mean.get(recs[i]['type'],0.44)),Hz)
             for i in range(len(recs))])
# r head: the protected release's own TEST r, used unchanged, exactly as on validation
pt=read_submission(pkg/'test_submission.csv')
A=np.array([pt[i] for i in ids]); A=A/A.sum(1,keepdims=True)
qp=1.0-A[:,0]; rp=A[:,1:]/np.maximum(qp,1e-12)[:,None]; rp=rp/rp.sum(1,keepdims=True)
P=np.column_stack((1.0-q2, q2[:,None]*rp)); P=P/P.sum(1,keepdims=True)
out={'test_queries':len(ids),'max_tick_minus_cutoff':int(worst),
     'distinct_actual_tick_gaps':gaps,
     'observation_count_min_median_max':[int(min(len(r['y']) for r in recs)),
        int(np.median([len(r['y']) for r in recs])), int(max(len(r['y']) for r in recs))],
     'max_abs_q_c1_minus_release_test_q':float(np.max(np.abs(q_c1-qp))),
     'mean_abs_q_T2_minus_q_release':float(np.mean(np.abs(q2-qp))),
     'simplex_ok':bool(np.isfinite(P).all() and P.min()>=0 and abs(P.sum(1)-1).max()<1e-12),
     'sha256_of_forecast':__import__('hashlib').sha256(P.round(12).tobytes()).hexdigest()[:32],
     'seconds':time.perf_counter()-start}
Path(ROOT+'/artifacts/verify_family_t2_independent/test_recipe_check.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
