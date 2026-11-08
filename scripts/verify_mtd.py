import sys,json
from pathlib import Path
from dataclasses import replace
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
from slot_model.mtd import MTDModel,inputs,index,em
from slot_forecasting.io import load_inputs
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/mtd_01'
q,h=load_inputs(D/'validation');q=q[:3];m=MTDModel.load(O/'plain.json');a=m.predict(q,h)
# Alter/add outcomes at and after each target cutoff; no forecast may change.
for query in q:
 modified=[replace(r,payout=20) if r.episode_id==query.episode_id and r.tick>=query.cutoff else r for r in h]
 sample=next(r for r in h if r.episode_id==query.episode_id)
 modified.append(replace(sample,tick=query.cutoff,payout=20))
 assert m.predict([query],modified)[query.query_id]==a[query.query_id]
for p in a.values():assert np.isfinite(p).all() and min(p)>=0 and abs(sum(p)-1)<1e-12
idx=index(h);query=q[0];idx[query.episode_id,query.machine_id].pop(query.cutoff-2,None)
assert inputs(query,query.cutoff,idx)[1]==5
# Deterministic EM and an independent scalar mixture prediction check.
rng=np.random.default_rng(42);x=rng.integers(0,6,(100,11));y=rng.integers(0,5,100)
l,Q=em(x,y,np.ones(100),np.ones(5)/5,4)
l2,Q2=em(x,y,np.ones(100),np.ones(5)/5,4)
assert np.array_equal(l,l2) and np.array_equal(Q,Q2)
p=np.einsum('j,jc->c',l,Q[np.arange(11),x[0]])
assert np.allclose(p,[sum(float(l[j])*float(Q[j,x[0,j],c]) for j in range(11)) for c in range(5)])
(O/'verification.json').write_text(json.dumps({'cutoff_invariance':True,'unknown_lag':True,'normalized':True,'deterministic_em':True,'scalar_mixture_reference':True},indent=2))
print('MTD checks passed')
