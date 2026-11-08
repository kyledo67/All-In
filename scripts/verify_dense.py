"""Check sparse generator parity and dense cutoff safety."""
import sys
from dataclasses import replace
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
from slot_forecasting.io import load_inputs,read_labels
from slot_model.features import causal_index
from slot_model.phase2_features import examples,observable,fit_prior
from scripts.dense_catboost import make_examples,idx,tq,labels,byepisode
sample=[q for q in tq if q.episode_id==tq[0].episode_id]
a=examples(sample,idx,labels,historical=True)
b=make_examples(sample,False)
assert len(a)==len(b)
for x,y in zip(a,b):
 q,h,target,known,weight=y
 assert x['q']==q and x['h']==h and x['y']==target and x['signal_known']==known
 assert abs(x['weight']-weight)<1e-15
for q,h,y,known,w in make_examples(sample,True):
 assert all(t<q.cutoff for t in h)
 if not known:
  assert q.signal==0 and q.query_id.endswith(str(q.cutoff))
 prior=fit_prior(tq,idx,{v.episode_id for v in tq if v.episode_id!=q.episode_id})
 f=observable(q,h,byepisode[q.episode_id],prior,known)
 for lag in range(1,9):
  assert f[f'lag{lag}_missing']==float(q.cutoff-lag not in h)
print('dense causal and sparse parity checks passed')
