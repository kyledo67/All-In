"""Diagnostic summary of fixed-architecture training density experiment."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_forecasting.io import read_targets,read_rows,read_submission
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/dense_catboost_01'
r=json.loads((O/'evaluation.json').read_text());t=read_targets(D/'validation/targets.csv');ids=[x.query_id for x in t]
groups={x['query_id']:x['condition'] for x in read_rows(D/'validation/conditions.csv',('query_id','condition'))}
truth={x.query_id:np.asarray(x.probabilities) for x in t};ref=read_submission(D/'validation/reference.csv')
p={n:read_submission(O/f'validation_{n}.csv') for n in ('sparse','dense')}
rows=[]
for e in sorted({x.episode_id for x in t}):
 qs=[x.query_id for x in t if x.episode_id==e]
 err=lambda pred:float(np.mean([np.sum((np.asarray(pred[q])-truth[q])**2) for q in qs]))
 rows.append({'episode':e,'condition':groups[qs[0]],'sparse':err(p['sparse']),'dense':err(p['dense']),'reference':err(ref)})
rng=np.random.default_rng(314159);delta=np.zeros(2000)
for c in sorted(set(groups.values())):
 sub=[v for v in rows if v['condition']==c];draw=rng.integers(0,len(sub),(2000,len(sub)))
 d=np.asarray([v['dense']-v['sparse'] for v in sub])[draw].mean(1)
 den=np.asarray([v['reference'] for v in sub])[draw].mean(1)
 delta+=d/np.maximum(den,1e-9)/4
r['paired_dense_minus_sparse_ratio_ci95']=np.quantile(delta,[.025,.975]).tolist();r['validation_episode_errors']=rows
r['p0_normalized_contribution']={}
for name,pred in p.items():
 r['p0_normalized_contribution'][name]={}
 for c in sorted(set(groups.values())):
  qs=[q for q in ids if groups[q]==c];err=float(np.mean([(pred[q][0]-truth[q][0])**2 for q in qs]))
  den=r['validation']['sparse']['by_condition'][c]['reference_error']
  r['p0_normalized_contribution'][name][c]=err/den/4
(O/'evaluation.json').write_text(json.dumps(r,indent=2)+'\n')
print({'paired_ci':r['paired_dense_minus_sparse_ratio_ci95'],'p0':{n:sum(v.values()) for n,v in r['p0_normalized_contribution'].items()}})
