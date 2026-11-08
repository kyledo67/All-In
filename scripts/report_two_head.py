import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_forecasting.io import read_targets,read_rows,read_submission
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/two_head_01'
r=json.loads((O/'evaluation.json').read_text());t=read_targets(D/'validation/targets.csv');ids=[x.query_id for x in t]
g={x['query_id']:x['condition'] for x in read_rows(D/'validation/conditions.csv',('query_id','condition'))}
truth={x.query_id:np.asarray(x.probabilities) for x in t};ref=read_submission(D/'validation/reference.csv')
p={'baseline':read_submission(R/'artifacts/dense_catboost_01/validation_sparse.csv'),'two_head':read_submission(O/'validation.csv')}
rows=[]
for e in sorted({x.episode_id for x in t}):
 qs=[x.query_id for x in t if x.episode_id==e];err=lambda pp:float(np.mean([np.sum((np.asarray(pp[q])-truth[q])**2) for q in qs]))
 rows.append({'episode':e,'condition':g[qs[0]],'baseline':err(p['baseline']),'two_head':err(p['two_head']),'reference':err(ref)})
rng=np.random.default_rng(314159);delta=np.zeros(2000)
for c in sorted(set(g.values())):
 sub=[x for x in rows if x['condition']==c];draw=rng.integers(0,len(sub),(2000,len(sub)))
 d=np.asarray([x['two_head']-x['baseline'] for x in sub])[draw].mean(1);den=np.asarray([x['reference'] for x in sub])[draw].mean(1)
 delta+=d/np.maximum(den,1e-9)/4
r['paired_delta_ratio_ci95']=np.quantile(delta,[.025,.975]).tolist();r['validation_episode_errors']=rows
r['p0_normalized_contribution']={}
for n,pp in p.items():
 r['p0_normalized_contribution'][n]={}
 for c in sorted(set(g.values())):
  qs=[q for q in ids if g[q]==c];err=float(np.mean([(pp[q][0]-truth[q][0])**2 for q in qs]));den=r['validation']['baseline']['by_condition'][c]['reference_error']
  r['p0_normalized_contribution'][n][c]=err/den/4
(O/'evaluation.json').write_text(json.dumps(r,indent=2)+'\n')
print({'ci':r['paired_delta_ratio_ci95'],'p0':{n:sum(v.values()) for n,v in r['p0_normalized_contribution'].items()}})
