import os,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_forecasting.io import read_targets,read_rows,read_submission
# Kit location: $SLOT_PUBLIC_DATA, else the copy beside the repo. It used to be one
# machine's home directory, which made this report unrunnable anywhere else.
D=Path(os.environ.get('SLOT_PUBLIC_DATA') or R/'main-participants-60-40/public');O=R/'artifacts/mtd_pairs_01'
r=json.loads((O/'evaluation.json').read_text());t=read_targets(D/'validation/targets.csv');ids=[x.query_id for x in t]
groups={x['query_id']:x['condition'] for x in read_rows(D/'validation/conditions.csv',('query_id','condition'))};truth={x.query_id:np.asarray(x.probabilities) for x in t}
ref=read_submission(D/'validation/reference.csv');names=('additive','lag7','control')
pred={n:read_submission(O/f'combined_{n}.csv') for n in names};pred['frozen_stack']=read_submission(R/'artifacts/blend_release/evaluated_oof_predictions.csv')
rows=[]
for e in sorted({x.episode_id for x in t}):
 qs=[x.query_id for x in t if x.episode_id==e];v={'episode':e,'condition':groups[qs[0]]}
 for n,p in {**pred,'reference':ref}.items():v[n]=float(np.mean([np.sum((np.asarray(p[q])-truth[q])**2) for q in qs]))
 rows.append(v)
rng=np.random.default_rng(314159);ci={}
for n in ('lag7','control'):
 result=np.zeros(2000)
 for c in sorted(set(groups.values())):
  sub=[v for v in rows if v['condition']==c];draw=rng.integers(0,len(sub),(2000,len(sub)))
  delta=np.asarray([v[n]-v['additive'] for v in sub])[draw].mean(1)
  denom=np.asarray([v['reference'] for v in sub])[draw].mean(1)
  result+=delta/np.maximum(denom,1e-9)/4
 ci[n]=np.quantile(result,[.025,.975]).tolist()
contrib={}
for n,p in pred.items():
 contrib[n]={}
 for c in sorted(set(groups.values())):
  qs=[q for q in ids if groups[q]==c]
  z=np.mean([(p[q][0]-truth[q][0])**2 for q in qs]);den=r['frozen_stack']['by_condition'][c]['reference_error']
  contrib[n][c]={'p0_error':float(z),'normalized_contribution_to_mean_ratio':float(z/den/4)}
r['paired_interval_vs_additive']=ci;r['per_episode']=rows;r['zero_payout_contribution']=contrib
(O/'evaluation.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'selection':r['selection']['selected'],'intervals':ci,'zero_contribution':contrib},indent=2))
