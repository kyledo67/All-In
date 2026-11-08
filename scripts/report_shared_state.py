import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_forecasting.io import read_targets,read_rows,read_submission
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/shared_state_01'
r=json.loads((O/'evaluation.json').read_text());t=read_targets(D/'validation/targets.csv');ids=[x.query_id for x in t];groups={x['query_id']:x['condition'] for x in read_rows(D/'validation/conditions.csv',('query_id','condition'))};truth={x.query_id:np.asarray(x.probabilities) for x in t}
ref=read_submission(D/'validation/reference.csv');a=read_submission(R/'artifacts/mtd_pairs_01/combined_additive.csv');b=read_submission(O/'validation.csv')
rows=[]
for e in sorted({x.episode_id for x in t}):
 qs=[x.query_id for x in t if x.episode_id==e];err=lambda p:float(np.mean([np.sum((np.asarray(p[q])-truth[q])**2) for q in qs]))
 rows.append({'episode':e,'condition':groups[qs[0]],'baseline':err(a),'candidate':err(b),'reference':err(ref)})
rng=np.random.default_rng(314159);delta=np.zeros(2000)
for c in sorted(set(groups.values())):
 sub=[x for x in rows if x['condition']==c];draw=rng.integers(0,len(sub),(2000,len(sub)))
 d=np.asarray([x['candidate']-x['baseline'] for x in sub])[draw].mean(1)
 referr=np.asarray([x['reference'] for x in sub])[draw].mean(1)
 delta+=d/np.maximum(referr,1e-9)/4
r['paired_delta_ratio_ci95']=np.quantile(delta,[.025,.975]).tolist();r['per_episode']=rows
r['zero_payout_contribution']={}
for name,p in [('baseline',a),('candidate',b)]:
 r['zero_payout_contribution'][name]={}
 for c in sorted(set(groups.values())):
  qs=[q for q in ids if groups[q]==c];err=float(np.mean([(p[q][0]-truth[q][0])**2 for q in qs]));den=r['baseline']['by_condition'][c]['reference_error']
  r['zero_payout_contribution'][name][c]={'error':err,'normalized_contribution':err/den/4}
(O/'evaluation.json').write_text(json.dumps(r,indent=2)+'\n')
print('paired',r['paired_delta_ratio_ci95'])
