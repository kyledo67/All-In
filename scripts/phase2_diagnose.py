"""Read-only diagnostics of frozen predictions; writes only phase2 artifacts."""
import sys,json,hashlib
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,'organizer')
import numpy as np
from slot_forecasting.io import load_inputs,read_targets,read_submission,read_rows
from slot_forecasting.contest import contest_score
from slot_model.features import causal_index
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=Path('artifacts/phase2')
qs,hs=load_inputs(D/'validation');idx=causal_index(qs,hs)
ts=read_targets(D/'validation/targets.csv');truth={t.query_id:np.array(t.probabilities) for t in ts}
pred=read_submission(Path('artifacts/final/validation_predictions.csv'))
groups={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))}
ref=read_submission(D/'validation/reference.csv')
rows=[]
for q in qs:
 h=idx[q.episode_id,q.machine_id];recent=[v for t,v in h.items() if t>=q.cutoff-16];older=[v for t,v in h.items() if q.cutoff-64<=t<q.cutoff-16]
 rp=np.bincount(recent,minlength=5)/max(1,len(recent));op=np.bincount(older,minlength=5)/max(1,len(older))
 effects=[]
 for lag in range(1,9):
  pairs=[(h[t-lag]>0,v>0) for t,v in h.items() if t>=q.cutoff-128 and t-lag in h]
  a=[v for c,v in pairs if c];b=[v for c,v in pairs if not c]
  effects.append(abs((sum(a)+4)/(len(a)+8)-(sum(b)+4)/(len(b)+8)))
 residual=np.array(pred[q.query_id])-truth[q.query_id]
 rows.append(dict(query=q.query_id,episode=q.episode_id,condition=groups[q.query_id],machine_type=q.machine_type,signal=q.signal,n_history=len(h),coverage_40=sum(t>=q.cutoff-40 for t in h),recent_win=float(1-rp[0]),win_change=float(op[0]-rp[0]),distribution_change=float(abs(rp-op).sum()),lag_strength=max(effects),error=float(residual@residual),residual=residual.tolist(),category_error=(residual**2).tolist()))
def summary(sub):
 es=defaultdict(list)
 for r in sub:es[r['episode']].append(r)
 return dict(queries=len(sub),episodes=len(es),error=float(np.mean([np.mean([r['error'] for r in rs]) for rs in es.values()])),bias=np.mean([np.mean([r['residual'] for r in rs],axis=0) for rs in es.values()],axis=0).tolist(),category_error=np.mean([np.mean([r['category_error'] for r in rs],axis=0) for rs in es.values()],axis=0).tolist())
result={'official':contest_score(pred,ts,groups,ref,max_points=60),'overall':summary(rows),'groups':{}}
for field in ('machine_type','signal','n_history','coverage_40'):
 if field=='n_history':
  result['groups'][field]={k:summary([r for r in rows if (r[field]<100)==b]) for k,b in [('short',True),('long',False)]}
 else:result['groups'][field]={str(v):summary([r for r in rows if r[field]==v]) for v in sorted({r[field] for r in rows})}
for field in ('recent_win','win_change','distribution_change','lag_strength'):
 ordered=sorted(rows,key=lambda r:r[field]);result['groups'][field]={f'quartile_{k+1}':summary(ordered[k*len(rows)//4:(k+1)*len(rows)//4]) for k in range(4)}
 result.setdefault('correlations',{})[field]={'error':float(np.corrcoef([r[field] for r in rows],[r['error'] for r in rows])[0,1]),'p0_bias':float(np.corrcoef([r[field] for r in rows],[r['residual'][0] for r in rows])[0,1])}
(O/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');(O/'diagnostic_rows.json').write_text(json.dumps(rows)+'\n')
print(json.dumps({'overall':result['overall'],'type':result['groups']['machine_type'],'signal':result['groups']['signal'],'history':result['groups']['n_history'],'correlations':result['correlations']},indent=2))
