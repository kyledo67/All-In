"""Score fixed MTD addition against the frozen OOF stack; no weight optimization."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'organizer')]
import numpy as np
from slot_forecasting.io import read_targets,read_rows,read_submission,write_submission
from slot_forecasting.contest import contest_score
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=ROOT/'artifacts/mtd_01'
t=read_targets(D/'validation/targets.csv');ids=[x.query_id for x in t];groups={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))};ref=read_submission(D/'validation/reference.csv')
base=read_submission(ROOT/'artifacts/blend_release/evaluated_oof_predictions.csv')
fixed=.25;plain=read_submission(O/'plain.csv');change=read_submission(O/'change.csv')
score=lambda p:contest_score(p,t,groups,ref,max_points=60)
make=lambda p:{q:[(1-fixed)*a+fixed*b for a,b in zip(base[q],p[q])] for q in ids}
results={name:score(make(p)) for name,p in [('plain',plain),('change',change)]}
# Paired whole-episode bootstrap, stratified by condition, with reference recomputed.
truth={x.query_id:np.array(x.probabilities) for x in t};episode={x.query_id:x.episode_id for x in t}
rows=[]
for e in sorted(set(episode.values())):
 qs=[q for q in ids if episode[q]==e];c=groups[qs[0]]
 err=lambda pred:float(np.mean([np.sum((np.array(pred[q])-truth[q])**2) for q in qs]))
 rows.append({'episode':e,'condition':c,'reference':err(ref),'base':err(base),'plain':err(make(plain)),'change':err(make(change))})
rng=np.random.default_rng(314159);intervals={}
for name in ('plain','change'):
 draws=np.zeros(2000)
 for c in sorted(set(groups.values())):
  sub=[r for r in rows if r['condition']==c];samples=rng.integers(0,len(sub),(2000,len(sub)))
  d=np.array([r[name]-r['base'] for r in sub])[samples].mean(axis=1)
  rr=np.array([r['reference'] for r in sub])[samples].mean(axis=1)
  draws+=d/np.maximum(rr,1e-9)/4
 intervals[name]=np.quantile(draws,[.025,.975]).tolist()
folds=json.loads((ROOT/'artifacts/blend_release/model/validation_fold_map.json').read_text())
foldresults={}
for name,p in [('plain',plain),('change',change)]:
 pred=make(p);foldresults[name]=[]
 for k in range(4):
  subset=[x for x in t if folds[x.episode_id]==k];qs={x.query_id for x in subset}
  ps={q:pred[q] for q in qs};bs={q:base[q] for q in qs}
  opts=lambda z:contest_score(z,subset,{q:groups[q] for q in qs},{q:ref[q] for q in qs},max_points=60)
  foldresults[name].append({'fold':k,'base':opts(bs)['technical_points'],'candidate':opts(ps)['technical_points']})
report={'fixed_weight':fixed,'selection':'No fitted or changed stack weights. The three requested weights are descriptive sensitivity checks only.','score':results,'paired_delta_ratio_ci95':intervals,'folds':foldresults,'per_episode':rows,'exposure':'Public validation previously inspected; exploratory only. OOF stack is fold-routed; deployed equal mean has no independent score.'}
(O/'stack_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({n:{'points':r['technical_points'],'ci':intervals[n],'folds':foldresults[n]} for n,r in results.items()},indent=2))
