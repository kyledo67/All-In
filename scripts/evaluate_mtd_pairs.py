"""Select a pair family using train-episode OOF labels, then one exploratory validation audit."""
import sys,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_model.mtd import MTDModel
from slot_model.mtd_pairs import PairMTDModel,LAG7_PAIRS,CONTROL_PAIRS
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_submission
from slot_forecasting.contest import contest_score
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/mtd_pairs_01';O.mkdir(exist_ok=True)
tq,th=load_inputs(D/'train');vq,vh=load_inputs(D/'validation');labels=read_labels(D/'train/labels.csv',tq)
folds=json.loads((R/'artifacts/phase2/folds.json').read_text())['train']
configs={'additive':(), 'lag7':LAG7_PAIRS, 'control':CONTROL_PAIRS}
train_oof={k:{} for k in configs};fold_runtime=[]
for fold in range(4):
 fit=[q for q in tq if folds[q.episode_id]!=fold];held=[q for q in tq if folds[q.episode_id]==fold]
 allowed={q.episode_id for q in fit};fit_h=[h for h in th if h.episode_id in allowed]
 start=time.perf_counter();base=MTDModel(changepoint=True).fit(fit,fit_h,{q.query_id:labels[q.query_id] for q in fit})
 for name,pairs in configs.items():
  model=PairMTDModel(pairs=pairs,base=base)
  train_oof[name].update(model.predict(held,[h for h in th if h.episode_id in {q.episode_id for q in held}]))
 fold_runtime.append(time.perf_counter()-start)
assert all(set(p)==set(labels) for p in train_oof.values())
# Training outcomes supply one-hot Brier surrogate. Equal mass per episode and query.
training_metric={}
for name,pred in train_oof.items():
 ep={}
 for q in tq:
  y=(0,1,2,5,20).index(labels[q.query_id]);one=np.eye(5)[y];p=np.asarray(pred[q.query_id])
  ep.setdefault(q.episode_id,[]).append(float(np.sum((p-one)**2)))
 training_metric[name]=float(np.mean([np.mean(v) for v in ep.values()]))
 write_submission(O/f'train_oof_{name}.csv',pred)
selected=min(training_metric,key=lambda n:(training_metric[n],n))
# Freeze selection before touching validation targets.
selection={'metric':'episode-equal observed-outcome Brier, because train true probability vectors are unavailable','scores':training_metric,'selected':selected,'folds_hash':hashlib.sha256((R/'artifacts/phase2/folds.json').read_bytes()).hexdigest(),'fold_seconds':fold_runtime,'prior_strength':40.,'configs':{k:list(v) for k,v in configs.items()}}
(O/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
start=time.perf_counter();base=MTDModel.load(R/'artifacts/mtd_01/change.json');val={}
for name,pairs in configs.items():
 val[name]=PairMTDModel(pairs=pairs,base=base).predict(vq,vh);write_submission(O/f'validation_{name}.csv',val[name])
score_t=read_targets(D/'validation/targets.csv');group={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))};ref=read_submission(D/'validation/reference.csv')
stack=read_submission(R/'artifacts/blend_release/evaluated_oof_predictions.csv')
sc=lambda x:contest_score(x,score_t,group,ref,max_points=60)
combined={name:{q.query_id:[.75*a+.25*b for a,b in zip(stack[q.query_id],val[name][q.query_id])] for q in vq} for name in configs}
for name,p in combined.items():write_submission(O/f'combined_{name}.csv',p)
report={'selection':selection,'validation_exposure':'Previously inspected; exploratory. Selection was frozen using train episodes.','standalone':{n:sc(p) for n,p in val.items()},'combined':{n:sc(p) for n,p in combined.items()},'frozen_stack':sc(stack),'validation_seconds':time.perf_counter()-start}
(O/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'selected':selected,'train_brier':training_metric,'combined_points':{n:v['technical_points'] for n,v in report['combined'].items()},'validation_seconds':report['validation_seconds']},indent=2))
