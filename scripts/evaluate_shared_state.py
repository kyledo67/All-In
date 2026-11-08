"""Train-only episode-state calibration after pair corrections failed."""
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_model.shared_state import neighbor_state,adjust
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_submission
from slot_forecasting.contest import contest_score
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/shared_state_01';O.mkdir(exist_ok=True)
tq,th=load_inputs(D/'train');vq,vh=load_inputs(D/'validation');labels=read_labels(D/'train/labels.csv',tq)
folds=json.loads((R/'artifacts/phase2/folds.json').read_text())['train']
train_base=read_submission(R/'artifacts/mtd_pairs_01/train_oof_additive.csv')
# For held train episodes, estimate the emission rate outside their entire fold.
state={}
for fold in range(4):
 held=[q for q in tq if folds[q.episode_id]==fold];allowed={q.episode_id for q in tq if folds[q.episode_id]!=fold}
 observed=[r.payout>0 for r in th if r.episode_id in allowed]
 held_history=[r for r in th if r.episode_id in {q.episode_id for q in held}]
 state.update(neighbor_state(held,held_history,float(np.mean(observed))))
# Fit a one-parameter win probability adjustment on training-only OOF predictions.
# Equal episode mass; no public validation target enters this step.
by_episode={}
for q in tq:by_episode.setdefault(q.episode_id,[]).append(q)
def brier(beta,episodes):
 return float(np.mean([np.mean([np.sum((np.asarray(adjust(train_base[q.query_id],state[q.query_id],beta))-
      np.eye(5)[(0,1,2,5,20).index(labels[q.query_id])])**2) for q in by_episode[e]]) for e in episodes]))
all_eps=set(by_episode);candidate=np.linspace(-.16,.16,17)
fold_fit=[];oof={}
for fold in range(4):
 held={e for e in all_eps if folds[e]==fold};fit=all_eps-held
 beta=min(candidate,key=lambda v:(brier(v,fit),abs(v)))
 fold_fit.append({'fold':fold,'beta':float(beta),'held_brier_baseline':brier(0,held),'held_brier_candidate':brier(beta,held)})
 for q in tq:
  if q.episode_id in held:oof[q.query_id]=adjust(train_base[q.query_id],state[q.query_id],beta)
selected=min(candidate,key=lambda v:(brier(v,all_eps),abs(v)))
selection={'beta':float(selected),'train_oof_brier_baseline':brier(0,all_eps),'train_oof_brier_candidate':brier(selected,all_eps),'folds':fold_fit,'emission_rate':'estimated outside held training episode fold','state':'two-state Markov filter from earlier other-machine wins; 0.97 self-transition; low/high rates global training mean +/-0.08','fold_hash':hashlib.sha256((R/'artifacts/phase2/folds.json').read_bytes()).hexdigest()}
(O/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');write_submission(O/'train_oof.csv',oof)
# One exploratory audit of the frozen training-selected adjustment.
rate=float(np.mean([r.payout>0 for r in th]));s=neighbor_state(vq,vh,rate)
base=read_submission(R/'artifacts/mtd_pairs_01/combined_additive.csv')
forecast={q.query_id:adjust(base[q.query_id],s[q.query_id],selected) for q in vq}
write_submission(O/'validation.csv',forecast)
t=read_targets(D/'validation/targets.csv');groups={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))};ref=read_submission(D/'validation/reference.csv')
sc=lambda p:contest_score(p,t,groups,ref,max_points=60)
report={'selection':selection,'baseline':sc(base),'candidate':sc(forecast),'exposure':'Previously inspected public validation; exploratory, not independent.','no_test_prediction':True}
(O/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'selection':selection,'base_points':report['baseline']['technical_points'],'candidate_points':report['candidate']['technical_points']},indent=2))
