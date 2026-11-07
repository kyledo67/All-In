import json,hashlib,sys,subprocess
from pathlib import Path
sys.path.insert(0,'organizer')
from slot_forecasting.io import load_inputs,read_rows
O=Path('artifacts/phase2');D=Path('/Users/kyledo/Downloads/main-participants-60-40/public')
folds={}
for split in ('train','validation'):
 q,_=load_inputs(D/split);groups={r['query_id']:r['condition'] for r in read_rows(D/split/'conditions.csv',('query_id','condition'))};ep={x.episode_id:groups[x.query_id] for x in q}
 assignment={}
 for c in sorted(set(ep.values())):
  ordered=sorted([e for e in ep if ep[e]==c],key=lambda e:hashlib.sha256(('phase2-v1|'+e).encode()).hexdigest())
  assignment.update({e:i%4 for i,e in enumerate(ordered)})
 folds[split]=assignment
(O/'folds.json').write_text(json.dumps(folds,indent=2)+'\n')
config={'version':'phase2-v1','seeds':[20261108,20261109],'folds':4,'local':[{'half_life':h,'strength':s} for h in (16,48,128) for s in (8,32)],'catboost':[{'depth':4,'l2_leaf_reg':8},{'depth':6,'l2_leaf_reg':16}], 'learning_rate':.04,'max_iterations':400,'early_stopping_rounds':50,'thread_count':2,'classes':[0,1,2,5,20], 'historical_cutoffs':'192..cutoff-1 every8; max8; only observed labels and complete preceding40 ticks; alternate thinning to last40+24 old', 'train_weights':'equal episode, equal machine within episode, divided over that machine examples; no class/value weighting', 'base_training':'Only public/train observed outcomes and histories. No validation probability target in base fits.', 'prior_and_hmm_crossfit':'Four train-episode folds; early-stop fold0 uses nested crossfit within other3 folds; validation/test upstream models fitted on all train only.', 'evaluation':'Four stratified episode development folds; all validation previously exposed. Configs and convex weights selected on other3 folds; concatenate held-fold predictions and call official scorer.', 'independent_evaluation':'None available. Hidden test remains only unexamined generalization evaluation.', 'ensemble':'Global simplex weights on four predeclared components, only if residual complementarity is present; optimizer fits other3 development folds, no condition input.', 'replacement_gate':{'minimum_points_gain':.5,'minimum_folds_improved':3,'max_condition_ratio_regression':.10,'both_seeds_improve':True},'ensemble_gate':{'minimum_points_gain_over_best_single':.25,'minimum_folds_improved':3,'bootstrap_upper_delta_ratio_below':0},'budget':'6 local configs; 2 CatBoost configs x2 feature variants x2 seeds; 8 early-stop plus8 full-train fits. No additional architectures.'}
(O/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(O/'requirements.lock').open('w'),check=True)
print('Protocol and episode folds frozen.')
