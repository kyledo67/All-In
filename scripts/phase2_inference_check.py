"""Verify saved challenger inference and measure feature/prediction costs once."""
import sys,time,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,'organizer')
import numpy as np
from catboost import CatBoostClassifier
from slot_forecasting.io import load_inputs,read_submission
from slot_model.features import causal_index
from slot_model.phase2_features import examples,matrix,local_probability
from slot_model.phase2_predict import hmm_summary
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=Path('artifacts/phase2')
start=time.perf_counter();qs,hs=load_inputs(D/'validation');idx=causal_index(qs,hs);es=examples(qs,idx);loading=time.perf_counter()-start
p=json.loads((O/'population_prior.json').read_text());config=json.loads((O/'release_config.json').read_text());proto=json.loads((O/'protocol.json').read_text())
start=time.perf_counter();local=np.array([local_probability(e['q'],e['h'],p,**proto['local'][config['config_indices']['local']]) for e in es]);local_time=time.perf_counter()-start
start=time.perf_counter();x,names=matrix(es,idx,{q.episode_id:p for q in qs});feature_time=time.perf_counter()-start
population=json.loads(Path('artifacts/experiment/population.json').read_text())
start=time.perf_counter();h,hsummary=hmm_summary(population,es);hmm_time=time.perf_counter()-start
with np.load(O/'cache/validation_ready.npz') as z:
 assert np.array_equal(x,z['x']);hmm_diff=float(np.max(np.abs(hsummary-z['hmm'])));assert hmm_diff<1e-12
metrics={'validation_queries':len(qs),'load_index_seconds':loading,'local_expert_seconds':local_time,'observable_features_seconds':feature_time,'hmm_summary_seconds':hmm_time,'observable_features_exact_match':True,'portable_vs_jax_hmm_summary_max_abs':hmm_diff,'catboost':{}}
for variant in ('observable','hmm'):
 values=x if variant=='observable' else np.column_stack((x,hsummary));i=config['config_indices'][variant]
 start=time.perf_counter();probs=[]
 for seed in proto['seeds']:
  m=CatBoostClassifier();m.load_model(O/'models'/f'cb_{variant}_{i}_{seed}.cbm');pred=m.predict_proba(values,thread_count=2);saved=read_submission(O/'predictions'/f'cb_{variant}_{i}_{seed}.csv')
  reference=np.array([saved[e['q'].query_id] for e in es]);diff=float(np.max(abs(pred-reference)));assert diff<1e-12;probs.append(pred)
 seconds=time.perf_counter()-start
 metrics['catboost'][variant]={'load_and_predict_two_models_seconds':seconds,'total_with_required_features_seconds':loading+feature_time+seconds+(hmm_time if variant=='hmm' else 0),'last_seed_max_abs_difference':diff}
(O/'inference_checks.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
