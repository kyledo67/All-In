"""Bounded CatBoost experiment with nested upstream feature cross-fitting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/all-in-mpl')
import json,hashlib,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slot_model import runtime
import numpy as np
from catboost import CatBoostClassifier,Pool
from slot_model.phase2_features import fit_prior,examples,matrix,hmm_arrays,local_probability
from slot_model.features import causal_index,cached_prepare
from slot_model.model import fit_population,predict_batch,serialize,deserialize
from slot_forecasting.io import load_inputs,read_labels,write_submission
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=Path('artifacts/phase2');C=json.loads((O/'protocol.json').read_text());F=json.loads((O/'folds.json').read_text())

def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
def hash_files(paths):
 h=hashlib.sha256()
 for p in paths:h.update(str(p).encode());h.update(Path(p).read_bytes())
 return h.hexdigest()

def prepare():
 inputs={s:load_inputs(D/s) for s in ('train','validation')}
 tq,th=inputs['train'];idx=causal_index(tq,th);vq,vh=inputs['validation'];vidx=causal_index(vq,vh)
 labels=read_labels(D/'train/labels.csv',tq)
 train=examples(tq,idx,labels,historical=True);val=examples(vq,vidx)
 seq,_=cached_prepare(D/'train',tq,th,training=True)
 owner_ep={q.query_id:q.episode_id for q in tq};seqep=np.array([owner_ep[x] for x in seq['owners']])
 all_ep=set(F['train'])
 provenance={};pops={}
 def population(allowed):
  key=hashlib.sha256('|'.join(sorted(allowed)).encode()).hexdigest()[:16]
  path=O/'hmm_models'/f'{key}.json'
  if key not in pops:
   if path.exists():p=deserialize(json.loads(path.read_text()))
   elif allowed==all_ep:
    p=deserialize(json.loads(Path('artifacts/experiment/population.json').read_text()));save(path,serialize(p))
   else:
    mask=np.array([e in allowed for e in seqep]);subset={k:v[mask] for k,v in seq.items() if k in ('x','y','owners')}
    _,p,initial,diag=fit_population(subset);save(path,serialize(p));save(path.with_name(key+'_diagnostics.json'),diag)
   pops[key]=p
  provenance[key]={'trained_episodes':sorted(allowed),'model_file':str(path),'hash':hashlib.sha256(path.read_bytes()).hexdigest()}
  return pops[key],key
 def produce(es,index,scope,upstream_sets):
  # upstream_sets maps each example episode to its permitted fitting episode set.
  digest=hash_files(['slot_model/phase2_features.py','slot_model/features.py','slot_model/model.py',__file__,'organizer/slot_forecasting/transition.py',O/'protocol.json',O/'folds.json',O/'requirements.lock',D/'train/history.csv',D/'train/queries.csv',D/'train/labels.csv',D/'validation/queries.csv',D/'validation/history.csv'])
  path=O/'cache'/f'{scope}-{digest}.npz';path.parent.mkdir(exist_ok=True)
  if path.exists():
   with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
  priors={};hmm=np.zeros((len(es),7));sets={tuple(sorted(a)) for a in upstream_sets.values()}
  model_keys=[]
  for allowed_tuple in sorted(sets):
   allowed=set(allowed_tuple);p,key=population(allowed);prior=fit_prior(tq,idx,allowed)
   selected=[i for i,e in enumerate(es) if upstream_sets[e['q'].episode_id]==allowed]
   assert all(es[i]['q'].episode_id not in allowed for i in selected)
   batch=[es[i] for i in selected]
   for e in batch:priors[e['q'].episode_id]=prior
   arrays=hmm_arrays(batch);prob,_,_=predict_batch(p,arrays)
   # Same upstream filter gives terminal regime entropy; no new fitted component.
   import jax,jax.numpy as jnp
   from slot_model.model import new_control
   m=new_control()
   f=jax.jit(jax.vmap(lambda y,x:m.filter(p,y,x).filtered_probs[-1]))
   states=np.asarray(f(jnp.asarray(arrays['y']),jnp.asarray(arrays['x'])))
   entropy=-np.sum(states*np.log(np.maximum(states,1e-15)),axis=1)
   hmm[selected,:5]=prob;hmm[selected,5]=entropy;hmm[selected,6]=states.max(1)
   model_keys.append(key)
  x,names=matrix(es,index,priors)
  local=np.array([[local_probability(e['q'],e['h'],priors[e['q'].episode_id],**c) for c in C['local']] for e in es])
  result=dict(x=x,hmm=hmm,local=local,names=np.array(names),ids=np.array([e['q'].query_id for e in es]),ep=np.array([e['q'].episode_id for e in es]),y=np.array([-1 if e['y'] is None else e['y'] for e in es]),weight=np.array([e['weight'] for e in es]),known=np.array([e['signal_known'] for e in es]),cutoff=np.array([e['q'].cutoff for e in es]))
  np.savez_compressed(path,**result);save(path.with_suffix('.json'),{'cache_hash':digest,'scope':scope,'rows':len(es),'features':len(names),'upstream_model_keys':model_keys,'own_episode_excluded':True})
  return result
 fullsets={e:{a for a in all_ep if F['train'][a]!=F['train'][e]} for e in all_ep}
 print('Preparing training cross-fit features',flush=True);tr=produce(train,idx,'train',fullsets)
 remaining={e for e in all_ep if F['train'][e]!=0}
 stoptrain=[e for e in train if e['q'].episode_id in remaining]
 stopeval=[e for e in train if e['q'].episode_id not in remaining]
 nested={e:{a for a in remaining if F['train'][a]!=F['train'][e]} for e in remaining}
 print('Preparing nested early-stopping features',flush=True);st=produce(stoptrain,idx,'stop_train',nested)
 se=produce(stopeval,idx,'stop_eval',{e:remaining for e in all_ep-remaining})
 print('Preparing validation features',flush=True);va=produce(val,vidx,'validation',{q.episode_id:all_ep for q in vq})
 fullprior=fit_prior(tq,idx,all_ep);save(O/'population_prior.json',fullprior)
 save(O/'upstream_feature_provenance.json',provenance)
 save(O/'feature_schema.json',{'observable':tr['names'].tolist(),'hmm':['hmm_p0','hmm_p1','hmm_p2','hmm_p5','hmm_p20','regime_entropy','regime_max']})
 return tr,st,se,va

def run():
 start=time.perf_counter();tr,st,se,va=prepare();save(O/'preparation.json',{'seconds':time.perf_counter()-start,'train_rows':len(tr['y']),'historical_rows':int(sum(~tr['known'])),'actual_query_rows':int(sum(tr['known']))})
 # Persist prepared matrix lookup to reuse for release without rebuilding.
 for name,a in [('train',tr),('validation',va)]:np.savez_compressed(O/'cache'/f'{name}_ready.npz',**a)
 for i,c in enumerate(C['local']):write_submission(O/'predictions'/f'local_{i}.csv',dict(zip(va['ids'].tolist(),va['local'][:,i].tolist())))
 def xy(data,variant):return data['x'] if variant=='observable' else np.column_stack((data['x'],data['hmm']))
 records=[]
 for variant in ('observable','hmm'):
  for ci,config in enumerate(C['catboost']):
   for seed in C['seeds']:
    name=f'cb_{variant}_{ci}_{seed}';path=O/'models'/f'{name}.cbm';diag=O/'models'/f'{name}.json'
    if path.exists() and diag.exists() and (O/'predictions'/f'{name}.csv').exists():records.append(json.loads(diag.read_text()));continue
    begin=time.perf_counter();params=dict(loss_function='MultiClass',classes_count=5,iterations=C['max_iterations'],learning_rate=C['learning_rate'],random_seed=seed,thread_count=2,allow_writing_files=False,verbose=False,**config)
    # Weights normalize episodes/machines; no automatic class balancing.
    stop=CatBoostClassifier(**params)
    stop.fit(Pool(xy(st,variant),st['y'],weight=st['weight']*len(st['weight'])/sum(st['weight'])),eval_set=Pool(xy(se,variant),se['y'],weight=se['weight']*len(se['weight'])/sum(se['weight'])),early_stopping_rounds=C['early_stopping_rounds'],use_best_model=True)
    trees=stop.tree_count_;params['iterations']=trees
    m=CatBoostClassifier(**params);m.fit(Pool(xy(tr,variant),tr['y'],weight=tr['weight']*len(tr['weight'])/sum(tr['weight'])))
    training_seconds=time.perf_counter()-begin
    assert list(m.classes_)==list(range(5));begin=time.perf_counter();p=m.predict_proba(xy(va,variant),thread_count=2);inference_seconds=time.perf_counter()-begin
    assert np.isfinite(p).all() and np.max(abs(p.sum(1)-1))<1e-12
    path.parent.mkdir(exist_ok=True);m.save_model(path)
    write_submission(O/'predictions'/f'{name}.csv',dict(zip(va['ids'].tolist(),p.tolist())))
    record={'name':name,'variant':variant,'config':config,'seed':seed,'trees':trees,'best_iteration':stop.get_best_iteration(),'training_seconds':training_seconds,'validation_inference_seconds':inference_seconds,'parameters':m.get_all_params(),'feature_importance':m.get_feature_importance().tolist()}
    save(diag,record);records.append(record);print(name,'trees',trees,'seconds',training_seconds,flush=True)
 save(O/'training_runs.json',records)

if __name__=='__main__':run()
