"""Fixed-architecture sparse/dense CatBoost comparison with episode folds."""
import hashlib,json,os,sys,time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'organizer')]
import numpy as np
from catboost import CatBoostClassifier,Pool
from slot_model.features import causal_index,PAYOUTS
from slot_model.phase2_features import fit_prior,observable,sample_history
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_submission
from slot_forecasting.contest import contest_score
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');OUT=ROOT/'artifacts/dense_catboost_01';OUT.mkdir(exist_ok=True)
F=json.loads((ROOT/'artifacts/phase2/folds.json').read_text())['train'];schema=json.loads((ROOT/'artifacts/phase2/feature_schema.json').read_text())['observable']
protocol=json.loads((ROOT/'artifacts/phase2/protocol.json').read_text());seeds=protocol['seeds'];trees={20261108:214,20261109:307}
tq,th=load_inputs(D/'train');vq,vh=load_inputs(D/'validation');labels=read_labels(D/'train/labels.csv',tq);idx=causal_index(tq,th);vidx=causal_index(vq,vh)
byepisode={}
for key,h in idx.items():byepisode.setdefault(key[0],{})[key]=h
vbyepisode={}
for key,h in vidx.items():vbyepisode.setdefault(key[0],{})[key]=h

# Exact historical target distribution except density: same range, full recent40,
# alternating sampled-older pattern, missing signal flag, and equal machine mass.
def make_examples(queries,dense):
 out=[]
 for q in sorted(queries,key=lambda q:q.query_id):
  machine=idx[q.episode_id,q.machine_id];local=[]
  cutoffs=range(192,q.cutoff) if dense else list(range(192,q.cutoff,8))[-8:]
  for cutoff in cutoffs:
   if cutoff not in machine:continue
   qq=replace(q,query_id=q.query_id+f'-t{cutoff}',cutoff=cutoff,signal=0)
   h=sample_history(qq,idx,thin=(cutoff//8)%2==0)
   if any(t not in h for t in range(cutoff-40,cutoff)):continue
   local.append((qq,h,machine[cutoff],False))
  # Keep each machine's original real-query versus historical-example mass.
  # The dense arm divides the same historical mass over more eligible cutoffs.
  sparse_count=0
  for cutoff in list(range(192,q.cutoff,8))[-8:]:
   if cutoff not in machine:continue
   qq=replace(q,query_id=q.query_id+f'-t{cutoff}',cutoff=cutoff,signal=0)
   h=sample_history(qq,idx,thin=(cutoff//8)%2==0)
   if all(t in h for t in range(cutoff-40,cutoff)):sparse_count+=1
  historical_mass=sparse_count/(1+sparse_count)
  each_historical=historical_mass/len(local) if local else 0.
  for qq,h,y,known in local:out.append((qq,h,y,known,each_historical/8))
  out.append((q,sample_history(q,idx),PAYOUTS.index(labels[q.query_id]),True,1/(1+sparse_count)/8))
 return out

def matrix_examples(es,priors,source):
 records=[observable(q,h,source[q.episode_id],priors[q.episode_id],known) for q,h,y,known,w in es]
 assert all(list(r)==schema for r in records)
 return np.asarray([[r[n] for n in schema] for r in records],float)

def train_features(queries,dense,held):
 allowed={q.episode_id for q in queries};es=make_examples(queries,dense)
 # The prior for an example excludes its whole train-episode inner fold AND
 # the held outer fold. For full release, this is the frozen train fold map.
 priors={}
 for fold in range(4):
  fit={e for e in allowed if F[e]!=fold}
  prior=fit_prior(tq,idx,fit)
  for e in allowed:
   if F[e]==fold:priors[e]=prior
 assert set(priors)==allowed
 cache_key=hashlib.sha256((str(dense)+'|'+str(held)+'|'+str(sorted(allowed))+'|'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'|'+hashlib.sha256((D/'train/history.csv').read_bytes()).hexdigest()).encode()).hexdigest()[:20]
 cache=OUT/f'features_{cache_key}.npz'
 if cache.exists():
  with np.load(cache) as z:return z['x'],z['y'],z['w'],cache_key
 x=matrix_examples(es,priors,byepisode);y=np.array([e[2] for e in es]);w=np.array([e[4] for e in es]);np.savez_compressed(cache,x=x,y=y,w=w)
 return x,y,w,cache_key

def held_features(queries,fit_episodes,source):
 prior=fit_prior(tq,idx,fit_episodes);priors={q.episode_id:prior for q in queries}
 es=[(q,sample_history(q,idx) if source is byepisode else sample_history(q,vidx),None,True,1) for q in queries]
 return matrix_examples(es,priors,source)

def fit_predict(x,y,w,test,seed):
 m=CatBoostClassifier(loss_function='MultiClass',classes_count=5,iterations=trees[seed],learning_rate=.04,random_seed=seed,thread_count=2,allow_writing_files=False,verbose=False,depth=4,l2_leaf_reg=8)
 m.fit(Pool(x,y,weight=w*len(w)/w.sum()))
 assert list(m.classes_)==list(range(5))
 return m,m.predict_proba(test,thread_count=2)

def main():
 start=time.perf_counter();results={};oof={k:{} for k in ('sparse','dense')};cache_ids={};folds=[]
 for fold in range(4):
  fit=[q for q in tq if F[q.episode_id]!=fold];held=[q for q in tq if F[q.episode_id]==fold];allowed={q.episode_id for q in fit}
  hx=held_features(held,allowed,byepisode);fr={'fold':fold,'held_episodes':len({q.episode_id for q in held}),'rows':{}}
  for name,dense in [('sparse',False),('dense',True)]:
   x,y,w,key=train_features(fit,dense,fold);cache_ids[f'{name}_{fold}']=key;ps=[]
   for seed in seeds:
    _,p=fit_predict(x,y,w,hx,seed);ps.append(p)
   p=np.mean(ps,axis=0)
   for q,v in zip(held,p):oof[name][q.query_id]=v.tolist()
   fr['rows'][name]=len(y)
  folds.append(fr);print('fold',fold,fr['rows'],round(time.perf_counter()-start,1),flush=True)
 for name,p in oof.items():write_submission(OUT/f'train_oof_{name}.csv',p)
 # Equal episode, equal query observed-outcome Brier. No validation targets.
 brier={};per_episode={}
 for name,p in oof.items():
  per_episode[name]={}
  for e in sorted(set(F)):
   qs=[q for q in tq if q.episode_id==e]
   per_episode[name][e]=float(np.mean([np.sum((np.asarray(p[q.query_id])-np.eye(5)[PAYOUTS.index(labels[q.query_id])])**2) for q in qs]))
  brier[name]=float(np.mean(list(per_episode[name].values())))
 selection={'method':'fixed depth4/L2=8 observable CatBoost, 2 saved seeds, saved 214/307 tree counts; only historical cutoff density changes','sparse':brier['sparse'],'dense':brier['dense'],'folds':folds,'cache_ids':cache_ids,'no_validation_targets_used':True,'seconds':time.perf_counter()-start}
 (OUT/'train_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 print('train OOF',brier,flush=True)
 # One exploratory full-train validation comparison. No setting is chosen here.
 valx=held_features(vq,set(F),vbyepisode)
 vp={}
 for name,dense in [('sparse',False),('dense',True)]:
  x,y,w,key=train_features(tq,dense,'all');cache_ids[f'{name}_all']=key;ps=[]
  for seed in seeds:
   m,p=fit_predict(x,y,w,valx,seed);ps.append(p)
   m.save_model(OUT/f'{name}_{seed}.cbm')
  vp[name]={q.query_id:p.tolist() for q,p in zip(vq,np.mean(ps,axis=0))};write_submission(OUT/f'validation_{name}.csv',vp[name])
 t=read_targets(D/'validation/targets.csv');g={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))};ref=read_submission(D/'validation/reference.csv')
 score=lambda p:contest_score(p,t,g,ref,max_points=60)
 report={'train_oof_brier':brier,'train_episode_errors':per_episode,'validation':{n:score(p) for n,p in vp.items()},'exposure':'Public validation repeatedly examined; exploratory only. No validation target in training or fitting.','runtime_seconds':time.perf_counter()-start,'cache_ids':cache_ids,'folds':folds,'selection':'No architecture or mixture weight tuning; only density varied.','data_hashes':{str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'train/history.csv',D/'train/queries.csv',D/'train/labels.csv',D/'validation/queries.csv',D/'validation/history.csv')}}
 (OUT/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
 print('validation', {n:x['technical_points'] for n,x in report['validation'].items()},'seconds',report['runtime_seconds'])
if __name__=='__main__':main()
