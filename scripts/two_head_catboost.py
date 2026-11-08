"""Separate loss and positive-payout heads, matched to original sparse CatBoost."""
import json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from catboost import CatBoostClassifier,Pool
from scripts.dense_catboost import D,F,tq,th,vq,vh,labels,seeds,trees,train_features,held_features,byepisode,vbyepisode
from slot_forecasting.io import read_submission,read_targets,read_rows,write_submission
from slot_forecasting.contest import contest_score
O=R/'artifacts/two_head_01';O.mkdir(exist_ok=True)

def train_predict(x,y,w,xt,seed):
 params=dict(depth=4,l2_leaf_reg=8,learning_rate=.04,iterations=trees[seed],random_seed=seed,thread_count=2,allow_writing_files=False,verbose=False)
 win=CatBoostClassifier(loss_function='Logloss',**params)
 win.fit(Pool(x,(y>0).astype(int),weight=w*len(w)/w.sum()))
 pos=y>0;py=y[pos]-1;pw=w[pos];size=CatBoostClassifier(loss_function='MultiClass',classes_count=4,**params)
 size.fit(Pool(x[pos],py,weight=pw*len(pw)/pw.sum()))
 assert list(size.classes_)==list(range(4))
 chance=win.predict_proba(xt,thread_count=2)[:,1]
 conditional=size.predict_proba(xt,thread_count=2)
 p=np.column_stack((1-chance,chance[:,None]*conditional))
 assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-12 and p.min()>=0
 return win,size,p
start=time.perf_counter();oof={};folds=[]
for fold in range(4):
 fit=[q for q in tq if F[q.episode_id]!=fold];held=[q for q in tq if F[q.episode_id]==fold]
 x,y,w,key=train_features(fit,False,fold);xt=held_features(held,{q.episode_id for q in fit},byepisode)
 ps=[]
 for seed in seeds:
  _,_,p=train_predict(x,y,w,xt,seed);ps.append(p)
 for q,p in zip(held,np.mean(ps,axis=0)):oof[q.query_id]=p.tolist()
 folds.append({'fold':fold,'fit_rows':len(y),'held_episodes':len({q.episode_id for q in held})})
 print('fold',fold,round(time.perf_counter()-start,1),flush=True)
write_submission(O/'train_oof.csv',oof)
base=read_submission(R/'artifacts/dense_catboost_01/train_oof_sparse.csv')
per={}
for name,p in [('baseline',base),('two_head',oof)]:
 per[name]={}
 for e in sorted(set(F)):
  qs=[q for q in tq if q.episode_id==e]
  per[name][e]=float(np.mean([np.sum((np.asarray(p[q.query_id])-np.eye(5)[(0,1,2,5,20).index(labels[q.query_id])])**2) for q in qs]))
train_score={n:float(np.mean(list(v.values()))) for n,v in per.items()}
print('train OOF Brier',train_score,flush=True)
# No model tuning or head substitution using validation targets.
x,y,w,key=train_features(tq,False,'all');xt=held_features(vq,set(F),vbyepisode);ps=[]
for seed in seeds:
 win,size,p=train_predict(x,y,w,xt,seed);ps.append(p)
 win.save_model(O/f'win_{seed}.cbm');size.save_model(O/f'size_{seed}.cbm')
forecast={q.query_id:p.tolist() for q,p in zip(vq,np.mean(ps,axis=0))};write_submission(O/'validation.csv',forecast)
old=read_submission(R/'artifacts/dense_catboost_01/validation_sparse.csv')
t=read_targets(D/'validation/targets.csv');groups={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))};ref=read_submission(D/'validation/reference.csv')
score=lambda p:contest_score(p,t,groups,ref,max_points=60)
report={'train_oof_brier':train_score,'train_episode_errors':per,'validation':{'baseline':score(old),'two_head':score(forecast)},'runtime_seconds':time.perf_counter()-start,'folds':folds,'architecture':'same sparse observable features, machine/episode weights, depth4/L2=8, learning rate .04, seeds, 214/307 trees; binary Logloss win head and 4-class positive MultiClass head','validation_exposure':'Public validation repeatedly examined; exploratory. No validation probability targets in fitting.'}
(O/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
print('validation', {n:x['technical_points'] for n,x in report['validation'].items()},'seconds',report['runtime_seconds'])
