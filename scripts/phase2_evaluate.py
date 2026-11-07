"""Fold-held configuration/blend selection on previously exposed development data."""
import sys,json,csv,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.experiment import scoring_inputs,score,paired_bootstrap
from slot_forecasting.io import read_submission,write_submission
import numpy as np
from scipy.optimize import minimize
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=Path('artifacts/phase2');F=json.loads((O/'folds.json').read_text())['validation'];C=json.loads((O/'protocol.json').read_text())
S=scoring_inputs(D);targets,groups,reference=S
ids=sorted(reference);ep=np.array([next(t.episode_id for t in targets if t.query_id==q) for q in ids]);T={t.query_id:t for t in targets};truth=np.array([T[q].probabilities for q in ids]);ref=np.array([reference[q] for q in ids]);episodes=set(ep)
def load(p):d=read_submission(p);return np.array([d[q] for q in ids])
def mapping(a):return dict(zip(ids,a.tolist()))
def official(a,es=episodes):return score(mapping(a),S,es)
def row_weights(es):
 w=np.zeros(len(ids))
 for c in sorted(set(groups.values())):
  ces={e for i,e in enumerate(ep) if e in es and groups[ids[i]]==c}
  b=np.mean([np.mean(np.sum((ref[ep==e]-truth[ep==e])**2,axis=1)) for e in ces])
  for e in ces:w[ep==e]=1/(4*len(ces)*sum(ep==e)*max(b,1e-9))
 return w
hmm=load(Path('artifacts/final/validation_predictions.csv'))
local=[load(O/'predictions'/f'local_{i}.csv') for i in range(len(C['local']))]
cat={v:{i:{s:load(O/'predictions'/f'cb_{v}_{i}_{s}.csv') for s in C['seeds']} for i in range(2)} for v in ('observable','hmm')}
avg={v:[np.mean(list(cat[v][i].values()),axis=0) for i in range(2)] for v in cat}

def select(es):
 result={};result['local']=min(range(len(local)),key=lambda i:official(local[i],es)['mean_loss_ratio'])
 for v in cat:result[v]=min(range(2),key=lambda i:official(avg[v][i],es)['mean_loss_ratio'])
 return result

def blend_weights(arrays,es):
 p=np.array(arrays);w=row_weights(es)
 residual=p-truth;flat=(residual*np.sqrt(w)[None,:,None]).reshape(len(p),-1)
 gram=flat@flat.T;norm=np.sqrt(np.diag(gram));cos=gram/np.maximum(norm[:,None]*norm[None,:],1e-15)
 complementary=bool(np.min(cos)<.98)
 if not complementary:
  weights=np.eye(len(p))[np.argmin(np.diag(gram))];return weights,cos,False
 f=lambda x:float(x@gram@x)
 res=minimize(f,np.ones(len(p))/len(p),jac=lambda x:2*gram@x,method='SLSQP',bounds=[(0,1)]*len(p),constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones_like(x)},options={'ftol':1e-12,'maxiter':200})
 assert res.success,res.message
 return res.x/res.x.sum(),cos,True

names=['hmm','local','cat_observable','cat_hmm'];oof={n:np.empty_like(hmm) for n in names+['ensemble']};fold_reports=[];seed_oof={s:{n:np.empty_like(hmm) for n in ('cat_observable','cat_hmm')} for s in C['seeds']}
for fold in range(4):
 held={e for e in episodes if F[e]==fold};fit=episodes-held;sel=select(fit);arrays=[hmm,local[sel['local']],avg['observable'][sel['observable']],avg['hmm'][sel['hmm']]]
 w,cos,complement=blend_weights(arrays,fit);mask=np.array([e in held for e in ep])
 for n,a in zip(names,arrays):oof[n][mask]=a[mask]
 oof['ensemble'][mask]=np.einsum('m,mnc->nc',w,np.array(arrays)) [mask]
 for s in C['seeds']:
  for v in cat:seed_oof[s]['cat_'+v][mask]=cat[v][sel[v]][s][mask]
 fr={'fold':fold,'selection':sel,'blend_weights':dict(zip(names,w.tolist())),'training_residual_cosine':cos.tolist(),'complementary':complement,'held_episodes':sorted(held),'score':{n:official(a,held) for n,a in zip(names,arrays)}}
 fr['score']['ensemble']=official(oof['ensemble'],held);fold_reports.append(fr)
 # Assert that our optimization weights implement the official score exactly.
 for a in arrays:assert abs(np.sum(row_weights(fit)*np.sum((a-truth)**2,axis=1))-official(a,fit)['mean_loss_ratio'])<1e-12
scores={n:official(a) for n,a in oof.items()}
boot={n:paired_bootstrap(mapping(a),mapping(hmm),S,episodes) for n,a in oof.items() if n!='hmm'}
best_single=min(names,key=lambda n:scores[n]['mean_loss_ratio'])
ensemble_vs_best=paired_bootstrap(mapping(oof['ensemble']),mapping(oof[best_single]),S,episodes)
ensemble_gain=scores['ensemble']['technical_points']-scores[best_single]['technical_points']
ensemble_folds=sum(r['score']['ensemble']['technical_points']>r['score'][best_single]['technical_points'] for r in fold_reports)
keep_ensemble=(ensemble_gain>=.25 and ensemble_folds>=3 and ensemble_vs_best['delta_loss_ratio_ci95'][1]<0)
candidate='ensemble' if keep_ensemble else best_single
seed_scores={str(s):{n:official(a) for n,a in outputs.items()} for s,outputs in seed_oof.items()}
gain=scores[candidate]['technical_points']-scores['hmm']['technical_points']
fold_wins=sum(r['score'][candidate]['technical_points']>r['score']['hmm']['technical_points'] for r in fold_reports)
worst_regression=max(scores[candidate]['by_condition'][c]['loss_ratio']-scores['hmm']['by_condition'][c]['loss_ratio'] for c in scores['hmm']['by_condition'])
seed_consistent=all(v.get(candidate,scores[candidate])['technical_points']>scores['hmm']['technical_points'] for v in seed_scores.values())
# For a blended candidate, repeat the same fold weights with each seed's base outputs.
if candidate=='ensemble':
 seed_consistent=True
 for s in C['seeds']:
  a=np.empty_like(hmm)
  for fr in fold_reports:
   mask=np.array([e in fr['held_episodes'] for e in ep]);w=np.array(list(fr['blend_weights'].values()))
   components=[hmm,oof['local'],seed_oof[s]['cat_observable'],seed_oof[s]['cat_hmm']]
   a[mask]=np.einsum('m,mnc->nc',w,np.array(components))[mask]
  seed_scores[str(s)]['ensemble']=official(a);seed_consistent &= official(a)['technical_points']>scores['hmm']['technical_points']
change=(candidate!='hmm' and gain>=.5 and fold_wins>=3 and worst_regression<=.10 and seed_consistent)
selected=candidate if change else 'hmm'
release_selection=select(episodes);release_arrays=[hmm,local[release_selection['local']],avg['observable'][release_selection['observable']],avg['hmm'][release_selection['hmm']]]
rw,rcos,_=blend_weights(release_arrays,episodes)
if selected!='ensemble':rw=np.eye(4)[names.index(selected)]
release_pred=np.einsum('m,mnc->nc',rw,np.array(release_arrays))
for n,a in oof.items():write_submission(O/'predictions'/f'oof_{n}.csv',mapping(a))
write_submission(O/'predictions'/'release_validation.csv',mapping(release_pred))
with (O/'per_episode.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['episode','condition']+list(oof)+['selected_minus_hmm'])
 for e in sorted(episodes):
  errors={n:float(np.mean(np.sum((a[ep==e]-truth[ep==e])**2,axis=1))) for n,a in oof.items()}
  writer.writerow([e,groups[ids[np.flatnonzero(ep==e)[0]]]]+list(errors.values())+[errors[selected]-errors['hmm']])
losses=np.array([np.sum((oof[n]-truth)**2,axis=1) for n in names]);correlations=np.corrcoef(losses)
report={'exposure':'Exploratory: all public validation previously examined. No untouched evaluation set.','aggregation':'Official score on concatenated held-fold predictions, not mean fold points.','scores':scores,'folds':fold_reports,'paired_vs_champion':boot,'ensemble_vs_best':ensemble_vs_best,'best_single':best_single,'ensemble_retained':keep_ensemble,'candidate':candidate,'deployed':selected,'champion_changed':change,'decision_checks':{'points_gain':gain,'folds_improved':fold_wins,'worst_condition_ratio_regression':worst_regression,'seed_consistent':bool(seed_consistent)},'seed_scores':seed_scores,'error_correlation':correlations.tolist(),'release':{'selected':selected,'config_indices':release_selection,'weights':dict(zip(names,rw.tolist())),'validation_descriptive_score':official(release_pred),'population_training':'public/train only','blend_fit':'all public validation after fold-held evaluation; no base model fitted to probability targets'},'all_candidate_development':{'local':{str(i):official(a) for i,a in enumerate(local)},'catboost':{v:{str(i):official(a) for i,a in enumerate(arrays)} for v,arrays in avg.items()}},'completed_unix':time.time()}
(O/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');(O/'release_config.json').write_text(json.dumps(report['release'],indent=2)+'\n')
for n,r in scores.items():print(n,round(r['technical_points'],6),{c:round(v['loss_ratio'],4) for c,v in r['by_condition'].items()})
print('DECISION',selected,'checks',report['decision_checks'],'release weights',report['release']['weights'])
