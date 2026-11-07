"""Inference for the four saved recipes. Never fits weights or model parameters."""
import json,time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from .features import causal_index
from .phase2_features import examples,matrix,local_probability
from .phase2_predict import hmm_summary
from slot_forecasting.io import load_inputs,write_submission

NAMES=('hmm','local','cat_observable','cat_hmm')

def predict(data,output,model,*,split='test',mode=None,audit_dir=None):
    start=time.perf_counter();model=Path(model)
    if mode is None:
        deployment=json.loads((model/'deployment.json').read_text());mode=deployment['mode']
    if mode not in ('heldfold','mean','fold0','fold1','fold2','fold3'):raise ValueError('Unknown frozen recipe routing')
    if mode=='heldfold' and split!='validation':raise ValueError('Held-fold routing is evaluation-only; test episodes have no saved fold assignment.')
    recipes=json.loads((model/'recipes.json').read_text());protocol=json.loads((model/'protocol.json').read_text())
    if [r['fold'] for r in recipes]!=[0,1,2,3]:raise ValueError('The four saved recipe identities/order changed')
    qs,hs=load_inputs(Path(data)/split);idx=causal_index(qs,hs);es=examples(qs,idx)
    query_ids=[e['q'].query_id for e in es]
    if len(set(query_ids))!=len(query_ids) or set(query_ids)!={q.query_id for q in qs}:raise ValueError('Query alignment changed')
    prior=json.loads((model/'population_prior.json').read_text())
    x,names=matrix(es,idx,{q.episode_id:prior for q in qs});schema=json.loads((model/'feature_schema.json').read_text())
    if names!=schema['observable']:raise ValueError('Feature schema changed')
    population=json.loads((model/'population.json').read_text());hp,hsummary=hmm_summary(population,es)
    local={};cat={};recipe_predictions=[]
    for recipe in recipes:
        selected=recipe['config_indices'];li=selected['local']
        if li not in local:
            local[li]=np.array([local_probability(e['q'],e['h'],prior,**protocol['local'][li]) for e in es])
        for variant in ('observable','hmm'):
            key=(variant,selected[variant])
            if key not in cat:
                values=x if variant=='observable' else np.column_stack((x,hsummary));ps=[]
                for seed in protocol['seeds']:
                    m=CatBoostClassifier();m.load_model(model/'models'/f'cb_{variant}_{key[1]}_{seed}.cbm')
                    if list(m.classes_)!=list(range(5)):raise ValueError('Payout class order changed')
                    ps.append(m.predict_proba(values,thread_count=2))
                cat[key]=np.mean(ps,axis=0)
        components=np.array([hp,local[li],cat['observable',selected['observable']],cat['hmm',selected['hmm']]])
        weights=np.array([recipe['weights'][n] for n in NAMES])
        assert (weights>=0).all() and abs(weights.sum()-1)<1e-12
        # Keep every saved weight, including numerical boundary values. No clipping or tuning.
        recipe_predictions.append(np.einsum('m,mnc->nc',weights,components))
    stack=np.asarray(recipe_predictions)
    if not np.isfinite(stack).all() or np.min(stack)<0 or np.max(abs(stack.sum(2)-1))>=1e-12:
        raise ValueError('A frozen recipe produced invalid probabilities')
    if mode=='heldfold':
        folds=json.loads((model/'validation_fold_map.json').read_text())
        assignment=np.array([folds[e['q'].episode_id] for e in es]);pred=stack[assignment,np.arange(len(es))]
    elif mode=='mean':pred=np.mean(stack,axis=0)
    else:pred=stack[int(mode[-1])]
    if not np.isfinite(pred).all() or np.min(pred)<0:raise ValueError('Invalid probabilities')
    assert np.max(abs(pred.sum(1)-1))<1e-12
    write_submission(output,{e['q'].query_id:p.tolist() for e,p in zip(es,pred)})
    if audit_dir is not None:
        audit_dir=Path(audit_dir);audit_dir.mkdir(parents=True,exist_ok=True)
        for recipe,values in zip(recipes,stack):
            write_submission(audit_dir/f'fold{recipe["fold"]}.csv',dict(zip(query_ids,values.tolist())))
        (audit_dir/'audit.json').write_text(json.dumps({'mode':mode,'query_ids_unique':True,
            'query_count':len(query_ids),'component_alignment':'Every component uses the same query-ID-indexed feature rows; serialized recipe CSVs retain those IDs.',
            'payout_order':[0,1,2,5,20],'recipe_order':[r['fold'] for r in recipes],
            'deployment_weights':[0.25]*4 if mode=='mean' else None,
            'recipe_max_sum_error':float(np.max(abs(stack.sum(2)-1))),
            'ensemble_max_sum_error':float(np.max(abs(pred.sum(1)-1)))},indent=2)+'\n')
    return {'queries':len(es),'seconds':time.perf_counter()-start,'mode':mode,'split':split,'recipes':len(recipes),'payout_order':[0,1,2,5,20]}
