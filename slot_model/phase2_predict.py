"""Offline prediction from frozen phase-two artifacts, reusing causal utilities."""
import json,math,time
from pathlib import Path
import numpy as np
from .features import causal_index
from .phase2_features import examples,matrix,local_probability,hmm_arrays
from .portable import state_emissions
from slot_forecasting.io import load_inputs,write_submission


def hmm_summary(population,examples_):
    data=hmm_arrays(examples_);transition=population['transition'];predictions=[];summaries=[]
    for y,x,nx in zip(data['y'],data['x'],data['next_x']):
        state=population['initial'][:]
        for yy,xx in zip(y,x):
            em=state_emissions(population,xx);post=[s*e[yy] for s,e in zip(state,em)];norm=math.fsum(post);post=[v/norm for v in post]
            state=[math.fsum(post[j]*transition[j][k] for j in range(len(state))) for k in range(len(state))]
        em=state_emissions(population,nx);prob=[math.fsum(s*e[c] for s,e in zip(state,em)) for c in range(5)]
        predictions.append(prob);summaries.append(prob+[-math.fsum(v*math.log(max(v,1e-15)) for v in post),max(post)])
    return np.array(predictions),np.array(summaries)


def predict(data,output,release,split='test'):
    start=time.perf_counter();release=Path(release)
    config=json.loads((release/'release_config.json').read_text());protocol=json.loads((release/'protocol.json').read_text())
    weights=config['weights'];queries,history=load_inputs(Path(data)/split);idx=causal_index(queries,history);es=examples(queries,idx)
    out=np.zeros((len(es),5));components={}
    prior=json.loads((release/'population_prior.json').read_text())
    if weights['local']>1e-10:
        c=protocol['local'][config['config_indices']['local']]
        components['local']=np.array([local_probability(e['q'],e['h'],prior,**c) for e in es])
    hsummary=None
    if weights['hmm']>1e-10 or weights['cat_hmm']>1e-10:
        population=json.loads((release/'population.json').read_text());hp,hsummary=hmm_summary(population,es)
        components['hmm']=hp
    if weights['cat_observable']>1e-10 or weights['cat_hmm']>1e-10:
        from catboost import CatBoostClassifier
        x,names=matrix(es,idx,{q.episode_id:prior for q in queries})
        schema=json.loads((release/'feature_schema.json').read_text());assert names==schema['observable']
        for variant in ('observable','hmm'):
            if weights['cat_'+variant]<=1e-10:continue
            values=x if variant=='observable' else np.column_stack((x,hsummary));probs=[]
            for seed in protocol['seeds']:
                m=CatBoostClassifier();m.load_model(release/'models'/f'cb_{variant}_{config["config_indices"][variant]}_{seed}.cbm')
                assert list(m.classes_)==list(range(5));probs.append(m.predict_proba(values,thread_count=2))
            components['cat_'+variant]=np.mean(probs,axis=0)
    for name,weight in weights.items():
        if weight>1e-10:out+=weight*components[name]
    # Numerical optimizer boundary weights smaller than 1e-10 are ignored and renormalized.
    out/=out.sum(1,keepdims=True)
    assert np.isfinite(out).all() and np.min(out)>=0
    write_submission(output,{e['q'].query_id:p.tolist() for e,p in zip(es,out)})
    return {'queries':len(es),'seconds':time.perf_counter()-start,'selected':config['selected'],'weights':weights,'payout_order':[0,1,2,5,20]}
