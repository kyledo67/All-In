"""Causal observable features and small local expert. No target-probability access."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import numpy as np
from .features import PAYOUTS,TYPES,feature_at,causal_index
from slot_forecasting.transition import context

LOCAL_CONFIGS=[{'half_life':h,'strength':s} for h in (16,48,128) for s in (8,32)]

def fit_prior(queries,index,allowed):
    """Equal total mass per training episode, then per machine; histories only."""
    counts=np.full((3,5),.1);episodes=defaultdict(list)
    for q in queries:
        if q.episode_id in allowed:episodes[q.episode_id].append(q)
    for qs in episodes.values():
        for q in qs:
            h=index[q.episode_id,q.machine_id]
            c=np.bincount(list(h.values()),minlength=5)
            counts[TYPES.index(q.machine_type)]+=c/max(1,c.sum())/len(qs)
    return (counts/counts.sum(1,keepdims=True)).tolist()

def local_probability(q,h,prior,half_life=48,strength=8):
    counts=strength*np.asarray(prior[TYPES.index(q.machine_type)],float)
    for tick,y in h.items():
        if tick<q.cutoff:counts[y]+=2**(-(q.cutoff-1-tick)/half_life)
    return counts/counts.sum()

def sample_history(q,index,thin=False):
    h={t:v for t,v in index[q.episode_id,q.machine_id].items() if t<q.cutoff}
    if thin:
        old=[t for t in h if t<q.cutoff-40]
        old=sorted(old,key=lambda t:hashlib.sha256(f'{q.episode_id}|{q.machine_id}|{q.cutoff}|{t}'.encode()).hexdigest())[:24]
        keep=set(old)|{t for t in h if t>=q.cutoff-40};h={t:v for t,v in h.items() if t in keep}
    return h

def observable(q,h,index,prior,signal_known=True):
    """Every outcome-derived feature reads only ticks strictly below q.cutoff."""
    h={t:y for t,y in h.items() if t<q.cutoff}
    f={f'type_{typ}':float(q.machine_type==typ) for typ in TYPES}
    f.update(signal=float(q.signal) if signal_known else 0.,signal_known=float(signal_known),cutoff=float(q.cutoff),history_count=float(len(h)))
    p=np.asarray(prior[TYPES.index(q.machine_type)])
    for lag in range(1,9):
        y=h.get(q.cutoff-lag)
        f[f'lag{lag}_missing']=float(y is None)
        for c in range(5):f[f'lag{lag}_c{c}']=float(y==c) if y is not None else 0.
    windows={}
    for w in (8,16,40,128):
        vals=[v for t,v in h.items() if q.cutoff-w<=t<q.cutoff]
        counts=np.bincount(vals,minlength=5);n=len(vals)
        f[f'coverage_{w}']=n/w
        f[f'n_{w}']=n
        windows[w]=(counts+8*p)/(n+8)
        for c in range(5):
            f[f'count_{w}_{c}']=float(counts[c]);f[f'prop_{w}_{c}']=float(windows[w][c])
    for half in (16,48,128):
        weights=[2**(-(q.cutoff-1-t)/half) for t in h]
        f[f'decay_mass_{half}']=sum(weights)
        f[f'decay_ess_{half}']=sum(weights)**2/max(sum(w*w for w in weights),1e-12)
        pred=local_probability(q,h,prior,half,8)
        for c in range(5):f[f'decay_{half}_{c}']=float(pred[c])
    older=[v for t,v in h.items() if q.cutoff-128<=t<q.cutoff-40]
    old=(np.bincount(older,minlength=5)+8*p)/(len(older)+8)
    for c in range(5):
        f[f'prior_{c}']=float(p[c]);f[f'change_{c}']=float(windows[16][c]-old[c])
    # Reuse organizer's actual-tick categorical context, not adjacent-row positions.
    payout_seq={t:PAYOUTS[v] for t,v in h.items()}
    for lags in ((1,),(2,),(4,),(8,),(1,2),(4,8)):
        name='_'.join(map(str,lags));wanted=context(payout_seq,q.cutoff,lags)
        counts=8*local_probability(q,h,prior,48,8);mass=0.
        if wanted is not None:
            for t,y in h.items():
                if context(payout_seq,t,lags)==wanted:
                    weight=2**(-(q.cutoff-1-t)/128);counts[y]+=weight;mass+=weight
        f[f'conditional_mass_{name}']=mass
        f[f'conditional_missing_{name}']=float(wanted is None)
        for c,v in enumerate(counts/counts.sum()):f[f'conditional_{name}_{c}']=float(v)
    other=[]
    for (episode,machine),seq in index.items():
        if episode==q.episode_id and machine!=q.machine_id:
            for t,y in seq.items():
                if q.cutoff-16<=t<q.cutoff:other.append(y)
    counts=np.bincount(other,minlength=5);otherp=(counts+8*p)/(len(other)+8)
    f['other_recent_count']=len(other)
    for c,v in enumerate(otherp):f[f'other_recent_{c}']=float(v)
    return f

def examples(queries,index,labels=None,historical=False):
    result=[]
    for q in sorted(queries,key=lambda q:q.query_id):
        local=[]
        if historical:
            # At most eight historical targets; original released outcomes only.
            for cutoff in list(range(192,q.cutoff,8))[-8:]:
                if cutoff not in index[q.episode_id,q.machine_id]:continue
                qq=replace(q,query_id=q.query_id+f'-t{cutoff}',cutoff=cutoff,signal=0)
                h=sample_history(qq,index,thin=(cutoff//8)%2==0)
                if any(t not in h for t in range(cutoff-40,cutoff)):continue
                local.append((qq,h,index[q.episode_id,q.machine_id][cutoff],False))
        local.append((q,sample_history(q,index),None if labels is None else PAYOUTS.index(labels[q.query_id]),True))
        for qq,h,y,known in local:result.append({'q':qq,'h':h,'y':y,'signal_known':known,'weight':1/len(local)})
    # Organizer episodes have eight machines; normalize explicitly for portability.
    counts=defaultdict(set)
    for e in result:counts[e['q'].episode_id].add(e['q'].machine_id)
    for e in result:e['weight']/=len(counts[e['q'].episode_id])
    return result

def matrix(examples,index,priors):
    records=[observable(e['q'],e['h'],index,priors[e['q'].episode_id],e['signal_known']) for e in examples]
    names=list(records[0]);assert all(list(r)==names for r in records)
    return np.array([[r[n] for n in names] for r in records],dtype=np.float64),names

def hmm_arrays(examples):
    xs=[];ys=[];nx=[]
    for e in examples:
        q,h=e['q'],e['h'];x=[feature_at(h,t,q.machine_type) for t in range(q.cutoff-32,q.cutoff)]
        assert all(v is not None for v in x)
        xs.append(x);ys.append([h[t] for t in range(q.cutoff-32,q.cutoff)]);nx.append(feature_at(h,q.cutoff,q.machine_type))
    return {'x':np.asarray(xs),'y':np.asarray(ys),'next_x':np.asarray(nx)}
