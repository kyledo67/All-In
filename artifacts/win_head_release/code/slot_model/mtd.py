"""Sparse additive MTDg EM. See research/mtd_reference for MIT source and deviations."""
import json
from pathlib import Path
import numpy as np
PAYOUTS=(0,1,2,5,20)
NAMES=[f'own_lag_{i}' for i in range(1,9)]+[f'neighbor_win_lag_{i}' for i in range(1,4)]

def index(history):
    out={}
    for r in history:out.setdefault((r.episode_id,r.machine_id),{})[r.tick]=PAYOUTS.index(r.payout)
    return out

def inputs(q,t,idx):
    h=idx[q.episode_id,q.machine_id]
    x=[h.get(t-k,5) for k in range(1,9)]
    for k in range(1,4):
        vals=[v[t-k]>0 for (e,m),v in idx.items() if e==q.episode_id and m!=q.machine_id and t-k in v]
        # Exchangeable categorical neighbor summary: none/low/high win fraction; unknown=5.
        x.append(5 if not vals else min(2,int(3*sum(vals)/len(vals))))
    return x

def em(x,y,w,prior,strength,iterations=30):
    x=np.asarray(x,int);y=np.asarray(y,int);w=np.asarray(w,float)
    lam=np.ones(11)/11;q=np.broadcast_to(prior,(11,6,5)).copy()
    for _ in range(iterations):
        resp=np.stack([lam[j]*q[j,x[:,j],y] for j in range(11)],axis=1)
        resp/=np.maximum(resp.sum(1,keepdims=True),1e-300);resp*=w[:,None]
        lam=(resp.sum(0)+1e-3);lam/=lam.sum()
        for j in range(11):
            c=np.broadcast_to(strength*prior/66,(6,5)).copy()
            np.add.at(c,(x[:,j],y),resp[:,j]);q[j]=c/np.maximum(c.sum(1,keepdims=True),1e-300)
    return lam,q

class MTDModel:
    def __init__(self,changepoint=False):self.changepoint=changepoint;self.contexts={};self.strength=8.
    def fit(self,queries,history,labels=None):
        idx=index(history);bytype={}
        for query in queries:
            h=idx[query.episode_id,query.machine_id]
            rows=[(inputs(query,t,idx),v) for t,v in sorted(h.items()) if t<query.cutoff]
            bytype.setdefault(query.machine_type,[]).extend(rows)
        # Empirical dispersion of machine proportions estimates Dirichlet concentration.
        props=[(np.bincount(list(h.values()),minlength=5)+.5)/(len(h)+2.5) for h in idx.values()]
        p=np.mean(props,axis=0);v=np.var(props,axis=0)
        self.strength=float(np.clip(np.median(p*(1-p)/np.maximum(v,1e-8)-1),1,40/3))
        for typ,rows in bytype.items():
            x,y=zip(*rows);prior=(np.bincount(y,minlength=5)+1);prior=prior/prior.sum()
            lam,mat=em(x,y,np.ones(len(y)),prior,self.strength)
            self.contexts[typ+'|unknown']={'prior':prior.tolist(),'lambda':lam.tolist(),'Q':mat.tolist()}
        # Historical signals are unknown: known-signal contexts use only released query labels.
        for typ in bytype:
            for signal in (-1,0,1):
                qs=[q for q in queries if q.machine_type==typ and q.signal==signal and labels and q.query_id in labels]
                base=self.contexts[typ+'|unknown'];prior=np.asarray(base['prior'])
                if qs:
                    x=[inputs(q,q.cutoff,idx) for q in qs];y=[PAYOUTS.index(labels[q.query_id]) for q in qs]
                    lam,mat=em(x,y,np.ones(len(y)),prior,self.strength)
                    self.contexts[f'{typ}|{signal}']={'prior':prior.tolist(),'lambda':lam.tolist(),'Q':mat.tolist()}
                else:self.contexts[f'{typ}|{signal}']=base
        return self
    def predict(self,queries,history):
        idx=index(history);out={}
        for query in queries:
            h={t:v for t,v in idx[query.episode_id,query.machine_id].items() if t<query.cutoff}
            rows=sorted(h);weights=np.ones(len(rows));shift=None
            if self.changepoint and len(rows)>=40:
                candidates=[]
                for t in rows[20:-19]:
                    a=[h[s]>0 for s in range(t-20,t) if s in h];b=[h[s]>0 for s in range(t,t+20) if s in h]
                    if len(a)>=15 and len(b)>=15:candidates.append((abs(np.mean(a)-np.mean(b)),t))
                if candidates:
                    size,shift=max(candidates)
                    if size>=.25:weights=np.array([.25 if t-8<shift else 1. for t in rows])
            base=self.contexts[query.machine_type+'|unknown'];prior=np.asarray(base['prior'])
            x=[inputs(query,t,idx) for t in rows];y=[h[t] for t in rows]
            lam,mat=em(x,y,weights,prior,min(self.strength,weights.sum()/3))
            xx=inputs(query,query.cutoff,idx);local=sum(lam[j]*mat[j,xx[j]] for j in range(11))
            ctx=self.contexts[f'{query.machine_type}|{query.signal}'];glob=sum(ctx['lambda'][j]*np.array(ctx['Q'])[j,xx[j]] for j in range(11))
            # Context evidence is capped at one third of observed effective local mass.
            alpha=min(self.strength,weights.sum()/3);p=(weights.sum()*local+alpha*glob)/(weights.sum()+alpha)
            out[query.query_id]=(p/p.sum()).tolist()
        return out
    def save(self,path):Path(path).write_text(json.dumps(self.__dict__,sort_keys=True))
    @classmethod
    def load(cls,path):m=cls();m.__dict__.update(json.loads(Path(path).read_text()));return m
