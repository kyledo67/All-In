"""Two-state filter over synchronized earlier neighbor wins in an episode."""
import math
import numpy as np
from .mtd import index

def neighbor_state(queries,history,win_rate):
    idx=index(history);result={}
    by_episode={}
    for (e,m),h in idx.items():by_episode.setdefault(e,{})[m]=h
    low=max(.01,win_rate-.08);high=min(.99,win_rate+.08)
    for q in queries:
        machines=by_episode[q.episode_id]
        prior=np.array([.5,.5]);prev=np.array([[.97,.03],[.03,.97]])
        # Include only other machines' observations from earlier actual ticks.
        start=max(0,q.cutoff-96)
        for t in range(start,q.cutoff):
            prior=prior@prev
            vals=[h[t]>0 for m,h in machines.items() if m!=q.machine_id and t in h]
            if vals:
                n=len(vals);wins=sum(vals)
                likelihood=np.array([low**wins*(1-low)**(n-wins),high**wins*(1-high)**(n-wins)])
                prior*=likelihood;prior/=prior.sum()
        result[q.query_id]=float(prior[1]-.5)
    return result

def adjust(p,state,beta):
    p=np.asarray(p,float);delta=float(np.clip(beta*state,-.08,.08))
    oldwin=1-p[0];newwin=float(np.clip(oldwin+delta,1e-6,1-1e-6))
    out=p.copy();out[0]=1-newwin;out[1:]*=newwin/max(oldwin,1e-12)
    return (out/out.sum()).tolist()
