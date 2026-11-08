"""Strongly shrunk pairwise correction to additive MTDg probabilities."""
from collections import defaultdict
import numpy as np
from .mtd import MTDModel,index,inputs

LAG7_PAIRS=((7,1),(7,2),(7,4),(7,8))
CONTROL_PAIRS=((1,2),(4,8))

class PairMTDModel:
    """Use paired categorical cell residuals; fit base from training episodes only."""
    def __init__(self, pairs=LAG7_PAIRS, prior=40., base=None):
        self.pairs=tuple(tuple(p) for p in pairs)
        self.prior=float(prior)
        self.base=base
    def fit(self,queries,history,labels=None):
        self.base=MTDModel(changepoint=True).fit(queries,history,labels)
        return self
    def predict(self,queries,history):
        if self.base is None:raise ValueError('Fit a base model first')
        original=self.base.predict(queries,history);idx=index(history);out={}
        for query in queries:
            p=np.asarray(original[query.query_id],float)
            h={t:v for t,v in idx[query.episode_id,query.machine_id].items() if t<query.cutoff}
            if not h or not self.pairs:
                out[query.query_id]=p.tolist();continue
            target=inputs(query,query.cutoff,idx)
            rows=sorted(h)
            historical=np.asarray([inputs(query,t,idx) for t in rows],dtype=int)
            outcomes=np.asarray([h[t] for t in rows],dtype=int)
            decay=np.asarray([2.**(-(query.cutoff-1-t)/48.) for t in rows])
            corrections=[]
            for i,j in self.pairs:
                a,b=i-1,j-1
                if target[a]==5 or target[b]==5:continue
                valid=(historical[:,a]!=5)&(historical[:,b]!=5)
                left_mask=valid&(historical[:,a]==target[a])
                right_mask=valid&(historical[:,b]==target[b])
                cell_mask=left_mask&right_mask
                counts=lambda mask:np.bincount(outcomes[mask],weights=decay[mask],minlength=5)
                total=counts(valid);left=counts(left_mask);right=counts(right_mask);cell=counts(cell_mask)
                n=cell.sum()
                if n==0:continue
                # Double-centered residual removes marginal lag effects. Every
                # empirical distribution shrinks to the additive forecast.
                smooth=lambda c:(c+self.prior*p)/(c.sum()+self.prior)
                residual=smooth(cell)-smooth(left)-smooth(right)+smooth(total)
                # Cell support limits short-history adaptation further.
                corrections.append((n/(n+self.prior))*residual)
            if corrections:
                p=np.maximum(p+np.mean(corrections,axis=0),1e-9);p/=p.sum()
            out[query.query_id]=p.tolist()
        return out
