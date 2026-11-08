import sys
sys.path.insert(0,'/Users/ayan/Downloads/All-In'); sys.path.insert(0,'/Users/ayan/Downloads/All-In/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
from slot_model.official_score import CONDITIONS
from slot_forecasting.io import read_submission

b = H.load(Path('main-participants-60-40/public'))
targets, groups, reference = b.scoring
def rows_for(a, c):
    sel=[t for t in targets if t.episode_id in b.episodes]
    out={}
    for cond in CONDITIONS:
        es=sorted({t.episode_id for t in sel if groups[t.query_id]==cond})
        vals=[]
        for e in es:
            ts=[t for t in sel if t.episode_id==e]
            vals.append([np.mean([np.sum((np.asarray(p[t.query_id])-t.probabilities)**2) for t in ts]) for p in (a,c,reference)])
        out[cond]=np.asarray(vals)
    return out

def boot(rows, seed, repeats):
    rng=np.random.default_rng(seed); tot=np.zeros(repeats)
    for cond in CONDITIONS:
        v=rows[cond]
        dr=rng.integers(0,len(v),(repeats,len(v)))
        m=v[dr].mean(1)
        tot += ((m[:,0]-m[:,1])/np.maximum(m[:,2],1e-9))/4
    return np.quantile(tot,[.025,.975]), tot

d = read_submission(Path('artifacts/endpoint_supervised/predictions/oof_E2x_calibrator_no_uprot.csv'))
P = np.array([d[i] for i in b.ids]); P=P/P.sum(axis=1,keepdims=True)
A = b.mapping(P); C = b.mapping(b.protected)
rows = rows_for(A, C)
ci, tot = boot(rows, 314159, 2000)
print('harness seed 314159, 2000 reps:', [round(x,9) for x in ci], ' P(delta>0)=%.4f'%float((tot>0).mean()))
ci2,_ = boot(rows, 314159, 400000)
print('seed 314159, 400000 reps      :', [round(x,9) for x in ci2])
ups=[]
for s in range(1,41):
    c,_ = boot(rows, 1000+s, 2000)
    ups.append(c[1])
ups=np.array(ups)
print('40 alternative seeds, 2000 reps: upper bound mean %.6f sd %.6f ; frac upper<0: %.2f'%(ups.mean(), ups.std(), float((ups<0).mean())))
print('   upper-bound quantiles', np.round(np.quantile(ups,[0,.25,.5,.75,1]),6).tolist())
for s in range(1,11):
    c,t = boot(rows, 5000+s, 100000)
    print('   seed %5d 100k reps: [%+.6f, %+.6f] P(delta>0)=%.4f'%(5000+s, c[0], c[1], float((t>0).mean())))
