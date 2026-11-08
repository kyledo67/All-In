import sys
sys.path.insert(0,'/Users/ayan/Downloads/All-In'); sys.path.insert(0,'/Users/ayan/Downloads/All-In/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
import scripts.endpoint_supervised as E
from slot_forecasting.io import load_inputs

b = H.load(Path('main-participants-60-40/public'))
vq,_ = load_inputs(b.data/'validation')
meta = {q.query_id:(q.episode_id,q.machine_id,q.cutoff) for q in vq}
# 1. index is strictly pre-cutoff
bad = 0
for qid in b.ids:
    ep,mid,cut = meta[qid]
    obs = b.index[(ep,mid)]
    if obs and max(obs) >= cut: bad += 1
print('queries whose causal index contains a tick >= cutoff:', bad)
# gap structure: are histories contiguous?
gaps = []
for qid in b.ids[:5]:
    ep,mid,cut = meta[qid]
    ks = sorted(b.index[(ep,mid)])
    gaps.append((qid, cut, len(ks), ks[:3], ks[-3:]))
for g in gaps: print(g)

base = E.gather(b)
# 2. inject fake FUTURE ticks and re-gather; features must be unchanged
rng = np.random.default_rng(5)
for qid in b.ids:
    ep,mid,cut = meta[qid]
    obs = b.index[(ep,mid)]
    for d in range(0, 12):
        obs[cut+d] = int(rng.integers(0,5))
pert = E.gather(b)
def flat(rec):
    out=[rec['has_query_state'], rec['n_state_rows'], tuple(rec['older'])]
    out += [tuple(rec['counts'][w]) for w in E.WINDOWS]
    out += [tuple(rec['ham'])]
    out += [tuple(map(tuple, rec['lagm'][l])) for l in E.LAGS]
    out += [tuple(map(tuple, rec['pairm'][p])) for p in E.PAIRS]
    return tuple(out)
diff = sum(1 for a,c in zip(base,pert) if flat(a)!=flat(c))
print('queries whose evidence changed after injecting fake post-cutoff ticks:', diff)
