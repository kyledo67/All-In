import sys
sys.path.insert(0,'/Users/ayan/Downloads/All-In'); sys.path.insert(0,'/Users/ayan/Downloads/All-In/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
import scripts.endpoint_supervised as E

b = H.load(Path('main-participants-60-40/public'))
qh, r_prot = b.split_heads(b.protected); u_prot = r_prot[:,0]
raw = E.gather(b); n=len(b.ids)
ctx = {'bundle': b, 'raw': raw, 'n': n, 'q': qh, 'u_prot': u_prot, 'train_mixture': None}

class Lad(E.Calibrator):
    def __init__(self, rungs, blend, **kw):
        super().__init__(**kw); self.rungs=rungs; self.blend=blend
    def fit_grid(self):
        return [(nm, lam) for nm,_ in E.LADDER if nm in self.rungs for lam in E.RIDGE]

def run(label, arm):
    res = E.run_arm(b, arm, ctx, label)
    o = b.official(res['P'])
    pv = b.paired(res['P'], b.protected)
    perfold=[]
    for f in range(4):
        eps={e for i,e in enumerate(b.ep) if b.outer_mask(f)[i]}
        perfold.append(round(b.official(res['P'],episodes=eps)['technical_points']-b.official(b.protected,episodes=eps)['technical_points'],4))
    print('%-34s points=%.6f gain=%+.6f paired=[%+.6f,%+.6f] perfold=%s cfgs=%s'%(
        label,o['technical_points'],o['technical_points']-53.532917808538166,
        pv['delta_loss_ratio_ci95'][0],pv['delta_loss_ratio_ci95'][1],perfold,
        [(c['config'],c['blend']) for c in res['chosen']]))
    return res

# 1. literal E2x
run('E2x (as reported)', E.Calibrator(drop_uprot=True))
# 2. pure per-type recalibration only (no per-machine evidence): F0 rung only
run('F0 only (type-mean recalibration)', Lad(('F0_prior_only',), E.BLEND, drop_uprot=True))
# 3. E2x restricted to beta=1.0 : NO u_prot in the final u at all
run('E2x beta forced to 1.0', Lad(tuple(nm for nm,_ in E.LADDER), (1.0,), drop_uprot=True))
# 4. E2x with only F6 rung, beta grid (the rung it always picks)
run('F6 only, nested beta', Lad(('F6_lagall',), E.BLEND, drop_uprot=True))
