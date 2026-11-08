import sys
sys.path.insert(0,'/Users/ayan/Downloads/All-In'); sys.path.insert(0,'/Users/ayan/Downloads/All-In/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
import scripts.endpoint_supervised as E
b = H.load(Path('main-participants-60-40/public'))
qh, r_prot = b.split_heads(b.protected); u_prot = r_prot[:,0]
raw = E.gather(b); n=len(b.ids)
ctx = {'bundle': b,'raw': raw,'n': n,'q': qh,'u_prot': u_prot,'train_mixture': None}
class Fix(E.Calibrator):
    def __init__(s, beta):
        super().__init__(drop_uprot=True); s.blend=(beta,)
    def fit_grid(s): return [('F6_lagall', lam) for lam in E.RIDGE]
base = 53.532917808538166
print('beta  points     gain     paired95 vs protected')
for beta in (0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0):
    res = E.run_arm(b, Fix(beta), ctx, 'b%.3f'%beta)
    o=b.official(res['P']); pv=b.paired(res['P'], b.protected)
    print('%.3f %10.6f %+8.6f  [%+.6f,%+.6f]'%(beta,o['technical_points'],o['technical_points']-base,*pv['delta_loss_ratio_ci95']))
