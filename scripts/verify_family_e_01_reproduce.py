import sys, json
sys.path.insert(0,'/Users/ayan/Downloads/All-In'); sys.path.insert(0,'/Users/ayan/Downloads/All-In/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
from slot_forecasting.io import read_submission

b = H.load(Path('main-participants-60-40/public'))
print('n ids', len(b.ids), 'n episodes', len(b.episodes))
# fold coverage
import collections
cnt = collections.Counter(b.folds[e] for e in sorted(b.episodes))
print('fold counts', dict(sorted(cnt.items())))
print('episodes missing from folds', sorted(e for e in b.episodes if e not in b.folds))
masks = [b.outer_mask(f) for f in range(4)]
tot = sum(m.astype(int) for m in masks)
print('rows covered exactly once:', bool((tot==1).all()), 'min/max cover', tot.min(), tot.max())

# distinct true u per type
rt = b.truth[:,1:]/np.maximum(b.truth[:,1:].sum(axis=1,keepdims=True),1e-15)
u = rt[:,0]
for t in H.TYPES:
    m = b.types==t
    vals = sorted({round(float(x),9) for x in u[m]})
    print(t, 'n', int(m.sum()), 'distinct true u', vals)
    # check tail split identical
    tl = rt[m][:,1:]; tl = tl/np.maximum(tl.sum(axis=1,keepdims=True),1e-15)
    print('   tail spread max-min per col', (tl.max(0)-tl.min(0)).round(8))

# E2x official + paired, recomputed from Family E's own published oof CSV
for name in ['E2x_calibrator_no_uprot','E2_calibrator','E0a_zero_correction']:
    d = read_submission(Path('artifacts/endpoint_supervised/predictions/oof_%s.csv'%name))
    P = np.array([d[i] for i in b.ids]); P = P/P.sum(axis=1,keepdims=True)
    o = b.official(P)
    pv = b.paired(P, b.protected)
    print('%-26s points=%.9f R=%.12f paired=%s' % (name, o['technical_points'], o['mean_loss_ratio'], [round(x,9) for x in pv['delta_loss_ratio_ci95']]))
    print('   by_condition', {k: round(v['loss_ratio'],6) for k,v in sorted(o['by_condition'].items())})
    # identity of control vs protected
    if name=='E0a_zero_correction':
        print('   max|control-protected| =', float(np.abs(P-b.protected).max()))
po = b.official(b.protected)
print('protected points', po['technical_points'], po['mean_loss_ratio'])
