"""Reproduce artifacts/mtd_01/change.json from public/train only, then reproduce
artifacts/mtd_pairs_01/validation_lag7.csv from it. Provenance check, no fitting on validation."""
import json, os, sys, time, hashlib
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from slot_model.mtd import MTDModel
from slot_model.mtd_pairs import PairMTDModel, LAG7_PAIRS
from slot_forecasting.io import load_inputs, read_labels, read_submission
D = R/'main-participants-60-40/public'
tq, th = load_inputs(D/'train'); vq, vh = load_inputs(D/'validation')
labels = read_labels(D/'train/labels.csv', tq)
t0 = time.perf_counter()
m = MTDModel(changepoint=True).fit(tq, th, labels)
fit_s = time.perf_counter()-t0
saved = json.loads((R/'artifacts/mtd_01/change.json').read_text())
mine = json.loads(json.dumps(m.__dict__, sort_keys=True))
def flat(d):
    out = {}
    for k, v in sorted(d['contexts'].items()):
        out[k] = np.concatenate([np.asarray(v['prior']).ravel(), np.asarray(v['lambda']).ravel(),
                                 np.asarray(v['Q']).ravel()])
    return out
a, b = flat(mine), flat(saved)
assert set(a) == set(b), (sorted(a), sorted(b))
worst = max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
t1 = time.perf_counter()
pred = PairMTDModel(pairs=LAG7_PAIRS, base=MTDModel.load(R/'artifacts/mtd_01/change.json')).predict(vq, vh)
pred_s = time.perf_counter()-t1
want = read_submission(R/'artifacts/mtd_pairs_01/validation_lag7.csv')
assert set(pred) == set(want)
dev = max(float(np.max(np.abs(np.asarray(pred[k])-np.asarray(want[k])))) for k in want)
rep = {'refit_from_public_train_only': True,
       'strength_mine': m.strength, 'strength_saved': saved['strength'],
       'changepoint_mine': m.changepoint, 'changepoint_saved': saved['changepoint'],
       'max_abs_parameter_deviation_vs_change_json': worst,
       'max_abs_deviation_validation_lag7_reproduction': dev,
       'fit_seconds': fit_s, 'predict_seconds': pred_s,
       'inputs_read': ['train/queries.csv','train/history.csv','train/labels.csv',
                       'validation/queries.csv','validation/history.csv'],
       'validation_targets_read': False}
(R/'artifacts/audit/provenance/mtd_reproduction.json').write_text(json.dumps(rep, indent=2, sort_keys=True)+'\n')
print(json.dumps(rep, indent=2, sort_keys=True))
