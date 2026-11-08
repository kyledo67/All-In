"""Re-derive artifacts/two_head_01/validation.csv from the committed .cbm models plus a
feature matrix rebuilt here from public/{train,validation} inputs only.
Does not import scripts/dense_catboost.py (which hard-codes /Users/kyledo)."""
import json, os, sys, time, hashlib
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from catboost import CatBoostClassifier
from slot_model.features import causal_index
from slot_model.phase2_features import fit_prior, observable, sample_history
from slot_forecasting.io import load_inputs, read_submission

D = R/'main-participants-60-40/public'
F = json.loads((R/'artifacts/phase2/folds.json').read_text())['train']
schema = json.loads((R/'artifacts/phase2/feature_schema.json').read_text())['observable']
protocol = json.loads((R/'artifacts/phase2/protocol.json').read_text()); seeds = protocol['seeds']
t0 = time.perf_counter()
tq, th = load_inputs(D/'train'); vq, vh = load_inputs(D/'validation')
idx = causal_index(tq, th); vidx = causal_index(vq, vh)
vbyepisode = {}
for key, h in vidx.items(): vbyepisode.setdefault(key[0], {})[key] = h
prior = fit_prior(tq, idx, set(F))
records = [observable(q, sample_history(q, vidx), vbyepisode[q.episode_id], prior, True) for q in vq]
assert all(list(r) == schema for r in records), 'schema mismatch'
xt = np.asarray([[r[n] for n in schema] for r in records], float)
ps = []
for seed in seeds:
    win = CatBoostClassifier(); win.load_model(str(R/f'artifacts/two_head_01/win_{seed}.cbm'))
    size = CatBoostClassifier(); size.load_model(str(R/f'artifacts/two_head_01/size_{seed}.cbm'))
    chance = win.predict_proba(xt, thread_count=2)[:, 1]
    cond = size.predict_proba(xt, thread_count=2)
    ps.append(np.column_stack((1-chance, chance[:, None]*cond)))
mine = np.mean(ps, axis=0)
want_map = read_submission(R/'artifacts/two_head_01/validation.csv')
want = np.array([want_map[q.query_id] for q in vq])
dev = float(np.max(np.abs(mine - want)))
# also replay the sparse 5-class dense_catboost model for the same matrix
ds = []
for seed in seeds:
    m = CatBoostClassifier(); m.load_model(str(R/f'artifacts/dense_catboost_01/sparse_{seed}.cbm'))
    ds.append(m.predict_proba(xt, thread_count=2))
dmine = np.mean(ds, axis=0)
dwant_map = read_submission(R/'artifacts/dense_catboost_01/validation_sparse.csv')
dwant = np.array([dwant_map[q.query_id] for q in vq])
ddev = float(np.max(np.abs(dmine - dwant)))
rep = {'two_head_validation_max_abs_deviation': dev,
       'sparse_cb_validation_max_abs_deviation': ddev,
       'validation_rows': int(xt.shape[0]), 'features': int(xt.shape[1]),
       'files_read': ['train/queries.csv','train/history.csv','validation/queries.csv','validation/history.csv',
                      'artifacts/phase2/{folds,protocol,feature_schema}.json',
                      'artifacts/two_head_01/{win,size}_{seed}.cbm',
                      'artifacts/dense_catboost_01/sparse_{seed}.cbm'],
       'note': 'No label or target file of either split was opened by this script.',
       'seconds': time.perf_counter()-t0}
(R/'artifacts/audit/provenance/verify/v_two_head_replay.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
print(json.dumps(rep,indent=2,sort_keys=True))
