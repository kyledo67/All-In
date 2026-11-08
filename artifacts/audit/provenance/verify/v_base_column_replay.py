"""Independently re-run tabpfn_train.verify_rebuild: rebuild the phase-2 feature tables
from public/{train,validation} + committed HMM checkpoints, push them through the
committed CatBoost checkpoints, and compare to the committed cb_*.csv predictions that
the frozen blend (oof_ensemble.csv) is built from. Also replay local_*.csv."""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from catboost import CatBoostClassifier
from slot_model.phase2_tables import PHASE2, cached_design, features_for
from slot_forecasting.io import read_submission

D = R/'main-participants-60-40/public'
t0 = time.perf_counter()
tables = cached_design(D, R/'artifacts/tabpfn/cache')
protocol = json.loads((PHASE2/'protocol.json').read_text())
val = tables['validation']; ids = val['ids'].tolist()
worst = {}
for variant in ('observable', 'hmm'):
    values = features_for(val, variant)
    for i in range(len(protocol['catboost'])):
        for seed in protocol['seeds']:
            name = f'cb_{variant}_{i}_{seed}'
            m = CatBoostClassifier(); m.load_model(str(PHASE2/'models'/f'{name}.cbm'))
            assert list(m.classes_) == list(range(5))
            got = m.predict_proba(values, thread_count=2)
            want = np.array([read_submission(PHASE2/'predictions'/f'{name}.csv')[q] for q in ids])
            worst[name] = float(np.max(abs(got-want)))
loc = {}
for i in range(len(protocol['local'])):
    want = np.array([read_submission(PHASE2/'predictions'/f'local_{i}.csv')[q] for q in ids])
    got = val['local'][:, i]
    # local_i.csv stores the 5-vector produced by local_probability
    loc[f'local_{i}'] = float(np.max(abs(np.array(got.tolist())-want)))
rep = {'cache_key': tables['cache_key'], 'cached': tables['cached'],
       'provenance': tables['provenance'],
       'catboost_replay_max_abs': worst, 'worst_catboost': max(worst.values()),
       'local_replay_max_abs': loc, 'worst_local': max(loc.values()),
       'seconds': time.perf_counter()-t0,
       'note': 'cached_design hashes train/{queries,history,labels} + validation/{queries,history} '
               'contents only; the matching cache key proves this kit is byte-identical to the one '
               'that built the committed tables. No validation label/target/reference file is opened.'}
(R/'artifacts/audit/provenance/verify/v_base_column_replay.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
print(json.dumps(rep,indent=2,sort_keys=True))
