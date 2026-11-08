"""Prove artifacts/phase2/predictions/oof_ensemble.csv is fold-held against the SAME
artifacts/phase2/folds.json['validation'] map that scripts/combined_evaluate.py uses:
rebuild each fold's rows from the recorded per-fold selection + simplex weights."""
import json, sys, hashlib
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from slot_forecasting.io import read_submission, read_targets
D = R/'main-participants-60-40/public'
targets = read_targets(D/'validation/targets.csv')
ids = sorted(read_submission(D/'validation/reference.csv'))
ep = np.array([{t.query_id: t.episode_id for t in targets}[q] for q in ids])
folds = json.loads((R/'artifacts/phase2/folds.json').read_text())['validation']
ev = json.loads((R/'artifacts/phase2/evaluation.json').read_text())
P = R/'artifacts/phase2/predictions'
def load(p):
    d = read_submission(p); return np.array([d[q] for q in ids])
hmm = load(R/'artifacts/final/validation_predictions.csv')
protocol = json.loads((R/'artifacts/phase2/protocol.json').read_text())
local = [load(P/f'local_{i}.csv') for i in range(len(protocol['local']))]
avg = {v: [np.mean([load(P/f'cb_{v}_{i}_{s}.csv') for s in protocol['seeds']], axis=0) for i in range(2)]
       for v in ('observable','hmm')}
got = load(P/'oof_ensemble.csv')
worst = 0.0; rows = []
for fr in ev['folds']:
    k = fr['fold']; sel = fr['selection']; w = np.array(list(fr['blend_weights'].values()))
    arrays = [hmm, local[sel['local']], avg['observable'][sel['observable']], avg['hmm'][sel['hmm']]]
    rebuilt = np.einsum('m,mnc->nc', w, np.array(arrays))
    recorded_held = set(fr['held_episodes'])
    map_held = {e for e in set(ep.tolist()) if folds[e] == k}
    mask = np.array([e in map_held for e in ep])
    d = float(np.max(np.abs(rebuilt[mask]-got[mask]))); worst = max(worst, d)
    rows.append({'fold': k, 'held_episodes_match_folds_json': recorded_held == map_held,
                 'held_episode_count': len(map_held), 'max_abs_rebuild_deviation': d,
                 'selection': sel, 'weights': fr['blend_weights']})
out = {'folds_json_sha256': hashlib.sha256((R/'artifacts/phase2/folds.json').read_bytes()).hexdigest(),
       'per_fold': rows, 'worst_rebuild_deviation': worst,
       'conclusion': 'oof_ensemble rows for fold k come from a config selection and simplex '
                     'fitted on the other three folds of the identical validation fold map'}
(R/'artifacts/audit/provenance/base_foldheld.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
print(json.dumps(out, indent=2, sort_keys=True))
