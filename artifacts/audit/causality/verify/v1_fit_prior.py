"""Does fit_prior() have its OWN cutoff guard, or does it rely solely on causal_index?

The prior that enters EVERY validation feature (prior_c, prop_*, decay_*, conditional_*,
other_recent_*) is fit_prior(train_queries, train_index, allowed) -- built from the TRAIN
index. The prior agent's bypass tests injected post-cutoff ticks only into the VALIDATION
index, so they never exercised this path. Inject into the TRAIN index instead.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from slot_model.features import PAYOUTS, causal_index
from slot_model.phase2_features import examples, fit_prior, matrix
from slot_model.phase2_tables import PHASE2
from slot_forecasting.io import load_inputs

DATA = ROOT / 'main-participants-60-40/public'
folds = json.loads((PHASE2 / 'folds.json').read_text())
tq, th = load_inputs(DATA / 'train')
vq, vh = load_inputs(DATA / 'validation')
tidx0 = causal_index(tq, th)
vidx = causal_index(vq, vh)
allowed = set(folds['train'])

p0 = fit_prior(tq, tidx0, allowed)

# inject post-cutoff ticks DIRECTLY into the TRAIN index (bypassing causal_index)
tidx1 = {k: dict(v) for k, v in tidx0.items()}
cut = {(q.episode_id, q.machine_id): q.cutoff for q in tq}
n = 0
for (e, m), c in sorted(cut.items()):
    for off in range(8):
        if c + off not in tidx1[(e, m)]:
            tidx1[(e, m)][c + off] = 4   # 20x jackpot: maximally distorting
            n += 1
p1 = fit_prior(tq, tidx1, allowed)
pd = float(np.max(np.abs(np.asarray(p0) - np.asarray(p1))))

es = examples(vq, vidx)
X0, names = matrix(es, vidx, {q.episode_id: p0 for q in vq})
X1, _ = matrix(es, vidx, {q.episode_id: p1 for q in vq})
xd = float(np.max(np.abs(X0 - X1)))
cols = sorted({names[j] for j in np.where(np.abs(X0 - X1).max(0) > 0)[0]})

# also: does fit_prior read ticks at/after cutoff when they are present?
reads_future = []
for q in tq[:5]:
    h = tidx1[(q.episode_id, q.machine_id)]
    reads_future.append({'query': q.query_id, 'cutoff': q.cutoff,
                         'max_tick_in_index': max(h), 'ticks_at_or_after_cutoff':
                         sum(1 for t in h if t >= q.cutoff)})

out = {'ticks_injected_into_train_index': n,
       'fit_prior_max_abs_change': pd,
       'fit_prior_has_own_cutoff_guard': pd == 0.0,
       'validation_feature_matrix_max_abs_change': xd,
       'validation_columns_changed': len(cols),
       'validation_columns_changed_sample': cols[:25],
       'prior_before': p0, 'prior_after': p1,
       'sample_machines': reads_future,
       'conclusion': ('fit_prior() applies NO t < cutoff filter: it bincounts every value in '
                      'index[episode,machine]. Its causality depends ENTIRELY on the caller '
                      'passing a causal_index() result. Single layer of defence.')
       if pd > 0 else 'fit_prior is internally guarded'}
print(json.dumps(out, indent=2)[:4000])
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
