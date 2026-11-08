"""Full refit of two_head from public/train only (labels of TRAIN, never validation),
reproducing scripts/two_head_catboost.py's final full-train arm with the /Users/kyledo
path replaced. Compares to the committed artifacts/two_head_01/validation.csv."""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from dataclasses import replace
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from catboost import CatBoostClassifier, Pool
from slot_model.features import causal_index, PAYOUTS
from slot_model.phase2_features import fit_prior, observable, sample_history
from slot_forecasting.io import load_inputs, read_labels, read_submission

D = R/'main-participants-60-40/public'
F = json.loads((R/'artifacts/phase2/folds.json').read_text())['train']
schema = json.loads((R/'artifacts/phase2/feature_schema.json').read_text())['observable']
protocol = json.loads((R/'artifacts/phase2/protocol.json').read_text())
seeds = protocol['seeds']; trees = {20261108: 214, 20261109: 307}
t0 = time.perf_counter()
tq, th = load_inputs(D/'train'); vq, vh = load_inputs(D/'validation')
labels = read_labels(D/'train/labels.csv', tq)
idx = causal_index(tq, th); vidx = causal_index(vq, vh)
byepisode = {}
for key, h in idx.items(): byepisode.setdefault(key[0], {})[key] = h
vbyepisode = {}
for key, h in vidx.items(): vbyepisode.setdefault(key[0], {})[key] = h

def make_examples(queries, dense):
    out = []
    for q in sorted(queries, key=lambda q: q.query_id):
        machine = idx[q.episode_id, q.machine_id]; local = []
        cutoffs = range(192, q.cutoff) if dense else list(range(192, q.cutoff, 8))[-8:]
        for cutoff in cutoffs:
            if cutoff not in machine: continue
            qq = replace(q, query_id=q.query_id+f'-t{cutoff}', cutoff=cutoff, signal=0)
            h = sample_history(qq, idx, thin=(cutoff//8) % 2 == 0)
            if any(t not in h for t in range(cutoff-40, cutoff)): continue
            local.append((qq, h, machine[cutoff], False))
        sparse_count = 0
        for cutoff in list(range(192, q.cutoff, 8))[-8:]:
            if cutoff not in machine: continue
            qq = replace(q, query_id=q.query_id+f'-t{cutoff}', cutoff=cutoff, signal=0)
            h = sample_history(qq, idx, thin=(cutoff//8) % 2 == 0)
            if all(t in h for t in range(cutoff-40, cutoff)): sparse_count += 1
        historical_mass = sparse_count/(1+sparse_count)
        each_historical = historical_mass/len(local) if local else 0.
        for qq, h, y, known in local: out.append((qq, h, y, known, each_historical/8))
        out.append((q, sample_history(q, idx), PAYOUTS.index(labels[q.query_id]), True, 1/(1+sparse_count)/8))
    return out

def matrix_examples(es, priors, source):
    rec = [observable(q, h, source[q.episode_id], priors[q.episode_id], known) for q, h, y, known, w in es]
    assert all(list(r) == schema for r in rec)
    return np.asarray([[r[n] for n in schema] for r in rec], float)

def train_features(queries, dense):
    allowed = {q.episode_id for q in queries}; es = make_examples(queries, dense)
    priors = {}
    for fold in range(4):
        fit = {e for e in allowed if F[e] != fold}
        p = fit_prior(tq, idx, fit)
        for e in allowed:
            if F[e] == fold: priors[e] = p
    assert set(priors) == allowed
    x = matrix_examples(es, priors, byepisode)
    return x, np.array([e[2] for e in es]), np.array([e[4] for e in es])

def held_features(queries, fit_episodes, source, index):
    p = fit_prior(tq, idx, fit_episodes); priors = {q.episode_id: p for q in queries}
    es = [(q, sample_history(q, index), None, True, 1) for q in queries]
    return matrix_examples(es, priors, source)

x, y, w = train_features(tq, False)
feat_s = time.perf_counter()-t0
xt = held_features(vq, set(F), vbyepisode, vidx)
ps = []
params_seen = []
for seed in seeds:
    params = dict(depth=4, l2_leaf_reg=8, learning_rate=.04, iterations=trees[seed], random_seed=seed,
                  thread_count=2, allow_writing_files=False, verbose=False)
    win = CatBoostClassifier(loss_function='Logloss', **params)
    win.fit(Pool(x, (y > 0).astype(int), weight=w*len(w)/w.sum()))
    pos = y > 0; py = y[pos]-1; pw = w[pos]
    size = CatBoostClassifier(loss_function='MultiClass', classes_count=4, **params)
    size.fit(Pool(x[pos], py, weight=pw*len(pw)/pw.sum()))
    chance = win.predict_proba(xt, thread_count=2)[:, 1]
    cond = size.predict_proba(xt, thread_count=2)
    ps.append(np.column_stack((1-chance, chance[:, None]*cond)))
    params_seen.append({'seed': seed, 'win_trees': win.tree_count_, 'size_trees': size.tree_count_})
mine = np.mean(ps, axis=0)
want_map = read_submission(R/'artifacts/two_head_01/validation.csv')
want = np.array([want_map[q.query_id] for q in vq])
rep = {'train_rows': int(x.shape[0]), 'features': int(x.shape[1]),
       'max_abs_deviation_vs_committed_validation_csv': float(np.max(np.abs(mine-want))),
       'per_seed': params_seen, 'feature_seconds': feat_s, 'total_seconds': time.perf_counter()-t0,
       'label_files_opened': ['train/labels.csv'],
       'note': 'No validation label, target, condition or reference file was opened by this script.'}
(R/'artifacts/audit/provenance/verify/v_two_head_refit.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
print(json.dumps(rep,indent=2,sort_keys=True))
