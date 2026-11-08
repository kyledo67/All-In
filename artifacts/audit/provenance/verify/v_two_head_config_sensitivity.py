"""Analogue of the auditor's mtd changepoint test, for two_head: how much of the
candidate's score depends on hard-coding CatBoost config index 0 (depth 4 / l2 8,
trees 214/307) rather than the other predeclared config (depth 6 / l2 16, 400/396)?"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from dataclasses import replace
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In'); sys.path[:0] = [str(R), str(R/'organizer')]
import numpy as np
from scipy.optimize import minimize
from catboost import CatBoostClassifier, Pool
from slot_model.features import causal_index, PAYOUTS
from slot_model.phase2_features import fit_prior, observable, sample_history
from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets, read_rows
from slot_forecasting.contest import contest_score
from slot_forecasting.schema import CONDITIONS

D = R/'main-participants-60-40/public'
F = json.loads((R/'artifacts/phase2/folds.json').read_text())['train']
schema = json.loads((R/'artifacts/phase2/feature_schema.json').read_text())['observable']
protocol = json.loads((R/'artifacts/phase2/protocol.json').read_text()); seeds = protocol['seeds']
runs = {r['name']: r for r in json.loads((R/'artifacts/phase2/training_runs.json').read_text())}
CONFIGS = {0: (dict(depth=4, l2_leaf_reg=8), {s: runs[f'cb_observable_0_{s}']['trees'] for s in seeds}),
           1: (dict(depth=6, l2_leaf_reg=16), {s: runs[f'cb_observable_1_{s}']['trees'] for s in seeds})}
tq, th = load_inputs(D/'train'); vq, vh = load_inputs(D/'validation')
labels = read_labels(D/'train/labels.csv', tq)
idx = causal_index(tq, th); vidx = causal_index(vq, vh)
byepisode = {}
for k, h in idx.items(): byepisode.setdefault(k[0], {})[k] = h
vbyepisode = {}
for k, h in vidx.items(): vbyepisode.setdefault(k[0], {})[k] = h

def make_examples(queries):
    out = []
    for q in sorted(queries, key=lambda q: q.query_id):
        machine = idx[q.episode_id, q.machine_id]; local = []
        for cutoff in list(range(192, q.cutoff, 8))[-8:]:
            if cutoff not in machine: continue
            qq = replace(q, query_id=q.query_id+f'-t{cutoff}', cutoff=cutoff, signal=0)
            h = sample_history(qq, idx, thin=(cutoff//8) % 2 == 0)
            if any(t not in h for t in range(cutoff-40, cutoff)): continue
            local.append((qq, h, machine[cutoff], False))
        sparse_count = len(local)
        hist = sparse_count/(1+sparse_count)
        each = hist/len(local) if local else 0.
        for qq, h, y, kn in local: out.append((qq, h, y, kn, each/8))
        out.append((q, sample_history(q, idx), PAYOUTS.index(labels[q.query_id]), True, 1/(1+sparse_count)/8))
    return out

def mat(es, priors, source):
    rec = [observable(q, h, source[q.episode_id], priors[q.episode_id], kn) for q, h, y, kn, w in es]
    assert all(list(r) == schema for r in rec)
    return np.asarray([[r[n] for n in schema] for r in rec], float)

es = make_examples(tq); allowed = {q.episode_id for q in tq}
priors = {}
for fold in range(4):
    p = fit_prior(tq, idx, {e for e in allowed if F[e] != fold})
    for e in allowed:
        if F[e] == fold: priors[e] = p
x = mat(es, priors, byepisode); y = np.array([e[2] for e in es]); w = np.array([e[4] for e in es])
vp = fit_prior(tq, idx, set(F))
xt = mat([(q, sample_history(q, vidx), None, True, 1) for q in vq], {q.episode_id: vp for q in vq}, vbyepisode)

def two_head(cfg, trees):
    ps = []
    for seed in seeds:
        params = dict(learning_rate=.04, iterations=trees[seed], random_seed=seed, thread_count=2,
                      allow_writing_files=False, verbose=False, **cfg)
        win = CatBoostClassifier(loss_function='Logloss', **params)
        win.fit(Pool(x, (y > 0).astype(int), weight=w*len(w)/w.sum()))
        pos = y > 0; py = y[pos]-1; pw = w[pos]
        size = CatBoostClassifier(loss_function='MultiClass', classes_count=4, **params)
        size.fit(Pool(x[pos], py, weight=pw*len(pw)/pw.sum()))
        c = win.predict_proba(xt, thread_count=2)[:, 1]
        ps.append(np.column_stack((1-c, c[:, None]*size.predict_proba(xt, thread_count=2))))
    return {q.query_id: p.tolist() for q, p in zip(vq, np.mean(ps, axis=0))}

built = {k: two_head(*CONFIGS[k]) for k in CONFIGS}
# ---- score the candidate with each two_head column -------------------------
targets = read_targets(D/'validation/targets.csv')
groups = {r['query_id']: r['condition'] for r in read_rows(D/'validation/conditions.csv', ('query_id','condition'))}
reference = read_submission(D/'validation/reference.csv')
folds = json.loads((R/'artifacts/phase2/folds.json').read_text())['validation']
ids = sorted(reference)
ep = np.array([{t.query_id: t.episode_id for t in targets}[q] for q in ids])
truth = np.array([{t.query_id: t.probabilities for t in targets}[q] for q in ids])
ref = np.array([reference[q] for q in ids]); episodes = sorted(set(ep.tolist()))
def arr(d): return np.array([d[q] for q in ids])
def load(rel): return arr(read_submission(R/rel))
frozen = load('artifacts/phase2/predictions/oof_ensemble.csv')
tab = (load('artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv')
       + load('artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv'))/2
mtd = load('artifacts/mtd_pairs_01/validation_lag7.csv')
def official(a): return contest_score(dict(zip(ids, a.tolist())), targets, groups, reference, max_points=60)
def rw(es_):
    es_ = set(es_); v = np.zeros(len(ids))
    for c in sorted(CONDITIONS):
        ces = {e for i, e in enumerate(ep) if e in es_ and groups[ids[i]] == c}
        b = float(np.mean([np.mean(np.sum((ref[ep == e]-truth[ep == e])**2, axis=1)) for e in ces]))
        for e in ces: v[ep == e] = 1/(4*len(ces)*int(np.sum(ep == e))*max(b, 1e-9))
    return v
def fit(comps, es_):
    p = np.array(comps); v = rw(es_)
    f = ((p-truth)*np.sqrt(v)[None, :, None]).reshape(len(p), -1); g = f@f.T
    r = minimize(lambda z: float(z@g@z), np.ones(len(p))/len(p), jac=lambda z: 2*g@z, method='SLSQP',
                 bounds=[(0, 1)]*len(p), constraints={'type':'eq','fun':lambda z: z.sum()-1,
                 'jac': lambda z: np.ones_like(z)}, options={'ftol':1e-12,'maxiter':200})
    assert r.success
    return r.x/r.x.sum()
def build(comps):
    held = np.empty_like(comps[0]); ws = []
    for k in range(4):
        h = {e for e in episodes if folds[e] == k}
        wt = fit(comps, sorted(set(episodes)-h)); ws.append(wt.tolist())
        m = np.array([e in h for e in ep]); held[m] = np.einsum('m,mnc->nc', wt, np.array(comps))[m]
    return held, ws
rep = {}
committed = load('artifacts/two_head_01/validation.csv')
rep['refit_cfg0_vs_committed_max_abs'] = float(np.max(np.abs(arr(built[0])-committed)))
for k in CONFIGS:
    col = arr(built[k])
    cand, ws = build([frozen, tab, mtd, col])
    rep[f'config{k}'] = {'params': CONFIGS[k][0], 'trees': CONFIGS[k][1],
                         'two_head_standalone_points': official(col)['technical_points'],
                         'candidate_points': official(cand)['technical_points'],
                         'fold_weights': ws}
rep['delta_points_config0_minus_config1'] = (rep['config0']['candidate_points']
                                             - rep['config1']['candidate_points'])
rep['max_abs_col0_minus_col1'] = float(np.max(np.abs(arr(built[0])-arr(built[1]))))
(R/'artifacts/audit/provenance/verify/v_two_head_config_sensitivity.json').write_text(
    json.dumps(rep, indent=2, sort_keys=True)+'\n')
print(json.dumps({k: v for k, v in rep.items() if k != 'config0' and k != 'config1'}, indent=2, sort_keys=True))
for k in CONFIGS:
    print('config', k, rep[f'config{k}']['params'], rep[f'config{k}']['trees'],
          'standalone %.6f candidate %.6f' % (rep[f'config{k}']['two_head_standalone_points'],
                                              rep[f'config{k}']['candidate_points']))
