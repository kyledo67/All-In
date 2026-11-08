"""3-component candidate (no frozen blend at all) + paired whole-episode bootstrap."""
import hashlib, json, sys
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(R), str(R / 'organizer')]
import numpy as np
from scipy.optimize import minimize
from slot_forecasting.io import read_targets, read_rows, read_submission
from slot_forecasting.contest import contest_score
from slot_forecasting.schema import CONDITIONS

D = R / 'main-participants-60-40/public'
OUT = R / 'artifacts/audit/provenance'
targets = read_targets(D / 'validation/targets.csv')
groups = {r['query_id']: r['condition'] for r in read_rows(D / 'validation/conditions.csv', ('query_id', 'condition'))}
reference = read_submission(D / 'validation/reference.csv')
folds = json.loads((R / 'artifacts/phase2/folds.json').read_text())['validation']
ids = sorted(reference)
episode_of = {t.query_id: t.episode_id for t in targets}
ep = np.array([episode_of[q] for q in ids])
truth = np.array([{t.query_id: t.probabilities for t in targets}[q] for q in ids])
ref = np.array([reference[q] for q in ids])
episodes = sorted(set(ep.tolist()))

def load(rel):
    d = read_submission(R / rel)
    return np.array([d[q] for q in ids])

frozen = load('artifacts/phase2/predictions/oof_ensemble.csv')
tab = (load('artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv')
       + load('artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv')) / 2.0
mtd = load('artifacts/mtd_pairs_01/validation_lag7.csv')
two = load('artifacts/two_head_01/validation.csv')

def mapping(a): return dict(zip(ids, a.tolist()))
def official(a):
    return contest_score(mapping(a), targets, groups, reference, max_points=60)
def row_weights(es):
    es = set(es); w = np.zeros(len(ids))
    for c in sorted(CONDITIONS):
        ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == c}
        b = float(np.mean([np.mean(np.sum((ref[ep == e] - truth[ep == e])**2, axis=1)) for e in ces]))
        for e in ces: w[ep == e] = 1.0/(4*len(ces)*int(np.sum(ep == e))*max(b, 1e-9))
    return w
def fit_simplex(arrays, es):
    p = np.array(arrays); w = row_weights(es)
    flat = ((p - truth) * np.sqrt(w)[None, :, None]).reshape(len(p), -1)
    g = flat @ flat.T
    r = minimize(lambda x: float(x@g@x), np.ones(len(p))/len(p), jac=lambda x: 2*g@x, method='SLSQP',
                 bounds=[(0, 1)]*len(p), constraints={'type': 'eq', 'fun': lambda x: x.sum()-1,
                 'jac': lambda x: np.ones_like(x)}, options={'ftol': 1e-12, 'maxiter': 200})
    assert r.success
    return r.x / r.x.sum()
def build(comps):
    held = np.empty_like(comps[0]); ws = []
    for f in range(4):
        h = {e for e in episodes if folds[e] == f}
        w = fit_simplex(comps, sorted(set(episodes)-h)); ws.append(w.tolist())
        m = np.array([e in h for e in ep])
        held[m] = np.einsum('m,mnc->nc', w, np.array(comps))[m]
    return held, ws

four, w4 = build([frozen, tab, mtd, two])
three, w3 = build([tab, mtd, two])

def paired_bootstrap(a, b, repeats=2000, seed=314159):
    am, bm = mapping(a), mapping(b)
    rows = {}
    for c in CONDITIONS:
        es = sorted({t.episode_id for t in targets if groups[t.query_id] == c})
        vals = []
        for e in es:
            ts = [t for t in targets if t.episode_id == e]
            vals.append([np.mean([np.sum((np.asarray(p[t.query_id]) - t.probabilities)**2) for t in ts])
                         for p in (am, bm, reference)])
        rows[c] = np.asarray(vals)
    rng = np.random.default_rng(seed)
    totals = np.zeros(repeats); conds = {}
    for c, vals in rows.items():
        draws = rng.integers(0, len(vals), (repeats, len(vals)))
        means = vals[draws].mean(1)
        diff = (means[:, 0] - means[:, 1]) / np.maximum(means[:, 2], 1e-9)
        totals += diff/4
        conds[c] = np.quantile(diff, [.025, .975]).tolist()
    return dict(delta_loss_ratio_ci95=np.quantile(totals, [.025, .975]).tolist(),
                by_condition_ci95=conds, median=float(np.median(totals)),
                fraction_below_zero=float(np.mean(totals < 0)), repeats=repeats, seed=seed)

boot = paired_bootstrap(four, frozen)
boot3 = paired_bootstrap(three, frozen)
rep = {'four_component_points': official(four)['technical_points'],
       'three_component_no_frozen_points': official(three)['technical_points'],
       'four_minus_three_points': official(four)['technical_points']-official(three)['technical_points'],
       'frozen_points': official(frozen)['technical_points'],
       'fold_weights_four': w4, 'fold_weights_three': w3,
       'paired_candidate_vs_frozen': boot, 'paired_three_vs_frozen': boot3,
       'note': 'All 96 public-validation episodes were examined many times before this. '
               'These intervals are exploratory, unadjusted for the model selection that '
               'preceded them, and no untouched holdout exists.'}
(OUT / 'bootstrap_and_dropbase.json').write_text(json.dumps(rep, indent=2, sort_keys=True)+'\n')
print(json.dumps(rep, indent=2, sort_keys=True))
