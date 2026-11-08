"""Independent recompute of the 32.260531 candidate from raw component CSVs.

Written for the provenance audit. Does not import scripts/combined_evaluate.py.
Scores with the organizer's contest_score directly, and separately re-derives
R from the written definition as a cross-check.
"""
import hashlib, json, sys, math
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(R), str(R / 'organizer')]
import numpy as np
from scipy.optimize import minimize
from slot_forecasting.io import read_targets, read_rows, read_submission, write_submission
from slot_forecasting.contest import contest_score
from slot_forecasting.schema import CONDITIONS

D = R / 'main-participants-60-40/public'
OUT = R / 'artifacts/audit/provenance'
OUT.mkdir(parents=True, exist_ok=True)

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

INPUTS = {
    'frozen_blend': 'artifacts/phase2/predictions/oof_ensemble.csv',
    'tabpfn_seed_20261108': 'artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv',
    'tabpfn_seed_20261109': 'artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv',
    'mtd_lag7': 'artifacts/mtd_pairs_01/validation_lag7.csv',
    'two_head': 'artifacts/two_head_01/validation.csv',
    'tabpfn_oof_composite': 'artifacts/tabpfn/predictions/oof_tabpfn.csv',
    'cb_hmm_0_20261108': 'artifacts/phase2/predictions/cb_hmm_0_20261108.csv',
    'cb_hmm_0_20261109': 'artifacts/phase2/predictions/cb_hmm_0_20261109.csv',
    'orchestrator_oof': 'artifacts/audit/oof_candidate_raw.csv',
    'folds.json': 'artifacts/phase2/folds.json',
}
sha = {k: hashlib.sha256((R / v).read_bytes()).hexdigest() for k, v in INPUTS.items()}

def load(rel):
    d = read_submission(R / rel)
    return np.array([d[q] for q in ids])

A = {k: load(v) for k, v in INPUTS.items() if v.endswith('.csv')}
tabpfn_raw = (A['tabpfn_seed_20261108'] + A['tabpfn_seed_20261109']) / 2.0
cb_hmm_avg = (A['cb_hmm_0_20261108'] + A['cb_hmm_0_20261109']) / 2.0

def mapping(a):
    return dict(zip(ids, a.tolist()))

def official(a, es=None):
    es = set(episodes if es is None else es)
    sel = [t for t in targets if t.episode_id in es]
    k = {t.query_id for t in sel}
    m = mapping(a)
    return contest_score({q: m[q] for q in k}, sel, {q: groups[q] for q in k},
                         {q: reference[q] for q in k}, max_points=60)

def manual_R(a, es=None):
    """R straight from the written definition, no organizer code."""
    es = set(episodes if es is None else es)
    total = 0.0
    for cond in sorted(CONDITIONS):
        ces = sorted({e for i, e in enumerate(ep) if e in es and groups[ids[i]] == cond})
        def cond_err(pred):
            return float(np.mean([np.mean(np.sum((pred[ep == e] - truth[ep == e]) ** 2, axis=1)) for e in ces]))
        total += cond_err(a) / max(cond_err(ref), 1e-9)
    return total / 4.0

def row_weights(es):
    """Per-row weight making sum(w * rowloss) == the official mean loss ratio on `es`."""
    es = set(es)
    w = np.zeros(len(ids))
    for cond in sorted(CONDITIONS):
        ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == cond}
        b = float(np.mean([np.mean(np.sum((ref[ep == e] - truth[ep == e]) ** 2, axis=1)) for e in ces]))
        for e in ces:
            w[ep == e] = 1.0 / (4 * len(ces) * int(np.sum(ep == e)) * max(b, 1e-9))
    return w

def simplex_fit(arrays, es):
    p = np.array(arrays)
    w = row_weights(es)
    flat = ((p - truth) * np.sqrt(w)[None, :, None]).reshape(len(p), -1)
    gram = flat @ flat.T
    res = minimize(lambda x: float(x @ gram @ x), np.ones(len(p)) / len(p), jac=lambda x: 2 * gram @ x,
                   method='SLSQP', bounds=[(0, 1)] * len(p),
                   constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1, 'jac': lambda x: np.ones_like(x)},
                   options={'ftol': 1e-12, 'maxiter': 200})
    assert res.success, res.message
    return res.x / res.x.sum()

NAMES = ['frozen_blend', 'tabpfn_raw', 'mtd_lag7', 'two_head']

def run(base_array, label):
    comps = [base_array, tabpfn_raw, A['mtd_lag7'], A['two_head']]
    held = np.empty_like(comps[0])
    fold_w, fold_fit_episodes = [], []
    for fold in range(4):
        h = {e for e in episodes if folds[e] == fold}
        fit = sorted(set(episodes) - h)
        w = simplex_fit(comps, fit)
        fold_w.append(dict(zip(NAMES, w.tolist())))
        fold_fit_episodes.append({'fold': fold, 'held': len(h), 'fit': len(fit),
                                  'overlap': sorted(set(fit) & h)})
        mask = np.array([e in h for e in ep])
        held[mask] = np.einsum('m,mnc->nc', w, np.array(comps))[mask]
    s = official(held)
    return {'label': label, 'technical_points': s['technical_points'], 'mean_loss_ratio': s['mean_loss_ratio'],
            'manual_R': manual_R(held), 'fold_weights': fold_w, 'fold_episode_check': fold_fit_episodes,
            'by_condition': {c: v['loss_ratio'] for c, v in s['by_condition'].items()}}, held

main, held_main = run(A['frozen_blend'], 'candidate(frozen_blend, tabpfn_raw, mtd_lag7, two_head)')
sub_avg, _ = run(cb_hmm_avg, 'substitute base = raw cb_hmm_0 seed average')
sub_one, _ = run(A['cb_hmm_0_20261108'], 'substitute base = raw cb_hmm_0_20261108')
drop3, held_drop = run(tabpfn_raw, 'substitute base = tabpfn_raw (base column duplicated)')

write_submission(OUT / 'oof_candidate_independent.csv', mapping(held_main))

report = {
  'sha256_inputs': sha,
  'sha256_my_oof': hashlib.sha256((OUT / 'oof_candidate_independent.csv').read_bytes()).hexdigest(),
  'candidate': main,
  'substitutions': [sub_avg, sub_one, drop3],
  'max_abs_vs_orchestrator_oof': float(np.max(np.abs(held_main - A['orchestrator_oof']))),
  'tabpfn_raw_vs_oof_composite_max_abs': float(np.max(np.abs(tabpfn_raw - A['tabpfn_oof_composite']))),
  'frozen_blend_points': official(A['frozen_blend'])['technical_points'],
  'tabpfn_raw_points': official(tabpfn_raw)['technical_points'],
  'mtd_lag7_points': official(A['mtd_lag7'])['technical_points'],
  'two_head_points': official(A['two_head'])['technical_points'],
  'reference_points': official(ref)['technical_points'],
  'truth_points': official(truth)['technical_points'],
  'n_queries': len(ids), 'n_episodes': len(episodes),
  'fold_sizes': {str(f): sum(1 for e in episodes if folds[e] == f) for f in range(4)},
}
(OUT / 'independent_recompute.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'sha256_inputs'}, indent=2, sort_keys=True))
