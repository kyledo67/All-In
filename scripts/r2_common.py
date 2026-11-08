"""Round-2 (adaptation) data access and official-equivalent scoring.

Reimplements the organizer's contest metric from RULES.md rather than importing the
downloaded kit, then is cross-checked against the kit's own check_submission.py so the
reimplementation is verified, not assumed:

  per query   e = sum_k (p_k - q_k)^2        over the five payout categories
  per episode mean of e over that episode's queries
  per cond    E[c] = mean of episode means
  R           mean over the four conditions of E[c] / max(B[c], 1e-9)
  points      max_points / (1 + R)

B[c] comes from public/validation/reference.csv, the frozen pooled starter this round.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

R2 = Path('/Users/ayan/Downloads/adaptation-participants/public')
R1 = Path('/Users/ayan/Downloads/All-In/main-participants-60-40/public')
PAYOUTS = (0, 1, 2, 5, 20)
CONDITIONS = ('stable', 'short_history', 'drift', 'shared_change')
COLS = ('p_0', 'p_1', 'p_2', 'p_5', 'p_20')


def _rows(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def queries(root, split):
    out = {}
    for r in _rows(root / split / 'queries.csv'):
        out[r['query_id']] = {'episode_id': r['episode_id'], 'machine_id': r['machine_id'],
                              'machine_type': r['machine_type'], 'signal': int(r['signal']),
                              'cutoff': int(r['cutoff'])}
    return out


def history(root, split):
    """episode -> machine -> {tick: payout}; ticks are sparse (missing = unobserved)."""
    out = {}
    for r in _rows(root / split / 'history.csv'):
        out.setdefault(r['episode_id'], {}).setdefault(r['machine_id'], {})[int(r['tick'])] = int(r['payout'])
    return out


def labels(root, split):
    return {r['query_id']: int(r['outcome']) for r in _rows(root / split / 'labels.csv')}


def conditions(root, split):
    return {r['query_id']: r['condition'] for r in _rows(root / split / 'conditions.csv')}


def targets(root, split='validation'):
    out = {}
    for r in _rows(root / split / 'targets.csv'):
        out[r['query_id']] = np.array([float(r[c]) for c in COLS])
    return out


def read_sub(path):
    out = {}
    for r in _rows(Path(path)):
        v = np.array([float(r[c]) for c in COLS])
        s = v.sum()
        out[r['query_id']] = v / s if s > 0 else np.full(5, 0.2)
    return out


def write_sub(path, preds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(('query_id',) + COLS)
        for qid in sorted(preds):
            v = np.asarray(preds[qid], dtype=float)
            v = np.clip(v, 0.0, 1.0)
            v = v / v.sum()
            w.writerow([qid] + [repr(float(x)) for x in v])
    return path


def episode_errors(pred, tgt, qmeta):
    """query_id-keyed dicts -> {episode: mean squared-probability error}."""
    by_ep = {}
    for qid, q in tgt.items():
        p = pred[qid]
        by_ep.setdefault(qmeta[qid]['episode_id'], []).append(
            math.fsum((float(a) - float(b)) ** 2 for a, b in zip(p, q)))
    return {e: math.fsum(v) / len(v) for e, v in by_ep.items()}


def score(pred, tgt, qmeta, cond, ref_ep=None, max_points=20.0):
    ep = episode_errors(pred, tgt, qmeta)
    ep_cond = {}
    for qid, c in cond.items():
        if qid in tgt:
            ep_cond[qmeta[qid]['episode_id']] = c
    out = {'by_condition': {}, 'episode_errors': ep}
    ratios = []
    for c in CONDITIONS:
        eps = sorted(e for e, cc in ep_cond.items() if cc == c)
        E = math.fsum(ep[e] for e in eps) / len(eps)
        rec = {'error': E, 'episodes': len(eps)}
        if ref_ep is not None:
            B = math.fsum(ref_ep[e] for e in eps) / len(eps)
            rec['reference_error'] = B
            rec['loss_ratio'] = E / max(B, 1e-9)
            ratios.append(rec['loss_ratio'])
        out['by_condition'][c] = rec
    if ratios:
        R = math.fsum(ratios) / len(ratios)
        out['R'] = R
        out['points_20'] = max_points / (1 + R)
        out['points_40'] = 40.0 / (1 + R)
    out['overall_error'] = math.fsum(ep.values()) / len(ep)
    return out


def reference_episode_errors(root=R2):
    tgt = targets(root)
    qm = queries(root, 'validation')
    return episode_errors(read_sub(root / 'validation' / 'reference.csv'), tgt, qm)


def full_score(sub_path_or_pred, root=R2, max_points=20.0):
    pred = read_sub(sub_path_or_pred) if isinstance(sub_path_or_pred, (str, Path)) else sub_path_or_pred
    return score(pred, targets(root), queries(root, 'validation'),
                 conditions(root, 'validation'), reference_episode_errors(root), max_points)


def paired_bootstrap(pred_a, pred_b, root=R2, repeats=4000, seed=12345):
    """Resample whole episodes within each condition; report the ΔR distribution (b - a)."""
    tgt, qm, cond = targets(root), queries(root, 'validation'), conditions(root, 'validation')
    refep = reference_episode_errors(root)
    epa, epb = episode_errors(pred_a, tgt, qm), episode_errors(pred_b, tgt, qm)
    ep_cond = {qm[q]['episode_id']: c for q, c in cond.items() if q in tgt}
    buckets = {c: sorted(e for e, cc in ep_cond.items() if cc == c) for c in CONDITIONS}
    rng = np.random.default_rng(seed)
    out = np.empty(repeats)
    arrays = {c: (np.array([epa[e] for e in buckets[c]]), np.array([epb[e] for e in buckets[c]]),
                  np.array([refep[e] for e in buckets[c]])) for c in CONDITIONS}
    for i in range(repeats):
        ra = rb = 0.0
        for c in CONDITIONS:
            a, b, r = arrays[c]
            idx = rng.integers(0, len(a), len(a))
            denom = max(r[idx].mean(), 1e-9)
            ra += a[idx].mean() / denom
            rb += b[idx].mean() / denom
        out[i] = (rb - ra) / len(CONDITIONS)
    return out
