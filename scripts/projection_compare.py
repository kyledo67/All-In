"""Verify the projection operator, then compare it against the deployed tail correction.

PROTOCOL DECLARATION. Part A verifies the operator. Part B is a saved-prediction
comparison on the ORIGINAL win_head_release_dirichlet array using TRAIN-derived tail
constants, so nothing is retrained. Part C is a NEW PROTOCOL, declared explicitly rather
than slipped in: it uses legitimately released validation probability targets as
supervision, cross-fitted over complete-episode outer folds, with every target-dependent
constant fitted only on fitting-side episodes. That is a different rule from the train-only
selection used previously, and results under it are labelled separately. The frozen base
recipe was itself selected on these same episodes, so every number here is conditional on
that recipe and is not a pristine evaluation of the whole research process.

Candidates, q and all other components matched except where stated:
  original          the win_head_release_dirichlet forecast            53.376905
  tail_rescale      the DEPLOYED operation: preserve a = P(1x|pos), rescale the remainder
  tail_projection   exact Euclidean projection onto conv{e0, e1, (0,0,v)} with the SAME
                    train tail constants - isolates the projection operation
  endpoint_proj     exact projection onto conv{e0, (0,r_L), (0,r_H)} with CROSS-FITTED
                    endpoints - permits continuous uncertainty between endpoints, unlike
                    the hard snap that previously lost 1.036641 points
  tail_proj_crossfit exact projection with cross-fitted tail constants - isolates whether
                    better structural constants, not the operator, carry the gain

Exposure: 96 repeatedly inspected validation episodes. Exploratory; new folds do not
create an independent holdout.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.optimize import minimize

from slot_model.simplex_project import (endpoint_family_vertices, project_to_hull,
                                        tail_family_vertices)
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission

TAILS = (2, 5, 20)
CHECKS = []


def record(name, ok, detail):
    CHECKS.append({'check': name, 'passed': bool(ok), 'detail': detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def verify():
    rng = np.random.default_rng(0)
    v = np.array([0.7, 0.25, 0.05])
    V = tail_family_vertices(v)

    # independent constrained optimiser as the reference
    worst = 0.0
    for _ in range(300):
        p = rng.random(5)
        p /= p.sum()
        got, _ = project_to_hull(p, V)
        f = lambda lam: float(np.sum((V.T @ lam - p) ** 2))
        res = minimize(f, np.ones(3) / 3, method='SLSQP', bounds=[(0, 1)] * 3,
                       constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1},
                       options={'ftol': 1e-14, 'maxiter': 500})
        ref = V.T @ (np.clip(res.x, 0, None) / max(np.clip(res.x, 0, None).sum(), 1e-15))
        worst = max(worst, float(np.sum((got - p) ** 2) - np.sum((ref - p) ** 2)))
    record('projection_optimal', worst < 1e-10,
           f'over 300 random vectors the active-set solution is never worse than SLSQP; '
           f'max excess squared distance {worst:.2e}')

    # idempotence: a point already in the family must be returned unchanged
    worst_idem = 0.0
    for _ in range(300):
        lam = rng.random(3)
        lam /= lam.sum()
        q = V.T @ lam
        got, _ = project_to_hull(q, V)
        worst_idem = max(worst_idem, float(np.max(abs(got - q))))
    record('idempotent', worst_idem < 1e-10,
           f'points already in conv(V) return unchanged to {worst_idem:.2e}')

    # boundary cases: each vertex, and vectors far outside
    ok = True
    for i in range(3):
        got, lam = project_to_hull(V[i], V)
        ok &= float(np.max(abs(got - V[i]))) < 1e-12 and lam[i] > 1 - 1e-9
    far, lamf = project_to_hull(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), V)
    record('boundary', ok and np.isfinite(far).all() and abs(far.sum() - 1) < 1e-12,
           f'all three vertices are fixed points; an extreme 20x-only input maps to '
           f'{np.round(far, 4)} with weights {np.round(lamf, 4)}')

    # the deployed rescale is NOT the projection - demonstrate numerically
    diffs = []
    for _ in range(300):
        p = rng.random(5)
        p /= p.sum()
        qq = 1 - p[0]
        a = p[1] / max(qq, 1e-12)
        res = np.concatenate(([p[0]], [qq * a], qq * (1 - a) * v))
        res = res / res.sum()
        proj, _ = project_to_hull(p, V)
        diffs.append(float(np.max(abs(res - proj))))
    record('rescale_is_not_projection', np.mean(diffs) > 1e-3,
           f'the deployed rescale differs from the exact projection by mean max-abs '
           f'{np.mean(diffs):.4f} (p90 {np.percentile(diffs, 90):.4f}) on random vectors, '
           f'so the two operations are genuinely different')
    return all(c['passed'] for c in CHECKS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--original', type=Path,
                        default=ROOT / 'artifacts/win_head_release_dirichlet/validation_submission.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/projection')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    print('=== Part A: operator verification ===')
    if not verify():
        (args.out / 'verification.json').write_text(
            json.dumps({'checks': CHECKS, 'passed': False}, indent=2) + '\n')
        raise SystemExit('projection verification failed')
    (args.out / 'verification.json').write_text(
        json.dumps({'checks': CHECKS, 'passed': True}, indent=2) + '\n')

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    vq, _ = load_inputs(args.data / 'validation')
    vtype = {q.query_id: q.machine_type for q in vq}
    types = np.array([vtype[i] for i in ids])

    original = read_submission(args.original)
    original = np.array([original[q] for q in ids])

    # train-derived tail constants (unchanged from the deployed candidate)
    tq, _ = load_inputs(args.data / 'train')
    labels = read_labels(args.data / 'train/labels.csv', tq)
    ttype = {q.query_id: q.machine_type for q in tq}
    counts = {}
    for k, v in labels.items():
        if v in TAILS:
            counts.setdefault(ttype[k], np.zeros(3))[TAILS.index(v)] += 1
    tail_train = {t: (c / c.sum()) for t, c in counts.items()}

    r_true = truth[:, 1:] / np.maximum(truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def official(a):
        a = a / a.sum(axis=1, keepdims=True)
        return score(mapping(a), scoring, episodes)

    # (1) deployed rescale, recomputed from the ORIGINAL array
    rescale = np.empty_like(original)
    for k in range(len(ids)):
        q = 1 - original[k, 0]
        a = original[k, 1] / max(q, 1e-12)
        tail = tail_train[types[k]]
        rescale[k] = np.concatenate(([original[k, 0]], [q * a], q * (1 - a) * tail))
    rescale /= rescale.sum(axis=1, keepdims=True)

    # (2) exact projection, SAME train constants
    tail_proj = np.empty_like(original)
    for k in range(len(ids)):
        tail_proj[k], _ = project_to_hull(original[k], tail_family_vertices(tail_train[types[k]]))

    # (3)+(4) cross-fitted structural constants from released targets
    tail_cross = np.empty_like(original)
    end_proj = np.empty_like(original)
    cross_param = []
    for fold in range(4):
        held = np.array([folds[x] == fold for x in ep])
        fit = ~held
        params = {}
        for t in sorted(set(types.tolist())):
            m = fit & (types == t)
            tl = r_true[m]
            tail = tl[:, 1:] / np.maximum(tl[:, 1:].sum(axis=1, keepdims=True), 1e-15)
            uniq = sorted({tuple(np.round(x, 9)) for x in tl})
            params[t] = {'tail': tail.mean(axis=0),
                         'r_low': np.array(uniq[0]), 'r_high': np.array(uniq[-1]),
                         'n_endpoints_seen': len(uniq)}
        cross_param.append({t: {'tail': params[t]['tail'].tolist(),
                                'r_low': params[t]['r_low'].tolist(),
                                'r_high': params[t]['r_high'].tolist(),
                                'n_endpoints_seen': params[t]['n_endpoints_seen']}
                            for t in params})
        for k in np.flatnonzero(held):
            pt = params[types[k]]
            tail_cross[k], _ = project_to_hull(original[k], tail_family_vertices(pt['tail']))
            end_proj[k], _ = project_to_hull(
                original[k], endpoint_family_vertices(pt['r_low'], pt['r_high']))

    variants = {'original': original, 'tail_rescale_deployed': rescale,
                'tail_projection_train': tail_proj, 'tail_projection_crossfit': tail_cross,
                'endpoint_projection_crossfit': end_proj}
    results = {k: official(v) for k, v in variants.items()}
    base = results['original']
    paired = {k: paired_bootstrap(mapping(v / v.sum(axis=1, keepdims=True)),
                                  mapping(original / original.sum(axis=1, keepdims=True)),
                                  scoring, episodes)
              for k, v in variants.items() if k != 'original'}

    report = {
        'protocol': __doc__,
        'what_tail_train_actually_is':
            'It preserves a = P(1x | positive), hence p_0 and p_1, and rescales the remaining '
            'mass to the fixed tail ratio. It lands in the declared family but is NOT the '
            'nearest point in it. Confirmed numerically: it differs from the exact projection '
            'by a mean max-abs of about 0.08 on random probability vectors.',
        'verification': CHECKS,
        'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                        'gain_over_original': v['technical_points'] - base['technical_points'],
                        'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                    for k, v in results.items()},
        'paired_vs_original': {k: v['delta_loss_ratio_ci95'] for k, v in paired.items()},
        'crossfit_parameters_per_fold': cross_param,
        'train_tail_constants': {t: v.tolist() for t, v in tail_train.items()},
        'attribution':
            'tail_projection_train vs tail_rescale_deployed isolates the OPERATOR at fixed '
            'constants. tail_projection_crossfit vs tail_projection_train isolates the '
            'CONSTANTS at a fixed operator. endpoint_projection_crossfit tests the richer '
            'geometry that hard snapping could not.',
        'exposure': 'Repeatedly inspected public validation; exploratory. Cross-fitted arms use '
                    'released probability supervision under the newly declared protocol.',
        'seconds': time.perf_counter() - start}
    for k, v in variants.items():
        write_submission(args.out / 'predictions' / f'oof_{k}.csv',
                         mapping(v / v.sum(axis=1, keepdims=True)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(base['by_condition'])
    print(f"\n=== Part B/C: matched comparison on the ORIGINAL array ===")
    print(f"{'candidate':<32}{'points':>12}{'gain':>9}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('original', 'tail_rescale_deployed', 'tail_projection_train',
              'tail_projection_crossfit', 'endpoint_projection_crossfit'):
        v = results[k]
        g = v['technical_points'] - base['technical_points']
        print(f"{k:<32}{v['technical_points']:>12.6f}{g:>+9.4f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print('\npaired 95% intervals against the original (negative favours the variant):')
    for k, full in paired.items():
        ci = full['delta_loss_ratio_ci95']
        print(f"   {k:<32} [{ci[0]:+.5f}, {ci[1]:+.5f}]"
              + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
