"""Measure whether adding more model classes to the blend actually helps.

Same four-fold protocol as scripts/phase2_evaluate.py and scripts/tabpfn_evaluate.py:
convex weights fitted on three folds, held-fold predictions concatenated, organizer
scorer. The frozen 31.702332 blend enters as a single component, so every number is
directly comparable to the existing bar.

Each challenger is supplied as `--component name=path/to/validation.csv`. For each one
and for each combination, the script reports the official score, per-condition loss
ratios, per-fold wins, the fitted simplex weight, and a paired whole-episode bootstrap
against the frozen blend.

The point is to separate "the development number went up" from "this component carries
information the blend does not already have". With six components fitted on 96
previously examined validation episodes, the former happens almost by construction.
The residual-cosine complementarity test and the per-fold win count are the checks that
do not inflate that way.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.optimize import minimize

from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import read_submission, write_submission

PHASE2 = ROOT / 'artifacts/phase2'
BASE = 'frozen_blend'
FIXED_WEIGHTS = (0.15, 0.25, 0.35)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=PHASE2 / 'predictions/oof_ensemble.csv',
                        help='The frozen four-recipe blend that defines the bar.')
    parser.add_argument('--component', action='append', default=[], metavar='NAME=PATH',
                        help='Challenger validation predictions; repeatable.')
    parser.add_argument('--weights', nargs='+', type=float, default=list(FIXED_WEIGHTS))
    parser.add_argument('--max-combination', type=int, default=3,
                        help='Largest number of challengers to add to the base at once.')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/combined/evaluation.json')
    args = parser.parse_args()
    start = time.perf_counter()

    folds = json.loads((PHASE2 / 'folds.json').read_text())['validation']
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    episode_of = {t.query_id: t.episode_id for t in targets}
    ep = np.array([episode_of[q] for q in ids])
    truth_of = {t.query_id: t.probabilities for t in targets}
    truth = np.array([truth_of[q] for q in ids])
    ref = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())

    def load(path):
        d = read_submission(path)
        return np.array([d[q] for q in ids])

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def official(a, es=episodes):
        return score(mapping(a), scoring, es)

    def row_weights(es):
        w = np.zeros(len(ids))
        for condition in sorted(set(groups.values())):
            ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == condition}
            b = np.mean([np.mean(np.sum((ref[ep == e] - truth[ep == e]) ** 2, axis=1)) for e in ces])
            for e in ces:
                w[ep == e] = 1 / (4 * len(ces) * sum(ep == e) * max(b, 1e-9))
        return w

    def simplex_fit(arrays, es):
        p = np.array(arrays)
        w = row_weights(es)
        flat = ((p - truth) * np.sqrt(w)[None, :, None]).reshape(len(p), -1)
        gram = flat @ flat.T
        norm = np.sqrt(np.diag(gram))
        cos = gram / np.maximum(norm[:, None] * norm[None, :], 1e-15)
        res = minimize(lambda x: float(x @ gram @ x), np.ones(len(p)) / len(p), jac=lambda x: 2 * gram @ x,
                       method='SLSQP', bounds=[(0, 1)] * len(p),
                       constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1, 'jac': lambda x: np.ones_like(x)},
                       options={'ftol': 1e-12, 'maxiter': 200})
        assert res.success, res.message
        return res.x / res.x.sum(), float(np.min(cos))

    base = load(args.base)
    challengers = {}
    for item in args.component:
        name, _, path = item.partition('=')
        if not path:
            raise SystemExit(f'--component needs NAME=PATH, got {item!r}')
        challengers[name] = load(path)
    if not challengers:
        raise SystemExit('Supply at least one --component NAME=PATH')
    for name, array in {BASE: base, **challengers}.items():
        if not np.isfinite(array).all() or array.min() < 0 or np.max(abs(array.sum(1) - 1)) > 1e-6:
            raise SystemExit(f'{name} is not a valid simplex')

    names = list(challengers)
    combinations = [()] + [c for size in range(1, min(args.max_combination, len(names)) + 1)
                           for c in itertools.combinations(names, size)]

    results = {}
    for combo in combinations:
        label = BASE if not combo else BASE + ' + ' + ' + '.join(combo)
        arrays_for = lambda: [base] + [challengers[n] for n in combo]
        held_free = np.empty_like(base)
        held_fixed = {w: np.empty_like(base) for w in args.weights}
        fold_weights, cosines, fold_points = [], [], []
        for fold in range(4):
            held = {e for e in episodes if folds[e] == fold}
            fit = episodes - held
            mask = np.array([e in held for e in ep])
            if not combo:
                held_free[mask] = base[mask]
                for w in args.weights:
                    held_fixed[w][mask] = base[mask]
                fold_points.append(official(base, held)['technical_points'])
                continue
            arrays = arrays_for()
            weights, cos = simplex_fit(arrays, fit)
            fold_weights.append(dict(zip([BASE] + list(combo), weights.tolist())))
            cosines.append(cos)
            free = np.einsum('m,mnc->nc', weights, np.array(arrays))
            held_free[mask] = free[mask]
            # Fixed weight spreads equally over the challengers, base keeps the rest.
            for w in args.weights:
                mixed = (1 - w) * base + w * np.mean([challengers[n] for n in combo], axis=0)
                held_fixed[w][mask] = mixed[mask]
            fold_points.append(official(free, held)['technical_points'])

        entry = {'components': list(combo), 'free': official(held_free),
                 'fixed': {f'{w:g}': official(held_fixed[w]) for w in args.weights},
                 'fold_points_free': fold_points,
                 'fold_weights': fold_weights,
                 'min_residual_cosine': cosines,
                 'complementary_all_folds': bool(cosines and all(c < 0.98 for c in cosines))}
        if combo:
            entry['paired_free_vs_base'] = paired_bootstrap(mapping(held_free), mapping(base), scoring, episodes)
            for w in args.weights:
                entry[f'paired_fixed_{w:g}_vs_base'] = paired_bootstrap(
                    mapping(held_fixed[w]), mapping(base), scoring, episodes)
            entry['challenger_weight_mean'] = float(np.mean(
                [sum(fw[n] for n in combo) for fw in fold_weights]))
        results[label] = entry

    base_points = results[BASE]['free']['technical_points']
    for label, entry in results.items():
        entry['free']['gain_over_base'] = entry['free']['technical_points'] - base_points
        folds_won = sum(p > q for p, q in zip(entry['fold_points_free'], results[BASE]['fold_points_free']))
        entry['folds_improved_free'] = folds_won

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'base_points': base_points, 'results': results,
                                    'exposure': 'Exploratory: all 96 public validation episodes previously examined.',
                                    'seconds': time.perf_counter() - start}, indent=2, sort_keys=True) + '\n')

    conds = sorted(results[BASE]['free']['by_condition'])
    print(f"\n{'combination':<52}{'free':>9}{'gain':>8}{'folds':>7}{'wt':>7}{'CI95 vs base':>26}  " +
          '  '.join(f'{c[:9]:>9}' for c in conds))
    ordered = sorted(results.items(), key=lambda kv: -kv[1]['free']['technical_points'])
    for label, entry in ordered:
        free = entry['free']
        ratios = '  '.join(f"{free['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds)
        if entry['components']:
            lo, hi = entry['paired_free_vs_base']['delta_loss_ratio_ci95']
            ci = f'[{lo:+.4f},{hi:+.4f}]' + ('*' if hi < 0 else ' ')
            wt = f"{entry['challenger_weight_mean']:.3f}"
        else:
            ci, wt = '(bar)', '-'
        print(f"{label:<52}{free['technical_points']:>9.4f}{free['gain_over_base']:>+8.4f}"
              f"{entry['folds_improved_free']:>6}/4{wt:>7}{ci:>26}  {ratios}")
    print('\n* = paired 95% interval entirely below zero (significant at whole-episode level)')
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
