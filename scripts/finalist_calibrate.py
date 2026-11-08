"""Calibration ladder, signal correction, and conditional-r refit for the frozen compact q.

PREREGISTRATION (recorded before any public-validation number in this script was read).

Hypotheses
  H1 a small monotone calibration of q improves the official five-vector score
  H2 a signal-dependent correction adds information beyond probability-only calibration
  H3 refitting the conditional positive-payout weights r AFTER q is frozen improves the
     score, because the deployed r was extracted from a full-vector blend selected under
     the older DENSE win masses

Candidates and ranges
  calibration   identity (a,b,c)=(1,1,0); affine-logit (equal-slope subfamily, a=b);
                beta calibration g(q)=sigmoid(a*log q - b*log(1-q) + c), a,b >= 0
  signal        logit(q') = a*logit(q) + b + gamma*signal, gamma unconstrained
  r formulation two_head only; existing full-vector-derived r; uniform mix of the three
                primitive conditional r; official-weighted simplex over those three

Primary metric    official technical points / R on concatenated held-fold predictions
Selection rule    every coefficient is fitted on the FITTING episodes of each outer fold
                  only, by minimizing the exact official-weighted five-vector squared
                  error with reference normalization taken from those fitting episodes
Matched controls  identity calibration, and each r formulation above
Follow-up rule    a mechanism is carried into the finalist only if it improves the
                  concatenated held-fold points AND its paired whole-episode interval
                  against the frozen baseline lies below zero

Exposure: these 96 public-validation episodes have been inspected repeatedly. This is
another adaptive round. Paired intervals describe variation in fixed forecasts; they do
not correct for repeated research selection (P9), and no split of these episodes creates
untouched data.
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

from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--q-source', type=Path,
                        default=ROOT / 'artifacts/pair_dict/predictions/oof_best_r_two_head_only.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/finalist')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    vq, _ = load_inputs(args.data / 'validation')
    signal = np.array([{q.query_id: float(q.signal) for q in vq}[i] for i in ids])

    def load(p):
        d = read_submission(p)
        return np.array([d[q] for q in ids])

    # Frozen compact q: the win mass of the shipped candidate.
    q_base = 1.0 - load(args.q_source)[:, 0]

    # Primitive conditional r candidates.
    prim = {
        'tabpfn': np.mean([load(ROOT / f'artifacts/tabpfn/predictions/tabpfn_hmm_{s}.csv')
                           for s in (20261108, 20261109)], axis=0),
        'mtd_lag7': load(ROOT / 'artifacts/mtd_pairs_01/validation_lag7.csv'),
        'two_head': load(ROOT / 'artifacts/two_head_01/validation.csv'),
    }
    R = {}
    for name, block in prim.items():
        qq = 1.0 - block[:, 0]
        R[name] = block[:, 1:] / np.maximum(qq, 1e-12)[:, None]
        R[name] = R[name] / R[name].sum(axis=1, keepdims=True)
    full_vector = load(ROOT / 'artifacts/audit/oof_candidate_raw.csv')
    qf = 1.0 - full_vector[:, 0]
    R['full_vector_derived'] = full_vector[:, 1:] / np.maximum(qf, 1e-12)[:, None]
    R['full_vector_derived'] /= R['full_vector_derived'].sum(axis=1, keepdims=True)
    R['uniform_mix'] = np.mean([R['tabpfn'], R['mtd_lag7'], R['two_head']], axis=0)
    R['uniform_mix'] /= R['uniform_mix'].sum(axis=1, keepdims=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def official(a, es=episodes):
        return score(mapping(a), scoring, es)

    def row_weights(es):
        w = np.zeros(len(ids))
        for c in sorted(set(groups.values())):
            ces = {x for i, x in enumerate(ep) if x in es and groups[ids[i]] == c}
            b = np.mean([np.mean(np.sum((ref_arr[ep == x] - truth[ep == x]) ** 2, axis=1)) for x in ces])
            for x in ces:
                w[ep == x] = 1 / (4 * len(ces) * sum(ep == x) * max(b, 1e-9))
        return w

    def compose(q, r):
        out = np.column_stack((1.0 - q, q[:, None] * r))
        return out / out.sum(axis=1, keepdims=True)

    def wloss(q, r, w):
        return float(np.sum(w * np.sum((compose(q, r) - truth) ** 2, axis=1)))

    # ---- calibration families ------------------------------------------------
    def apply_cal(kind, theta, q):
        if kind == 'identity':
            return q
        lg = logit(q)
        if kind == 'affine':
            a, b = theta
            return 1.0 / (1.0 + np.exp(-(a * lg + b)))
        if kind == 'beta':
            a, b, c = theta
            z = (abs(a) * np.log(np.clip(q, EPS, 1 - EPS))
                 - abs(b) * np.log(np.clip(1 - q, EPS, 1 - EPS)) + c)
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        if kind == 'signal':
            a, b, g = theta
            return 1.0 / (1.0 + np.exp(-(a * lg + b + g * signal)))
        raise ValueError(kind)

    START = {'identity': None, 'affine': np.array([1.0, 0.0]),
             'beta': np.array([1.0, 1.0, 0.0]), 'signal': np.array([1.0, 0.0, 0.0])}

    def fit_cal(kind, r, es):
        if kind == 'identity':
            return None
        w = row_weights(es)
        f = lambda th: wloss(apply_cal(kind, th, q_base), r, w)
        best, best_v = None, np.inf
        for scale in (1.0, 0.5):
            x0 = START[kind] * np.array([scale] + [1.0] * (len(START[kind]) - 1))
            res = minimize(f, x0, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 4000})
            if res.fun < best_v:
                best, best_v = res.x, res.fun
        return best

    def fit_r_simplex(q, es):
        names = ['tabpfn', 'mtd_lag7', 'two_head']
        w = row_weights(es)
        def obj(x):
            x = np.clip(x, 0, None)
            x = x / max(x.sum(), 1e-12)
            r = np.einsum('m,mnc->nc', x, np.array([R[n] for n in names]))
            r = r / r.sum(axis=1, keepdims=True)
            return wloss(q, r, w)
        res = minimize(obj, np.ones(3) / 3, method='SLSQP', bounds=[(0, 1)] * 3,
                       constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1},
                       options={'ftol': 1e-12, 'maxiter': 300})
        x = np.clip(res.x, 0, None); x = x / x.sum()
        return x, names

    # ---- outer-fold evaluation ----------------------------------------------
    results, detail = {}, {}
    for kind in ('identity', 'affine', 'beta', 'signal'):
        for rname in ('two_head', 'full_vector_derived', 'uniform_mix', 'fitted_simplex'):
            held = np.empty_like(truth)
            thetas, rweights = [], []
            for fold in range(4):
                m = np.array([folds[x] == fold for x in ep])
                fit = {x for x in episodes if folds[x] != fold}
                r_fit = R['two_head'] if rname == 'fitted_simplex' else R[rname]
                th = fit_cal(kind, r_fit, fit)
                thetas.append(None if th is None else [float(v) for v in th])
                q_cal = apply_cal(kind, th, q_base)
                if rname == 'fitted_simplex':
                    x, names = fit_r_simplex(q_cal, fit)
                    rweights.append(dict(zip(names, x.tolist())))
                    r_use = np.einsum('m,mnc->nc', x, np.array([R[n] for n in names]))
                    r_use = r_use / r_use.sum(axis=1, keepdims=True)
                else:
                    r_use = r_fit
                held[m] = compose(q_cal, r_use)[m]
            res = official(held)
            key = f'{kind}+{rname}'
            results[key] = res
            detail[key] = {'theta_per_fold': thetas, 'r_weights_per_fold': rweights}
            write_submission(args.out / 'predictions' / f'oof_{kind}_{rname}.csv', mapping(held))

    baseline = official(compose(q_base, R['two_head']))
    best_key = max(results, key=lambda k: results[k]['technical_points'])
    best = results[best_key]
    best_vec = load(args.out / 'predictions' / f"oof_{best_key.replace('+', '_')}.csv")
    paired = paired_bootstrap(mapping(best_vec), mapping(compose(q_base, R['two_head'])),
                              scoring, episodes)

    report = {
        'preregistration': __doc__,
        'exposure': 'Another adaptive round on repeatedly inspected public validation. Not '
                    'independent, not confirmed. Paired intervals do not correct for research '
                    'selection.',
        'frozen_q_source': str(args.q_source),
        'baseline_two_head_identity': {'points': baseline['technical_points'],
                                       'R': baseline['mean_loss_ratio']},
        'grid': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                     'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                 for k, v in results.items()},
        'fitted_parameters': detail,
        'best': {'key': best_key, 'points': best['technical_points'], 'R': best['mean_loss_ratio'],
                 'gain_over_baseline': best['technical_points'] - baseline['technical_points'],
                 'target_50_met': bool(best['technical_points'] >= 50.0),
                 'gap_to_50': float(50.0 - best['technical_points'])},
        'paired_best_vs_baseline': paired,
        'seconds': time.perf_counter() - start,
    }
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(baseline['by_condition'])
    print(f"\n{'candidate':<34}{'points':>11}{'R':>9}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    print(f"{'BASELINE two_head identity':<34}{baseline['technical_points']:>11.6f}"
          f"{baseline['mean_loss_ratio']:>9.4f}  "
          + '  '.join(f"{baseline['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    for k in sorted(results, key=lambda k: -results[k]['technical_points']):
        v = results[k]
        print(f"{k:<34}{v['technical_points']:>11.6f}{v['mean_loss_ratio']:>9.4f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\nbest {best_key}: {best['technical_points']:.6f}/60 R={best['mean_loss_ratio']:.6f} "
          f"gain {best['technical_points'] - baseline['technical_points']:+.6f}")
    print(f"paired 95% vs baseline [{lo:+.5f}, {hi:+.5f}] "
          f"{'significant' if hi < 0 else 'overlaps zero'}")
    print(f"target 50 {'MET' if best['technical_points'] >= 50 else 'NOT met'}; "
          f"gap {50 - best['technical_points']:+.6f}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
