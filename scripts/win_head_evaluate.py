"""Matched-control evaluation of the hierarchical logistic win head.

Controls, all sharing feature availability, recency schedule, training data, tuning
budget and evaluation procedure:
  0. existing three-component forecaster (the reference predictor)
  1. population win model, same temporal basis, no machine-specific deviations
  2. local adaptation of the intercept and the eight main lag effects
  3. local adaptation including the 28 pair interactions

Each candidate win estimate q_new is composed with the REFERENCE predictor's
positive-payout proportions r, so the five-vector is
    [1-q, q*r_1, q*r_2, q*r_5, q*r_20]
and the complete vector is scored by the organizer's scorer.

Outer-fold discipline. Population effects and all shrinkage/recency hyperparameters
come from public/train only, so they are independent of every validation fold. The
only validation-fitted quantity is the single bounded blend coefficient alpha, and for
held fold k it is fitted on the fitting episodes of fold k alone, against the exact
official full-vector loss with reference normalization computed from those fitting
episodes only. r comes from the reference predictor's held-fold prediction, whose own
simplex weights were fitted outside fold k. Held-fold predictions are concatenated and
the official scorer runs once.

Exposure: all 96 public-validation episodes have been inspected repeatedly across this
project. New splits of them do not create untouched data. Every number here is
exploratory; none is independently confirmed.
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

from slot_model import win_head as W
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission

CONTROLS = [('population', None), ('local_main', 'main'), ('local_interactions', 'full')]
SUPPORT_BUCKETS = ((0, 0), (1, 49), (50, 119), (120, 199), (200, 10 ** 9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--population', type=Path, default=ROOT / 'artifacts/win_head/population.json')
    parser.add_argument('--averaged', type=Path, default=ROOT / 'artifacts/win_head/population_averaged.json',
                        help='Per-machine half-life averaging settings; adds a fourth control when present.')
    parser.add_argument('--reference', type=Path, default=ROOT / 'artifacts/audit/oof_candidate_raw.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/win_head')
    args = parser.parse_args()
    start = time.perf_counter()
    controls = list(CONTROLS)

    payload = json.loads(args.population.read_text())
    chosen = payload['selected']
    half_life = chosen['half_life']
    lam = W.penalty(chosen['lam_intercept'], chosen['lam_main'], chosen['lam_inter'])
    beta = {t: np.asarray(v) for t, v in payload['beta'].items()}
    print(f"frozen from train: H={half_life} lam_main={chosen['lam_main']} "
          f"lam_inter={chosen['lam_inter']} (train surrogate Brier "
          f"{chosen['train_oof_episode_win_brier']:.6f})", flush=True)

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    base = read_submission(args.reference)
    base = np.array([base[q] for q in ids])
    q_exist = 1.0 - base[:, 0]
    proportions = base[:, 1:] / np.maximum(q_exist, 1e-12)[:, None]

    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    by_id = {q.query_id: q for q in vq}

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def official(a, es=episodes):
        return score(mapping(a), scoring, es)

    def row_weights(es):
        w = np.zeros(len(ids))
        for condition in sorted(set(groups.values())):
            ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == condition}
            b = np.mean([np.mean(np.sum((ref_arr[ep == e] - truth[ep == e]) ** 2, axis=1)) for e in sorted(ces)])
            for e in ces:
                w[ep == e] = 1 / (4 * len(ces) * sum(ep == e) * max(b, 1e-9))
        return w

    # ---- win estimates for every control ----------------------------------
    estimates, details, failures = {}, {}, {}
    for name, scope in CONTROLS:
        began = time.perf_counter()
        q = np.empty(len(ids))
        info = []
        fallbacks = 0
        for i, qid in enumerate(ids):
            query = by_id[qid]
            ticks = index[(query.episode_id, query.machine_id)]
            phi, y, _, used = W.design(ticks, query.cutoff, half_life=None)
            star = W.query_row(ticks, query.cutoff)
            if star is None:
                # Population fallback is impossible without the query basis, so defer to
                # the reference predictor rather than invent a lag value.
                q[i] = q_exist[i]
                info.append({'transitions': int(len(y)), 'adapted': False, 'fallback': True})
                fallbacks += 1
                continue
            weights = (np.ones(len(y)) if not half_life
                       else 2.0 ** (-(query.cutoff - 1 - np.asarray(used, dtype=np.float64)) / half_life))
            value, detail = W.predict_q(beta.get(query.machine_type, np.zeros(W.DIM)),
                                        phi, y, weights, star, lam, adapt=scope)
            if not np.isfinite(value):
                failures.setdefault(name, []).append(qid)
                value = q_exist[i]
            q[i] = value
            detail['fallback'] = False
            info.append(detail)
        estimates[name] = q
        details[name] = info
        print(f'  {name:<26} {time.perf_counter() - began:6.1f}s  '
              f'{fallbacks} query-basis fallbacks  '
              f'{len(failures.get(name, []))} non-finite', flush=True)

    # ---- fourth control: per-machine half-life averaging -------------------
    if args.averaged and args.averaged.exists():
        extra = json.loads(args.averaged.read_text())
        sel = extra['selected']
        lam_avg = W.penalty(sel['lam_intercept'], sel['lam_main'], sel['lam_inter'])
        beta_by_h = {}
        for key, table in extra['beta_by_half_life'].items():
            h = float(key)
            beta_by_h[None if h == 0.0 else h] = {t: np.asarray(v) for t, v in table.items()}
        name = 'local_interactions_averaged'
        began = time.perf_counter()
        q = np.empty(len(ids))
        info = []
        for i, qid in enumerate(ids):
            query = by_id[qid]
            ticks = index[(query.episode_id, query.machine_id)]
            phi, y, _, used = W.design(ticks, query.cutoff, half_life=None)
            star = W.query_row(ticks, query.cutoff)
            if star is None or len(y) == 0:
                q[i] = q_exist[i]
                info.append({'transitions': int(len(y)), 'adapted': False, 'fallback': True})
                continue
            by_type = {h: tbl.get(query.machine_type, np.zeros(W.DIM)) for h, tbl in beta_by_h.items()}
            value, detail = W.predict_q_averaged(by_type, phi, y, used, query.cutoff, star,
                                                 lam_avg, adapt='full')
            if not np.isfinite(value):
                failures.setdefault(name, []).append(qid)
                value = q_exist[i]
            q[i] = value
            info.append(detail)
        estimates[name] = q
        details[name] = info
        controls.append((name, 'full'))
        print(f'  {name:<26} {time.perf_counter() - began:6.1f}s  '
              f"lam_inter={sel['lam_inter']} over half-lives {extra['half_lives']}  "
              f"train surrogate {sel['train_oof_episode_win_brier']:.6f}", flush=True)

    # ---- alpha blend per outer fold ---------------------------------------
    def alpha_star(a, b, es):
        """Closed-form minimiser of the exact official loss over alpha in [0,1]."""
        w = row_weights(es)
        A = 1.0 + np.sum(proportions ** 2, axis=1)
        B = 1.0 - truth[:, 0] + np.sum(proportions * truth[:, 1:], axis=1)
        d = b - a
        num = float(np.sum(w * d * (a * A - B)))
        den = float(np.sum(w * d * d * A))
        if den <= 0:
            return 0.0
        return float(np.clip(-num / den, 0.0, 1.0))

    def compose_all(q):
        return np.array([W.compose(q[i], proportions[i]) for i in range(len(ids))])

    results = {'reference_three_component': official(base)}
    per_fold_alpha = {}
    blended = {}
    for name, _ in controls:
        q_new = estimates[name]
        held = np.empty(len(ids))
        alphas = []
        for fold in range(4):
            mask = np.array([folds[e] == fold for e in ep])
            fit = {e for e in episodes if folds[e] != fold}
            a = alpha_star(q_exist, q_new, fit)
            alphas.append(a)
            held[mask] = ((1 - a) * q_exist + a * q_new)[mask]
        per_fold_alpha[name] = alphas
        blended[name] = held
        results[name] = {
            'alpha0_reference_only': official(compose_all(q_exist))['technical_points'],
            'alpha1_new_only': official(compose_all(q_new)),
            'alpha_fitted_per_fold': official(compose_all(held)),
            'alpha_per_fold': alphas,
        }
        print(f"  {name:<20} alpha={['%.4f' % a for a in alphas]}  "
              f"a=1 -> {results[name]['alpha1_new_only']['technical_points']:.6f}  "
              f"fitted -> {results[name]['alpha_fitted_per_fold']['technical_points']:.6f}", flush=True)

    # ---- the headline candidate -------------------------------------------
    best_name = max(controls, key=lambda c: results[c[0]]['alpha_fitted_per_fold']['technical_points'])[0]
    best_vec = compose_all(blended[best_name])
    best = official(best_vec)
    write_submission(args.out / 'predictions' / 'oof_win_head.csv', mapping(best_vec))
    for name, _ in controls:
        write_submission(args.out / 'predictions' / f'oof_{name}_alpha1.csv', mapping(compose_all(estimates[name])))

    # ---- required diagnostics ---------------------------------------------
    q_true = 1.0 - truth[:, 0]
    payout_contrib = {}
    for name, vec in [('reference', base), (best_name, best_vec)]:
        err = (vec - truth) ** 2
        payout_contrib[name] = {f'p_{k}': {'mse': float(err[:, i].mean()),
                                           'share': float(err[:, i].sum() / err.sum()),
                                           'signed_bias': float((vec[:, i] - truth[:, i]).mean())}
                                for i, k in enumerate((0, 1, 2, 5, 20))}
    support = {}
    counts = np.array([d['transitions'] for d in details[best_name]])
    for lo, hi in SUPPORT_BUCKETS:
        mask = (counts >= lo) & (counts <= hi)
        if not mask.any():
            continue
        label = f'{lo}-{hi if hi < 10 ** 9 else "inf"}'
        support[label] = {
            'queries': int(mask.sum()),
            'mean_full_vector_loss_candidate': float(np.mean(np.sum((best_vec[mask] - truth[mask]) ** 2, axis=1))),
            'mean_full_vector_loss_reference': float(np.mean(np.sum((base[mask] - truth[mask]) ** 2, axis=1))),
            'q_rmse_candidate': float(np.sqrt(np.mean((blended[best_name][mask] - q_true[mask]) ** 2))),
            'q_rmse_reference': float(np.sqrt(np.mean((q_exist[mask] - q_true[mask]) ** 2))),
        }
    stability = {}
    for name, _ in controls:
        adapted = [d for d in details[name] if d.get('adapted')]
        if not adapted:
            stability[name] = {'adapted': 0}
            continue
        def med(key):
            vals = [d[key] for d in adapted if key in d]
            return float(np.median(vals)) if vals else None
        stability[name] = {
            'adapted': len(adapted),
            'logit_var_median': med('logit_var'),
            'logit_var_p90': (float(np.percentile([d['logit_var'] for d in adapted], 90))
                              if any('logit_var' in d for d in adapted) else None),
            'u_norm_median': med('u_norm'),
            'u_intercept_abs_median': (float(np.median([abs(d['u_intercept']) for d in adapted
                                                        if 'u_intercept' in d]))
                                       if any('u_intercept' in d for d in adapted) else None),
            'u_main_abs_mean_median': med('u_main_abs_mean'),
            'u_inter_abs_mean_median': med('u_inter_abs_mean'),
            # Averaging control only: which half-life each machine's own history favoured.
            'selected_half_life_histogram': ({str(int(h)): int(sum(1 for d in adapted
                                                                   if d.get('selected_half_life') == h))
                                              for h in sorted({d['selected_half_life'] for d in adapted
                                                               if 'selected_half_life' in d})}
                                             if any('selected_half_life' in d for d in adapted) else None),
            'mix_entropy_median': med('mix_entropy'),
        }
    paired = paired_bootstrap(mapping(best_vec), mapping(base), scoring, episodes)

    report = {
        'exposure': 'All 96 public-validation episodes previously inspected. Exploratory; '
                    'no untouched organizer evaluation set exists, so nothing here is independently confirmed.',
        'frozen_from_train': chosen,
        'controls': results,
        'headline': {'control': best_name, 'points': best['technical_points'],
                     'R': best['mean_loss_ratio'],
                     'by_condition': {c: v['loss_ratio'] for c, v in best['by_condition'].items()},
                     'target_50_met': bool(best['technical_points'] >= 50.0),
                     'gap_to_50': float(50.0 - best['technical_points'])},
        'alpha_per_fold': per_fold_alpha,
        'payout_coordinate_contributions': payout_contrib,
        'by_adaptation_support': support,
        'local_estimate_stability': stability,
        'q_rmse': {'reference': float(np.sqrt(np.mean((q_exist - q_true) ** 2))),
                   **{n: float(np.sqrt(np.mean((estimates[n] - q_true) ** 2))) for n, _ in controls},
                   f'{best_name}_blended': float(np.sqrt(np.mean((blended[best_name] - q_true) ** 2)))},
        'paired_vs_reference': paired,
        'non_finite_failures': {k: v[:10] for k, v in failures.items()},
        'seconds': time.perf_counter() - start,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(best['by_condition'])
    print(f"\n{'control':<26}{'points':>10}{'R':>9}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    rr = results['reference_three_component']
    print(f"{'existing 3-component':<26}{rr['technical_points']:>10.6f}{rr['mean_loss_ratio']:>9.4f}  "
          + '  '.join(f"{rr['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    for name, _ in controls:
        r1 = results[name]['alpha_fitted_per_fold']
        print(f"{name + ' (fitted a)':<26}{r1['technical_points']:>10.6f}{r1['mean_loss_ratio']:>9.4f}  "
              + '  '.join(f"{r1['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\nheadline: {best_name} at {best['technical_points']:.6f}/60, R={best['mean_loss_ratio']:.6f}")
    print(f"paired 95% vs reference: [{lo:+.5f}, {hi:+.5f}]  "
          f"{'significant' if hi < 0 else 'overlaps zero'}")
    print(f"target 50/60 {'MET' if best['technical_points'] >= 50 else 'NOT met'}; "
          f"gap {50 - best['technical_points']:+.4f} points")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
