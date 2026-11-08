"""Matched comparison: dense win head vs compact pair models, hard-selected and averaged.

Controls, all sharing the same transition rows, weight normalization, per-machine
half-lives from the verified existing procedure, population offset and causal feature
construction, so the comparison isolates local complexity and pair uncertainty:

  dense_37          the deployed 37-coefficient local model              (baseline)
  compact_hard      the single highest-evidence pair model per machine
  compact_average   predictive averaging over all 29 candidates
  alpha_blend       one bounded coefficient between dense and compact_average,
                    fitted per outer fold on the exact official loss

The pair-identity prior and the coefficient prior scale are learned on public/train
only, with a predeclared grid. No validation target enters any fitted quantity except
the single alpha, which is fitted on the fitting episodes of each outer fold.
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

from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from scripts.win_head_train import build, pooled

HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)
LAM_GRID = (0.03125, 0.0625, 0.125, 0.25)   # extended downward: 0.25 sat at the original grid edge
PRIOR_SMOOTH = 1.0                   # add-one smoothing on pair identity
PAYOUTS = (0, 1, 2, 5, 20)


def chosen_half_life(beta_by_h, entry, lam_dense):
    """Reuse the verified existing per-machine half-life selection, unchanged."""
    _, detail = W.predict_q_averaged(
        {h: beta_by_h[h].get(entry['type'], np.zeros(len(lam_dense))) for h in HALF_LIVES},
        entry['phi'], entry['y'], entry['used'], entry['cutoff'], entry['star'],
        lam_dense, adapt='full')
    sel = detail['selected_half_life']
    return (None if sel == 0.0 else sel), detail


def weights_for(entry, half_life):
    if not half_life:
        return np.ones(len(entry['y']))
    return 2.0 ** (-(entry['cutoff'] - 1 - entry['used']) / half_life)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/pair_dict')
    parser.add_argument('--reference', type=Path, default=ROOT / 'artifacts/audit/oof_candidate_raw.csv')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    cfg = json.loads((ROOT / 'artifacts/win_head/population_averaged.json').read_text())
    sel = cfg['selected']
    lam_dense = W.penalty(sel['lam_intercept'], sel['lam_main'], sel['lam_inter'])
    beta_by_h = {}
    for key, table in cfg['beta_by_half_life'].items():
        h = float(key)
        beta_by_h[None if h == 0.0 else h] = {t: np.asarray(v) for t, v in table.items()}

    # ---- learn the pair-identity prior and lambda on TRAIN only -----------
    tq, th = load_inputs(args.data / 'train')
    tindex = causal_index(tq, th)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    folds_t = json.loads((ROOT / 'artifacts/phase2/folds.json'))['train'] \
        if False else json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['train']
    tentries = build(tq, tindex)
    wins = {k: 1.0 if PAYOUTS.index(labels[k]) > 0 else 0.0 for k in tentries}
    epof = {k: tentries[k]['episode'] for k in tentries}
    byfold = {f: [k for k in tentries if folds_t[epof[k]] == f] for f in range(4)}

    def ebrier(pred):
        per = {}
        for k, v in pred.items():
            per.setdefault(epof[k], []).append((v - wins[k]) ** 2)
        return float(np.mean([np.mean(v) for v in per.values()]))

    pop_cache = {}

    def pop(fold, h):
        if (fold, h) not in pop_cache:
            keys = [k for f in range(4) if f != fold for k in byfold[f]]
            pop_cache[(fold, h)] = pooled(tentries, keys, h, lam_dense)[0]
        return pop_cache[(fold, h)]

    records = []
    best = None
    for lam_scale in LAM_GRID:
        oof_avg, oof_hard, counts = {}, {}, np.zeros(len(PD.CANDIDATES))
        for fold in range(4):
            betas = {h: pop(fold, h) for h in HALF_LIVES}
            for k in byfold[fold]:
                e = tentries[k]
                if e['star'] is None or len(e['y']) == 0:
                    continue
                h, _ = chosen_half_life(betas, e, lam_dense)
                beta = betas[h].get(e['type'], np.zeros(len(lam_dense)))
                w = weights_for(e, h)
                population = (e['phi'] @ beta, float(beta @ e['star']))
                om, qs, _ = PD.machine_posterior(e['phi'], e['y'], w, e['star'], population,
                                                 lam_scale * 0.25, lam_scale)
                oof_avg[k] = PD.average(om, qs)
                oof_hard[k] = float(qs[int(np.argmax(om))])
                counts[int(np.argmax(om))] += 1
        rec = {'lam_scale': lam_scale, 'train_oof_brier_average': ebrier(oof_avg),
               'train_oof_brier_hard': ebrier(oof_hard),
               'argmax_counts': counts.tolist()}
        records.append(rec)
        print(f"  lam_scale={lam_scale:<5} train OOF Brier  averaged={rec['train_oof_brier_average']:.6f}  "
              f"hard={rec['train_oof_brier_hard']:.6f}  ({time.perf_counter()-start:.0f}s)", flush=True)
        key = min(rec['train_oof_brier_average'], rec['train_oof_brier_hard'])
        if best is None or key < best['_key']:
            rec['_key'] = key
            best = rec
            prior_counts = counts.copy()
    lam_scale = best['lam_scale']
    log_prior = np.log((prior_counts + PRIOR_SMOOTH) / (prior_counts.sum() + PRIOR_SMOOTH * len(prior_counts)))
    print(f"\nselected lam_scale={lam_scale} on the train surrogate "
          f"(dense baseline was 0.147693); pair-identity prior learned from train argmax counts "
          f"with add-{PRIOR_SMOOTH} smoothing", flush=True)

    # ---- validation ------------------------------------------------------
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds_v = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    base = read_submission(args.reference)
    base = np.array([base[q] for q in ids])
    q_exist = 1 - base[:, 0]
    props = base[:, 1:] / np.maximum(q_exist, 1e-12)[:, None]

    dense_vec = read_submission(ROOT / 'artifacts/win_head/predictions/oof_win_head.csv')
    dense_vec = np.array([dense_vec[q] for q in ids])
    q_dense = 1 - dense_vec[:, 0]

    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    ventries = build(vq, vindex)
    full_beta = beta_by_h
    q_avg = np.empty(len(ids)); q_hard = np.empty(len(ids))
    conc, picks, nfail = [], [], 0
    for i, qid in enumerate(ids):
        e = ventries[qid]
        if e['star'] is None or len(e['y']) == 0:
            q_avg[i] = q_hard[i] = q_dense[i]; nfail += 1; continue
        h, _ = chosen_half_life(full_beta, e, lam_dense)
        beta = full_beta[h].get(e['type'], np.zeros(len(lam_dense)))
        w = weights_for(e, h)
        population = (e['phi'] @ beta, float(beta @ e['star']))
        om, qs, _ = PD.machine_posterior(e['phi'], e['y'], w, e['star'], population,
                                         lam_scale * 0.25, lam_scale, log_prior=log_prior)
        q_avg[i] = PD.average(om, qs)
        q_hard[i] = float(qs[int(np.argmax(om))])
        conc.append(float(om.max())); picks.append(int(np.argmax(om)))

    def compose(q):
        return np.array([W.compose(q[i], props[i]) for i in range(len(ids))])

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

    A = 1 + np.sum(props ** 2, axis=1)
    B = 1 - truth[:, 0] + np.sum(props * truth[:, 1:], axis=1)

    def alpha_star(a, b, es):
        w = row_weights(es); d = b - a
        den = float(np.sum(w * d * d * A))
        return 0.0 if den <= 0 else float(np.clip(-np.sum(w * d * (a * A - B)) / den, 0, 1))

    held = np.empty(len(ids)); alphas = []
    for fold in range(4):
        m = np.array([folds_v[x] == fold for x in ep])
        fit = {x for x in episodes if folds_v[x] != fold}
        a = alpha_star(q_dense, q_avg, fit); alphas.append(a)
        held[m] = ((1 - a) * q_dense + a * q_avg)[m]

    held_hard = np.empty(len(ids)); alphas_hard = []
    for fold in range(4):
        m = np.array([folds_v[x] == fold for x in ep])
        fit = {x for x in episodes if folds_v[x] != fold}
        a = alpha_star(q_dense, q_hard, fit); alphas_hard.append(a)
        held_hard[m] = ((1 - a) * q_dense + a * q_hard)[m]
    held_mix = np.empty(len(ids)); alphas_mix = []
    for fold in range(4):
        m = np.array([folds_v[x] == fold for x in ep])
        fit = {x for x in episodes if folds_v[x] != fold}
        a = alpha_star(q_hard, q_avg, fit); alphas_mix.append(a)
        held_mix[m] = ((1 - a) * q_hard + a * q_avg)[m]
    results = {
        'dense_37': official(compose(q_dense)),
        'compact_hard': official(compose(q_hard)),
        'compact_average': official(compose(q_avg)),
        'alpha_blend_dense_x_average': official(compose(held)),
        'alpha_blend_dense_x_hard': official(compose(held_hard)),
        'alpha_blend_hard_x_average': official(compose(held_mix)),
    }
    # Re-measure the two r sources under the NEW win model; the previous 0.1156-point
    # gap was measured with the dense q and must not be assumed to carry over.
    best_q = max([('compact_hard', q_hard), ('compact_average', q_avg),
                  ('alpha_blend_hard_x_average', held_mix)],
                 key=lambda kv: official(compose(kv[1]))['technical_points'])
    r_variants = {}
    th_pred = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    th_pred = np.array([th_pred[q] for q in ids])
    q_th = 1 - th_pred[:, 0]
    props_th = th_pred[:, 1:] / np.maximum(q_th, 1e-12)[:, None]
    for label, pr in (('full_stack_three_component', props), ('two_head_only', props_th)):
        vec = np.array([W.compose(best_q[1][i], pr[i]) for i in range(len(ids))])
        r_variants[label] = official(vec)['technical_points']
        write_submission(args.out / 'predictions' / f'oof_best_r_{label}.csv', mapping(vec))
    alphas = {'dense_x_average': alphas, 'dense_x_hard': alphas_hard, 'hard_x_average': alphas_mix}
    q_true = 1 - truth[:, 0]
    report = {
        'exposure': 'All 96 public-validation episodes previously inspected. Exploratory; nested '
                    'evaluation does not undo prior inspection and no untouched set exists.',
        'weighting_method': 'Laplace marginal likelihood under a proper Gaussian prior, times a '
                            'smoothed pair-identity prior learned on train. The decay-weighted '
                            'likelihood makes this a generalized/approximate Bayesian model weight, '
                            'not an exact Bayes factor.',
        'lam_scale': lam_scale, 'train_records': records,
        'controls': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                         'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                     for k, v in results.items()},
        'alpha_per_fold': alphas,
        'r_source_remeasured_under_new_q': r_variants,
        'best_q_control': best_q[0],
        'q_rmse': {'dense': float(np.sqrt(np.mean((q_dense - q_true) ** 2))),
                   'compact_hard': float(np.sqrt(np.mean((q_hard - q_true) ** 2))),
                   'compact_average': float(np.sqrt(np.mean((q_avg - q_true) ** 2)))},
        'pair_weight_concentration': {'median_max_weight': float(np.median(conc)),
                                      'p10': float(np.percentile(conc, 10)),
                                      'p90': float(np.percentile(conc, 90))},
        'argmax_candidate_histogram': {PD.candidate_names()[i]: int(np.sum(np.asarray(picks) == i))
                                       for i in sorted(set(picks))},
        'fallbacks': nfail,
        'paired_compact_average_vs_dense': paired_bootstrap(mapping(compose(q_avg)),
                                                            mapping(compose(q_dense)), scoring, episodes),
        'seconds': time.perf_counter() - start,
    }
    for name, q in (('compact_average', q_avg), ('compact_hard', q_hard), ('alpha_blend', held)):
        write_submission(args.out / 'predictions' / f'oof_{name}.csv', mapping(compose(q)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(results['dense_37']['by_condition'])
    print(f"\n{'control':<30}{'points':>11}{'R':>9}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k, v in results.items():
        print(f"{k:<30}{v['technical_points']:>11.6f}{v['mean_loss_ratio']:>9.4f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = report['paired_compact_average_vs_dense']['delta_loss_ratio_ci95']
    print(f"\nalphas {json.dumps({k: [round(a,4) for a in v] for k,v in alphas.items()})}")
    print(f"r source under the new q: " + '  '.join(f'{k}={v:.6f}' for k,v in r_variants.items())
          + f"   (best q control: {best_q[0]})")
    print(f"paired compact_average vs dense: [{lo:+.5f}, {hi:+.5f}] "
          f"{'favours compact' if hi < 0 else 'overlaps zero' if lo < 0 else 'favours dense'}")
    print(f"pair weight concentration median max omega = {report['pair_weight_concentration']['median_max_weight']:.4f}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
