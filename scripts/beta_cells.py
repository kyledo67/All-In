"""Four-cell Beta-Bernoulli pair challenger (our conjugate adaptation, not P1 or P2).

PREREGISTRATION. Hypothesis: the remaining error concentrates on low-support machines
(short_history is the worst condition at 0.3284 and the ESS 10th percentile is 11.3
transitions), where a conjugate four-cell model with learned priors should estimate the
win probability more stably than a four-coefficient logistic fit with a Laplace
predictive. Candidates: the same 29 (null plus 28 lag pairs). Ranges: prior strength
kappa in {2, 4, 8, 16, 32, 64}. Primary metric: official points / R on concatenated
held-fold predictions. Selection: kappa, the per-type prior mean and the pair-identity
prior are learned on TRAIN only. Matched controls: the frozen logistic compact model at
48.436543 with identical eligible transitions, identical per-machine half-lives and
identical r. Both hard selection and predictive averaging are reported.

For a pair (i,j) the four contexts are 00, 01, 10, 11 on (s_i>0, s_j>0). With
decay-weighted fractional counts W_c, L_c and theta_c ~ Beta(a_c, b_c):
    predictive mean  (a_c + W_c) / (a_c + b_c + W_c + L_c)
    log evidence     sum_c [ logB(a_c + W_c, b_c + L_c) - logB(a_c, b_c) ]

DOCUMENTED DIFFERENCES FROM THE LOGISTIC PAIR MODEL, which this comparison does NOT
isolate and which must not be attributed to removing Laplace error alone:
  1. The logistic model carries a POPULATION OFFSET that varies within a pair context
     (beta_type^T phi differs per transition). A constant-probability Beta cell cannot
     represent that; it replaces a varying offset with one number per cell.
  2. Independent Beta priors per cell differ from the cell dependence that a Gaussian
     prior on (a, b, c, d) induces across the four cells.
  3. Fractional decay-weighted counts keep the Beta integral algebraically available,
     but the result is integration under a GENERALIZED posterior. These are not exact
     stationary-model Bayes factors for a drifting process.
The existing logistic four-coefficient pair model is itself the closest available
logistic control, differing exactly by 1 and 2 above; no separate offset-matched
logistic four-cell control was implemented, so differences cannot be decomposed further.

Empty, all-win and all-loss cells are handled by the prior alone, which is the point of
the conjugate form; a cell with zero weight returns its prior mean.
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
from scipy.special import betaln

from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from scripts.win_head_train import build

HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)   # extended down: 2.0 sat at the grid edge
PAYOUTS = (0, 1, 2, 5, 20)


def cells(signs, candidate):
    """Cell index per transition: 0 for the null model, else 2*(s_i>0) + (s_j>0)."""
    if candidate is None:
        return np.zeros(len(signs), dtype=np.int64), 1
    i, j = candidate
    return ((signs[:, i - 1] > 0).astype(np.int64) * 2
            + (signs[:, j - 1] > 0).astype(np.int64)), 4


def posterior(phi, y, w, star_row, mu, kappa, log_prior):
    """Return (weights, per-candidate predictive q) under the conjugate cells."""
    signs = PD.signs_from(phi)
    star_signs = star_row[1:1 + W.HORIZON][None, :]
    evid = np.empty(len(PD.CANDIDATES))
    qs = np.empty(len(PD.CANDIDATES))
    a0, b0 = kappa * mu, kappa * (1.0 - mu)
    for index, candidate in enumerate(PD.CANDIDATES):
        idx, ncell = cells(signs, candidate)
        star_idx, _ = cells(star_signs, candidate)
        wins = np.bincount(idx, weights=w * y, minlength=ncell)
        losses = np.bincount(idx, weights=w * (1.0 - y), minlength=ncell)
        evid[index] = float(np.sum(betaln(a0 + wins, b0 + losses) - betaln(a0, b0)))
        c = int(star_idx[0])
        qs[index] = (a0 + wins[c]) / (a0 + b0 + wins[c] + losses[c])
    scores = evid + log_prior
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    omega = np.exp(scores - scores.max())
    omega /= omega.sum()
    return omega, qs, evid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/beta_cells')
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

    # ---- learn kappa, the per-type prior mean, and the pair prior on TRAIN ----
    tq, th_hist = load_inputs(args.data / 'train')
    tindex = causal_index(tq, th_hist)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    folds_t = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['train']
    tentries = build(tq, tindex)
    wins_t = {k: 1.0 if PAYOUTS.index(labels[k]) > 0 else 0.0 for k in tentries}
    epof = {k: tentries[k]['episode'] for k in tentries}
    byfold = {f: [k for k in tentries if folds_t[epof[k]] == f] for f in range(4)}

    def half_life_for(e):
        _, d = W.predict_q_averaged(
            {h: beta_by_h[h].get(e['type'], np.zeros(len(lam_dense))) for h in HALF_LIVES},
            e['phi'], e['y'], e['used'], e['cutoff'], e['star'], lam_dense, adapt='full')
        p = d['selected_half_life']
        return None if p == 0.0 else p

    hl_t = {k: half_life_for(tentries[k]) for k in tentries
            if tentries[k]['star'] is not None and len(tentries[k]['y'])}
    print(f'train half-lives resolved in {time.perf_counter()-start:.0f}s', flush=True)

    def ebrier(pred):
        per = {}
        for k, v in pred.items():
            per.setdefault(epof[k], []).append((v - wins_t[k]) ** 2)
        return float(np.mean([np.mean(v) for v in per.values()]))

    uniform = np.zeros(len(PD.CANDIDATES))
    records, best = [], None
    for kappa in KAPPAS:
        oof_h, oof_a, counts = {}, {}, np.zeros(len(PD.CANDIDATES))
        for fold in range(4):
            fit_keys = [k for f in range(4) if f != fold for k in byfold[f] if k in hl_t]
            mu_by_type = {}
            for k in fit_keys:
                e = tentries[k]
                mu_by_type.setdefault(e['type'], []).append(wins_t[k])
            mu_by_type = {t: float(np.mean(v)) for t, v in mu_by_type.items()}
            for k in byfold[fold]:
                if k not in hl_t:
                    continue
                e = tentries[k]
                h = hl_t[k]
                w = (np.ones(len(e['y'])) if not h else
                     2.0 ** (-(e['cutoff'] - 1 - e['used']) / h))
                mu = mu_by_type.get(e['type'], 0.44)
                om, qs, _ = posterior(e['phi'], e['y'], w, e['star'], mu, kappa, uniform)
                oof_h[k] = float(qs[int(np.argmax(om))])
                oof_a[k] = float(np.dot(om, qs))
                counts[int(np.argmax(om))] += 1
        rec = {'kappa': kappa, 'train_oof_brier_hard': ebrier(oof_h),
               'train_oof_brier_average': ebrier(oof_a), 'argmax_counts': counts.tolist()}
        records.append(rec)
        key = min(rec['train_oof_brier_hard'], rec['train_oof_brier_average'])
        print(f"  kappa={kappa:<5} train OOF Brier hard={rec['train_oof_brier_hard']:.6f} "
              f"avg={rec['train_oof_brier_average']:.6f}", flush=True)
        if best is None or key < best['_k']:
            rec['_k'] = key; best = rec; prior_counts = counts.copy()
    kappa = best['kappa']
    use_hard = best['train_oof_brier_hard'] <= best['train_oof_brier_average']
    log_prior = np.log((prior_counts + 1.0) / (prior_counts.sum() + len(prior_counts)))
    mu_all = {}
    for k in hl_t:
        mu_all.setdefault(tentries[k]['type'], []).append(wins_t[k])
    mu_all = {t: float(np.mean(v)) for t, v in mu_all.items()}
    print(f"\nselected kappa={kappa}, rule={'hard' if use_hard else 'average'} "
          f"(logistic compact train baseline was 0.109608)", flush=True)

    # ---- validation -------------------------------------------------------
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    episodes = set(ep.tolist())
    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    ventries = build(vq, vindex)

    base = read_submission(ROOT / 'artifacts/pair_dict/predictions/oof_best_r_two_head_only.csv')
    base = np.array([base[q] for q in ids])
    q_log = 1.0 - base[:, 0]
    thp = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    thp = np.array([thp[q] for q in ids])
    r_two = thp[:, 1:] / np.maximum(1.0 - thp[:, 0], 1e-12)[:, None]
    r_two /= r_two.sum(axis=1, keepdims=True)

    q_hard = np.empty(len(ids)); q_avg = np.empty(len(ids)); empties = 0
    for i, qid in enumerate(ids):
        e = ventries[qid]
        if e['star'] is None or len(e['y']) == 0:
            q_hard[i] = q_avg[i] = q_log[i]; empties += 1; continue
        h = half_life_for(e)
        w = np.ones(len(e['y'])) if not h else 2.0 ** (-(e['cutoff'] - 1 - e['used']) / h)
        om, qs, _ = posterior(e['phi'], e['y'], w, e['star'],
                              mu_all.get(e['type'], 0.44), kappa, log_prior)
        q_hard[i] = float(qs[int(np.argmax(om))])
        q_avg[i] = float(np.dot(om, qs))

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(q):
        out = np.column_stack((1.0 - q, q[:, None] * r_two))
        return out / out.sum(axis=1, keepdims=True)

    def official(a):
        return score(mapping(a), scoring, episodes)

    out = {'logistic_compact (control)': official(compose(q_log)),
           'beta_cells_hard': official(compose(q_hard)),
           'beta_cells_average': official(compose(q_avg))}
    # one bounded blend of the conjugate and logistic win estimates
    folds_v = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    def row_weights(es):
        w = np.zeros(len(ids))
        for c in sorted(set(groups.values())):
            ces = {x for i, x in enumerate(ep) if x in es and groups[ids[i]] == c}
            b = np.mean([np.mean(np.sum((ref_arr[ep == x] - truth[ep == x]) ** 2, axis=1)) for x in ces])
            for x in ces:
                w[ep == x] = 1 / (4 * len(ces) * sum(ep == x) * max(b, 1e-9))
        return w
    A = 1 + np.sum(r_two ** 2, axis=1)
    B = 1 - truth[:, 0] + np.sum(r_two * truth[:, 1:], axis=1)
    q_best = q_hard if use_hard else q_avg
    held = np.empty(len(ids)); alphas = []
    for fold in range(4):
        m = np.array([folds_v[x] == fold for x in ep])
        fit = {x for x in episodes if folds_v[x] != fold}
        w = row_weights(fit); d = q_best - q_log
        den = float(np.sum(w * d * d * A))
        a = 0.0 if den <= 0 else float(np.clip(-np.sum(w * d * (q_log * A - B)) / den, 0, 1))
        alphas.append(a); held[m] = ((1 - a) * q_log + a * q_best)[m]
    out['alpha_blend_logistic_x_beta'] = official(compose(held))

    paired = paired_bootstrap(mapping(compose(q_best)), mapping(compose(q_log)), scoring, episodes)
    report = {'preregistration': __doc__, 'kappa': kappa,
              'rule': 'hard' if use_hard else 'average', 'train_records': records,
              'prior_mean_by_type': mu_all, 'empty_design_fallbacks': empties,
              'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                              'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                          for k, v in out.items()},
              'alpha_per_fold': alphas, 'paired_beta_vs_logistic': paired,
              'exposure': 'Repeatedly inspected public validation; exploratory.',
              'seconds': time.perf_counter() - start}
    for name, q in (('beta_hard', q_hard), ('beta_average', q_avg), ('alpha_blend', held)):
        write_submission(args.out / 'predictions' / f'oof_{name}.csv', mapping(compose(q)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(out['logistic_compact (control)']['by_condition'])
    print(f"\n{'candidate':<34}{'points':>11}{'R':>9}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k, v in out.items():
        print(f"{k:<34}{v['technical_points']:>11.6f}{v['mean_loss_ratio']:>9.4f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\nalpha per fold {[round(a,4) for a in alphas]}")
    print(f"paired beta vs logistic [{lo:+.5f}, {hi:+.5f}] "
          f"{'favours beta' if hi < 0 else 'overlaps zero' if lo < 0 else 'favours logistic'}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
