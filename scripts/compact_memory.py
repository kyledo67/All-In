"""Select the recency half-life with the COMPACT predictor, at historical origins.

PREREGISTRATION. Hypothesis: the deployed compact pair model inherits its half-life from
the old dense 37-coefficient model's held-out likelihood, which is a procedure mismatch;
selecting memory with the compact predictor itself should fit better. Candidates: the
frozen half-life set 8, 16, 32, 64, 128, 256, infinity. Primary metric: official points /
R on concatenated held-fold predictions. Selection: the half-life is chosen per machine
from that machine's own pre-cutoff history only; the pair prior and lambda come from
training folds. Matched control: the frozen compact baseline at 48.436543 (two_head r,
identity calibration). A follow-up is justified only if held-fold points improve.

Corrected procedure, per machine and per candidate half-life H:
  1. split the machine's complete-lag transitions by ACTUAL tick into a prefix and a
     held-out recent block (the same block for every H, so candidates are compared on
     identical eligible target ticks)
  2. anchor the decay at the historical origin, i.e. the first held-out tick, NOT at the
     final cutoff - this is what the frozen implementation got wrong
  3. fit all 29 compact candidates on the prefix alone and select by evidence + prior,
     so pair identity is chosen without seeing the block it is scored on
  4. predict every held-out tick and accumulate UNWEIGHTED binary log loss, which keeps
     the comparison on a common scale across half-lives regardless of their total
     observation weight
  5. choose H, then refit on the full pre-cutoff history with decay anchored at the
     cutoff and emit the prediction for the cutoff tick

Step 3 is the part the frozen selector could not do: it chose pair identity from the
complete final history and then scored earlier outcomes with it.
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
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.win_head_train import build

HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)


def decay(ticks, anchor, half_life):
    if not half_life:
        return np.ones(len(ticks))
    return 2.0 ** (-(anchor - 1 - np.asarray(ticks, dtype=np.float64)) / half_life)


def fit_select(phi, y, w, beta, star_row, lam_scale, lam_icept, log_prior):
    """Fit the 29 candidates, return (weights, per-candidate q at star_row)."""
    population = (phi @ beta, float(beta @ star_row))
    return PD.machine_posterior(phi, y, w, star_row, population, lam_icept, lam_scale,
                                log_prior=log_prior)


def candidate_block_loss(phi, y, w, beta, fit_idx, val_idx, lam_scale, lam_icept, log_prior):
    """Select on the prefix, then score the held-out block with unweighted log loss."""
    sub_phi, sub_y, sub_w = phi[fit_idx], y[fit_idx], w[fit_idx]
    signs = PD.signs_from(sub_phi)
    offset = sub_phi @ beta
    best_score, best_index = -np.inf, 0
    evidences = np.empty(len(PD.CANDIDATES))
    thetas = []
    for index, candidate in enumerate(PD.CANDIDATES):
        z = PD.local_design(signs, candidate)
        lam = np.full(z.shape[1], lam_scale)
        lam[0] = lam_icept
        theta, hess, loglik = PD._fit(z, sub_y, sub_w, offset, lam)
        evidences[index] = PD.log_evidence(theta, hess, loglik, lam)
        thetas.append((theta, hess))
    scores = evidences + log_prior
    best_index = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
    theta, hess = thetas[best_index]
    candidate = PD.CANDIDATES[best_index]
    vz = PD.local_design(PD.signs_from(phi[val_idx]), candidate)
    mu = phi[val_idx] @ beta + vz @ theta
    try:
        var = np.einsum('ij,jk,ik->i', vz, np.linalg.inv(hess), vz)
    except np.linalg.LinAlgError:
        var = np.zeros(len(mu))
    var = np.maximum(var, 0.0)
    eta = np.clip(mu / np.sqrt(1.0 + np.pi * var / 8.0), -30, 30)
    return float(np.sum(y[val_idx] * eta - np.logaddexp(0.0, eta))), best_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/compact_memory')
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
    pair_cfg = json.loads((ROOT / 'artifacts/win_head/pair_dict_deploy.json').read_text())
    lam_scale = float(pair_cfg['lam_scale'])
    lam_icept = lam_scale * float(pair_cfg['lam_intercept_scale'])
    log_prior = np.asarray(pair_cfg['log_prior'], dtype=np.float64)

    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    entries = build(vq, index)

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    episodes = set(ep.tolist())

    base_vec = read_submission(ROOT / 'artifacts/pair_dict/predictions/oof_best_r_two_head_only.csv')
    base_vec = np.array([base_vec[q] for q in ids])
    q_frozen = 1.0 - base_vec[:, 0]
    th = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    th = np.array([th[q] for q in ids])
    r_two = th[:, 1:] / np.maximum(1.0 - th[:, 0], 1e-12)[:, None]
    r_two /= r_two.sum(axis=1, keepdims=True)

    q_new = np.empty(len(ids))
    picks, same_as_dense, fallbacks = [], 0, 0
    for i, qid in enumerate(ids):
        e = entries[qid]
        if e['star'] is None or len(e['y']) == 0:
            q_new[i] = q_frozen[i]; fallbacks += 1; picks.append(None); continue
        n = len(e['y'])
        order = np.argsort(e['used'])
        hold = min(40, max(8, n // 3))
        if n - hold < 12:
            # Too little prefix to select memory honestly; keep the frozen choice.
            h_pick = None
            _, dense_detail = W.predict_q_averaged(
                {h: beta_by_h[h].get(e['type'], np.zeros(len(lam_dense))) for h in HALF_LIVES},
                e['phi'], e['y'], e['used'], e['cutoff'], e['star'], lam_dense, adapt='full')
            picked = dense_detail['selected_half_life']
            h_pick = None if picked == 0.0 else picked
        else:
            fit_idx, val_idx = order[:-hold], order[-hold:]
            origin = float(e['used'][val_idx].min())
            best, h_pick = -np.inf, None
            for h in HALF_LIVES:
                beta = beta_by_h[h].get(e['type'], np.zeros(len(lam_dense)))
                w = decay(e['used'], origin, h)
                ll, _ = candidate_block_loss(e['phi'], e['y'], w, beta, fit_idx, val_idx,
                                             lam_scale, lam_icept, log_prior)
                if ll > best:
                    best, h_pick = ll, h
        beta = beta_by_h[h_pick].get(e['type'], np.zeros(len(lam_dense)))
        w = decay(e['used'], e['cutoff'], h_pick)
        omega, qs, _ = fit_select(e['phi'], e['y'], w, beta, e['star'],
                                  lam_scale, lam_icept, log_prior)
        q_new[i] = float(qs[int(np.argmax(omega))])
        picks.append(h_pick if h_pick else 0.0)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(q, r):
        out = np.column_stack((1.0 - q, q[:, None] * r))
        return out / out.sum(axis=1, keepdims=True)

    def official(a):
        return score(mapping(a), scoring, episodes)

    frozen = official(compose(q_frozen, r_two))
    corrected = official(compose(q_new, r_two))
    paired = paired_bootstrap(mapping(compose(q_new, r_two)),
                              mapping(compose(q_frozen, r_two)), scoring, episodes)
    hist = {}
    for p in picks:
        if p is None:
            continue
        hist[str(int(p))] = hist.get(str(int(p)), 0) + 1
    report = {
        'preregistration': __doc__,
        'frozen_baseline': {'points': frozen['technical_points'], 'R': frozen['mean_loss_ratio'],
                            'by_condition': {c: v['loss_ratio'] for c, v in frozen['by_condition'].items()}},
        'compact_memory_selection': {'points': corrected['technical_points'],
                                     'R': corrected['mean_loss_ratio'],
                                     'by_condition': {c: v['loss_ratio'] for c, v in corrected['by_condition'].items()}},
        'gain': corrected['technical_points'] - frozen['technical_points'],
        'half_life_histogram': hist, 'fallbacks': fallbacks,
        'q_max_abs_change': float(np.max(abs(q_new - q_frozen))),
        'paired_vs_frozen': paired,
        'exposure': 'Repeatedly inspected public validation; exploratory, another adaptive round.',
        'seconds': time.perf_counter() - start,
    }
    write_submission(args.out / 'predictions' / 'oof_compact_memory.csv', mapping(compose(q_new, r_two)))
    write_submission(args.out / 'predictions' / 'q_compact_memory.csv',
                     mapping(np.column_stack((1 - q_new, q_new[:, None] * r_two))))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(frozen['by_condition'])
    print(f"{'candidate':<34}{'points':>11}{'R':>9}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for name, v in (('frozen (dense-selected H)', frozen), ('compact-selected H', corrected)):
        print(f"{name:<34}{v['technical_points']:>11.6f}{v['mean_loss_ratio']:>9.4f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\ngain {report['gain']:+.6f} points; paired 95% [{lo:+.5f}, {hi:+.5f}] "
          f"{'significant' if hi < 0 else 'overlaps zero'}")
    print(f"half-life picks {hist}; fallbacks {fallbacks}; max |dq| {report['q_max_abs_change']:.4f}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
