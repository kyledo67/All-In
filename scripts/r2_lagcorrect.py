"""Did the predictive lag relationships change, and can a compact correction capture it?

The frozen local q head is a recency-weighted Beta-Bernoulli rate: it uses HOW OFTEN the
machine won, not the ORDER of recent wins. The Round-2 targets show a lag-1 coefficient on
logit q of -0.25 where Round 1 had -0.03, so order may now carry signal the rate model
discards. Each candidate below adds one small, named block of order information.

Specification matters: the correction is fitted in logit space, where the model is
well-posed, but weighted by (dq/dlogit)^2 = (q(1-q))^2 so that least squares there
approximates least squares in the probability space the contest actually scores. The blend
weight is then chosen on the exact official objective, inside the fitting folds only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ayan/Downloads/All-In')
import numpy as np

from scripts.r2_common import (R1, R2, CONDITIONS, targets, queries, history, conditions,
                               read_sub, score, reference_episode_errors)
from scripts.r2_features import build
from scripts.r2_qmodel import logit, sigmoid, split_qr, folds_by_condition, ridge_fit, ridge_pred, SP

ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
LAMS = np.linspace(0.0, 1.2, 25)

BLOCKS = {
    'lag_wins': [f'lag{k}_win' for k in range(1, 9)],
    'lag_miss': [f'lag{k}_missing' for k in range(1, 9)],
    'pairs': ['pair_1_2', 'pair_2_4', 'pair_4_8', 'pair_1_8', 'pair_1_3', 'pair_2_3',
              'pair_3_6', 'pair_5_7'],
    'types': ['type_steady', 'type_balanced', 'type_jackpot'],
    'recency': ['since_win', 'streak_len', 'streak_is_win', 'trend_ew8_ew64', 'trend_20_prev20'],
    'neighbour': ['nb_rate20', 'nb_ew8', 'nb_ew32', 'nb_last_win', 'nb_minus_self',
                  'nb_ew8_minus_ew32', 'nb_sd_rate20'],
    'horizon': ['n_obs', 'log_n_obs', 'gap_frac_gt1', 'last_tick_lag'],
    'signal': ['signal_-1', 'signal_0', 'signal_1'],
}
CANDIDATES = [
    ('lag_wins',), ('lag_wins', 'lag_miss'), ('lag_wins', 'pairs'),
    ('lag_wins', 'pairs', 'types'), ('lag_wins', 'recency'), ('neighbour',),
    ('lag_wins', 'neighbour'), ('lag_wins', 'pairs', 'neighbour', 'horizon'),
    ('signal',), ('lag_wins', 'pairs', 'types', 'recency', 'neighbour', 'horizon', 'signal'),
]


def main():
    tg = targets(R2)
    qm = queries(R2, 'validation')
    cond = conditions(R2, 'validation')
    ids = sorted(tg)
    _, X, names = build(qm, history(R2, 'validation'), ids)
    A = np.array([tg[i] for i in ids])
    qt = 1 - A[:, 0]
    rt = A[:, 1:] / np.maximum(qt, 1e-12)[:, None]
    refep = reference_episode_errors(R2)
    frozen = read_sub(Path(SP) / 'frozen_validation.csv')
    qf, rf = split_qr(frozen, ids)
    D = json.load(open(Path(SP) / 'frozen_validation_diag.json'))
    q_local = np.array([D['q'][i] for i in ids])
    fold, _ = folds_by_condition(ids, qm, cond)
    eps = np.array([qm[i]['episode_id'] for i in ids])

    base = q_local
    lb = logit(base)
    yt = logit(qt)
    wls = (base * (1 - base)) ** 2

    def qobj(qh, sel):
        p = np.column_stack([1 - qh[sel], qh[sel][:, None] * rt[sel]])
        t = np.column_stack([1 - qt[sel], qt[sel][:, None] * rt[sel]])
        return float(((p - t) ** 2).sum(axis=1).sum())

    def rep(name, qh):
        pred = {i: np.concatenate([[1 - qh[k]], qh[k] * rf[k]]) for k, i in enumerate(ids)}
        s = score(pred, tg, qm, cond, refep)
        print(f"  {name:48s} R={s['R']:.5f} pts/20={s['points_20']:.4f} pts/40={s['points_40']:.4f}"
              f" ratios=" + " ".join(f"{s['by_condition'][c]['loss_ratio']:.3f}" for c in CONDITIONS))
        return s

    print('=== univariate evidence: weighted correlation of the logit residual with each feature ===')
    resid = yt - lb
    w = wls / wls.sum()
    for blk, feats in BLOCKS.items():
        out = []
        for f in feats:
            j = names.index(f)
            x = X[:, j]
            mx = np.dot(w, x)
            mr = np.dot(w, resid)
            cov = np.dot(w, (x - mx) * (resid - mr))
            sd = np.sqrt(np.dot(w, (x - mx) ** 2) * np.dot(w, (resid - mr) ** 2))
            out.append(f'{f}={cov/max(sd,1e-12):+.3f}')
        print(f'  {blk:10s} ' + '  '.join(out))

    print('\n=== out-of-fold corrections on the local q, frozen r kept ===')
    rep('frozen (local q, alpha=1)', base)
    results = {}
    for cand in CANDIDATES:
        feats = [f for b in cand for f in BLOCKS[b]]
        cols = [names.index(f) for f in feats]
        oof = np.zeros(len(ids))
        picks = []
        for f in range(4):
            te = fold == f
            tr = ~te
            uniq = sorted(set(eps[tr].tolist()))
            blocks = [set(uniq[i::4]) for i in range(4)]
            best, bestv = None, np.inf
            for a in ALPHAS:
                inner = {lam: 0.0 for lam in LAMS}
                for b in blocks:
                    ite = np.array([e in b for e in eps]) & tr
                    itr = tr & ~ite
                    if ite.sum() == 0:
                        continue
                    m = ridge_fit(X[itr][:, cols], (yt - lb)[itr], a, wls[itr])
                    d = ridge_pred(m, X[ite][:, cols])
                    for lam in LAMS:
                        full = base.copy()
                        full[ite] = sigmoid(lb[ite] + lam * d)
                        inner[lam] += qobj(full, ite)
                k = min(inner, key=inner.get)
                if inner[k] < bestv:
                    bestv, best = inner[k], (a, k)
            a, lam = best
            m = ridge_fit(X[tr][:, cols], (yt - lb)[tr], a, wls[tr])
            oof[te] = sigmoid(lb[te] + lam * ridge_pred(m, X[te][:, cols]))
            picks.append((a, round(float(lam), 2)))
        s = rep('+'.join(cand) + f' [{len(feats)}f]', oof)
        results['+'.join(cand)] = {'R': s['R'], 'points_20': s['points_20'], 'picks': picks}
        print(f"        picks {picks}")
        np.save(Path(SP) / f"lagcorr_{'_'.join(cand)}.npy", oof)

    json.dump(results, open(Path(SP) / 'lagcorrect.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
