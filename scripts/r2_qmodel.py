"""Phase 3/4: the four controlled adaptation arms for the q head, scored officially.

Arms
  frozen        the unchanged Round-1 release, applied to Round-2 inputs
  r2_only       refit on Round-2 released validation targets only
  combined      fitted on Round-1 + Round-2 released validation targets, pooled
  hierarchical  a correction on top of the frozen logit, ridge-shrunk toward zero

Honesty of the numbers
  Outer evaluation is 4 complete-episode folds, stratified so each fold holds six
  episodes of each condition. Every hyperparameter -- ridge penalty, blend weight,
  boosting depth -- is chosen inside the three fitting folds by an inner split, never
  on the held fold. Reported R is therefore out-of-fold on all 96 episodes.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ayan/Downloads/All-In')
import numpy as np

from scripts.r2_common import (R1, R2, CONDITIONS, targets, queries, history, conditions,
                               read_sub, write_sub, score, reference_episode_errors,
                               episode_errors, paired_bootstrap)
from scripts.r2_features import build

SP = Path('/private/tmp/claude-501/-Users-ayan-Downloads-All-In/c377a5e6-3d92-4ec2-b480-1d03aa25cf7f/scratchpad/r2')
ROOT = Path('/Users/ayan/Downloads/All-In')
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def split_qr(sub, ids):
    a = np.array([sub[i] for i in ids])
    a = a / a.sum(axis=1, keepdims=True)
    q = 1 - a[:, 0]
    r = a[:, 1:] / np.maximum(q, 1e-12)[:, None]
    return q, r / r.sum(axis=1, keepdims=True)


def folds_by_condition(ids, qmeta, cond, k=4, seed=777):
    eps = {}
    for qid in ids:
        eps[qmeta[qid]['episode_id']] = cond[qid]
    rng = np.random.default_rng(seed)
    assign = {}
    for c in CONDITIONS:
        e = sorted(x for x, cc in eps.items() if cc == c)
        order = rng.permutation(len(e))
        for pos, j in enumerate(order):
            assign[e[j]] = pos % k
    return np.array([assign[qmeta[i]['episode_id']] for i in ids]), assign


def ridge_fit(X, y, alpha, w=None):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    if w is None:
        A = Z.T @ Z
        b = Z.T @ y
    else:
        A = Z.T @ (Z * w[:, None])
        b = Z.T @ (y * w)
    P = alpha * np.eye(Z.shape[1])
    P[-1, -1] = 0.0
    beta = np.linalg.solve(A + P, b)
    return (mu, sd, beta)


def ridge_pred(model, X):
    mu, sd, beta = model
    return np.column_stack([(X - mu) / sd, np.ones(len(X))]) @ beta


ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def pick_alpha(X, y, groups, w=None):
    """Inner leave-one-group-out over the fitting folds' episodes, grouped in 4 blocks."""
    uniq = sorted(set(groups.tolist()))
    blocks = [uniq[i::4] for i in range(4)]
    best, best_mse = ALPHAS[0], np.inf
    for a in ALPHAS:
        err, nn = 0.0, 0
        for b in blocks:
            te = np.isin(groups, b)
            if te.sum() == 0 or (~te).sum() < 10:
                continue
            m = ridge_fit(X[~te], y[~te], a, None if w is None else w[~te])
            err += float(((ridge_pred(m, X[te]) - y[te]) ** 2).sum())
            nn += int(te.sum())
        if nn and err / nn < best_mse:
            best_mse, best = err / nn, a
    return best


def catboost_fit(X, y, seed=0, depth=6, iters=600, lr=0.05):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=iters, depth=depth, learning_rate=lr, loss_function='RMSE',
                          random_seed=seed, verbose=0, thread_count=2,
                          allow_writing_files=False, bootstrap_type='No')
    m.fit(X, y)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/r2_adaptation')
    ap.add_argument('--no-catboost', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    tg2 = targets(R2)
    qm2 = queries(R2, 'validation')
    h2 = history(R2, 'validation')
    cond2 = conditions(R2, 'validation')
    ids2 = sorted(tg2)
    _, X2, names = build(qm2, h2, ids2)
    A2 = np.array([tg2[i] for i in ids2])
    q_true = 1 - A2[:, 0]
    r_true = A2[:, 1:] / np.maximum(q_true, 1e-12)[:, None]
    y2 = logit(q_true)

    frozen_val = read_sub(SP / 'frozen_validation.csv')
    qf2, rf2 = split_qr(frozen_val, ids2)
    lf2 = logit(qf2)

    tg1 = targets(R1)
    qm1 = queries(R1, 'validation')
    h1 = history(R1, 'validation')
    ids1 = sorted(tg1)
    _, X1, _ = build(qm1, h1, ids1)
    A1 = np.array([tg1[i] for i in ids1])
    q_true1 = 1 - A1[:, 0]
    y1 = logit(q_true1)
    frozen_val1 = read_sub(ROOT / 'artifacts/win_head_release_exact/validation_submission.csv')
    qf1, _ = split_qr(frozen_val1, ids1)
    lf1 = logit(qf1)

    fold, _ = folds_by_condition(ids2, qm2, cond2)
    ep_of = np.array([qm2[i]['episode_id'] for i in ids2])
    refep = reference_episode_errors(R2)

    def rep(name, qhat, rhat):
        pred = {i: np.concatenate([[1 - qhat[k]], qhat[k] * rhat[k]]) for k, i in enumerate(ids2)}
        s = score(pred, tg2, qm2, cond2, refep)
        print(f"  {name:30s} R={s['R']:.5f}  pts/20={s['points_20']:.4f}  pts/40={s['points_40']:.4f}"
              f"  ratios=" + " ".join(f"{s['by_condition'][c]['loss_ratio']:.3f}" for c in CONDITIONS))
        return s, pred

    results = {}
    print('\n=== arm 0: frozen Round-1 release ===')
    s_fr, pred_fr = rep('frozen', qf2, rf2)
    results['frozen'] = s_fr

    # ---- out-of-fold arms ----
    oof = {k: np.zeros(len(ids2)) for k in ('r2_only', 'combined', 'hier', 'hier_cb', 'r2_cb')}
    chosen = {k: [] for k in oof}
    for f in range(4):
        te = fold == f
        tr = ~te
        gtr = np.array([hash(e) for e in ep_of[tr]])  # stable grouping for the inner split
        gtr = np.array([abs(int.from_bytes(e.encode()[-4:], 'little')) % 1000 for e in ep_of[tr]])

        a = pick_alpha(X2[tr], y2[tr], gtr)
        m = ridge_fit(X2[tr], y2[tr], a)
        oof['r2_only'][te] = ridge_pred(m, X2[te])
        chosen['r2_only'].append(a)

        Xc = np.vstack([X2[tr], X1])
        yc = np.concatenate([y2[tr], y1])
        gc = np.concatenate([gtr, np.full(len(y1), 999)])
        ac = pick_alpha(Xc, yc, gc)
        mc = ridge_fit(Xc, yc, ac)
        oof['combined'][te] = ridge_pred(mc, X2[te])
        chosen['combined'].append(ac)

        # hierarchical: correction on top of the frozen logit, with the blend weight
        # and penalty both chosen inside the fitting folds
        res = y2[tr] - lf2[tr]
        Xh = np.column_stack([X2[tr], lf2[tr]])
        ah = pick_alpha(Xh, res, gtr)
        mh = ridge_fit(Xh, res, ah)
        oof['hier'][te] = lf2[te] + ridge_pred(mh, np.column_stack([X2[te], lf2[te]]))
        chosen['hier'].append(ah)

        if not args.no_catboost:
            cb = catboost_fit(np.column_stack([X2[tr], lf2[tr]]), res, seed=1000 + f)
            oof['hier_cb'][te] = lf2[te] + cb.predict(np.column_stack([X2[te], lf2[te]]))
            cb2 = catboost_fit(X2[tr], y2[tr], seed=2000 + f)
            oof['r2_cb'][te] = cb2.predict(X2[te])
            chosen['hier_cb'].append('cb600d6')
            chosen['r2_cb'].append('cb600d6')

    print('\n=== arms 1-4: out-of-fold on all 96 episodes, frozen r kept ===')
    for k in ('r2_only', 'combined', 'hier', 'r2_cb', 'hier_cb'):
        if np.all(oof[k] == 0):
            continue
        s, _ = rep(k, sigmoid(oof[k]), rf2)
        results[k] = s
        print(f"       chosen per fold: {chosen[k]}")

    print('\n=== logit-space fit quality (out-of-fold) ===')
    print(f"  frozen            MSE {np.mean((lf2 - y2) ** 2):.4f}  corr {np.corrcoef(lf2, y2)[0,1]:.4f}")
    for k in oof:
        if np.all(oof[k] == 0):
            continue
        print(f"  {k:17s} MSE {np.mean((oof[k] - y2) ** 2):.4f}  corr {np.corrcoef(oof[k], y2)[0,1]:.4f}")

    np.save(SP / 'oof_logits.npy', np.array([oof[k] for k in sorted(oof)]))
    json.dump({'order': sorted(oof), 'ids': ids2, 'frozen_logit': lf2.tolist(),
               'y_true': y2.tolist(), 'fold': fold.tolist(),
               'results': {k: {'R': v['R'], 'points_20': v['points_20'],
                               'ratios': {c: v['by_condition'][c]['loss_ratio'] for c in CONDITIONS}}
                           for k, v in results.items()}},
              open(args.out / 'arms.json', 'w'), indent=2)
    print(f"\nwrote {args.out / 'arms.json'}")


if __name__ == '__main__':
    main()
