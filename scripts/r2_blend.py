"""Does blending the Round-1 release variants help on Round-2?

The official metric is exactly squared error in probability space, so a convex blend of
member forecasts is a simplex-constrained least-squares problem. Weights are solved on the
three fitting folds and applied to the held fold, so the reported R is out-of-fold. A
weight floor is never imposed: if the solver puts everything on one member, that is the
answer the evidence gives.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ayan/Downloads/All-In')
import numpy as np

from scripts.r2_common import (R2, CONDITIONS, targets, queries, conditions, read_sub,
                               score, reference_episode_errors, episode_errors)
from scripts.r2_qmodel import folds_by_condition, SP

MEM = {'frozen': 'frozen_validation.csv', 'tail': 'win_head_release_tail_validation.csv',
       'dirichlet': 'win_head_release_dirichlet_validation.csv',
       'parity': 'win_head_release_parity_validation.csv',
       'nofx': 'win_head_release_nofx_validation.csv',
       'hmm': 'final_validation.csv', 'blendrel': 'blend_validation.csv'}


def simplex_ls(P, T, iters=4000, lr=None):
    """min_w ||sum_m w_m P_m - T||_F^2  s.t.  w >= 0, sum w = 1.  Projected gradient."""
    M = P.shape[0]
    G = np.einsum('mij,nij->mn', P, P)
    b = np.einsum('mij,ij->m', P, T)
    w = np.full(M, 1.0 / M)
    L = np.linalg.eigvalsh(G).max()
    lr = 1.0 / max(L, 1e-9) if lr is None else lr
    for _ in range(iters):
        g = G @ w - b
        w = project_simplex(w - lr * g)
    return w


def project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def main():
    tg = targets(R2)
    qm = queries(R2, 'validation')
    cond = conditions(R2, 'validation')
    ids = sorted(tg)
    refep = reference_episode_errors(R2)
    T = np.array([tg[i] for i in ids])
    names = sorted(MEM)
    P = []
    for k in names:
        s = read_sub(Path(SP) / MEM[k])
        a = np.array([s[i] for i in ids])
        P.append(a / a.sum(axis=1, keepdims=True))
    ref = read_sub(R2 / 'validation' / 'reference.csv')
    names.append('reference')
    a = np.array([ref[i] for i in ids])
    P.append(a / a.sum(axis=1, keepdims=True))
    P = np.array(P)

    fold, _ = folds_by_condition(ids, qm, cond)

    def rep(name, A):
        pred = {i: A[k] for k, i in enumerate(ids)}
        s = score(pred, tg, qm, cond, refep)
        print(f"  {name:36s} R={s['R']:.5f}  pts/20={s['points_20']:.4f}  pts/40={s['points_40']:.4f}"
              f"  ratios=" + " ".join(f"{s['by_condition'][c]['loss_ratio']:.3f}" for c in CONDITIONS))
        return s

    print('=== in-sample optimum (an upper bound, NOT a claim) ===')
    w_all = simplex_ls(P, T)
    print('  weights ' + '  '.join(f'{n}={v:.3f}' for n, v in zip(names, w_all) if v > 1e-4))
    rep('simplex blend, fitted on all 96', np.einsum('m,mij->ij', w_all, P))

    print('\n=== out-of-fold blend ===')
    out = np.zeros_like(T)
    ws = []
    for f in range(4):
        te = fold == f
        w = simplex_ls(P[:, ~te], T[~te])
        ws.append({n: round(float(v), 4) for n, v in zip(names, w) if v > 1e-3})
        out[te] = np.einsum('m,mij->ij', w, P[:, te])
    out = np.clip(out, 1e-9, 1)
    out = out / out.sum(axis=1, keepdims=True)
    s_blend = rep('simplex blend, out-of-fold', out)
    for f, w in enumerate(ws):
        print(f'    fold {f} weights: {w}')

    print('\n=== pairwise: frozen + one other, out-of-fold weight ===')
    fi = names.index('frozen')
    for j, n in enumerate(names):
        if j == fi:
            continue
        o = np.zeros_like(T)
        wl = []
        for f in range(4):
            te = fold == f
            w = simplex_ls(P[[fi, j]][:, ~te], T[~te])
            wl.append(round(float(w[1]), 3))
            o[te] = w[0] * P[fi][te] + w[1] * P[j][te]
        rep(f'frozen + {n} (w_other per fold {wl})', o / o.sum(axis=1, keepdims=True))

    json.dump({'names': names, 'w_insample': w_all.tolist(), 'w_oof': ws,
               'R_oof': s_blend['R'], 'points_20_oof': s_blend['points_20']},
              open(Path(SP) / 'blend.json', 'w'), indent=2)
    np.save(Path(SP) / 'blend_oof.npy', out)


if __name__ == '__main__':
    main()
