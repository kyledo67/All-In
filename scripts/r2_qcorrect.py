"""Phase 4: q correction fitted and selected on the OFFICIAL objective.

The first pass fitted logit q by least squares. That lowered logit MSE and RAISED the
official R, because the metric is squared error on probabilities: a logit fit spends its
capacity on near-0/near-1 machines where probability error is already negligible. Here
every candidate is fitted to the probability-space residual q_true - q_frozen and selected
by mean squared probability error on inner folds, which is the quantity the contest scores.

Arms are nested so each added degree of freedom is attributable:
  cal_iso        monotone recalibration of the frozen q alone
  res_ridge      probability-space ridge correction, blend weight chosen inner
  res_cb         same with gradient boosting
  res_ridge_sh   ridge correction with a separate blend weight for short histories
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ayan/Downloads/All-In')
import numpy as np

from scripts.r2_common import (R1, R2, CONDITIONS, targets, queries, history, conditions,
                               read_sub, score, reference_episode_errors)
from scripts.r2_features import build
from scripts.r2_qmodel import (logit, sigmoid, split_qr, folds_by_condition, ridge_fit,
                               ridge_pred, catboost_fit, SP, ROOT)

LAMBDAS = np.linspace(0.0, 1.0, 21)
ALPHAS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


def q_objective(qhat, qt, rt):
    """Exact metric contribution of the q head with r held at its true value."""
    p = np.column_stack([1 - qhat, qhat[:, None] * rt])
    t = np.column_stack([1 - qt, qt[:, None] * rt])
    return float(((p - t) ** 2).sum(axis=1).mean())


def isotonic(x, y):
    o = np.argsort(x, kind='stable')
    xs, ys = x[o], y[o]
    v = ys.astype(float).copy()
    w = np.ones(len(v))
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-12:
            i += 1
            continue
        tw = w[i] + w[i + 1]
        tv = (v[i] * w[i] + v[i + 1] * w[i + 1]) / tw
        v[i] = tv
        w[i] = tw
        v = np.delete(v, i + 1)
        w = np.delete(w, i + 1)
        xs = np.delete(xs, i + 1)
        if i > 0:
            i -= 1
    return xs, v


def iso_apply(knots, vals, x):
    return np.interp(x, knots, vals, left=vals[0], right=vals[-1])


def inner_blocks(eps):
    uniq = sorted(set(eps.tolist()))
    return [set(uniq[i::4]) for i in range(4)]


def main():
    tg2 = targets(R2)
    qm2 = queries(R2, 'validation')
    h2 = history(R2, 'validation')
    cond2 = conditions(R2, 'validation')
    ids2 = sorted(tg2)
    _, X2, names = build(qm2, h2, ids2)
    A2 = np.array([tg2[i] for i in ids2])
    qt = 1 - A2[:, 0]
    rt = A2[:, 1:] / np.maximum(qt, 1e-12)[:, None]

    frozen = read_sub(SP / 'frozen_validation.csv')
    qf, rf = split_qr(frozen, ids2)
    fold, _ = folds_by_condition(ids2, qm2, cond2)
    eps = np.array([qm2[i]['episode_id'] for i in ids2])
    refep = reference_episode_errors(R2)
    short = X2[:, names.index('n_obs')] * 250.0 < 120

    print(f'q-space objective, frozen: {q_objective(qf, qt, rt):.6f}')
    print('\ncalibration of the frozen q against the truth (R2 validation):')
    for lo, hi in ((0, .02), (.02, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .98), (.98, 1.01)):
        s = (qf >= lo) & (qf < hi)
        if s.sum():
            print(f'  q_frozen in [{lo:.2f},{hi:.2f}) n={s.sum():4d}  mean frozen {qf[s].mean():.4f}'
                  f'  mean true {qt[s].mean():.4f}  bias {qt[s].mean()-qf[s].mean():+.4f}')

    oof = {}

    def run(name, fn):
        out = np.zeros(len(ids2))
        picks = []
        for f in range(4):
            te = fold == f
            out[te], pk = fn(~te, te)
            picks.append(pk)
        oof[name] = np.clip(out, 1e-6, 1 - 1e-6)
        pred = {i: np.concatenate([[1 - oof[name][k]], oof[name][k] * rf[k]])
                for k, i in enumerate(ids2)}
        s = score(pred, tg2, qm2, cond2, refep)
        print(f"  {name:16s} R={s['R']:.5f}  pts/20={s['points_20']:.4f}  pts/40={s['points_40']:.4f}"
              f"  qobj={q_objective(oof[name], qt, rt):.6f}  ratios="
              + " ".join(f"{s['by_condition'][c]['loss_ratio']:.3f}" for c in CONDITIONS)
              + f"  picks={picks}")
        return s

    # --- monotone recalibration of the frozen q
    def f_iso(tr, te):
        k, v = isotonic(qf[tr], qt[tr])
        cand = {}
        for lam in LAMBDAS:
            cand[lam] = []
        for b in inner_blocks(eps[tr]):
            itr = np.array([e not in b for e in eps[tr]])
            kk, vv = isotonic(qf[tr][itr], qt[tr][itr])
            ite = ~itr
            base = qf[tr][ite]
            adj = iso_apply(kk, vv, base)
            for lam in LAMBDAS:
                cand[lam].append(q_objective(np.clip((1 - lam) * base + lam * adj, 0, 1),
                                             qt[tr][ite], rt[tr][ite]) * ite.sum())
        best = min(LAMBDAS, key=lambda l: sum(cand[l]))
        return np.clip((1 - best) * qf[te] + best * iso_apply(k, v, qf[te]), 0, 1), round(float(best), 2)

    print('\n=== arms, out-of-fold, frozen r kept ===')
    run('frozen', lambda tr, te: (qf[te], '-'))
    run('cal_iso', f_iso)

    # --- probability-space ridge residual
    def make_res(use_cb=False, per_short=False, feat=None):
        def f(tr, te):
            Xtr = X2[tr] if feat is None else X2[tr][:, feat]
            Xte = X2[te] if feat is None else X2[te][:, feat]
            res = qt[tr] - qf[tr]
            blocks = inner_blocks(eps[tr])
            scores = {}
            for a in (['cb'] if use_cb else ALPHAS):
                inner = {lam: 0.0 for lam in LAMBDAS}
                if per_short:
                    inner = {(l1, l2): 0.0 for l1 in LAMBDAS[::2] for l2 in LAMBDAS[::2]}
                for b in blocks:
                    itr = np.array([e not in b for e in eps[tr]])
                    ite = ~itr
                    if ite.sum() == 0 or itr.sum() < 50:
                        continue
                    if use_cb:
                        m = catboost_fit(Xtr[itr], res[itr], seed=7, depth=4, iters=400, lr=0.05)
                        d = m.predict(Xte[ite] if False else Xtr[ite])
                    else:
                        m = ridge_fit(Xtr[itr], res[itr], a)
                        d = ridge_pred(m, Xtr[ite])
                    base = qf[tr][ite]
                    sh = short[tr][ite]
                    for key in inner:
                        if per_short:
                            l1, l2 = key
                            lam = np.where(sh, l2, l1)
                        else:
                            lam = key
                        inner[key] += q_objective(np.clip(base + lam * d, 1e-6, 1 - 1e-6),
                                                  qt[tr][ite], rt[tr][ite]) * ite.sum()
                best_key = min(inner, key=inner.get)
                scores[a] = (inner[best_key], best_key)
            a = min(scores, key=lambda k: scores[k][0])
            key = scores[a][1]
            if use_cb:
                m = catboost_fit(Xtr, res, seed=7, depth=4, iters=400, lr=0.05)
                d = m.predict(Xte)
            else:
                m = ridge_fit(Xtr, res, a)
                d = ridge_pred(m, Xte)
            lam = np.where(short[te], key[1], key[0]) if per_short else key
            return np.clip(qf[te] + lam * d, 1e-6, 1 - 1e-6), (a, key)
        return f

    run('res_ridge', make_res())
    run('res_ridge_sh', make_res(per_short=True))
    run('res_cb', make_res(use_cb=True))

    np.save(SP / 'oof_q.npy', np.array([oof[k] for k in sorted(oof)]))
    json.dump({'order': sorted(oof), 'ids': ids2,
               'q': {k: v.tolist() for k, v in oof.items()}},
              open(SP / 'oof_q.json', 'w'))
    print(f"\nsaved out-of-fold q to {SP / 'oof_q.json'}")


if __name__ == '__main__':
    main()
