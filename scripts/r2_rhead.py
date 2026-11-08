"""Phase 4, second priority: the r head as a posterior over a two-element dictionary.

The conditional payout distribution r = P(payout | win) takes exactly six values on
released targets, and the six are bit-identical in the main and adaptation rounds:
each observed machine_type admits exactly two. So r is not a regression problem, it is a
binary latent per query, and the forecast is the posterior MIXTURE of the two candidates.
A hard argmax would throw away the (large) uncertainty and scores worse than the frozen r.

Recency matters because the amounts can change within an episode, so the likelihood uses
exponentially weighted counts of the machine's own pre-cutoff nonzero payouts. The
half-life, the likelihood temperature and the blend against the frozen r are selected on
the official objective with complete-episode inner folds.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ayan/Downloads/All-In')
import numpy as np

from scripts.r2_common import (R1, R2, CONDITIONS, targets, queries, history, conditions,
                               read_sub, score, reference_episode_errors)
from scripts.r2_qmodel import split_qr, folds_by_condition, SP, ROOT

PAY = (1, 2, 5, 20)
PIDX = {1: 0, 2: 1, 5: 2, 20: 3}
DICT = {
    'steady':   (np.array([0.45512201, 0.43590239, 0.10679609, 0.00217951]),
                 np.array([0.75398939, 0.19680849, 0.04821808, 0.00098404])),
    'balanced': (np.array([0.21406615, 0.54673659, 0.20502622, 0.03417104]),
                 np.array([0.49985311, 0.34792827, 0.13047310, 0.02174552])),
    'jackpot':  (np.array([0.12078673, 0.32392068, 0.41646944, 0.13882315]),
                 np.array([0.33514603, 0.24494620, 0.31493083, 0.10497694])),
}
HALFLIVES = (8, 16, 24, 32, 48, 64, 96, 128, 1e9)
TEMPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def weighted_counts(root, split, hl):
    h = history(root, split)
    qmap = queries(root, split)
    out = {}
    for qid, m in qmap.items():
        obs = h[m['episode_id']].get(m['machine_id'], {})
        c = np.zeros(4)
        for t, p in obs.items():
            if t < m['cutoff'] and p > 0:
                c[PIDX[p]] += 2.0 ** (-(m['cutoff'] - 1 - t) / hl)
        out[qid] = c
    return out


def posterior(root, split, ids, hl, temp, prior=0.5):
    cnt = weighted_counts(root, split, hl)
    qmap = queries(root, split)
    w = np.zeros(len(ids))
    rhat = np.zeros((len(ids), 4))
    for i, qid in enumerate(ids):
        a, b = DICT[qmap[qid]['machine_type']]
        c = cnt[qid]
        la = float(np.dot(c, np.log(a))) * temp + np.log(max(prior, 1e-9))
        lb = float(np.dot(c, np.log(b))) * temp + np.log(max(1 - prior, 1e-9))
        mx = max(la, lb)
        pa = np.exp(la - mx) / (np.exp(la - mx) + np.exp(lb - mx))
        w[i] = pa
        rhat[i] = pa * a + (1 - pa) * b
    return w, rhat


def true_labels(root, ids):
    tt = targets(root)
    qmap = queries(root, 'validation')
    A = np.array([tt[i] for i in ids])
    q = 1 - A[:, 0]
    r = A[:, 1:] / np.maximum(q, 1e-12)[:, None]
    lab = np.array([int(np.argmin([np.abs(r[i] - x).sum()
                                   for x in DICT[qmap[ids[i]]['machine_type']]]))
                    for i in range(len(ids))])
    return q, r, lab


def main():
    ids = sorted(targets(R2))
    qm = queries(R2, 'validation')
    cond = conditions(R2, 'validation')
    qt, rt, lab = true_labels(R2, ids)
    frozen = read_sub(SP / 'frozen_validation.csv')
    qf, rf = split_qr(frozen, ids)
    refep = reference_episode_errors(R2)
    fold, _ = folds_by_condition(ids, qm, cond)
    eps = np.array([qm[i]['episode_id'] for i in ids])

    print('=== hard-assignment accuracy vs recency half-life (R2 then R1) ===')
    for tag, root in (('R2', R2), ('R1', R1)):
        idsx = sorted(targets(root))
        _, _, labx = true_labels(root, idsx)
        line = []
        for hl in HALFLIVES:
            w, _ = posterior(root, 'validation', idsx, hl, 1.0)
            line.append(f"hl={hl if hl < 1e8 else 'inf':>4}:{((w > .5).astype(int) == (1 - labx)).mean():.3f}")
        print(f'  {tag}  ' + '  '.join(line))

    def obj(rhat, qsrc):
        p = np.column_stack([1 - qsrc, qsrc[:, None] * rhat])
        t = np.column_stack([1 - qt, qt[:, None] * rt])
        return float(((p - t) ** 2).sum(axis=1).mean())

    def rep(name, rhat):
        pred = {i: np.concatenate([[1 - qf[k]], qf[k] * rhat[k]]) for k, i in enumerate(ids)}
        s = score(pred, targets(R2), qm, cond, refep)
        print(f"  {name:38s} R={s['R']:.5f}  pts/20={s['points_20']:.4f}  pts/40={s['points_40']:.4f}"
              f"  obj={obj(rhat, qf):.6f}  ratios=" +
              " ".join(f"{s['by_condition'][c]['loss_ratio']:.3f}" for c in CONDITIONS))
        return s

    print('\n=== r arms with the frozen q held fixed ===')
    rep('frozen r', rf)
    rep('true r (oracle)', rt)
    rep('type-mean of the two candidates', np.array([(DICT[qm[i]['machine_type']][0] +
                                                      DICT[qm[i]['machine_type']][1]) / 2 for i in ids]))
    w, rp = posterior(R2, 'validation', ids, 32, 1.0)
    rep('posterior hl=32 temp=1', rp)
    rep('hard argmax hl=32', np.array([DICT[qm[ids[i]]['machine_type']][0 if w[i] > .5 else 1]
                                        for i in range(len(ids))]))

    # project the frozen r onto the two-point dictionary segment: the truth is ALWAYS an
    # endpoint, so a forecast off the segment wastes error on a direction that cannot occur
    def project(rsrc, sharpen=1.0):
        out = np.zeros_like(rsrc)
        ws = np.zeros(len(rsrc))
        for i, qid in enumerate(ids):
            a, b = DICT[qm[qid]['machine_type']]
            d = a - b
            wopt = float(np.dot(rsrc[i] - b, d) / np.dot(d, d))
            wopt = min(1.0, max(0.0, (wopt - 0.5) * sharpen + 0.5))
            ws[i] = wopt
            out[i] = wopt * a + (1 - wopt) * b
        return out, ws

    for sh in (1.0, 1.5, 2.0, 3.0, 5.0):
        pr, _ = project(rf, sh)
        rep(f'frozen r projected on segment, sharpen={sh}', pr)
    _, w_fr = project(rf, 1.0)

    # select (hl, temp, blend) inside the fitting folds only
    print('\n=== selected out-of-fold ===')
    grid = [(hl, t) for hl in HALFLIVES for t in TEMPS]
    cache = {g: posterior(R2, 'validation', ids, g[0], g[1])[1] for g in grid}
    # evidence combination: the count log-odds plus the frozen r's implied log-odds
    lo_fr = np.log(np.clip(w_fr, 1e-3, 1 - 1e-3) / (1 - np.clip(w_fr, 1e-3, 1 - 1e-3)))
    for hl in (32, 64, 128):
        wc, _ = posterior(R2, 'validation', ids, hl, 1.0)
        lo_c = np.log(np.clip(wc, 1e-6, 1 - 1e-6) / (1 - np.clip(wc, 1e-6, 1 - 1e-6)))
        for cw in (0.0, 0.25, 0.5, 1.0):
            for fw in (0.5, 1.0, 1.5):
                wm = 1 / (1 + np.exp(-(cw * lo_c + fw * lo_fr)))
                rr = np.array([wm[i] * DICT[qm[q]['machine_type']][0] +
                               (1 - wm[i]) * DICT[qm[q]['machine_type']][1]
                               for i, q in enumerate(ids)])
                cache[('comb', hl, cw, fw)] = rr
    grid = list(cache)
    BL = np.linspace(0, 1, 21)
    oof = np.zeros((len(ids), 4))
    picks = []
    for f in range(4):
        te = fold == f
        tr = ~te
        uniq = sorted(set(eps[tr].tolist()))
        blocks = [set(uniq[i::4]) for i in range(4)]
        best, bestv = None, np.inf
        for g in grid:
            for bl in BL:
                tot = 0.0
                for b in blocks:
                    sel = np.array([e in b for e in eps]) & tr
                    if sel.sum() == 0:
                        continue
                    mix = bl * cache[g][sel] + (1 - bl) * rf[sel]
                    p = np.column_stack([1 - qf[sel], qf[sel][:, None] * mix])
                    t = np.column_stack([1 - qt[sel], qt[sel][:, None] * rt[sel]])
                    tot += float(((p - t) ** 2).sum(axis=1).sum())
                if tot < bestv:
                    bestv, best = tot, (g, bl)
        g, bl = best
        oof[te] = bl * cache[g][te] + (1 - bl) * rf[te]
        picks.append((g[0] if g[0] < 1e8 else 'inf', g[1], round(float(bl), 2)))
    s = rep('posterior blend, selected in-fold', oof)
    print(f'  picks per fold (half-life, temp, blend): {picks}')

    json.dump({'ids': ids, 'r_oof': oof.tolist(), 'picks': [list(map(str, p)) for p in picks],
               'R': s['R'], 'points_20': s['points_20']},
              open(SP / 'r_head.json', 'w'))
    print(f"\nsaved {SP / 'r_head.json'}")


if __name__ == '__main__':
    main()
