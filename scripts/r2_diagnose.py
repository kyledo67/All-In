"""Phase 2: quantitative comparison of main-round and adaptation data.

Everything is measured on the two rounds' VALIDATION splits, the only places where both
rounds expose true probability vectors, so the comparison is like-for-like.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/ayan/Downloads/All-In')
import numpy as np
from scripts.r2_common import R1, R2, targets, queries, history, conditions, labels, PAYOUTS

np.set_printoptions(suppress=True, linewidth=200)


def lag_feats(obs, cutoff, nlag=8):
    """Win indicator and payout at the k-th previous OBSERVED tick before cutoff."""
    ticks = sorted(t for t in obs if t < cutoff)
    out = np.zeros(2 * nlag)
    for k in range(nlag):
        if k < len(ticks):
            p = obs[ticks[-1 - k]]
            out[k] = 1.0 if p > 0 else 0.0
            out[nlag + k] = p
        else:
            out[k] = -1.0
            out[nlag + k] = -1.0
    return out


def build(root, split='validation'):
    qm, hist, tgt = queries(root, split), history(root, split), targets(root, split)
    rows, qs, rs, meta = [], [], [], []
    for qid in sorted(tgt):
        m = qm[qid]
        obs = hist[m['episode_id']].get(m['machine_id'], {})
        pre = {t: v for t, v in obs.items() if t < m['cutoff']}
        wins = [1.0 if v > 0 else 0.0 for t, v in sorted(pre.items())]
        pay = [v for t, v in sorted(pre.items())]
        f = lag_feats(obs, m['cutoff'])
        n = len(wins)
        rate = float(np.mean(wins)) if n else -1.0
        rate20 = float(np.mean(wins[-20:])) if n else -1.0
        rate5 = float(np.mean(wins[-5:])) if n else -1.0
        nzmean = float(np.mean([p for p in pay if p > 0])) if any(p > 0 for p in pay) else -1.0
        # neighbours in the same episode, strictly before cutoff
        nb, nbn = [], 0
        for mid, o in hist[m['episode_id']].items():
            if mid == m['machine_id']:
                continue
            v = [1.0 if p > 0 else 0.0 for t, p in sorted(o.items()) if t < m['cutoff']]
            if v:
                nb.append(np.mean(v[-20:])); nbn += 1
        rows.append(np.concatenate([f, [rate, rate20, rate5, n, nzmean,
                                        float(np.mean(nb)) if nb else -1.0, nbn,
                                        m['signal'], m['cutoff'],
                                        float(m['machine_type'] == 'steady'),
                                        float(m['machine_type'] == 'balanced'),
                                        float(m['machine_type'] == 'jackpot')]]))
        t = tgt[qid]
        qs.append(1 - t[0])
        rs.append(t[1:] / max(1 - t[0], 1e-12))
        meta.append((qid, m['episode_id'], m['machine_type'], m['signal'], n))
    names = ([f'win_lag{k+1}' for k in range(8)] + [f'pay_lag{k+1}' for k in range(8)] +
             ['rate_all', 'rate20', 'rate5', 'n_obs', 'nz_mean_payout', 'nb_rate20', 'nb_count',
              'signal', 'cutoff', 'is_steady', 'is_balanced', 'is_jackpot'])
    return np.array(rows), np.array(qs), np.array(rs), meta, names


def section(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


def main():
    D = {}
    for tag, root in (('R1', R1), ('R2', R2)):
        D[tag] = build(root)

    section('1. conditional payout distribution r = P(payout | win)')
    for tag in ('R1', 'R2'):
        X, q, r, meta, names = D[tag]
        print(f'{tag}: mean r {r.mean(axis=0).round(5)}   sd {r.std(axis=0).round(5)}')
        u, c = np.unique(r.round(6), axis=0, return_counts=True)
        print(f'    distinct r vectors (6dp): {len(u)} of {len(r)}')
        # how much of r's variance is explained by its own principal directions
        Z = r - r.mean(axis=0)
        s = np.linalg.svd(Z, compute_uv=False)
        print(f'    r variance share by component: {(s**2/ (s**2).sum()).round(4)}')
        for mt, col in (('steady', -3), ('balanced', -2), ('jackpot', -1)):
            sel = X[:, col] == 1
            print(f'    {mt:9s} n={sel.sum():4d}  mean r {r[sel].mean(axis=0).round(5)}  '
                  f'mean q {q[sel].mean():.4f}')
        # tail ratio stability: p20/p5 and p2/p1 given win
        tr = r[:, 3] / np.maximum(r[:, 2], 1e-12)
        pr = r[:, 1] / np.maximum(r[:, 0], 1e-12)
        print(f'    p20/p5  median {np.median(tr):.5f}  iqr {np.percentile(tr,[25,75]).round(5)}')
        print(f'    p2/p1   median {np.median(pr):.5f}  iqr {np.percentile(pr,[25,75]).round(5)}')

    section('2. is r predictable from observables? (ridge on logit-ish targets)')
    for tag in ('R1', 'R2'):
        X, q, r, meta, names = D[tag]
        Xs = np.column_stack([X, np.ones(len(X))])
        mu = Xs[:, :-1].mean(0); sd = Xs[:, :-1].std(0) + 1e-9
        Xs = np.column_stack([(X - mu) / sd, np.ones(len(X))])
        for j, lbl in enumerate(['r_1x', 'r_2x', 'r_5x', 'r_20x']):
            y = r[:, j]
            beta = np.linalg.solve(Xs.T @ Xs + 1e-2 * np.eye(Xs.shape[1]), Xs.T @ y)
            pred = Xs @ beta
            ss = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            print(f'  {tag} {lbl}: in-sample R^2 {ss:.4f}  sd(y) {y.std():.5f}')

    section('3. q: dependence on lagged wins (logistic on logit q, true targets)')
    for tag in ('R1', 'R2'):
        X, q, r, meta, names = D[tag]
        y = np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
        keep = [names.index(f'win_lag{k+1}') for k in range(8)] + \
               [names.index(x) for x in ('rate_all', 'rate20', 'rate5', 'nb_rate20',
                                         'signal', 'is_steady', 'is_balanced', 'is_jackpot')]
        Xk = X[:, keep]
        mu, sd = Xk.mean(0), Xk.std(0) + 1e-9
        Z = np.column_stack([(Xk - mu) / sd, np.ones(len(Xk))])
        beta = np.linalg.solve(Z.T @ Z + 1e-2 * np.eye(Z.shape[1]), Z.T @ y)
        pred = Z @ beta
        print(f'  {tag}: R^2 on logit q = {1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum():.4f}')
        for nm, b in zip([names[i] for i in keep] + ['intercept'], beta):
            print(f'      {nm:16s} {b:+.4f}')

    section('4. marginal history statistics (train split, all observed ticks)')
    for tag, root in (('R1', R1), ('R2', R2)):
        for split in ('train', 'validation', 'test'):
            h = history(root, split)
            allp = [p for ep in h.values() for m in ep.values() for p in m.values()]
            a = np.array(allp)
            cnt = np.array([(a == v).mean() for v in PAYOUTS])
            nmach = sum(len(ep) for ep in h.values())
            print(f'  {tag} {split:10s} rows={len(a):7d} machines={nmach:5d} '
                  f'win_rate={ (a>0).mean():.4f} payout_mix={cnt.round(4)}')

    section('5. observed-tick density and gap structure')
    for tag, root in (('R1', R1), ('R2', R2)):
        for split in ('validation',):
            h = history(root, split)
            qm = queries(root, split)
            ncounts, gaps = [], []
            for qid, m in qm.items():
                obs = h[m['episode_id']].get(m['machine_id'], {})
                ticks = sorted(t for t in obs if t < m['cutoff'])
                ncounts.append(len(ticks))
                if len(ticks) > 1:
                    gaps.extend(np.diff(ticks).tolist())
            print(f'  {tag} {split}: n_obs median {np.median(ncounts):.0f} '
                  f'pct[5,25,75,95] {np.percentile(ncounts,[5,25,75,95]).round(0)} '
                  f'gap mean {np.mean(gaps):.3f} P(gap>1) {np.mean(np.array(gaps)>1):.4f}')

    section('6. lag autocorrelation of win indicator, computed from HISTORY only')
    for tag, root in (('R1', R1), ('R2', R2)):
        h = history(root, 'train')
        for mt_filter in (None,):
            ac = np.zeros(9); n = np.zeros(9)
            for ep in h.values():
                for mid, obs in ep.items():
                    ticks = sorted(obs)
                    w = {t: (1.0 if obs[t] > 0 else 0.0) for t in ticks}
                    mean = np.mean(list(w.values()))
                    for t in ticks:
                        for k in range(1, 9):
                            if t - k in w:
                                ac[k] += (w[t] - mean) * (w[t - k] - mean)
                                n[k] += 1
            print(f'  {tag}: lag-k autocov/var ' +
                  '  '.join(f'{k}:{ac[k]/max(n[k],1):+.5f}' for k in range(1, 9)))

    section('7. condition composition and q by condition')
    for tag, root in (('R1', R1), ('R2', R2)):
        X, q, r, meta, names = D[tag]
        cond = conditions(root, 'validation')
        for c in ('stable', 'short_history', 'drift', 'shared_change'):
            sel = np.array([cond[m[0]] == c for m in meta])
            nobs = X[sel, names.index('n_obs')]
            print(f'  {tag} {c:14s} n={sel.sum():4d} mean q {q[sel].mean():.4f} '
                  f'sd q {q[sel].std():.4f} mean n_obs {nobs.mean():7.1f} '
                  f'mean r {r[sel].mean(axis=0).round(4)}')


if __name__ == '__main__':
    main()
