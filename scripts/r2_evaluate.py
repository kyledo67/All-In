"""Phase 5: the full evaluation record for the Round-2 candidate.

Reported at both point scales, because the two released kits disagree: the adaptation kit
on disk is `temporal-adaptation-v2` with adaptation = 20 points, while the main kit we were
issued is `temporal-adaptation-v3-60-40` and its RULES.md states adaptation = 40. R is
identical either way; only the multiplier differs.

The deployed arm's alphas were fitted on all 96 validation episodes, so its score on those
episodes is a replay, not an estimate. The out-of-fold column is the figure to believe.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ayan/Downloads/All-In')
import numpy as np

from scripts.r2_common import (R2, CONDITIONS, targets, queries, history, conditions,
                               read_sub, score, reference_episode_errors, episode_errors,
                               paired_bootstrap)
from scripts.r2_features import build
from scripts.r2_qmodel import split_qr, folds_by_condition, SP

OUT = Path('/Users/ayan/Downloads/All-In/artifacts/r2_release')


def main():
    tg = targets(R2)
    qm = queries(R2, 'validation')
    cond = conditions(R2, 'validation')
    ids = sorted(tg)
    refep = reference_episode_errors(R2)
    A = np.array([tg[i] for i in ids])
    qt = 1 - A[:, 0]
    rt = A[:, 1:] / np.maximum(qt, 1e-12)[:, None]
    _, X, names = build(qm, history(R2, 'validation'), ids)
    nobs = X[:, names.index('n_obs')] * 250
    fold, _ = folds_by_condition(ids, qm, cond)
    eps = np.array([qm[i]['episode_id'] for i in ids])

    frozen = read_sub(Path(SP) / 'frozen_validation.csv')
    cand = read_sub(OUT / 'validation_predictions.csv')
    q_local, q_ref, _ = np.load(Path(SP) / 'q_parts.npy')
    _, rf = split_qr(frozen, ids)

    # the honest out-of-fold version of the deployed recipe
    grid = np.arange(0, 1.001, 0.01)

    def qobj(qh, sel):
        p = np.column_stack([1 - qh[sel], qh[sel][:, None] * rt[sel]])
        t = np.column_stack([1 - qt[sel], qt[sel][:, None] * rt[sel]])
        return float(((p - t) ** 2).sum(axis=1).sum())

    oof = np.zeros(len(ids))
    per_fold_alpha = []
    for f in range(4):
        te = fold == f
        tr = ~te
        al = min(grid, key=lambda a: qobj((1 - a) * q_ref + a * q_local, tr & (nobs >= 128)))
        ash = min(grid, key=lambda a: qobj((1 - a) * q_ref + a * q_local, tr & (nobs < 128)))
        av = np.where(nobs < 128, ash, al)
        oof[te] = ((1 - av) * q_ref + av * q_local)[te]
        per_fold_alpha.append({'fold': f, 'alpha_long': round(float(al), 3),
                               'alpha_short': round(float(ash), 3)})
    oof_pred = {i: np.concatenate([[1 - oof[k]], oof[k] * rf[k]]) for k, i in enumerate(ids)}

    rec = {'kit': 'adaptation-participants', 'split': 'validation',
           'note': 'R is scale-free; points shown at both released allocations',
           'per_fold_alpha_refit': per_fold_alpha,
           'deployed_alpha': {'long': 0.94, 'short': 0.83, 'min_observations': 128}}

    print('=== headline ===')
    for name, pred, kind in (('pooled reference', read_sub(R2 / 'validation/reference.csv'), 'frozen ref'),
                             ('Round-1 entry unchanged', frozen, 'no fitting on R2'),
                             ('Round-2 candidate, OUT-OF-FOLD', oof_pred, 'honest estimate'),
                             ('Round-2 candidate, as deployed', cand, 'replay: alphas saw these episodes')):
        s = score(pred, tg, qm, cond, refep)
        print(f'  {name:34s} R={s["R"]:.5f}  {s["points_20"]:7.4f}/20  {s["points_40"]:7.4f}/40'
              f'  [{kind}]')
        print(f'{"":36s} ratios ' + '  '.join(f'{c}={s["by_condition"][c]["loss_ratio"]:.4f}'
                                              for c in CONDITIONS))
        rec[name] = {'R': s['R'], 'points_20': s['points_20'], 'points_40': s['points_40'],
                     'by_condition': {c: s['by_condition'][c] for c in CONDITIONS},
                     'basis': kind}

    print('\n=== fold stability (out-of-fold recipe, scored within each fold) ===')
    fs = []
    for f in range(4):
        sel = fold == f
        sub_ids = [i for k, i in enumerate(ids) if sel[k]]
        t2 = {i: tg[i] for i in sub_ids}
        c2 = {i: cond[i] for i in sub_ids}
        sA = score({i: frozen[i] for i in sub_ids}, t2, qm, c2, refep)
        sB = score({i: oof_pred[i] for i in sub_ids}, t2, qm, c2, refep)
        print(f'  fold {f}: 24 episodes  frozen R={sA["R"]:.5f}  candidate R={sB["R"]:.5f}'
              f'  dR={sB["R"]-sA["R"]:+.5f}')
        fs.append({'fold': f, 'R_frozen': sA['R'], 'R_candidate': sB['R'],
                   'dR': sB['R'] - sA['R']})
    rec['fold_stability'] = fs
    print(f'  dR across folds: mean {np.mean([x["dR"] for x in fs]):+.5f}'
          f'  sd {np.std([x["dR"] for x in fs]):.5f}  all four negative:'
          f' {all(x["dR"] < 0 for x in fs)}')

    print('\n=== paired episode-level bootstrap, candidate(out-of-fold) vs frozen ===')
    d = paired_bootstrap(frozen, oof_pred, repeats=4000, seed=20261108)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'  dR mean {d.mean():+.5f}  95% [{lo:+.5f}, {hi:+.5f}]  P(dR<0)={np.mean(d<0):.4f}'
          f'  {"excludes zero" if hi < 0 else "overlaps zero"}')
    rec['bootstrap'] = {'repeats': 4000, 'mean_dR': float(d.mean()), 'ci95': [float(lo), float(hi)],
                        'p_improve': float(np.mean(d < 0)), 'excludes_zero': bool(hi < 0)}

    epA = episode_errors(frozen, tg, qm)
    epB = episode_errors(oof_pred, tg, qm)
    diffs = np.array([epB[e] - epA[e] for e in sorted(epA)])
    print(f'\n=== paired episode improvements (96 episodes) ===')
    print(f'  candidate better on {int((diffs < 0).sum())}/96 episodes, worse on'
          f' {int((diffs > 0).sum())}; mean dE {diffs.mean():+.6f}'
          f'  median {np.median(diffs):+.6f}  worst regression {diffs.max():+.6f}')
    rec['episode_paired'] = {'better': int((diffs < 0).sum()), 'worse': int((diffs > 0).sum()),
                             'mean_dE': float(diffs.mean()), 'max_regression': float(diffs.max())}

    print('\n=== short histories and changing regimes ===')
    cnd = np.array([cond[i] for i in ids])
    for lbl, sel in (('short history (n_obs=64)', nobs < 128),
                     ('long history', nobs >= 128),
                     ('drift episodes', cnd == 'drift'),
                     ('shared_change episodes', cnd == 'shared_change'),
                     ('stable episodes', cnd == 'stable')):
        eA = qobj(1 - np.array([frozen[i] for i in ids])[:, 0], sel) / max(sel.sum(), 1)
        eB = qobj(oof, sel) / max(sel.sum(), 1)
        print(f'  {lbl:26s} n={int(sel.sum()):4d}  mean sq prob error  frozen {eA:.6f}'
              f' -> candidate {eB:.6f}  ({100*(eB-eA)/eA:+.2f}%)')
        rec.setdefault('subgroups', {})[lbl] = {'n': int(sel.sum()), 'frozen': eA, 'candidate': eB}

    print('\n=== calibration of the full payout distribution (candidate, out-of-fold) ===')
    P = np.array([oof_pred[i] for i in ids])
    T = A
    print('  category    mean forecast   mean true   bias      rmse')
    for k, lab in enumerate(('p_0', 'p_1', 'p_2', 'p_5', 'p_20')):
        print(f'  {lab:10s} {P[:,k].mean():12.5f} {T[:,k].mean():11.5f}'
              f' {P[:,k].mean()-T[:,k].mean():+9.5f} {np.sqrt(((P[:,k]-T[:,k])**2).mean()):9.5f}')
        rec.setdefault('calibration', {})[lab] = {
            'mean_forecast': float(P[:, k].mean()), 'mean_true': float(T[:, k].mean()),
            'bias': float(P[:, k].mean() - T[:, k].mean()),
            'rmse': float(np.sqrt(((P[:, k] - T[:, k]) ** 2).mean()))}
    print('  win-probability reliability, out-of-fold:')
    qh = 1 - P[:, 0]
    for lo_, hi_ in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        s = (qh >= lo_) & (qh < hi_)
        if s.sum():
            print(f'    q in [{lo_:.1f},{hi_:.1f}) n={int(s.sum()):4d} forecast {qh[s].mean():.4f}'
                  f'  true {qt[s].mean():.4f}  bias {qh[s].mean()-qt[s].mean():+.4f}')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evaluation.json').write_text(json.dumps(rec, indent=2, sort_keys=True) + '\n')
    print(f'\nwrote {OUT / "evaluation.json"}')


if __name__ == '__main__':
    main()
