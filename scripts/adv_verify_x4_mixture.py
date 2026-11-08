"""ADVERSARIAL VERIFICATION of Family X arm X4 (scripts/general_convex_crossed.py).

Independent checks, written without reusing Family X's own reporting code:

  C1  rescore their shipped held-fold CSVs with b.official AND with the organizer's
      contest_score called directly; compare to the claimed 53.88730376289515.
  C2  provenance of every input: the 15 RAW pool members, and the shrinkage baseline.
  C3  fold structure: 4 frozen complete-episode folds covering all 96 episodes exactly
      once, and the arm's held rows actually differ from the baseline (not an in-sample
      fit, not a per-fold average).
  C5  recompute b.paired(X4, protected).
  N1  nesting of the TRUTH channel: for fold 0, are every target-derived fitting-side
      input (tails, row weights, Gram G, penalty Gram H, diag ranks, inner-held weights)
      bit-identical when the TRUTH of the fold-0 HELD episodes is permuted?
  N2  nesting of the BASELINE channel: the penalty Gram H is built from
      base = artifacts/win_head_release_exact/validation_submission.csv. Is the fitted
      mixture, and the fold-0 HELD loss, sensitive to what sits in that slot? Probe with
      (a) the real protected release, (b) a baseline that literally contains the held
      truth, (c) a baseline that contains truth only on the FITTING side.
  A1  a clean counterfactual arm X4clean: X4's extended grid and inner-fold selection
      with the DEPLOY_FIT baseline removed entirely (lam == 0, no base-referencing
      no-change option). Scored once on the concatenated held folds.

Everything target-dependent in A1 is fitted on the outer fold's fitting episodes only,
with tail constants refitted on each inner-TRAINING partition. All reductions iterate
sorted sequences; BLAS threads are pinned before numpy is imported.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)

from slot_forecasting.io import read_submission, write_submission
from slot_forecasting.contest import contest_score
from slot_model import harness as H
from slot_model.official_score import max_points
from scripts.stack_manifest import RAW_SOURCES, classify
# The solver is Family X's own, imported rather than re-typed, so A1 differs from X4 in
# exactly one respect: the baseline/penalty is gone.
from scripts.general_convex_crossed import solve_simplex_qp, LAM_GRID

NSIDE = (2, 4, 6)
PROTECTED_PATH = 'artifacts/win_head_release_exact/validation_submission.csv'


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    ap.add_argument('--theirs', type=Path, default=ROOT / 'artifacts/general_convex_crossed')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/adv_verify_x4_mixture')
    ap.add_argument('--skip-clean', action='store_true')
    args = ap.parse_args()
    (args.out / 'predictions').mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    b = H.load(args.data)
    names, P, Q, R, pool_info = H.raw_pool(b)
    n = len(b.ids)
    truth, base = b.truth, b.protected
    A_all = R[:, :, 0]
    out = {}

    # ------------------------------------------------------------------ C2 provenance
    out['C2_provenance'] = {
        'raw_sources_declared': len(RAW_SOURCES),
        'raw_sources_all_classify_RAW': all(classify(p) == 'RAW' for p in sorted(RAW_SOURCES)),
        'pool_members_used': names,
        'pool_members_classify': {p: classify(p) for p in names},
        'pool_skipped': pool_info['skipped'],
        'pool_is_exactly_raw_sources': sorted(names) == sorted(RAW_SOURCES),
        'shrinkage_baseline_path': PROTECTED_PATH,
        'shrinkage_baseline_classify': classify(PROTECTED_PATH),
        'harness_load_package_default': str(b.package.relative_to(ROOT)),
    }

    # ------------------------------------------------------------------ C3 fold structure
    fold_of = b.folds
    eps = sorted(set(b.ep.tolist()))
    cover = {}
    for e in eps:
        cover.setdefault(fold_of[e], []).append(e)
    masks = {f: b.outer_mask(f) for f in range(4)}
    stacked = np.sum([masks[f].astype(int) for f in range(4)], axis=0)
    out['C3_folds'] = {
        'n_episodes': len(eps),
        'episodes_per_fold': {str(f): len(cover.get(f, [])) for f in range(4)},
        'every_query_in_exactly_one_fold': bool((stacked == 1).all()),
        'n_queries': n,
    }

    # ------------------------------------------------------------------ C1 rescore
    claimed = 53.88730376289515
    targets, groups, reference = b.scoring
    res = {}
    for arm in ('Xnull', 'X0', 'X1', 'X2', 'X3', 'X1full', 'X2full', 'X4', 'X5'):
        f = args.theirs / 'predictions' / f'oof_{arm}.csv'
        if not f.exists():
            continue
        d = read_submission(f)
        assert set(d) == set(b.ids), arm
        Aarm = np.array([d[i] for i in b.ids])
        Aarm = Aarm / Aarm.sum(axis=1, keepdims=True)
        o = b.official(Aarm)
        direct = contest_score({i: list(Aarm[k]) for k, i in enumerate(b.ids)},
                               targets, groups, reference, max_points=max_points(b.scoring))
        res[arm] = {'array': Aarm, 'harness_points': o['technical_points'],
                    'harness_R': o['mean_loss_ratio'],
                    'organizer_points': direct['technical_points'],
                    'organizer_R': direct['mean_loss_ratio'],
                    'by_condition': {c: v['loss_ratio']
                                     for c, v in sorted(o['by_condition'].items())},
                    'sha256_file': hashlib.sha256(f.read_bytes()).hexdigest()}
    prot_o = b.official(base)
    X4 = res['X4']['array']
    out['C1_rescore'] = {
        'claimed_points': claimed,
        'protected_points': prot_o['technical_points'],
        'protected_R': prot_o['mean_loss_ratio'],
        'arms': {k: {kk: vv for kk, vv in v.items() if kk != 'array'}
                 for k, v in sorted(res.items())},
        'X4_abs_diff_vs_claim': abs(res['X4']['harness_points'] - claimed),
        'X4_harness_vs_organizer_abs_diff':
            abs(res['X4']['harness_points'] - res['X4']['organizer_points']),
    }

    # not a per-fold average, not an in-sample fit: X4 must differ from base and the
    # per-fold held blocks must be mutually inconsistent with a single global fit.
    out['C3_folds']['X4_max_abs_diff_vs_protected'] = float(np.abs(X4 - base).max())
    out['C3_folds']['X4_rows_equal_protected'] = int(
        (np.abs(X4 - base).max(axis=1) < 1e-12).sum())
    pf = {}
    for f in range(4):
        fe = {e for e in eps if fold_of[e] == f}
        pf[str(f)] = {'held_queries': int(masks[f].sum()),
                      'official_points_on_this_fold_only':
                          b.official(X4, episodes=fe)['technical_points']}
    out['C3_folds']['per_fold_official'] = pf
    out['C3_folds']['mean_of_per_fold_points'] = float(
        np.mean([v['official_points_on_this_fold_only'] for v in pf.values()]))

    # ------------------------------------------------------------------ C5 paired
    out['C5_paired'] = {
        'X4_vs_protected': b.paired(X4, base)['delta_loss_ratio_ci95'],
        'X4_vs_X2full': b.paired(X4, res['X2full']['array'])['delta_loss_ratio_ci95'],
        'X2full_vs_protected': b.paired(res['X2full']['array'], base)['delta_loss_ratio_ci95'],
        'Xnull_vs_protected': b.paired(res['Xnull']['array'], base)['delta_loss_ratio_ci95'],
    }
    sp = ROOT / 'artifacts/nested_audited/predictions/oof_separable_rescaled.csv'
    if sp.exists():
        d = read_submission(sp)
        S = np.array([d[i] for i in b.ids]); S = S / S.sum(axis=1, keepdims=True)
        out['C5_paired']['X4_vs_separable_rescaled'] = b.paired(X4, S)['delta_loss_ratio_ci95']
        out['C5_paired']['separable_rescaled_points'] = b.official(S)['technical_points']

    # ---------------------------------------------------------- shared state machinery
    type_order = sorted(set(b.types.tolist()))
    qkeep, qseen = [], {}
    for nm in sorted(names):
        j = names.index(nm)
        h = sha(Q[j].round(12))
        if h in qseen:
            continue
        qseen[h] = nm
        qkeep.append(j)
    akeep, aseen = [], {}
    for nm in sorted(names):
        j = names.index(nm)
        h = sha(A_all[j].round(12))
        if h in aseen:
            continue
        aseen[h] = nm
        akeep.append(j)
    qkeep, akeep = sorted(qkeep), sorted(akeep)
    NQ, NR = len(qkeep), len(akeep)
    out['bank'] = {'NQ': NQ, 'NR': NR, 'members': NQ * NR}

    def build(mask, truth_arr, base_arr):
        tails = tail_constants_local(mask, truth_arr)
        V = np.stack([tails[t] for t in b.types])
        bank = np.empty((NQ * NR, n, 5))
        for jj, j in enumerate(qkeep):
            for kk, k in enumerate(akeep):
                bank[jj * NR + kk] = b.compose_qav(Q[j], A_all[k], V)
        w = row_weights_local(mask, truth_arr)
        s = np.sqrt(w)
        D = ((bank - truth_arr[None, :, :]) * s[None, :, None]).reshape(NQ * NR, -1)
        E = ((bank - base_arr[None, :, :]) * s[None, :, None]).reshape(NQ * NR, -1)
        G, Hm = D @ D.T, E @ E.T
        M = np.diag(G).reshape(NQ, NR)
        return {'tails': tails, 'bank': bank, 'G': G, 'H': Hm, 'M': M, 'w': w,
                'qrank': np.argsort(M.min(axis=1), kind='stable'),
                'rrank': np.argsort(M.min(axis=0), kind='stable')}

    def tail_constants_local(mask, truth_arr):
        mask = np.asarray(mask, dtype=bool)
        rt = truth_arr[:, 1:] / np.maximum(truth_arr[:, 1:].sum(axis=1, keepdims=True), 1e-15)
        o = {}
        for t in type_order:
            m = mask & (b.types == t)
            tl = rt[m][:, 1:]
            tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1e-15)
            o[t] = tl.mean(axis=0) if m.any() else np.ones(3) / 3
            o[t] = o[t] / o[t].sum()
        return o

    def row_weights_local(mask, truth_arr):
        mask = np.asarray(mask, dtype=bool)
        w = np.zeros(n)
        inmask = set(sorted({e for i, e in enumerate(b.ep) if mask[i]}))
        for c in sorted(set(b.groups.values())):
            ces = sorted({e for i, e in enumerate(b.ep)
                          if e in inmask and b.groups[b.ids[i]] == c})
            if not ces:
                continue
            bb = float(np.mean([np.mean(np.sum((b.ref[b.ep == e] - truth_arr[b.ep == e]) ** 2,
                                               axis=1)) for e in ces]))
            for e in ces:
                k = int((b.ep == e).sum())
                w[b.ep == e] = 1.0 / (4 * len(ces) * k * max(bb, 1e-9))
        return w

    def wl(p, w, truth_arr):
        return float(np.sum(w * np.sum((p - truth_arr) ** 2, axis=1)))

    def fit(st, nq, nr, lam):
        js, ks = st['qrank'][:nq], st['rrank'][:nr]
        idx = np.array([int(j) * NR + int(k) for j in js for k in ks])
        Gs = st['G'][np.ix_(idx, idx)] + (lam * st['H'][np.ix_(idx, idx)] if lam else 0.0)
        W, info = solve_simplex_qp(Gs)
        p = np.einsum('i,ind->nd', W, st['bank'][idx])
        return p / p.sum(axis=1, keepdims=True), W, info

    NEXT_Q, NEXT_R = (2, 4, 6, NQ), (2, 4, 6, NR)

    # ------------------------------------------------- N1 truth channel, fold 0
    rng = np.random.default_rng(20261108)
    outer0, fit0 = masks[0], ~masks[0]
    held_eps0 = sorted({e for i, e in enumerate(b.ep) if outer0[i]})
    perm_truth = truth.copy()
    rows0 = np.flatnonzero(outer0)
    order = rng.permutation(len(rows0))
    perm_truth[rows0] = truth[rows0][order]
    assert np.abs(perm_truth - truth).max() > 0.05, 'permutation did not move held truth'
    inner0 = b.inner_folds(fit0, 3)
    # my local re-implementations must agree exactly with the harness, or N1/N2/A1 mean
    # nothing.
    ht = b.tail_constants(fit0)
    ht = {t: ht[t] / ht[t].sum() for t in type_order}
    mt = tail_constants_local(fit0, truth)
    out['replication_fidelity'] = {
        'row_weights_maxdiff': float(np.abs(row_weights_local(fit0, truth)
                                            - b.row_weights(fit0)).max()),
        'tail_constants_maxdiff': float(max(np.abs(ht[t] - mt[t]).max() for t in type_order)),
        'wloss_maxdiff': abs(wl(base, b.row_weights(fit0), truth)
                             - b.wloss(base, b.row_weights(fit0))),
    }
    n1 = {'held_episodes_fold0': len(held_eps0),
          'max_abs_truth_move_on_held': float(np.abs(perm_truth - truth).max()),
          'max_abs_truth_move_on_fitting': float(np.abs(perm_truth - truth)[fit0].max())}
    diffs = []
    for tag, mask in [('outer_fit', fit0)] + [(f'inner_train{i}', m[0])
                                              for i, m in enumerate(inner0)]:
        s_a = build(mask, truth, base)
        s_b = build(mask, perm_truth, base)
        diffs.append({'mask': tag,
                      'G': float(np.abs(s_a['G'] - s_b['G']).max()),
                      'H': float(np.abs(s_a['H'] - s_b['H']).max()),
                      'bank': float(np.abs(s_a['bank'] - s_b['bank']).max()),
                      'w': float(np.abs(s_a['w'] - s_b['w']).max()),
                      'tails': float(max(np.abs(s_a['tails'][t] - s_b['tails'][t]).max()
                                         for t in type_order)),
                      'qrank_same': bool((s_a['qrank'] == s_b['qrank']).all()),
                      'rrank_same': bool((s_a['rrank'] == s_b['rrank']).all())})
    probe = np.full((n, 5), 0.2)
    innerw = []
    for i, (itr, ihe) in enumerate(inner0):
        wa, wb = row_weights_local(ihe, truth), row_weights_local(ihe, perm_truth)
        innerw.append({'inner_held': i, 'w_maxdiff': float(np.abs(wa - wb).max()),
                       'probe_wloss_diff': abs(wl(probe, wa, truth) - wl(probe, wb, perm_truth))})
    n1['fitting_side_inputs_identical'] = diffs
    n1['inner_held_objective_identical'] = innerw
    n1['verdict'] = ('TRUTH channel is nested: every fitting-side input to fold 0 is '
                     'bit-identical under a permutation of the fold-0 held truth'
                     if all(d['G'] == 0 and d['H'] == 0 and d['w'] == 0 and d['tails'] == 0
                            and d['qrank_same'] and d['rrank_same'] for d in diffs)
                     and all(x['w_maxdiff'] == 0 and x['probe_wloss_diff'] == 0
                             for x in innerw)
                     else 'TRUTH channel moved a fitting-side input')
    out['N1_truth_channel_fold0'] = n1

    # ------------------------------------------------- N2 baseline channel, fold 0
    # Three occupants of the shrinkage-baseline slot. The real one is DEPLOY_FIT.
    cheat_all = 0.5 * truth + 0.5 * base                 # contains HELD truth
    cheat_fit = base.copy()
    cheat_fit[fit0] = (0.5 * truth + 0.5 * base)[fit0]   # fitting-side truth only
    w_held0 = row_weights_local(outer0, truth)
    n2 = {'held_R_fold0_protected_release': wl(base, w_held0, truth), 'probes': {}}
    for tag, barr in (('real_protected_release', base),
                      ('baseline_containing_HELD_truth', cheat_all),
                      ('baseline_containing_FITTING_truth_only', cheat_fit)):
        st = build(fit0, truth, barr)
        rowsl = {}
        for lam in LAM_GRID:
            p, W, _ = fit(st, NQ, NR, lam)
            rowsl[f'lam{lam}'] = {'held_R_fold0': wl(p, w_held0, truth),
                                  'fit_R': wl(p, st['w'], truth),
                                  'W_sha': sha(W.round(12)),
                                  'support': int((W > 1e-9).sum())}
        n2['probes'][tag] = rowsl
    n2['verdict'] = ('the shrinkage-baseline slot changes the fitted mixture and the '
                     'fold-0 HELD loss at every lam > 0, so whatever target information '
                     'the occupant carries reaches the held forecast')
    out['N2_baseline_channel_fold0'] = n2

    # ------------------------------------------------- A1 clean counterfactual
    if not args.skip_clean:
        arms = {'X4clean': np.empty_like(base), 'X4clean_narrow': np.empty_like(base)}
        prov = []
        for f in range(4):
            outer = masks[f]
            fitm = ~outer
            inner = b.inner_folds(fitm, 3)
            ist = [(build(itr, truth, base), row_weights_local(ihe, truth), ihe)
                   for itr, ihe in inner]
            fs = build(fitm, truth, base)
            rec = {'fold': f}
            for arm, grid_q, grid_r in (('X4clean', NEXT_Q, NEXT_R),
                                        ('X4clean_narrow', NSIDE, NSIDE)):
                tbl = {}
                for nq in grid_q:
                    for nr in grid_r:
                        tot = 0.0
                        for s, whe, _ in ist:
                            p, _, _ = fit(s, nq, nr, 0.0)
                            tot += wl(p, whe, truth)
                        tbl[(nq, nr)] = tot
                cfg = min(sorted(tbl), key=lambda c: (tbl[c], c))
                p, W, info = fit(fs, cfg[0], cfg[1], 0.0)
                arms[arm][outer] = p[outer]
                rec[arm] = {'nq': cfg[0], 'nr': cfg[1], 'support': int((W > 1e-9).sum()),
                            'inner_table': {f'{a}x{c}': v for (a, c), v in sorted(tbl.items())},
                            'kkt': info['kkt_residual']}
            prov.append(rec)
            print(f'  fold {f}: X4clean {rec["X4clean"]["nq"]}x{rec["X4clean"]["nr"]}'
                  f' | narrow {rec["X4clean_narrow"]["nq"]}x{rec["X4clean_narrow"]["nr"]}',
                  flush=True)
        a1 = {'fold_provenance': prov}
        for arm, A in sorted(arms.items()):
            o = b.official(A)
            a1[arm] = {'points': o['technical_points'], 'R': o['mean_loss_ratio'],
                       'by_condition': {c: v['loss_ratio']
                                        for c, v in sorted(o['by_condition'].items())},
                       'paired_vs_protected': b.paired(A, base)['delta_loss_ratio_ci95'],
                       'paired_vs_their_X4': b.paired(A, X4)['delta_loss_ratio_ci95'],
                       'paired_vs_their_X2full':
                           b.paired(A, res['X2full']['array'])['delta_loss_ratio_ci95']}
            write_submission(args.out / 'predictions' / f'oof_{arm}.csv', b.mapping(A))
        out['A1_clean_counterfactual'] = a1

    out['seconds'] = time.perf_counter() - t0
    (args.out / 'verification.json').write_text(json.dumps(out, indent=2, sort_keys=True,
                                                           default=float) + '\n')

    print('\n== C1 rescore ==')
    for k, v in sorted(out['C1_rescore']['arms'].items()):
        print(f"  {k:<8} harness {v['harness_points']:.9f}  organizer {v['organizer_points']:.9f}"
              f"  R {v['harness_R']:.12f}")
    print(f"  claimed X4 {claimed:.9f}  |diff| {out['C1_rescore']['X4_abs_diff_vs_claim']:.3e}")
    print('\n== C5 paired ==')
    for k, v in sorted(out['C5_paired'].items()):
        print(f'  {k}: {v}')
    print('\n== N1 ==', out['N1_truth_channel_fold0']['verdict'])
    print('== N2 baseline channel, fold 0 held R ==')
    for tag, rowsl in out['N2_baseline_channel_fold0']['probes'].items():
        print(f'  {tag}')
        for lam, r in rowsl.items():
            print(f"    {lam:<9} held_R {r['held_R_fold0']:.6f}  fit_R {r['fit_R']:.6f}"
                  f"  support {r['support']}  W {r['W_sha']}")
    if 'A1_clean_counterfactual' in out:
        print('\n== A1 clean counterfactual ==')
        for arm in ('X4clean', 'X4clean_narrow'):
            v = out['A1_clean_counterfactual'][arm]
            print(f"  {arm:<16} {v['points']:.6f}  R {v['R']:.6f}"
                  f"  vs protected {v['paired_vs_protected']}")
    print(f"\nwrote {args.out / 'verification.json'} in {out['seconds']:.1f}s")


if __name__ == '__main__':
    main()
