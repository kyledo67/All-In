"""Separable head stacking, then the structured tail projection, on archived forecasts only.

PREREGISTERED, fixed before any held-fold number below was read.

Two ideas, both from the stated protocol and both using only archived predictions:

1. SEPARABLE STACKING. The official query loss at a fixed conditional r is
       L(q) = (1 + ||r||^2) q^2 - 2 (u + r.t_+) q + const,
   so the five-vector loss is a quadratic in q for any given r, and the q and r heads are
   separable. Stacking five-vectors mixes both heads with one weight vector; stacking the
   heads separately gives each its own convex weights. Concretely:
       q_stack = convex combination of archived q heads
       r_stack = convex combination of archived r heads
       p       = (1 - q_stack, q_stack * r_stack)
   Both weight vectors are fitted on each outer fold's FITTING episodes against the exact
   official weighted squared error, alternating once (fit r given q, then q given r), with
   the pool size for each head chosen by inner complete-episode folds.

2. STRUCTURED PROJECTION OF THE STACKED OUTPUT. A convex combination of five-vectors need
   not keep the fixed type-specific tail split, even when every member has it, because the
   mixing weights interact with each member's win mass. Each final five-vector is therefore
   projected onto conv{e_0, e_1, (0,0,v_type)} by the exact Euclidean projection, with v
   cross-fitted from fitting-side episodes only. The preserve-a rescale is reported
   alongside, since it previously beat the exact projection at matched constants.

Pool: archived forecasts, EXCLUDING the privileged arrays and excluding this script's and
the stacking script's own outputs. Re-stacking on previous held-fold outputs is a
self-feedback loop that inflates the score on these same 96 episodes by construction; that
mistake was made once and is now blocked.

Comparator: the protected candidate at 53.532918, and the clean augmented stacking at
53.869175. Advancement requires points to improve AND the paired interval against the
protected forecast to lie below zero, applied per arm.

These 96 episodes underpin the whole project. Every number here is conditional
exploratory development, and the accumulated selection is not corrected by any interval.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
# DETERMINISM. Two runs of this script with an identical pool previously produced identical
# scores to about 1e-6 points but DIFFERENT selected pool sizes (fold 1 kq 12 vs 20, fold 3
# kq 30 vs 20) and byte-different output CSVs. There is no RNG here: simplex_fit starts from
# a fixed uniform point and every set iterated is a set of ints or is sorted. The cause is
# threaded BLAS, whose reduction order varies run to run, perturbing the matrix M in
# fit_heads at the 1e-16 level; argmin over M and argmin over the near-tied kq grid then
# flip discretely. Accelerate and OpenBLAS do not honour OMP_NUM_THREADS, so their own
# thread counts are pinned to 1 to make every reduction order fixed. This must be set
# before numpy is imported.
for _v in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.optimize import minimize

from slot_model.simplex_project import project_to_hull, tail_family_vertices
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.stack_manifest import RAW_SOURCES, classify

BLOCK = ('PRIVILEGED', 'privileged_envelope', 'true_r')
SELF_DIRS = ('artifacts/stacking/', 'artifacts/separable/')
# kr hit the previous maximum of 20 in 2 of 4 folds, so the grid is extended; the
# choice remains inside the inner complete-episode folds.
KGRID = (1, 2, 3, 5, 8, 12, 20, 30, 45)
PATTERNS = ('artifacts/*/validation_submission.csv', 'artifacts/*/predictions/*.csv',
            'artifacts/win_head/predictions/*.csv', 'artifacts/phase2/predictions/oof_*.csv',
            'artifacts/final/validation_predictions.csv', 'artifacts/two_head_01/validation.csv',
            'artifacts/mtd_pairs_01/validation_*.csv', 'artifacts/dense_catboost_01/validation_*.csv',
            'artifacts/shared_state_01/validation.csv', 'artifacts/tabpfn/predictions/*.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protected', type=Path,
                        default=ROOT / 'artifacts/win_head_release_exact/validation_submission.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/separable')
    parser.add_argument('--manifest', action='store_true',
                        help='build the pool from the explicit RAW manifest in '
                             'scripts/stack_manifest.py instead of directory globs. Only '
                             'candidates whose provenance is RAW (train-only fitting) are '
                             'admitted, so no member leaks an outer held fold through its '
                             'FITTING-side values.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    want = set(ids)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']
    vq, _ = load_inputs(args.data / 'validation')
    types = np.array([{q.query_id: q.machine_type for q in vq}[i] for i in ids])

    sources = sorted(RAW_SOURCES) if args.manifest else sorted(
        {f for pat in PATTERNS for f in glob.glob(pat)})
    seen, names, arrs, excluded = {}, [], [], []
    dup_q, dup_r, skipped = {}, {}, []
    for path in sources:
        if args.manifest:
            if classify(path) != 'RAW':
                excluded.append(path)
                continue
        elif any(t in path for t in BLOCK) or any(path.startswith(d) for d in SELF_DIRS):
            excluded.append(path)
            continue
        if not args.manifest and ('reference' in path or 'diagnostics' in path):
            continue
        try:
            d = read_submission(path)
        except Exception:
            skipped.append([path, 'unreadable'])
            continue
        if set(d) != want:
            skipped.append([path, 'misaligned query set'])
            continue
        a = np.array([d[i] for i in ids])
        if not (np.isfinite(a).all() and a.min() >= -1e-12 and abs(a.sum(1) - 1).max() < 1e-6):
            skipped.append([path, 'not on the simplex'])
            continue
        key = hashlib.sha256(a.round(12).tobytes()).hexdigest()
        if key in seen:
            skipped.append([path, f'duplicate five-vector of {seen[key]}'])
            continue
        seen[key] = path
        an = a / a.sum(axis=1, keepdims=True)
        qh = 1.0 - an[:, 0]
        rh = an[:, 1:] / np.maximum(qh, 1e-12)[:, None]
        rh = rh / rh.sum(axis=1, keepdims=True)
        hq = hashlib.sha256(qh.round(12).tobytes()).hexdigest()
        hr = hashlib.sha256(rh.round(12).tobytes()).hexdigest()
        dup_q.setdefault(hq, []).append(path)
        dup_r.setdefault(hr, []).append(path)
        names.append(path)
        arrs.append(an)
    dup_q_groups = [v for v in dup_q.values() if len(v) > 1]
    dup_r_groups = [v for v in dup_r.values() if len(v) > 1]
    P = np.stack(arrs)
    Q = 1.0 - P[:, :, 0]
    R = P[:, :, 1:] / np.maximum(Q, 1e-12)[:, :, None]
    R = R / R.sum(axis=2, keepdims=True)
    print(f'pool {len(names)} distinct archived forecasts; {len(excluded)} blocked; '
          f'{len(skipped)} skipped', flush=True)
    print(f'  head duplication: {len(dup_q_groups)} q-head groups and {len(dup_r_groups)} '
          f'r-head groups share an identical head, so they are not independent evidence',
          flush=True)

    prot = read_submission(args.protected)
    prot = np.array([prot[q] for q in ids])
    prot = prot / prot.sum(axis=1, keepdims=True)

    u = 1.0 - truth[:, 0]

    def row_weights(mask):
        w = np.zeros(len(ids))
        es = {x for i, x in enumerate(ep) if mask[i]}
        for c in sorted(set(groups.values())):
            ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == c}
            if not ces:
                continue
            b = np.mean([np.mean(np.sum((ref_arr[ep == e] - truth[ep == e]) ** 2, axis=1))
                         for e in sorted(ces)])
            for e in ces:
                w[ep == e] = 1 / (4 * len(ces) * sum(ep == e) * max(b, 1e-9))
        return w

    def compose(q, r):
        out = np.column_stack((1 - q, q[:, None] * r))
        return out / out.sum(axis=1, keepdims=True)

    def wloss_qr(q, r, w):
        return float(np.sum(w * np.sum((compose(q, r) - truth) ** 2, axis=1)))

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def official(a):
        a = a / a.sum(axis=1, keepdims=True)
        return score(mapping(a), scoring, episodes)

    def simplex_fit(obj, m):
        res = minimize(obj, np.ones(m) / m, method='SLSQP', bounds=[(0, 1)] * m,
                       constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1},
                       options={'ftol': 1e-14, 'maxiter': 800})
        x = np.clip(res.x, 0, None)
        return x / max(x.sum(), 1e-12)

    def fit_heads(mask, kq, kr):
        """Alternate once: pick r pool and weights given the best q, then q given that r."""
        w = row_weights(mask)
        A_ = 1.0 + np.sum(R ** 2, axis=2)
        Bc = u[None, :] + np.einsum('mnc,nc->mn', R, truth[:, 1:])
        M = ((Q ** 2) * w) @ A_.T - 2.0 * ((Q * w) @ Bc.T)
        iq0, ir0 = np.unravel_index(int(np.argmin(M)), M.shape)
        # r weights given q = Q[iq0]
        rl = np.array([wloss_qr(Q[iq0], R[j], w) for j in range(len(names))])
        ridx = np.argsort(rl)[:kr]
        def r_obj(x):
            xx = np.clip(x, 0, None)
            xx = xx / max(xx.sum(), 1e-12)
            rr = np.einsum('i,inc->nc', xx, R[ridx])
            rr = rr / rr.sum(axis=1, keepdims=True)
            return wloss_qr(Q[iq0], rr, w)
        xr = simplex_fit(r_obj, len(ridx))
        r_st = np.einsum('i,inc->nc', xr, R[ridx])
        r_st = r_st / r_st.sum(axis=1, keepdims=True)
        # q weights given r_st
        ql = np.array([wloss_qr(Q[i], r_st, w) for i in range(len(names))])
        qidx = np.argsort(ql)[:kq]
        def q_obj(x):
            xx = np.clip(x, 0, None)
            xx = xx / max(xx.sum(), 1e-12)
            return wloss_qr(np.einsum('i,in->n', xx, Q[qidx]), r_st, w)
        xq = simplex_fit(q_obj, len(qidx))
        q_st = np.einsum('i,in->n', xq, Q[qidx])
        # second alternation: refit r given the stacked q, then q again
        rl2 = np.array([wloss_qr(q_st, R[j], w) for j in range(len(names))])
        ridx = np.argsort(rl2)[:kr]
        def r_obj2(x):
            xx = np.clip(x, 0, None)
            xx = xx / max(xx.sum(), 1e-12)
            rr = np.einsum('i,inc->nc', xx, R[ridx])
            rr = rr / rr.sum(axis=1, keepdims=True)
            return wloss_qr(q_st, rr, w)
        xr = simplex_fit(r_obj2, len(ridx))
        r_st = np.einsum('i,inc->nc', xr, R[ridx])
        r_st = r_st / r_st.sum(axis=1, keepdims=True)
        ql2 = np.array([wloss_qr(Q[i], r_st, w) for i in range(len(names))])
        qidx = np.argsort(ql2)[:kq]
        def q_obj2(x):
            xx = np.clip(x, 0, None)
            xx = xx / max(xx.sum(), 1e-12)
            return wloss_qr(np.einsum('i,in->n', xx, Q[qidx]), r_st, w)
        xq = simplex_fit(q_obj2, len(qidx))
        q_st = np.einsum('i,in->n', xq, Q[qidx])
        return q_st, r_st, (qidx, xq, ridx, xr)

    held = {'separable': np.empty_like(prot), 'separable_projected': np.empty_like(prot),
            'separable_rescaled': np.empty_like(prot)}
    prov = []
    for fold in range(4):
        outer = np.array([folds[x] == fold for x in ep])
        fit = ~outer
        fit_eps = sorted({x for i, x in enumerate(ep) if fit[i]})
        inner = {e: i % 3 for i, e in enumerate(fit_eps)}
        best = None
        for kq in KGRID:
            for kr in KGRID:
                tot = 0.0
                for ib in range(3):
                    itr = np.array([fit[i] and inner.get(ep[i], -1) != ib for i in range(len(ids))])
                    ihe = np.array([fit[i] and inner.get(ep[i], -1) == ib for i in range(len(ids))])
                    q_s, r_s, _ = fit_heads(itr, kq, kr)
                    tot += wloss_qr(q_s, r_s, row_weights(ihe))
                if best is None or tot < best[0]:
                    best = (tot, kq, kr)
        _, kq, kr = best
        q_st, r_st, parts = fit_heads(fit, kq, kr)
        # cross-fitted tail constants from fitting episodes only
        r_true = truth[:, 1:] / np.maximum(truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)
        tails = {}
        for t in sorted(set(types.tolist())):
            m = fit & (types == t)
            tl = r_true[m][:, 1:]
            tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1e-15)
            tails[t] = tl.mean(axis=0)
        base = compose(q_st, r_st)
        proj = np.empty_like(base)
        resc = np.empty_like(base)
        for k in np.flatnonzero(outer):
            proj[k], _ = project_to_hull(base[k], tail_family_vertices(tails[types[k]]))
            qq = 1 - base[k, 0]
            a = base[k, 1] / max(qq, 1e-12)
            v = tails[types[k]] / tails[types[k]].sum()
            row = np.concatenate(([base[k, 0]], [qq * a], qq * (1 - a) * v))
            resc[k] = row / row.sum()
        held['separable'][outer] = base[outer]
        held['separable_projected'][outer] = proj[outer]
        held['separable_rescaled'][outer] = resc[outer]
        # which coordinates does the projection actually move, and by how much
        om = np.flatnonzero(outer)
        dlt = np.abs(proj[om] - base[om])
        active = dlt > 1e-9
        qidx, xq, ridx, xr = parts
        prov.append({'fold': fold, 'kq': kq, 'kr': kr,
                     'projection_audit': {
                         'rows_changed': int(np.any(active, axis=1).sum()),
                         'rows_held': int(len(om)),
                         'per_coordinate_rows_changed':
                             dict(zip(('p0', 'p1', 'p2', 'p5', 'p20'),
                                      active.sum(axis=0).tolist())),
                         'per_coordinate_mean_abs_change':
                             dict(zip(('p0', 'p1', 'p2', 'p5', 'p20'),
                                      dlt.mean(axis=0).tolist())),
                         'per_coordinate_max_abs_change':
                             dict(zip(('p0', 'p1', 'p2', 'p5', 'p20'),
                                      dlt.max(axis=0).tolist())),
                         'max_abs_change_any_coordinate': float(dlt.max()),
                         'mean_total_variation': float(dlt.sum(axis=1).mean() / 2)},
                     'q_members': [names[i] for i in qidx], 'q_weights': xq.tolist(),
                     'r_members': [names[i] for i in ridx], 'r_weights': xr.tolist(),
                     'tails': {t: v.tolist() for t, v in tails.items()}})
        print(f'  fold {fold}: kq={kq} kr={kr}', flush=True)

    res = {'protected': official(prot), **{k: official(v) for k, v in held.items()}}
    pairs = {k: paired_bootstrap(mapping(v / v.sum(axis=1, keepdims=True)), mapping(prot),
                                 scoring, episodes) for k, v in held.items()}
    qualified = {k: res[k]['technical_points'] - res['protected']['technical_points']
                 for k in held
                 if res[k]['technical_points'] > res['protected']['technical_points']
                 and pairs[k]['delta_loss_ratio_ci95'][1] < 0}

    report = {'protocol': __doc__, 'pool_size': len(names),
              'pool_mode': 'explicit RAW manifest' if args.manifest else 'directory globs',
              'pool_members': names,
              'pool_excluded_by_provenance': excluded,
              'pool_skipped': skipped,
              'duplicate_q_head_groups': dup_q_groups,
              'duplicate_r_head_groups': dup_r_groups,
              'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                              'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                          for k, v in res.items()},
              'gains_over_protected': {k: res[k]['technical_points'] - res['protected']['technical_points']
                                       for k in held},
              'paired_vs_protected': {k: v['delta_loss_ratio_ci95'] for k, v in pairs.items()},
              'qualified_arms': qualified, 'fold_provenance': prov,
              'exposure': 'Conditional exploratory development on 96 repeatedly reused episodes.',
              'seconds': time.perf_counter() - start}
    for k, v in held.items():
        write_submission(args.out / 'predictions' / f'oof_{k}.csv',
                         mapping(v / v.sum(axis=1, keepdims=True)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(res['protected']['by_condition'])
    print(f"\n{'arm':<26}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('protected', 'separable', 'separable_rescaled', 'separable_projected'):
        v = res[k]
        print(f"{k:<26}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print()
    for k, ci in report['paired_vs_protected'].items():
        print(f"   {k:<22} dR 95% [{ci[0]:+.5f}, {ci[1]:+.5f}]"
              + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    print(f"\nqualified arms: { {k: round(v, 6) for k, v in qualified.items()} }")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
