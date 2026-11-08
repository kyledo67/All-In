"""Family G: fit and select against the probability vector actually DELIVERED.

scripts/separable_stack.py fits its head weights and chooses its inner-fold settings against
the UNPROJECTED q/r composition, and only afterwards applies the tail operator. If the
delivered vector is the rescaled one, then fitting and selecting against the unprojected
composition optimises the wrong objective. This family tests whether aligning the stages
helps, and separates the two possible misalignments.

PARAMETERISATION. With q = P(positive), a = P(1x | positive) and a per-type tail vector v
over {2,5,20} summing to 1,
    p(q,a,v) = [1-q, q a, q(1-a) v2, q(1-a) v5, q(1-a) v20].
Writing p = e + q g with e = (1,0,0,0,0) and g = (-1, a, (1-a)v2, (1-a)v5, (1-a)v20),
    ||p - t||^2 = ||e-t||^2 + 2 q g.(e-t) + q^2 ||g||^2,
so at fixed (a, v) the official weighted loss is an exact convex quadratic in q, and
symmetrically it is an exact convex quadratic in a at fixed q. Both stacked heads are convex
combinations, so each alternation step is a simplex-constrained convex QP.

THE JOINT PROBLEM IS NOT CONVEX. The delivered vector contains the products q*a and
q*(1-a)*v, so the joint objective over (q weights, a weights) is not jointly convex and
alternation is not guaranteed to find a global optimum. What alternation does guarantee is a
monotone non-increasing objective, since each step solves its own subproblem exactly. Starts
are a small deterministic set, the alternation count is fixed in advance, and the per-step
objective sequence and solver feasibility are recorded.

ARMS, matched on pool, folds, shortlist budgets and numerical safeguards:
  G0  fit and select against the UNPROJECTED composition, then apply the declared rescale.
      This reproduces the current procedure's objective alignment.
  G1  identical fitting procedure to G0, but choose the inner-fold settings using the FINAL
      rescaled five-vector score.
  G2  fit the q and a weights AND choose settings using the final rescaled score throughout.
Interpretation: G1 - G0 isolates misalignment of SELECTION; G2 - G1 isolates misalignment of
FITTING. Confounding them would not identify which stage matters.

The brief's instruction is honoured: start with the rescaled formulation only. A
projection-aware variant is deliberately NOT included here, because the exact Euclidean
projection moves p0 and p1 by a shared multiplier (verified: dp0 = dp1 to 2.9e-16), so the
delivered operator is not a simple reparameterisation of (q, a, v) and would need the
optimizer to evaluate the projection itself.

NESTING. Tail constants v are refitted on each inner-TRAINING partition when scoring an
inner-held partition, and on the outer fitting episodes for the held-fold forecast. All head
weights, shortlists and the (kq, ka) choice come from fitting episodes only. Held folds are
concatenated and scored once.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.optimize import minimize

from slot_model import harness as H
from slot_forecasting.io import write_submission

KGRID = (1, 2, 3, 5, 8, 12, 15)
STARTS = (0, 1)          # 0 = uniform over shortlist, 1 = all mass on the best single head
ALTERNATIONS = 2


def simplex_qp(quad, lin, m, starts=STARTS):
    """Minimise 0.5 x'Qx + c'x over the simplex. Deterministic starts, SLSQP."""
    best = None
    for s in starts:
        x0 = np.ones(m) / m if s == 0 else np.eye(m)[int(np.argmin(np.diag(quad) + 2 * lin))]
        res = minimize(lambda x: float(0.5 * x @ quad @ x + lin @ x),
                       x0, jac=lambda x: quad @ x + lin, method='SLSQP',
                       bounds=[(0.0, 1.0)] * m,
                       constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1.0,
                                    'jac': lambda x: np.ones(m)},
                       options={'ftol': 1e-14, 'maxiter': 500})
        x = np.clip(res.x, 0.0, None)
        x = x / max(x.sum(), 1e-12)
        val = float(0.5 * x @ quad @ x + lin @ x)
        if best is None or val < best[0]:
            best = (val, x, bool(res.success), float(np.max(np.abs(quad @ x + lin
                                                                   - (quad @ x + lin).mean()))))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/family_g')
    a = ap.parse_args()
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    b = H.load(a.data)
    names, P, Q, R, info = H.raw_pool(b)
    A_HEADS = R[:, :, 0]                       # P(1x | positive) per candidate
    n = len(b.ids)
    print(f'pool {len(names)} RAW; distinct q {info["distinct_q_heads"]} '
          f'r {info["distinct_r_heads"]}', flush=True)

    # distinct head banks, deduplicated separately, canonical source kept
    def dedupe(M):
        keep, seen = [], {}
        for i in range(len(M)):
            h = hashlib.sha256(M[i].round(12).tobytes()).hexdigest()
            if h not in seen:
                seen[h] = i
                keep.append(i)
        return keep
    qidx_all, aidx_all = dedupe(Q), dedupe(A_HEADS)
    print(f'distinct q heads {len(qidx_all)}, distinct a heads {len(aidx_all)}', flush=True)

    def tailmat(v_by_type):
        return np.array([v_by_type[t] for t in b.types])

    def rescale(q, r, V):
        """The declared rescale: keep a = r[:,0], replace the tail by the per-type v."""
        return H.Bundle.compose_qav(q, r[:, 0], V)

    # ---------- objectives ----------
    def loss_unproj(xq, xa_full, w, qi, ri):
        q = xq @ Q[qi]
        r = np.einsum('i,inc->nc', xa_full, R[ri])
        return b.wloss(H.Bundle.compose(q, r), w)

    def loss_final(q, aa, V, w):
        return b.wloss(H.Bundle.compose_qav(q, aa, V), w)

    # ---------- fitting procedure A: stack q and FULL r against the unprojected vector ----
    def fit_unprojected(mask, kq, kr):
        w = b.row_weights(mask)
        ql = np.array([b.wloss(H.Bundle.compose(Q[i], R[i]), w) for i in qidx_all])
        qi = [qidx_all[i] for i in np.argsort(ql)[:kq]]
        rl = np.array([b.wloss(H.Bundle.compose(Q[qi[0]], R[i]), w) for i in range(len(R))])
        ri = list(np.argsort(rl)[:kr])
        xq = np.ones(len(qi)) / len(qi)
        xr = np.ones(len(ri)) / len(ri)
        feas, trace = True, []
        for _ in range(ALTERNATIONS):
            # r step at fixed q: compose is linear in r, so weighted SSE is a convex QP in xr
            q = xq @ Q[qi]
            Bm = np.einsum('n,inc->inc', q, R[ri])                      # (k, n, 4)
            Tt = b.truth[:, 1:]
            G = np.einsum('inc,jnc,n->ij', Bm, Bm, w)
            c = -np.einsum('inc,nc,n->i', Bm, Tt, w)
            val, xr, ok, _ = simplex_qp(2 * G, 2 * c, len(ri))
            feas &= ok
            # q step at fixed r: exact quadratic via A, B coefficients
            r = np.einsum('i,inc->nc', xr, R[ri])
            Aq, Bq = b.q_coeffs(r)
            Gq = np.einsum('in,jn,n,n->ij', Q[qi], Q[qi], w, Aq)
            cq = -np.einsum('in,n,n->i', Q[qi], w, Bq)
            val, xq, ok, _ = simplex_qp(2 * Gq, 2 * cq, len(qi))
            feas &= ok
            trace.append(loss_unproj(xq, xr, w, qi, ri))
        return dict(q=xq @ Q[qi], r=np.einsum('i,inc->nc', xr, R[ri]),
                    qi=qi, ri=ri, xq=xq.tolist(), xr=xr.tolist(),
                    feasible=feas, objective_trace=trace)

    # ---------- fitting procedure B: stack q and a against the FINAL vector ---------------
    def fit_final(mask, kq, ka):
        w = b.row_weights(mask)
        V = tailmat(b.tail_constants(mask))
        ql = np.array([loss_final(Q[i], A_HEADS[i], V, w) for i in qidx_all])
        qi = [qidx_all[i] for i in np.argsort(ql)[:kq]]
        al = np.array([loss_final(Q[qi[0]], A_HEADS[i], V, w) for i in aidx_all])
        ai = [aidx_all[i] for i in np.argsort(al)[:ka]]
        xq = np.ones(len(qi)) / len(qi)
        xa = np.ones(len(ai)) / len(ai)
        feas, trace = True, []
        for _ in range(ALTERNATIONS):
            # a step at fixed q: p = u + a * d, exactly quadratic in a
            q = xq @ Q[qi]
            u = np.column_stack((1 - q, np.zeros(n), q[:, None] * V))
            d = np.column_stack((np.zeros(n), q, -q[:, None] * V))
            Ah = A_HEADS[ai]
            res = u - b.truth
            G = np.einsum('in,jn,n,n->ij', Ah, Ah, w, np.sum(d * d, axis=1))
            c = np.einsum('in,n,n->i', Ah, w, np.sum(d * res, axis=1))
            val, xa, ok, _ = simplex_qp(2 * G, 2 * c, len(ai))
            feas &= ok
            # q step at fixed a: p = e + q * g, exactly quadratic in q
            aa = xa @ Ah
            g = np.column_stack((-np.ones(n), aa, (1 - aa)[:, None] * V))
            e = np.zeros((n, 5)); e[:, 0] = 1.0
            Aq = np.sum(g * g, axis=1)
            Bq = -np.sum(g * (e - b.truth), axis=1)
            Gq = np.einsum('in,jn,n,n->ij', Q[qi], Q[qi], w, Aq)
            cq = -np.einsum('in,n,n->i', Q[qi], w, Bq)
            val, xq, ok, _ = simplex_qp(2 * Gq, 2 * cq, len(qi))
            feas &= ok
            trace.append(loss_final(xq @ Q[qi], xa @ Ah, V, w))
        return dict(q=xq @ Q[qi], a=xa @ Ah, qi=qi, ai=ai,
                    xq=xq.tolist(), xa=xa.tolist(), feasible=feas, objective_trace=trace)

    # ---------- outer loop ----------
    held = {k: np.empty_like(b.protected) for k in ('G0', 'G1', 'G2')}
    prov = []
    for fold in range(4):
        outer = b.outer_mask(fold)
        fm = ~outer
        inner = b.inner_folds(fm)
        sel = {'G0': {}, 'G1': {}, 'G2': {}}
        for kq in KGRID:
            if kq > len(qidx_all):
                continue
            for kk in KGRID:
                u_un = u_fin = f_fin = 0.0
                for tr, he in inner:
                    wh = b.row_weights(he)
                    Vh = tailmat(b.tail_constants(tr))     # refit per inner-training set
                    fu = fit_unprojected(tr, kq, min(kk, len(R)))
                    u_un += b.wloss(H.Bundle.compose(fu['q'], fu['r']), wh)
                    u_fin += b.wloss(rescale(fu['q'], fu['r'], Vh), wh)
                    if kk <= len(aidx_all):
                        ff = fit_final(tr, kq, kk)
                        f_fin += loss_final(ff['q'], ff['a'], Vh, wh)
                sel['G0'][(kq, kk)] = u_un
                sel['G1'][(kq, kk)] = u_fin
                if kk <= len(aidx_all):
                    sel['G2'][(kq, kk)] = f_fin
        picks = {k: min(v, key=lambda t: v[t]) for k, v in sel.items()}
        Vo = tailmat(b.tail_constants(fm))
        for arm in ('G0', 'G1'):
            kq, kk = picks[arm]
            f = fit_unprojected(fm, kq, min(kk, len(R)))
            held[arm][outer] = rescale(f['q'], f['r'], Vo)[outer]
        kq, kk = picks['G2']
        f2 = fit_final(fm, kq, kk)
        held['G2'][outer] = H.Bundle.compose_qav(f2['q'], f2['a'], Vo)[outer]
        prov.append({'fold': fold,
                     'picks': {k: list(v) for k, v in picks.items()},
                     'G2_feasible': f2['feasible'],
                     'G2_objective_trace': f2['objective_trace'],
                     'G2_q_weights': f2['xq'], 'G2_a_weights': f2['xa'],
                     'G2_q_members': [names[i] for i in f2['qi']],
                     'G2_a_members': [names[i] for i in f2['ai']],
                     'tail_constants': {t: v.tolist() for t, v in b.tail_constants(fm).items()}})
        print(f"  fold {fold}: picks G0={picks['G0']} G1={picks['G1']} G2={picks['G2']}",
              flush=True)

    res = {'protected': b.official(b.protected)}
    for k, v in held.items():
        res[k] = b.official(v)
    pa = {k: b.paired(v, b.protected) for k, v in held.items()}
    pg = {'G1_vs_G0': b.paired(held['G1'], held['G0']),
          'G2_vs_G1': b.paired(held['G2'], held['G1']),
          'G2_vs_G0': b.paired(held['G2'], held['G0'])}
    rep = {'protocol': __doc__, 'pool': names, 'pool_info': info,
           'distinct_q_heads': len(qidx_all), 'distinct_a_heads': len(aidx_all),
           'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                           'by_condition': {c: x['loss_ratio']
                                            for c, x in v['by_condition'].items()}}
                       for k, v in res.items()},
           'paired_vs_protected': {k: v['delta_loss_ratio_ci95'] for k, v in pa.items()},
           'paired_within_family': {k: v['delta_loss_ratio_ci95'] for k, v in pg.items()},
           'fold_provenance': prov,
           'convexity_note': 'Each alternation step is an exact simplex-constrained convex '
                             'QP. The JOINT problem over (q weights, a weights) is NOT '
                             'convex because the delivered vector contains the products q*a '
                             'and q*(1-a)*v. Alternation guarantees only a monotone '
                             'non-increasing objective from a fixed deterministic start set.',
           'seconds': time.perf_counter() - t0}
    for k, v in held.items():
        write_submission(a.out / 'predictions' / f'oof_{k}.csv', b.mapping(v))
    (a.out / 'evaluation.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')

    conds = sorted(res['protected']['by_condition'])
    print(f"\n{'arm':<12}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('protected', 'G0', 'G1', 'G2'):
        v = res[k]
        print(f"{k:<12}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print()
    for k, ci in rep['paired_vs_protected'].items():
        print(f"  {k:<12} vs protected dR 95% [{ci[0]:+.5f}, {ci[1]:+.5f}]"
              + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    for k, ci in rep['paired_within_family'].items():
        print(f"  {k:<12} dR 95% [{ci[0]:+.5f}, {ci[1]:+.5f}]"
              + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    print(f"wrote {a.out / 'evaluation.json'} in {rep['seconds']:.1f}s")


if __name__ == '__main__':
    main()
