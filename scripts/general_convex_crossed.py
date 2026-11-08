"""FAMILY X -- general convex mixtures over a bank of CROSSED (q, r) head forecasts.

PREREGISTERED PROTOCOL, fixed before any held-fold number below was read.

The old recombination arm (scripts/stack_and_recombine.py, arm B) picks ONE (q, r) pair.
Separable stacking (scripts/separable_stack.py) mixes the two heads with their own weight
vectors, which is exactly the RANK-ONE restriction W[j,k] = alpha[j] * beta[k] of a general
mixture over crossed forecasts. This family tests whether releasing that restriction buys
anything.

STRUCTURE. For q head j and conditional-payout head k,
    p[j,k] = [1 - q[j], q[j] * a[k], q[j] * (1 - a[k]) * v_type]
where a[k] = r[k][payout 1] and v_type is the per-machine-type tail vector over {2, 5, 20}
fitted from the TRUTH of the fitting-side episodes only. The mixture is
    p = sum_{j,k} W[j,k] p[j,k],   W >= 0,  sum W = 1.
Because every bank member carries the SAME v_type, so does any convex mixture: the tail
block of the mixture is (sum_{j,k} W[j,k] q[j] (1 - a[k])) * v_type. The tail
transformation is therefore held FIXED across every arm of this family and no second
projection is applied to the mixture. That also means the effective identity of an r head
under this family is its a vector alone.

CONVEXITY. For a FIXED legal bank the officially weighted full five-vector squared error is
a convex quadratic in W with no linear term, because sum W = 1:
    L(W) = sum_i w_i || sum_m W_m (p_m,i - t_i) ||^2 = W' G W,  G = <D, D>_w, D_m = p_m - t.
A convex penalty on the mixture's departure from a legal baseline forecast is the same shape:
    pen(W) = sum_i w_i || sum_m W_m (p_m,i - base_i) ||^2 = W' H W,  H = <E, E>_w.
So every fit here is a simplex-constrained convex QP in W' (G + lam H) W, solved to a
reported KKT / projected-gradient residual rather than by a generic nonlinear optimiser
alone.

ARMS (bank = crossed heads of the 15 audited RAW pool members; 15 distinct q heads, 10
distinct a heads after separate per-side deduplication):
  Xnull  no change: the protected release, verbatim. Zero correction.
  X0     best single crossed (j, k) pair, chosen on each fold's FITTING episodes.
  X1     rank-one (separable) convex mixture, shortlist chosen by inner folds.
  X2     GENERAL convex mixture, shortlist chosen by inner folds, no shrinkage.
  X3     general convex mixture with a shrinkage penalty toward the protected release,
         lam and the shortlist chosen by inner folds, and the NO-CHANGE option (emit the
         baseline) inside the same selection, so "use this method at all" is fold-internal.
  X1full rank-one convex mixture over the COMPLETE 150-member bank, no selection.
  X2full general convex mixture over the COMPLETE 150-member bank, no selection. Declared
         in round 1 as a diagnostic outside the preregistered 2/4/6 shortlist grid, to
         document what the shortlists buy.
  X4     general convex mixture whose shortlist size per side AND shrinkage strength are
         chosen by the inner folds from an EXTENDED grid that can reach the complete bank,
         with the no-change option inside the same selection.
  X5     BLOCK-GENERAL weights: a separate W per machine type over the complete bank. The
         official objective is a weighted sum over queries, so splitting it by machine type
         and solving one simplex QP per block minimises the same total exactly, with no
         renormalisation. machine_type is a query attribute and is available at inference,
         unlike the hidden condition label. Per-type against pooled, the shrinkage strength,
         and the no-change option are all chosen together by the inner folds. This is the
         second, independent way to spend weight generality: if a general W over crossed
         heads is worth nothing pooled, it may still be worth something block-wise.

ARM-SET EXTENSION, disclosed. Round 1 ran Xnull, X0, X1, X2, X3 and the X2full diagnostic.
X2full, the one arm with no shortlist, beat every shortlisted arm by about 0.22 points, so
X1full (its matched rank-one control at the same bank width), X4 (the fitting-side
self-selecting version, which removes the hand-chosen width) and X5 (block-general weights)
were added and are reported here. Each added arm was specified before it was run, but the EXTENSION ITSELF was prompted
by a held-fold number on these same 96 episodes and no interval below corrects for it.

Matched controls: X0 against Xnull, X1 against X0, X2 against X1, X3 against X2, X1full
against X1, X2full against X1full (identical bank and tails, rank-one W against general W:
this is the family's hypothesis in its cleanest form), X4 against X2full, X5 against X2full. The paired
interval is reported against BOTH the protected release and the arm's own control, and
against the family's established clean reference separable_rescaled (53.892202).

NESTING. Four frozen complete-episode outer folds. Inside each fold everything
target-dependent -- the per-type tail vectors, the per-side head rankings, the shortlist
sizes, the shrinkage strength, the mixture weights, and the decision to depart from the
baseline at all -- is fitted or chosen on that fold's FITTING episodes only, through
b.inner_folds. Tail vectors are REFITTED on each inner-TRAINING partition before an inner
comparison, so an inner-held partition never sees its own tail constants. Held-fold
forecasts are concatenated and scored once.

LEAKAGE, stated plainly. The bank is built only from the 15 audited RAW forecasts, whose
construction used no validation target. The tail vectors and the mixture weights ARE fitted
against validation truth, but always on the fitting-side episodes of the fold being
predicted, which is the same nesting the clean reference separable_rescaled (53.892202)
uses. Arm X3 additionally shrinks toward the protected release, which is itself a
DEPLOY_FIT artifact fitted on all validation targets; its correction is nested but its
absolute level inherits the protected release's target fitting, and the fair reading of X3
is its paired interval against the protected release, not its absolute points.

DETERMINISM. episode_id is a string; every reduction iterates a sorted sequence. BLAS thread
counts are pinned before numpy is imported because threaded reduction order previously
flipped near-tied discrete grid choices in this exact family.

These 96 episodes underpin the whole project. Every number here is conditional exploratory
development and no interval corrects for the accumulated reuse.
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
from scipy.optimize import minimize

# macOS Accelerate raises spurious divide-by-zero/overflow/invalid FP flags from its
# vectorised gemm kernels on perfectly clean finite inputs (verified directly: a random
# 150x3840 matmul of values near 0.01 sets all three flags while every output stays
# finite and correct). The flags are suppressed for matmul only, and every solver below
# asserts finiteness of its own output, so a REAL non-finite value still stops the run.
warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)

from slot_forecasting.io import write_submission
from slot_model import harness as H

NSIDE = (2, 4, 6)                               # unique heads per side, preregistered
LAM_GRID = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)      # modest shrinkage grid, 0.0 = none


# ---------------------------------------------------------------- simplex QP --------------
def proj_simplex(v):
    """Exact Euclidean projection onto {x >= 0, sum x = 1}. Sort-based, deterministic."""
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - 1.0
    k = np.arange(1, len(v) + 1, dtype=float)
    pos = np.nonzero(u - c / k > 0)[0]
    rho = int(pos[-1]) if len(pos) else 0
    return np.maximum(v - c[rho] / (rho + 1.0), 0.0)


def kkt_residual(G, x):
    """Stationarity residual for min x'Gx on the simplex: g_j = lam on the support,
    g_j >= lam off it, with lam the support mean of g."""
    g = 2.0 * (G @ x)
    sup = x > 1e-12
    if not sup.any():
        return float('nan')
    lam = float(g[sup].mean())
    r_on = float(np.max(np.abs(g[sup] - lam)))
    r_off = float(np.max(np.maximum(lam - g[~sup], 0.0))) if (~sup).any() else 0.0
    return max(r_on, r_off)


def solve_simplex_qp(G, iters=4000, check=25):
    """min x' G x s.t. x on the simplex. Accelerated projected gradient with a monotone
    plain-PGD polish and an SLSQP cross-check; the lowest objective wins, ties by order."""
    G = 0.5 * (G + G.T)
    m = len(G)
    scale = max(float(np.max(np.abs(np.diag(G)))), 1e-30)
    Gs = G / scale                                   # scale-free stopping rule
    L = max(2.0 * float(np.linalg.eigvalsh(Gs)[-1]), 1e-12)
    f = lambda x: float(x @ Gs @ x)

    x = np.full(m, 1.0 / m)
    y, tk = x.copy(), 1.0
    for it in range(iters):
        xn = proj_simplex(y - (2.0 * (Gs @ y)) / L)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        y = xn + ((tk - 1.0) / tn) * (xn - x)
        x, tk = xn, tn
        if it % check == check - 1 and kkt_residual(Gs, x) < 1e-13:
            break
    fista_it, xa = it + 1, x.copy()

    xp = xa.copy()                                   # monotone polish, never increases f
    for _ in range(600):
        xn = proj_simplex(xp - (2.0 * (Gs @ xp)) / L)
        if f(xn) >= f(xp) - 1e-18:
            xp = xn if f(xn) < f(xp) else xp
            break
        xp = xn

    cands = [('fista', xa), ('pgd_polish', xp)]
    for tag, x0 in (('slsqp_uniform', np.full(m, 1.0 / m)), ('slsqp_warm', xp)):
        res = minimize(f, x0, jac=lambda z: 2.0 * (Gs @ z), method='SLSQP',
                       bounds=[(0.0, 1.0)] * m,
                       constraints={'type': 'eq', 'fun': lambda z: z.sum() - 1.0},
                       options={'ftol': 1e-16, 'maxiter': 800})
        z = np.clip(res.x, 0.0, None)
        s = z.sum()
        cands.append((tag + ('' if res.success else '_nosuccess'),
                      z / s if s > 0 else np.full(m, 1.0 / m)))
    vals = [f(z) for _, z in cands]
    best = int(np.argmin(np.asarray(vals)))
    xb = cands[best][1]
    xb = np.maximum(xb, 0.0)
    xb = xb / xb.sum()
    # Post-selection monotone polish. Only strictly improving projected-gradient steps are
    # accepted, so this can never raise the objective, but it does tighten the stationarity
    # residual on a flat face where SLSQP stops at its iteration limit.
    fb = f(xb)
    for _ in range(4000):
        xn = proj_simplex(xb - (2.0 * (Gs @ xb)) / L)
        fn = f(xn)
        if fn >= fb - 1e-19:
            break
        xb, fb = xn, fn
    xb = np.maximum(xb, 0.0)
    xb = xb / xb.sum()
    if not np.isfinite(xb).all():
        raise FloatingPointError('simplex QP produced a non-finite weight vector')
    pg = float(np.linalg.norm(xb - proj_simplex(xb - (2.0 * (Gs @ xb)) / L)))
    info = {'solver': cands[best][0], 'fista_iters': fista_it,
            'kkt_residual': kkt_residual(Gs, xb), 'projected_gradient_norm': pg,
            'objective': float(xb @ G @ xb), 'simplex_residual': float(abs(xb.sum() - 1.0)),
            'min_weight': float(xb.min()), 'gram_scale': scale}
    return xb, info


def solve_rank_one(G, nq, nr, rounds=40):
    """min (alpha x beta)' G (alpha x beta) by block-coordinate descent on two simplices.
    Each block is itself a simplex QP, so each step is exact and the objective is monotone."""
    Gt = 0.5 * (G + G.T)
    T = Gt.reshape(nq, nr, nq, nr)
    alpha = np.full(nq, 1.0 / nq)
    beta = np.full(nr, 1.0 / nr)
    prev, infos = None, []
    for it in range(rounds):
        Ga = np.einsum('k,l,jkil->ji', beta, beta, T)
        alpha, ia = solve_simplex_qp(Ga, iters=1500)
        Gb = np.einsum('j,i,jkil->kl', alpha, alpha, T)
        beta, ib = solve_simplex_qp(Gb, iters=1500)
        W = np.outer(alpha, beta).reshape(-1)
        cur = float(W @ Gt @ W)
        infos = [ia, ib]
        if prev is not None and abs(prev - cur) <= 1e-16 * max(abs(prev), 1e-12):
            break
        prev = cur
    W = np.outer(alpha, beta).reshape(-1)
    return W, {'solver': 'block_coordinate_rank_one', 'rounds': it + 1,
               'alpha_kkt': infos[0]['kkt_residual'], 'beta_kkt': infos[1]['kkt_residual'],
               'objective': float(W @ Gt @ W), 'simplex_residual': float(abs(W.sum() - 1.0)),
               'min_weight': float(W.min())}


# ---------------------------------------------------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/general_convex_crossed')
    args = ap.parse_args()
    (args.out / 'predictions').mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    b = H.load(args.data)
    names, P, Q, R, pool_info = H.raw_pool(b)
    n = len(b.ids)
    truth, base = b.truth, b.protected

    # --- separate per-side deduplication; canonical executable source per kept head -------
    qkeep, qseen = [], {}
    for i, nm in enumerate(sorted(names)):
        j = names.index(nm)
        h = hashlib.sha256(Q[j].round(12).tobytes()).hexdigest()
        if h in qseen:
            continue
        qseen[h] = nm
        qkeep.append(j)
    A_all = R[:, :, 0]                      # the only part of r this family's tail fix uses
    akeep, aseen, agroups = [], {}, {}
    for nm in sorted(names):
        j = names.index(nm)
        h = hashlib.sha256(A_all[j].round(12).tobytes()).hexdigest()
        agroups.setdefault(h, []).append(nm)
        if h in aseen:
            continue
        aseen[h] = nm
        akeep.append(j)
    qkeep, akeep = sorted(qkeep), sorted(akeep)
    NQ, NR = len(qkeep), len(akeep)
    print(f'pool {len(names)} members -> {NQ} distinct q heads x {NR} distinct a heads '
          f'= {NQ * NR} crossed bank members', flush=True)

    type_order = sorted(set(b.types.tolist()))
    cache = {}

    def state(mask):
        """Everything derived from one TRAINING mask: tails, bank, Grams, per-side ranks."""
        key = hashlib.sha1(np.asarray(mask, dtype=bool).tobytes()).hexdigest()
        if key in cache:
            return cache[key]
        tails = b.tail_constants(mask)
        tails = {t: tails[t] / tails[t].sum() for t in type_order}
        V = np.stack([tails[t] for t in b.types])
        bank = np.empty((NQ * NR, n, 5))
        for jj, j in enumerate(qkeep):
            for kk, k in enumerate(akeep):
                bank[jj * NR + kk] = b.compose_qav(Q[j], A_all[k], V)
        w = b.row_weights(mask)
        s = np.sqrt(w)
        D = ((bank - truth[None, :, :]) * s[None, :, None]).reshape(NQ * NR, -1)
        E = ((bank - base[None, :, :]) * s[None, :, None]).reshape(NQ * NR, -1)
        G, Hm = D @ D.T, E @ E.T
        Gty, Hty = {}, {}                    # per-machine-type blocks of the SAME objective
        for t in type_order:
            s2 = np.sqrt(w * (b.types == t))
            Dt = ((bank - truth[None, :, :]) * s2[None, :, None]).reshape(NQ * NR, -1)
            Et = ((bank - base[None, :, :]) * s2[None, :, None]).reshape(NQ * NR, -1)
            Gty[t], Hty[t] = Dt @ Dt.T, Et @ Et.T
        M = np.diag(G).reshape(NQ, NR)       # per-crossed-member fitting loss
        qrank = np.argsort(M.min(axis=1), kind='stable')
        rrank = np.argsort(M.min(axis=0), kind='stable')
        st = {'tails': tails, 'bank': bank, 'G': G, 'H': Hm, 'M': M, 'Gty': Gty,
              'Hty': Hty, 'qrank': qrank, 'rrank': rrank, 'base_loss': b.wloss(base, w)}
        cache[key] = st
        return st

    def fit(st, nq, nr, lam, mode):
        js, ks = st['qrank'][:nq], st['rrank'][:nr]
        idx = np.array([int(j) * NR + int(k) for j in js for k in ks])
        Gs = st['G'][np.ix_(idx, idx)] + (lam * st['H'][np.ix_(idx, idx)] if lam else 0.0)
        if mode == 'general':
            W, info = solve_simplex_qp(Gs)
        else:
            W, info = solve_rank_one(Gs, len(js), len(ks))
        p = np.einsum('i,ind->nd', W, st['bank'][idx])
        p = p / p.sum(axis=1, keepdims=True)
        info['members'] = [[names[qkeep[int(j)]], names[akeep[int(k)]]] for j in js for k in ks]
        return p, W, idx, info

    NEXT_Q, NEXT_R = (2, 4, 6, NQ), (2, 4, 6, NR)   # extended grid, reaches the whole bank
    def fit_pertype(st, lam):
        p = np.empty((n, 5))
        Ws, infos = {}, {}
        for t in type_order:
            Gb = st['Gty'][t] + (lam * st['Hty'][t] if lam else 0.0)
            W, info = solve_simplex_qp(Gb)
            rows = b.types == t
            p[rows] = np.einsum('i,ind->nd', W, st['bank'][:, rows, :])
            Ws[t] = W.tolist()
            infos[t] = {k: info[k] for k in info if k != 'members'}
        p = p / p.sum(axis=1, keepdims=True)
        return p, Ws, infos

    arms = ('Xnull', 'X0', 'X1', 'X2', 'X3', 'X1full', 'X2full', 'X4', 'X5')
    held = {k: np.empty_like(base) for k in arms}
    prov, solver_stats = [], []

    for fold in range(4):
        outer = b.outer_mask(fold)
        fitm = ~outer
        inner = b.inner_folds(fitm, 3)
        istates = [(state(itr), b.row_weights(ihe), ihe) for itr, ihe in inner]
        fs = state(fitm)
        rec = {'fold': fold, 'held_queries': int(outer.sum()),
               'tails_outer_fit': {t: fs['tails'][t].tolist() for t in type_order}}

        # Xnull: no change
        held['Xnull'][outer] = base[outer]

        # X0: best single crossed pair on the FITTING episodes
        j0, k0 = np.unravel_index(int(np.argmin(fs['M'])), fs['M'].shape)
        held['X0'][outer] = fs['bank'][int(j0) * NR + int(k0)][outer]
        rec['X0'] = {'q': names[qkeep[int(j0)]], 'a': names[akeep[int(k0)]],
                     'fit_loss': float(fs['M'][j0, k0])}

        # X1 / X2: shortlist by inner folds (lam = 0)
        for arm, mode in (('X1', 'rank_one'), ('X2', 'general')):
            tbl = {}
            for nq in NSIDE:
                for nr in NSIDE:
                    tot = 0.0
                    for ist, whe, _ in istates:
                        p, _, _, _ = fit(ist, nq, nr, 0.0, mode)
                        tot += b.wloss(p, whe)
                    tbl[(nq, nr)] = tot
            cfg = min(sorted(tbl), key=lambda c: (tbl[c], c))
            p, W, idx, info = fit(fs, cfg[0], cfg[1], 0.0, mode)
            held[arm][outer] = p[outer]
            rec[arm] = {'nq': cfg[0], 'nr': cfg[1],
                        'inner_table': {f'{a}x{c}': v for (a, c), v in sorted(tbl.items())},
                        'weights': W.tolist(), 'solver': info}
            solver_stats.append({'fold': fold, 'arm': arm, **{k: info[k] for k in info
                                                              if k != 'members'}})

        # X3: general mixture, shrinkage toward the protected release, plus the no-change
        #     option, all chosen by the same inner folds
        tbl3 = {}
        for nq in NSIDE:
            for nr in NSIDE:
                for lam in LAM_GRID:
                    tot = 0.0
                    for ist, whe, _ in istates:
                        p, _, _, _ = fit(ist, nq, nr, lam, 'general')
                        tot += b.wloss(p, whe)
                    tbl3[(nq, nr, lam)] = tot
        no_change = sum(b.wloss(base, whe) for _, whe, _ in istates)
        cfg3 = min(sorted(tbl3), key=lambda c: (tbl3[c], c))
        if no_change <= tbl3[cfg3]:
            held['X3'][outer] = base[outer]
            rec['X3'] = {'choice': 'no_change', 'inner_no_change': no_change,
                         'inner_best_mixture': tbl3[cfg3], 'best_cfg': list(cfg3)}
        else:
            p, W, idx, info = fit(fs, cfg3[0], cfg3[1], cfg3[2], 'general')
            held['X3'][outer] = p[outer]
            rec['X3'] = {'choice': 'mixture', 'nq': cfg3[0], 'nr': cfg3[1], 'lam': cfg3[2],
                         'inner_no_change': no_change, 'inner_best_mixture': tbl3[cfg3],
                         'inner_table': {f'{a}x{c}|lam{l}': v
                                         for (a, c, l), v in sorted(tbl3.items())},
                         'weights': W.tolist(), 'solver': info}
            solver_stats.append({'fold': fold, 'arm': 'X3',
                                 **{k: info[k] for k in info if k != 'members'}})

        # X1full / X2full: the COMPLETE bank, no selection of any kind. These are the
        # matched pair that isolates the family's hypothesis at full bank width: identical
        # members, identical tails, rank-one W against general W.
        for arm, mode in (('X1full', 'rank_one'), ('X2full', 'general')):
            p, W, idx, info = fit(fs, NQ, NR, 0.0, mode)
            held[arm][outer] = p[outer]
            rec[arm] = {'nq': NQ, 'nr': NR, 'support': int((W > 1e-9).sum()),
                        'weights': W.tolist(), 'solver': info}
            solver_stats.append({'fold': fold, 'arm': arm,
                                 **{k: info[k] for k in info if k != 'members'}})

        # X4: the same general QP, but the shortlist size per side AND the shrinkage are
        # chosen by the inner folds from an EXTENDED grid that can reach the complete bank,
        # with the no-change option inside the same selection. This is the fitting-side
        # self-selecting version of X2full, so nothing about its width is chosen by hand.
        tbl4 = {}
        for nq in NEXT_Q:
            for nr in NEXT_R:
                for lam in LAM_GRID:
                    tot = 0.0
                    for ist, whe, _ in istates:
                        p, _, _, _ = fit(ist, nq, nr, lam, 'general')
                        tot += b.wloss(p, whe)
                    tbl4[(nq, nr, lam)] = tot
        cfg4 = min(sorted(tbl4), key=lambda c: (tbl4[c], c))
        if no_change <= tbl4[cfg4]:
            held['X4'][outer] = base[outer]
            rec['X4'] = {'choice': 'no_change', 'inner_no_change': no_change,
                         'inner_best_mixture': tbl4[cfg4], 'best_cfg': list(cfg4)}
        else:
            p, W, idx, info = fit(fs, cfg4[0], cfg4[1], cfg4[2], 'general')
            held['X4'][outer] = p[outer]
            rec['X4'] = {'choice': 'mixture', 'nq': cfg4[0], 'nr': cfg4[1], 'lam': cfg4[2],
                         'support': int((W > 1e-9).sum()),
                         'inner_no_change': no_change, 'inner_best_mixture': tbl4[cfg4],
                         'inner_table': {f'{a}x{c}|lam{l}': v
                                         for (a, c, l), v in sorted(tbl4.items())},
                         'weights': W.tolist(), 'solver': info}
            solver_stats.append({'fold': fold, 'arm': 'X4',
                                 **{k: info[k] for k in info if k != 'members'}})

        # X5: block-general weights. Per-type against pooled, shrinkage strength, and the
        # no-change option are one single inner-fold selection.
        tbl5 = {}
        for kind in ('pertype', 'pooled'):
            for lam in LAM_GRID:
                tot = 0.0
                for ist, whe, _ in istates:
                    pp = (fit_pertype(ist, lam)[0] if kind == 'pertype'
                          else fit(ist, NQ, NR, lam, 'general')[0])
                    tot += b.wloss(pp, whe)
                tbl5[(kind, lam)] = tot
        cfg5 = min(sorted(tbl5), key=lambda c: (tbl5[c], c))
        if no_change <= tbl5[cfg5]:
            held['X5'][outer] = base[outer]
            rec['X5'] = {'choice': 'no_change', 'inner_no_change': no_change,
                         'inner_best': tbl5[cfg5], 'best_cfg': list(cfg5)}
        else:
            if cfg5[0] == 'pertype':
                p, Ws, infos = fit_pertype(fs, cfg5[1])
                rec['X5'] = {'choice': 'pertype', 'lam': cfg5[1], 'weights_by_type': Ws,
                             'solver_by_type': infos}
                for t in type_order:
                    solver_stats.append({'fold': fold, 'arm': f'X5[{t}]', **infos[t]})
            else:
                p, W, idx, info = fit(fs, NQ, NR, cfg5[1], 'general')
                rec['X5'] = {'choice': 'pooled', 'lam': cfg5[1], 'weights': W.tolist(),
                             'solver': info}
                solver_stats.append({'fold': fold, 'arm': 'X5',
                                     **{k: info[k] for k in info if k != 'members'}})
            held['X5'][outer] = p[outer]
            rec['X5']['inner_no_change'] = no_change
            rec['X5']['inner_table'] = {f'{k}|lam{l}': v for (k, l), v in sorted(tbl5.items())}

        prov.append(rec)
        print(f"  fold {fold}: X0 q={Path(rec['X0']['q']).stem} a={Path(rec['X0']['a']).stem}"
              f" | X1 {rec['X1']['nq']}x{rec['X1']['nr']}"
              f" | X2 {rec['X2']['nq']}x{rec['X2']['nr']}"
              f" | X3 {rec['X3'].get('choice')}"
              f" {rec['X3'].get('nq','-')}x{rec['X3'].get('nr','-')}"
              f" lam={rec['X3'].get('lam','-')}"
              f" | X4 {rec['X4'].get('choice')}"
              f" {rec['X4'].get('nq','-')}x{rec['X4'].get('nr','-')}"
              f" lam={rec['X4'].get('lam','-')}"
              f" | X5 {rec['X5'].get('choice')} lam={rec['X5'].get('lam','-')}", flush=True)

    # ------------------------------------------------------------ scoring -----------------
    res = {'protected': b.official(base)}
    for k in arms:
        res[k] = b.official(held[k])
    controls = {'Xnull': 'protected', 'X0': 'Xnull', 'X1': 'X0', 'X2': 'X1', 'X3': 'X2',
                'X1full': 'X1', 'X2full': 'X1full', 'X4': 'X2full', 'X5': 'X2full'}
    ref_arr = {'protected': base, **held}
    pv, pc = {}, {}
    for k in arms:
        pv[k] = b.paired(held[k], base)['delta_loss_ratio_ci95']
        pc[k] = b.paired(held[k], ref_arr[controls[k]])['delta_loss_ratio_ci95']

    # The family's established clean reference: separable (rank-one) stacking with the
    # rescale tail on the same 15-member audited pool, artifacts/nested_audited, 53.892202.
    ext = {}
    ref_path = ROOT / 'artifacts/nested_audited/predictions/oof_separable_rescaled.csv'
    if ref_path.exists():
        from slot_forecasting.io import read_submission
        d = read_submission(ref_path)
        if set(d) == set(b.ids):
            S = np.array([d[i] for i in b.ids])
            S = S / S.sum(axis=1, keepdims=True)
            so = b.official(S)
            ext['separable_rescaled'] = {
                'path': str(ref_path.relative_to(ROOT)),
                'points': so['technical_points'], 'R': so['mean_loss_ratio'],
                'paired_arm_vs_reference': {k: b.paired(held[k], S)['delta_loss_ratio_ci95']
                                            for k in ('X2', 'X1full', 'X2full', 'X4', 'X5')}}

    report = {
        'protocol': __doc__,
        'external_reference': ext,
        'bank': {'pool_members': names, 'distinct_q_heads': NQ, 'distinct_a_heads': NR,
                 'canonical_q_sources': [names[j] for j in qkeep],
                 'canonical_a_sources': [names[j] for j in akeep],
                 'a_head_duplicate_groups': [v for v in agroups.values() if len(v) > 1],
                 'pool_info': pool_info},
        'grids': {'heads_per_side': list(NSIDE), 'lambda': list(LAM_GRID),
                  'no_change_option': True},
        'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                        'by_condition': {c: x['loss_ratio']
                                         for c, x in sorted(v['by_condition'].items())}}
                    for k, v in sorted(res.items())},
        'gains_over_protected': {k: res[k]['technical_points'] - res['protected']['technical_points']
                                 for k in arms},
        'controls': controls,
        'paired_vs_protected': {k: pv[k] for k in arms},
        'paired_vs_control': {k: pc[k] for k in arms},
        'significant_vs_protected': {k: bool(pv[k][1] < 0) for k in arms},
        'significant_vs_control': {k: bool(pc[k][1] < 0) for k in arms},
        'extra_pairings': {f'{a}_vs_{c}': b.paired(held[a], held[c])['delta_loss_ratio_ci95']
                           for a, c in (('X2full', 'X2'), ('X4', 'X2'), ('X1full', 'X0'),
                                        ('X4', 'X1full'), ('X5', 'X1full'), ('X5', 'X4'))},
        'solver_diagnostics': solver_stats,
        'solver_feasibility': {
            'max_simplex_residual': max(s['simplex_residual'] for s in solver_stats),
            'min_weight_over_fits': min(s['min_weight'] for s in solver_stats),
            'max_kkt_residual_general': max(
                [s['kkt_residual'] for s in solver_stats if 'kkt_residual' in s]),
            'max_block_kkt_rank_one': max(
                [max(s['alpha_kkt'], s['beta_kkt']) for s in solver_stats
                 if 'alpha_kkt' in s] or [0.0]),
            'note': 'KKT and projected-gradient residuals are on the diagonal-scaled Gram, '
                    'so they are dimensionless; the Gram is PSD by construction, so the '
                    'simplex QP is always feasible (the uniform point is feasible) and the '
                    'only failure mode is a flat optimum, which near-duplicate bank members '
                    'do produce. A flat face leaves the fitted loss unchanged but the held '
                    'forecast weakly determined; the shrinkage penalty in X3 is the only arm '
                    'that strictly convexifies it.'},
        'fold_provenance': prov,
        'exposure': 'Conditional exploratory development on 96 repeatedly reused episodes.',
        'seconds': time.perf_counter() - start,
    }
    for k in arms:
        write_submission(args.out / 'predictions' / f'oof_{k}.csv',
                         b.mapping(held[k]))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    digest = []
    for k in arms:
        f = args.out / 'predictions' / f'oof_{k}.csv'
        digest.append(f'{k} {hashlib.sha256(f.read_bytes()).hexdigest()}')
    for r in prov:
        digest.append(f"fold{r['fold']} X0 {r['X0']['q']} {r['X0']['a']} "
                      f"X1 {r['X1']['nq']}x{r['X1']['nr']} X2 {r['X2']['nq']}x{r['X2']['nr']} "
                      f"X3 {r['X3'].get('choice')} {r['X3'].get('nq','-')}x"
                      f"{r['X3'].get('nr','-')} lam={r['X3'].get('lam','-')} "
                      f"X4 {r['X4'].get('choice')} {r['X4'].get('nq','-')}x"
                      f"{r['X4'].get('nr','-')} lam={r['X4'].get('lam','-')} "
                      f"X5 {r['X5'].get('choice')} lam={r['X5'].get('lam','-')}")
    for k in arms:
        digest.append(f"{k} points {res[k]['technical_points']:.9f} R {res[k]['mean_loss_ratio']:.12f}")
    (args.out / 'selection_digest.txt').write_text('\n'.join(digest) + '\n')

    conds = sorted(res['protected']['by_condition'])
    print(f"\n{'arm':<10}{'points':>12}{'R':>11}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('protected',) + arms:
        v = res[k]
        print(f"{k:<10}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>11.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print()
    for k in arms:
        print(f"  {k:<8} vs protected [{pv[k][0]:+.5f}, {pv[k][1]:+.5f}]"
              f"{'  SIG' if pv[k][1] < 0 else '  ns ':<6}"
              f"   vs {controls[k]:<9} [{pc[k][0]:+.5f}, {pc[k][1]:+.5f}]"
              f"{'  SIG' if pc[k][1] < 0 else '  ns'}")
    print(f"\nwrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
