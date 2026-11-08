"""ADVERSARIAL VERIFICATION of Family X arm X1full.

Independent reimplementation + leakage probes. Writes nothing outside
artifacts/verify_family_x/.

Checks implemented here (re-run reproduction is done separately by invoking the real
script into a fresh --out):

  C2  nesting probes:
        (a) row_weights(fit) is exactly zero on every held row of its own fold
        (b) tail_constants(fit) is a function of fitting-side truth only (recomputed
            from scratch here, not via the harness)
        (c) permuting the TRUTH of fold-f held episodes must leave fold-f's X1full
            held forecast byte-identical and must move the other folds
  C3  pool admissibility: exactly the 15 RAW_SOURCES whose classify() == 'RAW'
  C4  causality: this arm consumes no history; the only per-query inputs are the
        audited members' columns and machine_type.  Verified by hashing the inputs
        the arm actually touches.
  C5  recompute b.paired(X1full, protected)
  C6  score the concatenated held folds ONCE with b.official and independently with
        the organizer's own contest_score
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

from slot_model import harness as H
from slot_forecasting.contest import contest_score
from slot_forecasting.rounds import VERSION


# ------------------------------------------------------------------ my own solvers --------
def proj_simplex(v):
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - 1.0
    k = np.arange(1, len(v) + 1, dtype=float)
    pos = np.nonzero(u - c / k > 0)[0]
    rho = int(pos[-1]) if len(pos) else 0
    return np.maximum(v - c[rho] / (rho + 1.0), 0.0)


def pgd_simplex(G, x0, iters=20000):
    """Plain monotone projected gradient on the simplex. Independent of their FISTA."""
    G = 0.5 * (G + G.T)
    sc = max(float(np.max(np.abs(np.diag(G)))), 1e-30)
    Gs = G / sc
    L = max(2.0 * float(np.linalg.eigvalsh(Gs)[-1]), 1e-12)
    f = lambda z: float(z @ Gs @ z)
    x, fx = x0.copy(), f(x0)
    for _ in range(iters):
        xn = proj_simplex(x - (2.0 * (Gs @ x)) / L)
        fn = f(xn)
        if fn >= fx - 1e-20:
            break
        x, fx = xn, fn
    return x / x.sum()


def my_rank_one(G, nq, nr, rounds=400):
    """Block-coordinate descent with an independently written inner solver."""
    Gt = 0.5 * (G + G.T)
    T = Gt.reshape(nq, nr, nq, nr)
    alpha = np.full(nq, 1.0 / nq)
    beta = np.full(nr, 1.0 / nr)
    prev = None
    for it in range(rounds):
        Ga = np.einsum('k,l,jkil->ji', beta, beta, T)
        alpha = pgd_simplex(Ga, alpha)
        Gb = np.einsum('j,i,jkil->kl', alpha, alpha, T)
        beta = pgd_simplex(Gb, beta)
        W = np.outer(alpha, beta).reshape(-1)
        cur = float(W @ Gt @ W)
        if prev is not None and abs(prev - cur) <= 1e-17 * max(abs(prev), 1e-12):
            break
        prev = cur
    W = np.outer(alpha, beta).reshape(-1)
    return W, float(W @ Gt @ W), it + 1


# ------------------------------------------------------------------ pipeline --------------
def my_tail_constants(truth, types, mask, type_order):
    """Recomputed from scratch, not via b.tail_constants."""
    out = {}
    for t in type_order:
        m = mask & (types == t)
        rows = truth[m]
        r = rows[:, 1:] / np.maximum(rows[:, 1:].sum(axis=1, keepdims=True), 1e-15)
        tl = r[:, 1:]
        tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1e-15)
        v = tl.mean(axis=0) if m.any() else np.ones(3) / 3.0
        out[t] = v / v.sum()
    return out


def build(b, Q, A, qkeep, akeep, truth, types, type_order, mask):
    """Bank + weighted Gram from ONE mask. truth passed explicitly so a probe can swap it."""
    NQ, NR = len(qkeep), len(akeep)
    n = truth.shape[0]
    tails = my_tail_constants(truth, types, mask, type_order)
    V = np.stack([tails[t] for t in types])
    bank = np.empty((NQ * NR, n, 5))
    for jj, j in enumerate(qkeep):
        for kk, k in enumerate(akeep):
            bank[jj * NR + kk] = b.compose_qav(Q[j], A[k], V)
    w = b.row_weights(mask)
    s = np.sqrt(w)
    D = ((bank - truth[None, :, :]) * s[None, :, None]).reshape(NQ * NR, -1)
    return bank, D @ D.T, w, tails


def x1full(b, Q, A, qkeep, akeep, truth, types, type_order, folds=range(4)):
    """My own X1full: rank-one convex mixture over the complete bank, per outer fold."""
    out = np.zeros((truth.shape[0], 5))
    diag = []
    for fold in folds:
        outer = b.outer_mask(fold)
        fitm = ~outer
        bank, G, w, tails = build(b, Q, A, qkeep, akeep, truth, types, type_order, fitm)
        assert np.all(w[outer] == 0.0), 'row_weights leaked onto held rows'
        W, obj, its = my_rank_one(G, len(qkeep), len(akeep))
        p = np.einsum('i,ind->nd', W, bank)
        p = p / p.sum(axis=1, keepdims=True)
        out[outer] = p[outer]
        diag.append({'fold': int(fold), 'objective': obj, 'bcd_rounds': its,
                     'support': int((W > 1e-9).sum()),
                     'tails': {t: tails[t].tolist() for t in type_order}})
    return out, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/verify_family_x')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rep = {}

    b = H.load(args.data)
    names, P, Q, R, pool_info = H.raw_pool(b)
    n = len(b.ids)
    truth0 = b.truth.copy()
    types = b.types
    type_order = sorted(set(types.tolist()))

    # ---- C3 pool admissibility ----------------------------------------------------------
    from scripts.stack_manifest import RAW_SOURCES, classify
    want = sorted(p for p in RAW_SOURCES if classify(p) == 'RAW')
    rep['C3_pool'] = {
        'raw_sources_total': len(RAW_SOURCES),
        'classify_RAW': len(want),
        'pool_used': sorted(names),
        'pool_equals_raw_sources': sorted(names) == want,
        'skipped': pool_info['skipped'],
        'distinct_q_heads': pool_info['distinct_q_heads'],
        'distinct_r_heads': pool_info['distinct_r_heads'],
        'nonraw_in_pool': [x for x in names if classify(x) != 'RAW'],
        'member_sha256': {x: hashlib.sha256((ROOT / x).read_bytes()).hexdigest()[:16]
                          for x in sorted(names)},
    }

    # ---- per-side dedup, done independently by exact array equality ---------------------
    qkeep, akeep = [], []
    A_all = R[:, :, 0]
    for i in range(len(names)):
        if not any(np.array_equal(Q[i], Q[j]) for j in qkeep):
            qkeep.append(i)
    for i in range(len(names)):
        if not any(np.array_equal(A_all[i], A_all[j]) for j in akeep):
            akeep.append(i)
    rep['dedup'] = {'NQ': len(qkeep), 'NR': len(akeep),
                    'bank_members': len(qkeep) * len(akeep),
                    'q_sources': [names[j] for j in qkeep],
                    'a_sources': [names[j] for j in akeep]}

    # ---- C4 causality surface -----------------------------------------------------------
    # The arm's ONLY per-query inputs are: the audited members' five-vectors, machine_type,
    # and the fitting-side truth.  No history, no cutoff, no lag, no condition label.
    import scripts.general_convex_crossed as GX
    src = (ROOT / 'scripts/general_convex_crossed.py').read_text()
    rep['C4_causality'] = {
        'uses_b_index': 'b.index' in src or '.index[' in src,
        'uses_cutoffs': 'cutoffs' in src,
        'uses_signals': 'signals' in src,
        'uses_groups_condition_label': 'b.groups' in src,
        'uses_types': 'b.types' in src,
        'note': 'machine_type is a released query attribute; condition label is not used.',
    }

    # ---- C2a/C2b/C6: my independent X1full ----------------------------------------------
    mine, diag = x1full(b, Q, A_all, qkeep, akeep, truth0, types, type_order)
    rep['my_x1full'] = {'fold_diagnostics': diag}
    off = b.official(mine)
    rep['my_x1full']['official'] = {
        'points': off['technical_points'], 'R': off['mean_loss_ratio'],
        'by_condition': {c: v['loss_ratio'] for c, v in sorted(off['by_condition'].items())}}

    # organizer's own contest_score, independently, on the concatenated held folds
    targets, groups, reference = b.scoring
    cs = contest_score(b.mapping(mine), targets, groups, reference,
                       max_points=b.scoring.metadata['max_points'], version=VERSION)
    rep['my_x1full']['contest_score'] = {'points': cs['technical_points'],
                                         'R': cs['mean_loss_ratio']}

    # ---- the family's own published X1full ----------------------------------------------
    from slot_forecasting.io import read_submission
    theirs = {}
    for tag, pth in (('published', ROOT / 'artifacts/general_convex_crossed/predictions/oof_X1full.csv'),
                     ('rerun_a', ROOT / 'artifacts/verify_family_x_rerun_a/predictions/oof_X1full.csv'),
                     ('rerun_b', ROOT / 'artifacts/verify_family_x_rerun_b/predictions/oof_X1full.csv')):
        if not pth.exists():
            continue
        d = read_submission(pth)
        if set(d) != set(b.ids):
            continue
        X = np.array([d[i] for i in b.ids])
        X = X / X.sum(axis=1, keepdims=True)
        o = b.official(X)
        c2 = contest_score(b.mapping(X), targets, groups, reference,
                           max_points=b.scoring.metadata['max_points'], version=VERSION)
        theirs[tag] = {
            'path': str(pth.relative_to(ROOT)),
            'sha256': hashlib.sha256(pth.read_bytes()).hexdigest(),
            'points': o['technical_points'], 'R': o['mean_loss_ratio'],
            'contest_score_points': c2['technical_points'],
            'by_condition': {c: v['loss_ratio'] for c, v in sorted(o['by_condition'].items())},
            'paired_vs_protected': b.paired(X, b.protected)['delta_loss_ratio_ci95'],
            'max_abs_diff_vs_mine': float(np.max(np.abs(X - mine))),
        }
        theirs[tag]['upper_bound_below_zero'] = bool(
            theirs[tag]['paired_vs_protected'][1] < 0)
        theirs[tag]['paired_vs_mine'] = b.paired(X, mine)['delta_loss_ratio_ci95']
    rep['C1_C5_C6_theirs'] = theirs

    prot = b.official(b.protected)
    rep['protected'] = {'points': prot['technical_points'], 'R': prot['mean_loss_ratio']}
    rep['my_x1full']['paired_vs_protected'] = b.paired(mine, b.protected)['delta_loss_ratio_ci95']

    # ---- C2c: truth-permutation probe, per fold -----------------------------------------
    probes = []
    rng = np.random.default_rng(20261108)
    for pf in range(4):
        tp = truth0.copy()
        eps = sorted({e for i, e in enumerate(b.ep) if b.folds[e] == pf})
        rows = np.array([i for i, e in enumerate(b.ep) if b.folds[e] == pf])
        perm = rng.permutation(len(rows))
        tp[rows] = truth0[rows][perm]
        moved = float(np.max(np.abs(tp - truth0)))
        # the harness caches row weights off (ref, truth); clear it and swap truth in place
        b._wcache.clear()
        b.truth = tp
        pert, _ = x1full(b, Q, A_all, qkeep, akeep, tp, types, type_order)
        b.truth = truth0
        b._wcache.clear()
        per_fold = {}
        for f in range(4):
            m = b.outer_mask(f)
            per_fold[f'fold{f}_held_maxabsdiff'] = float(np.max(np.abs(pert[m] - mine[m])))
        probes.append({'perturbed_fold': pf, 'n_episodes': len(eps),
                       'truth_moved_maxabs': moved,
                       'self_fold_unchanged': per_fold[f'fold{pf}_held_maxabsdiff'] == 0.0,
                       'other_folds_moved': all(
                           v > 0 for k, v in per_fold.items() if not k.startswith(f'fold{pf}_')),
                       **per_fold})
    rep['C2c_truth_permutation_probe'] = probes

    # ---- C2a: weights/tails confinement, stated numerically -----------------------------
    conf = []
    for f in range(4):
        outer = b.outer_mask(f)
        w = b.row_weights(~outer)
        t_fit = my_tail_constants(truth0, types, ~outer, type_order)
        t_all = my_tail_constants(truth0, types, np.ones(n, bool), type_order)
        conf.append({'fold': f,
                     'max_weight_on_held_rows': float(np.max(np.abs(w[outer]))),
                     'held_rows': int(outer.sum()),
                     'tail_fit_vs_all_maxabs': max(
                         float(np.max(np.abs(t_fit[t] - t_all[t]))) for t in type_order)})
    rep['C2a_weight_and_tail_confinement'] = conf

    rep['seconds'] = time.perf_counter() - t0
    (args.out / 'verification.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')

    from slot_forecasting.io import write_submission
    write_submission(args.out / 'my_x1full.csv', b.mapping(mine))

    print(json.dumps({k: rep[k] for k in ('C3_pool', 'dedup', 'C4_causality',
                                          'C2a_weight_and_tail_confinement')},
                     indent=2, sort_keys=True)[:4000])
    print('\n-- my independent X1full --')
    print(json.dumps(rep['my_x1full']['official'], indent=2, sort_keys=True))
    print('paired vs protected', rep['my_x1full']['paired_vs_protected'])
    print('contest_score', rep['my_x1full']['contest_score'])
    print('\n-- theirs --')
    print(json.dumps(theirs, indent=2, sort_keys=True))
    print('\n-- truth permutation probe --')
    print(json.dumps(probes, indent=2, sort_keys=True))
    print(f"\nwrote {args.out / 'verification.json'} in {rep['seconds']:.1f}s")


if __name__ == '__main__':
    main()
