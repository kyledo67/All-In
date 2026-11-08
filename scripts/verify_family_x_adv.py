"""ADVERSARIAL VERIFICATION of Family X arm X2full.

Independent re-derivation, independent scoring, structural nesting proof, empirical leak
test, and paired-interval recomputation. Writes artifacts/verify_family_x_adv/.
Nothing here modifies another family's files.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
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
warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)

from slot_forecasting.io import read_submission
from slot_forecasting.contest import contest_score
from slot_model import harness as H
from slot_model.official_score import max_points, paired_bootstrap
from scripts.stack_manifest import RAW_SOURCES, classify

OUT = ROOT / 'artifacts/verify_family_x_adv'
FAM = ROOT / 'artifacts/general_convex_crossed'
DATA = ROOT / 'main-participants-60-40/public'
rep = {}


# ------------------------------------------------------------------ my own simplex solver --
def proj(v):
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - 1.0
    k = np.arange(1, len(v) + 1, dtype=float)
    pos = np.nonzero(u - c / k > 0)[0]
    rho = int(pos[-1]) if len(pos) else 0
    return np.maximum(v - c[rho] / (rho + 1.0), 0.0)


def my_qp(G):
    """Independent solver: exponentiated gradient + monotone PGD + SLSQP, best objective."""
    G = 0.5 * (G + G.T)
    m = len(G)
    sc = max(float(np.max(np.abs(np.diag(G)))), 1e-30)
    Gs = G / sc
    f = lambda x: float(x @ Gs @ x)
    L = max(2.0 * float(np.linalg.eigvalsh(Gs)[-1]), 1e-12)
    cands = []
    # (a) exponentiated gradient (mirror descent on the simplex) -- different geometry
    x = np.full(m, 1.0 / m)
    for step in (1.0, 0.3, 0.1, 0.03):
        for _ in range(8000):
            g = 2.0 * (Gs @ x)
            z = x * np.exp(-step * (g - float(g @ x)))
            s = z.sum()
            x = z / s if s > 0 else np.full(m, 1.0 / m)
    cands.append(('mirror', x.copy()))
    # (b) monotone projected gradient from uniform
    xp = np.full(m, 1.0 / m)
    for _ in range(60000):
        xn = proj(xp - (2.0 * (Gs @ xp)) / L)
        if f(xn) >= f(xp) - 1e-20:
            break
        xp = xn
    cands.append(('pgd', xp))
    # (c) SLSQP from uniform and from the mirror point
    for tag, x0 in (('slsqp_u', np.full(m, 1.0 / m)), ('slsqp_m', x), ('slsqp_p', xp)):
        r = minimize(f, x0, jac=lambda z: 2.0 * (Gs @ z), method='SLSQP',
                     bounds=[(0.0, 1.0)] * m,
                     constraints={'type': 'eq', 'fun': lambda z: z.sum() - 1.0},
                     options={'ftol': 1e-16, 'maxiter': 2000})
        z = np.clip(r.x, 0.0, None)
        s = z.sum()
        cands.append((tag, z / s if s > 0 else np.full(m, 1.0 / m)))
    vals = [f(z) for _, z in cands]
    bi = int(np.argmin(np.asarray(vals)))
    xb = cands[bi][1]
    # monotone polish
    fb = f(xb)
    for _ in range(60000):
        xn = proj(xb - (2.0 * (Gs @ xb)) / L)
        fn = f(xn)
        if fn >= fb - 1e-21:
            break
        xb, fb = xn, fn
    xb = np.maximum(xb, 0.0) / np.maximum(xb, 0.0).sum()
    return xb, {'solver': cands[bi][0], 'objective': float(xb @ G @ xb),
                'all': {t: float(v * sc) for (t, _), v in zip(cands, vals)}}


# ------------------------------------------------------------------ bank -------------------
def build(b):
    names, P, Q, R, info = H.raw_pool(b)
    A = R[:, :, 0]
    qk, seen = [], set()
    for j, nm in enumerate(names):
        h = hashlib.sha256(Q[j].round(12).tobytes()).hexdigest()
        if h not in seen:
            seen.add(h); qk.append(j)
    ak, seen = [], set()
    for j, nm in enumerate(names):
        h = hashlib.sha256(A[j].round(12).tobytes()).hexdigest()
        if h not in seen:
            seen.add(h); ak.append(j)
    return names, Q, A, sorted(qk), sorted(ak), info


def x2full_fold(b, Q, A, qk, ak, fitm, outer, ranked=True):
    """My independent X2full for one fold. Returns held forecast rows, W, diagnostics."""
    types = sorted(set(b.types.tolist()))
    tl = b.tail_constants(fitm)
    tl = {t: tl[t] / tl[t].sum() for t in types}
    V = np.stack([tl[t] for t in b.types])
    NQ, NR = len(qk), len(ak)
    bank = np.empty((NQ * NR, len(b.ids), 5))
    for jj, j in enumerate(qk):
        for kk, k in enumerate(ak):
            bank[jj * NR + kk] = b.compose_qav(Q[j], A[k], V)
    w = b.row_weights(fitm)
    assert float(np.abs(w[outer]).max()) == 0.0, 'row weights leak onto held rows'
    s = np.sqrt(w)
    D = ((bank - b.truth[None]) * s[None, :, None]).reshape(NQ * NR, -1)
    G = D @ D.T
    W, inf = my_qp(G)
    p = np.einsum('i,ind->nd', W, bank)
    p = p / p.sum(axis=1, keepdims=True)
    return p, W, inf, tl


def main():
    t0 = time.perf_counter()
    b = H.load(DATA)
    n = len(b.ids)
    names, Q, A, qk, ak, pinfo = build(b)
    outer = {f: b.outer_mask(f) for f in range(4)}
    rep['bank'] = {'members': len(names), 'NQ': len(qk), 'NR': len(ak),
                   'crossed': len(qk) * len(ak), 'skipped': pinfo['skipped']}

    # ---- CHECK 3: provenance of every pool member ----------------------------------------
    rep['check3_provenance'] = {
        'pool_equals_RAW_SOURCES': sorted(names) == sorted(RAW_SOURCES),
        'classify_values': sorted(set(classify(p) for p in names)),
        'non_RAW_members': [p for p in names if classify(p) != 'RAW'],
        'n_members': len(names)}

    # ---- CHECK 1 + 6: independent scoring of the family's written predictions -------------
    fam = {}
    for arm in ('Xnull', 'X0', 'X1', 'X2', 'X3', 'X1full', 'X2full', 'X4', 'X5'):
        f = FAM / 'predictions' / f'oof_{arm}.csv'
        d = read_submission(f)
        assert set(d) == set(b.ids)
        M = np.array([d[i] for i in b.ids])
        fam[arm] = M / M.sum(axis=1, keepdims=True)
    targets, groups, reference = b.scoring
    mp = max_points(b.scoring)
    rep['check1_6_independent_scoring'] = {'max_points_from_kit': mp}
    for arm in ('Xnull', 'X2full', 'X1full'):
        cs = contest_score({q: fam[arm][i].tolist() for i, q in enumerate(b.ids)},
                           targets, groups, reference, max_points=mp)
        hs = b.official(fam[arm])
        rep['check1_6_independent_scoring'][arm] = {
            'contest_score_points': cs['technical_points'],
            'contest_score_R': cs['mean_loss_ratio'],
            'harness_points': hs['technical_points'],
            'harness_R': hs['mean_loss_ratio'],
            'by_condition': {c: v['loss_ratio'] for c, v in sorted(cs['by_condition'].items())},
            'episodes_scored': sum(v['episodes'] for v in cs['by_condition'].values()),
            'queries_scored': sum(v['queries'] for v in cs['by_condition'].values())}
    po = b.official(b.protected)
    rep['check1_6_independent_scoring']['protected_recomputed'] = {
        'points': po['technical_points'], 'R': po['mean_loss_ratio']}
    # per-fold-average vs concatenated, to show which one the headline is
    perfold = [b.official(fam['X2full'], episodes=set(np.unique(b.ep[outer[f]]).tolist()))
               ['technical_points'] for f in range(4)]
    rep['check6_concatenated_not_averaged'] = {
        'per_fold_points': perfold, 'mean_of_per_fold': float(np.mean(perfold)),
        'concatenated_once': b.official(fam['X2full'])['technical_points']}

    # ---- CHECK 5: paired interval, recomputed, plus seed robustness -----------------------
    pr = b.paired(fam['X2full'], b.protected)
    rep['check5_paired'] = {'ci95_vs_protected': pr['delta_loss_ratio_ci95'],
                            'upper_below_zero': bool(pr['delta_loss_ratio_ci95'][1] < 0),
                            'by_condition': pr['by_condition_ci95'],
                            'seed': pr['seed'], 'repeats': pr['repeats']}
    rep['check5_paired']['vs_X1full'] = b.paired(fam['X2full'], fam['X1full'])['delta_loss_ratio_ci95']
    # seed sensitivity: the official seed is frozen at 314159; vary it to see fragility
    import slot_model.official_score as OS
    alt = []
    src = OS.paired_bootstrap
    for seed in (1, 2, 3, 7, 11, 101, 777, 2024, 20261108, 999983):
        def pb(a, bb, scoring, episodes, repeats=2000, _s=seed):
            tg, gr, rf = scoring
            sel = [t for t in tg if t.episode_id in set(episodes)]
            rows = {}
            from slot_forecasting.schema import CONDITIONS
            for cond in CONDITIONS:
                es = sorted({t.episode_id for t in sel if gr[t.query_id] == cond})
                vals = []
                for e in es:
                    ts = [t for t in sel if t.episode_id == e]
                    vals.append([np.mean([np.sum((np.asarray(p[t.query_id]) - t.probabilities) ** 2)
                                          for t in ts]) for p in (a, bb, rf)])
                rows[cond] = np.asarray(vals)
            rng = np.random.default_rng(_s)
            tot = np.zeros(repeats)
            for cond, v in rows.items():
                dr = rng.integers(0, len(v), (repeats, len(v)))
                mn = v[dr].mean(1)
                tot += ((mn[:, 0] - mn[:, 1]) / np.maximum(mn[:, 2], 1e-9)) / 4
            return np.quantile(tot, [.025, .975]).tolist()
        alt.append({'seed': seed,
                    'ci95': pb(b.mapping(fam['X2full']), b.mapping(b.protected),
                               b.scoring, b.episodes)})
    for a in alt:
        a['significant'] = bool(a['ci95'][1] < 0)
    rep['check5_seed_robustness'] = {'alt_seeds': alt,
                                     'n_significant': sum(a['significant'] for a in alt),
                                     'n_seeds': len(alt)}

    # ---- CHECK 2 part A: structural -- held rows carry zero fitting weight ---------------
    struct = []
    for f in range(4):
        fitm = ~outer[f]
        w = b.row_weights(fitm)
        tl = b.tail_constants(fitm)
        tl2 = b.tail_constants(fitm & ~outer[f])
        struct.append({'fold': f,
                       'max_abs_rowweight_on_held': float(np.abs(w[outer[f]]).max()),
                       'n_held_queries': int(outer[f].sum()),
                       'tails_identical_when_held_excluded_twice':
                           all(np.allclose(tl[t], tl2[t]) for t in tl)})
    rep['check2_structural'] = struct

    # ---- CHECK 1 part B: independent re-derivation of X2full ----------------------------
    mine = np.empty((n, 5))
    objs = []
    for f in range(4):
        p, W, inf, tl = x2full_fold(b, Q, A, qk, ak, ~outer[f], outer[f])
        mine[outer[f]] = p[outer[f]]
        objs.append({'fold': f, 'my_objective': inf['objective'], 'solver': inf['solver'],
                     'support': int((W > 1e-9).sum()), 'start_objectives': inf['all']})
    mo = b.official(mine)
    fam_obj = [s['objective'] for s in json.loads((FAM / 'evaluation.json').read_text())
               ['solver_diagnostics'] if s['arm'] == 'X2full']
    rep['check1_independent_rederivation'] = {
        'my_points': mo['technical_points'], 'my_R': mo['mean_loss_ratio'],
        'my_by_condition': {c: v['loss_ratio'] for c, v in sorted(mo['by_condition'].items())},
        'family_points': b.official(fam['X2full'])['technical_points'],
        'points_abs_diff': abs(mo['technical_points']
                               - b.official(fam['X2full'])['technical_points']),
        'max_abs_forecast_diff': float(np.abs(mine - fam['X2full']).max()),
        'my_fold_objectives': objs,
        'family_fold_objectives': fam_obj,
        'objective_abs_diff_per_fold': [abs(a['my_objective'] - c)
                                        for a, c in zip(objs, fam_obj)],
        'paired_mine_vs_family': b.paired(mine, fam['X2full'])['delta_loss_ratio_ci95'],
        'paired_mine_vs_protected': b.paired(mine, b.protected)['delta_loss_ratio_ci95']}

    # ---- CHECK 2 part B: empirical leak test on my re-derivation ------------------------
    # Scramble the TRUTH of every held episode of fold f. The fold-f held forecast must be
    # byte-identical; other folds' forecasts must move (proving the perturbation lands).
    leak = []
    for f in range(4):
        b2 = H.load(DATA)
        rows = np.nonzero(outer[f])[0]
        rng = np.random.default_rng(10_000 + f)
        perm = rng.permutation(len(rows))
        b2.truth[rows] = b2.truth[rows][perm]
        # also inject a hard corruption so a leak cannot hide behind a mild permutation
        b2.truth[rows] = 0.5 * b2.truth[rows] + 0.5 * rng.dirichlet(np.ones(5), len(rows))
        b2._wcache.clear()
        n2, Q2, A2, qk2, ak2, _ = build(b2)
        same, other = None, []
        for g in range(4):
            p, _, _, _ = x2full_fold(b2, Q2, A2, qk2, ak2, ~outer[g], outer[g])
            d = float(np.abs(p[outer[g]] - mine[outer[g]]).max())
            if g == f:
                same = d
            else:
                other.append(d)
        leak.append({'perturbed_fold': f, 'max_abs_diff_on_that_folds_held': same,
                     'max_abs_diff_on_other_folds_held': other,
                     'passes_both_directions': bool(same == 0.0 and max(other) > 1e-6)})
    rep['check2_empirical_leak_test'] = leak

    # ---- CHECK 4: causality surface ------------------------------------------------------
    src_txt = (ROOT / 'scripts/general_convex_crossed.py').read_text()
    rep['check4_causality'] = {
        'uses_bundle_index_history': 'b.index' in src_txt or 'causal_index' in src_txt,
        'uses_cutoffs': 'b.cutoffs' in src_txt,
        'uses_signals': 'b.signals' in src_txt,
        'uses_condition_labels_directly': 'b.groups' in src_txt,
        'uses_machine_type': 'b.types' in src_txt,
        'machine_type_in_test_queries': 'machine_type' in
            (DATA / 'test/queries.csv').read_text().splitlines()[0],
        'note': 'X2full consumes only (a) the 15 RAW members forecasts, whose causality is '
                'inherited, and (b) per-machine-type tail means from fitting-side TRUTH. It '
                'reads no raw history, no lags, no cutoffs.'}

    rep['seconds'] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'verification.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
    np.savetxt(OUT / 'my_x2full.csv', mine, delimiter=',', fmt='%.17g')
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
