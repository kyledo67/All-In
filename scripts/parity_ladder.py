"""Four-arm prior-calibration ladder on the parity-routed queries.

Each rung adds exactly one kind of freedom, so a gain can be attributed to the structure
that produced it rather than to "a learned prior" in general.

  A protected    the shipped prior: one Beta(kappa*mu_type, kappa*(1-mu_type)) shared by
                 both parity groups, each group's posterior mean taken independently.
                 Zero fitted parameters. Reproduces the packaged forecast exactly.
  B single_beta  the SAME single-Beta form, RECALIBRATED: one mean offset per machine type
                 plus one shared concentration. 4 parameters. This is the cheap rung the
                 earlier run skipped: it asks whether the shipped kappa and type means are
                 simply mistuned, separately from whether two components are needed.
  C mixture_ind  two-component Beta mixture marginals per type (low mean, high mean, mixing
                 weight) plus one shared concentration, groups independent. 10 parameters.
  D mixture_cpl  C's fitted marginals held FIXED, plus one shared coupling rho that changes
                 only the dependence between the two groups. 11 parameters. Marginals are
                 matched to C by construction, so D minus C isolates dependence.

Interpretation: B-A is recalibration of the existing form. C-B is the value of a second
component. D-C is the value of cross-group dependence. Confounding the three is exactly
what the earlier two-arm run did.

PERMISSION. organizer/RULES.md line 9 "Learn from the main training and validation data";
line 39 "Public validation is available for model selection."

FROZEN ACROSS ARMS. build_cache supplies each query's selected lag pair, template,
half-life and decayed group counts from its proper pre-cutoff history; every arm reuses
them byte-identically. Non-parity routes keep the control q. The conditional positive
vector r, its blend weight, tail constants and tail operation are frozen and numerically
identical in all four arms, so only the prior moves.

OBJECTIVE. The exact official full-vector loss at frozen r,
    L_i(q) = (1 + ||r_i||^2) q^2 - 2 (u_i + r_i . t_+,i) q + const,
with official episode and condition weighting and fitting-side reference normalization.
Not unweighted win Brier.

NESTING. Complete-episode outer folds frozen in advance. Every target-dependent parameter
is fitted on outer fitting episodes only. Each arm's ridge strength is chosen on inner
complete-episode folds carved from the outer fitting episodes, and all of that arm's
parameters are refitted inside each inner training partition. D's marginal is frozen from
C fitted on the same fitting partition before rho is estimated. Held-fold predictions are
concatenated and scored once.

GRID, predeclared: lambda in {inf, 100, 10, 1}. lambda = inf reproduces arm A exactly, so
"no change" is an available outcome for every rung.

Conditional exploratory development: the base recipe and these hypotheses were shaped by
the same 96 repeatedly inspected episodes. Nested fitting creates no independent holdout.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.optimize import minimize

from slot_model import parity_prior as PP
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import read_submission, write_submission
from scripts.parity_prior_run import TYPES, LAMBDAS, INITS, build_cache

ARMS = ('protected', 'single_beta', 'mixture_independent', 'mixture_coupled')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_exact')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/parity_ladder')
    args = ap.parse_args()
    (args.out / 'predictions').mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    cache, kappa, prior_mean = build_cache(args.data, args.package)
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    dep = read_submission(args.package / 'validation_submission.csv')
    dep = np.array([dep[q] for q in ids])
    q_dep = 1.0 - dep[:, 0]
    r_frozen = dep[:, 1:] / np.maximum(q_dep, 1e-12)[:, None]
    r_frozen /= r_frozen.sum(axis=1, keepdims=True)

    routed = np.array([cache[i]['template'] == 'parity' for i in ids])
    idxr = np.flatnonzero(routed)
    W0 = np.array([cache[ids[k]]['counts'][0][0] for k in idxr])
    L0 = np.array([cache[ids[k]]['counts'][0][1] for k in idxr])
    W1 = np.array([cache[ids[k]]['counts'][1][0] for k in idxr])
    L1 = np.array([cache[ids[k]]['counts'][1][1] for k in idxr])
    GRP = np.array([cache[ids[k]]['group'] for k in idxr])
    TIDX = np.array([TYPES.index(cache[ids[k]]['type']) for k in idxr])
    HALF = np.full(len(idxr), 0.5)
    BASE = np.array([np.log(prior_mean[t] / (1 - prior_mean[t])) for t in TYPES])
    PM = np.array([prior_mean[t] for t in TYPES])
    print(f'parity-routed {routed.sum()} of {len(ids)}; other routes keep the control q',
          flush=True)

    u = 1.0 - truth[:, 0]
    A_ = 1.0 + np.sum(r_frozen ** 2, axis=1)
    B_ = u + np.sum(r_frozen * truth[:, 1:], axis=1)

    def row_weights(mask):
        w = np.zeros(len(ids))
        es = {x for i, x in enumerate(ep) if mask[i]}
        for c in sorted(set(groups.values())):
            ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == c}
            if not ces:
                continue
            b = np.mean([np.mean(np.sum((ref_arr[ep == e] - truth[ep == e]) ** 2, axis=1))
                         for e in ces])
            for e in ces:
                w[ep == e] = 1 / (4 * len(ces) * sum(ep == e) * max(b, 1e-9))
        return w

    def wloss(q, w):
        return float(np.sum(w * (A_ * q ** 2 - 2 * B_ * q)))

    # ---- arm B: single Beta, 3 mean offsets + shared log-concentration ----
    def unpack_B(th):
        mu = 1.0 / (1.0 + np.exp(-(BASE + th[:3])))
        return mu, float(np.exp(th[3]))

    def predict_B(th):
        mu, conc = unpack_B(th)
        m = mu[TIDX]
        q = q_dep.copy()
        q[idxr] = PP.predict_batch(W0, L0, W1, L1, GRP, m, m, conc, HALF, 0.0)
        return q

    def pen_B(th, lam):
        if not np.isfinite(lam):
            return 0.0
        mu, conc = unpack_B(th)
        return (float(np.sum((mu - PM) ** 2)) + (np.log(conc) - np.log(kappa)) ** 2) / lam

    # ---- arms C/D: two-component mixture ----
    def unpack_CD(th):
        lo = 1.0 / (1.0 + np.exp(-(BASE - np.abs(th[0:9:3]))))
        hi = 1.0 / (1.0 + np.exp(-(BASE + np.abs(th[1:9:3]))))
        pi = 1.0 / (1.0 + np.exp(-th[2:9:3]))
        return np.minimum(lo, hi), np.maximum(lo, hi), pi, float(np.exp(th[9])), float(np.tanh(th[10]))

    def predict_CD(th, use_rho):
        lo, hi, pi, conc, rho = unpack_CD(th)
        q = q_dep.copy()
        q[idxr] = PP.predict_batch(W0, L0, W1, L1, GRP, lo[TIDX], hi[TIDX], conc, pi[TIDX],
                                   rho if use_rho else 0.0)
        return q

    def pen_CD(th, lam):
        if not np.isfinite(lam):
            return 0.0
        lo, hi, _, conc, _ = unpack_CD(th)
        return (float(np.sum((lo - PM) ** 2 + (hi - PM) ** 2))
                + (np.log(conc) - np.log(kappa)) ** 2) / lam

    def fit_B(mask, lam):
        if not np.isfinite(lam):
            return None
        w = row_weights(mask)
        best = None
        for s in INITS:
            th0 = np.array([s, s, s, np.log(kappa)])
            r = minimize(lambda z: wloss(predict_B(z), w) + pen_B(z, lam), th0,
                         method='Nelder-Mead',
                         options={'xatol': 1e-7, 'fatol': 1e-11, 'maxiter': 3000, 'maxfev': 3000})
            if best is None or r.fun < best[0]:
                best = (r.fun, r.x, r)
        return best[1]

    def fit_C(mask, lam):
        if not np.isfinite(lam):
            return None
        w = row_weights(mask)
        best = None
        for s in INITS:
            th0 = np.zeros(11)
            th0[0:9:3] = s
            th0[1:9:3] = s
            th0[9] = np.log(kappa)
            r = minimize(lambda z: wloss(predict_CD(z, False), w) + pen_CD(z, lam), th0,
                         method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 6000, 'maxfev': 6000})
            if best is None or r.fun < best[0]:
                best = (r.fun, r.x)
        return best[1]

    def fit_D(mask, base):
        if base is None:
            return None
        w = row_weights(mask)
        best = None
        for z0 in (0.0, 1.0, -1.0):
            def obj(z, b=base):
                th = b.copy(); th[10] = z[0]
                return wloss(predict_CD(th, True), w)
            r = minimize(obj, np.array([z0]), method='Nelder-Mead',
                         options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 400})
            if best is None or r.fun < best[0]:
                best = (r.fun, r.x)
        th = base.copy(); th[10] = float(best[1][0])
        return th

    held = {a: q_dep.copy() for a in ARMS}
    prov = []
    for fold in range(4):
        outer = np.array([folds[x] == fold for x in ep])
        fm = ~outer
        fit_eps = sorted({x for i, x in enumerate(ep) if fm[i]})
        inner = {e: i % 3 for i, e in enumerate(fit_eps)}
        parts = []
        for ib in range(3):
            itr = np.array([fm[i] and inner.get(ep[i], -1) != ib for i in range(len(ids))])
            parts.append((itr, row_weights(np.array(
                [fm[i] and inner.get(ep[i], -1) == ib for i in range(len(ids))]))))
        sc = {'single_beta': {}, 'mixture_independent': {}}
        for lam in LAMBDAS:
            tb = tc = 0.0
            for itr, wh in parts:
                thb = fit_B(itr, lam)
                tb += wloss(q_dep if thb is None else predict_B(thb), wh)
                thc = fit_C(itr, lam)
                tc += wloss(q_dep if thc is None else predict_CD(thc, False), wh)
            sc['single_beta'][lam] = tb
            sc['mixture_independent'][lam] = tc
        lam_b = min(sc['single_beta'], key=lambda L: sc['single_beta'][L])
        lam_c = min(sc['mixture_independent'], key=lambda L: sc['mixture_independent'][L])
        th_b = fit_B(fm, lam_b)
        th_c = fit_C(fm, lam_c)
        th_d = fit_D(fm, th_c)
        if th_b is not None:
            held['single_beta'][outer] = predict_B(th_b)[outer]
        if th_c is not None:
            held['mixture_independent'][outer] = predict_CD(th_c, False)[outer]
            held['mixture_coupled'][outer] = predict_CD(th_d, True)[outer]
        rec = {'fold': fold,
               'lambda_single_beta': 'inf' if not np.isfinite(lam_b) else lam_b,
               'lambda_mixture': 'inf' if not np.isfinite(lam_c) else lam_c,
               'inner_scores': {a: {('inf' if not np.isfinite(k) else str(k)): v
                                    for k, v in d.items()} for a, d in sc.items()}}
        if th_b is not None:
            mu, conc = unpack_B(th_b)
            rec['single_beta_fit'] = {'type_means': dict(zip(TYPES, mu.tolist())),
                                      'shipped_type_means': dict(zip(TYPES, PM.tolist())),
                                      'concentration': conc, 'shipped_concentration': kappa}
        if th_d is not None:
            lo, hi, pi, conc, rho = unpack_CD(th_d)
            rec['mixture_fit'] = {'low': dict(zip(TYPES, lo.tolist())),
                                  'high': dict(zip(TYPES, hi.tolist())),
                                  'pi': dict(zip(TYPES, pi.tolist())),
                                  'concentration': conc, 'rho': rho}
        prov.append(rec)
        print(f"  fold {fold}: lambda_B={rec['lambda_single_beta']} "
              f"lambda_C={rec['lambda_mixture']}"
              + (f"  B conc={rec['single_beta_fit']['concentration']:.4f}"
                 if th_b is not None else '  B=control')
              + (f"  rho={rec['mixture_fit']['rho']:+.4f}" if th_d is not None else ''),
              flush=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def assemble(q):
        o = np.column_stack((1 - q, q[:, None] * r_frozen))
        return o / o.sum(axis=1, keepdims=True)

    res = {a: score(mapping(assemble(held[a])), scoring, episodes) for a in ARMS}
    ctrl = float(np.max(abs(assemble(held['protected'])
                            - dep / dep.sum(axis=1, keepdims=True))))
    rungs = [('single_beta', 'protected'), ('mixture_independent', 'single_beta'),
             ('mixture_coupled', 'mixture_independent'), ('mixture_coupled', 'protected')]
    paired = {f'{a}_vs_{b}': paired_bootstrap(mapping(assemble(held[a])),
                                              mapping(assemble(held[b])), scoring, episodes)
              for a, b in rungs}

    rep = {'protocol': __doc__,
           'parity_routed': int(routed.sum()), 'total_queries': len(ids),
           'protected_reproduces_package_to': ctrl,
           'results': {a: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                           'by_condition': {c: x['loss_ratio']
                                            for c, x in v['by_condition'].items()}}
                       for a, v in res.items()},
           'rungs': {f'{a}_minus_{b}': res[a]['technical_points'] - res[b]['technical_points']
                     for a, b in rungs},
           'paired': {k: {'ci95_delta_R': v['delta_loss_ratio_ci95'],
                          'units': 'difference in normalized error R; negative favours the '
                                   'FIRST named arm'} for k, v in paired.items()},
           'fold_provenance': prov,
           'exposure': 'Conditional exploratory development on 96 repeatedly inspected '
                       'episodes; nested fitting creates no independent holdout.',
           'seconds': time.perf_counter() - start}
    for a in ARMS:
        write_submission(args.out / 'predictions' / f'oof_{a}.csv', mapping(assemble(held[a])))
    (args.out / 'evaluation.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')

    conds = sorted(res['protected']['by_condition'])
    print(f"\n{'arm':<24}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for a in ARMS:
        v = res[a]
        print(f"{a:<24}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print(f"\narm A reproduces the packaged forecast to {ctrl:.2e}")
    for k, v in rep['rungs'].items():
        print(f"  {k:<44}{v:+.6f}")
    for k, v in rep['paired'].items():
        ci = v['ci95_delta_R']
        print(f"  {k:<44}dR 95% [{ci[0]:+.5f}, {ci[1]:+.5f}]"
              + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    print(f"wrote {args.out / 'evaluation.json'} in {rep['seconds']:.1f}s")


if __name__ == '__main__':
    main()
