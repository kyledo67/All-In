"""Learn the population prior over the two parity probabilities; then test dependence.

PROTOCOL, frozen before any public-validation number below was read.

PERMISSION. organizer/RULES.md line 9: "Learn from the main training and validation data,
then submit forecasts for its test queries", and line 39: "Public validation is available
for model selection." Released probability supervision is therefore permitted and this
work is deployable, not merely diagnostic.

FROZEN SELECTIONS. The control's selected lag pair, template and half-life are cached per
query from their proper pre-cutoff histories and REUSED byte-identically in every arm, so
a changed prior cannot silently change model selection and contaminate the attribution.
Only parity-routed queries receive the new prior; null and full routes are untouched. The
conditional positive-payout vector r, its blend weight, tail constants and tail operation
are all frozen and numerically identical across arms.

ARMS
  1 control        the protected prior: one Beta(kappa*mu_type, kappa*(1-mu_type)) for
                   both groups, predicted independently. Exact, not a limiting case.
  2 marginal       learned two-component mixture, delta = 0 (exact independence)
  3 joint          the SAME fitted marginals plus one shared coupling rho
Interpretation: 2 vs 1 is marginal prior calibration; 3 vs 2 is the incremental value of
cross-group dependence; 3 vs 1 is the total.

PARAMETERS. mu_L and mu_H per type as adjustments to the existing type mean, pi per type,
one shared concentration, one shared coupling: 6 + 3 + 1 + 1 = 11 globals, no locally
fitted coefficients.

OBJECTIVE. The exact full-vector loss at frozen r,
    L_i(q) = (1 + ||r_i||^2) q^2 - 2 (u_i + r_i . t_+,i) q + const
with official episode and condition weighting and fitting-side reference normalization.
Not unweighted win Brier.

NESTING. Complete-episode outer folds frozen in advance. Every target-dependent parameter
is fitted on outer fitting episodes only. The regularization strength comes from inner
complete-episode folds, and ALL parameters are refitted inside each inner training
partition so no inner held target leaks through a previously fitted marginal. The marginal
is frozen before the coupling is estimated within each fitting partition. Held-fold
predictions are concatenated and scored once.

REGULARIZATION GRID, predeclared: lambda in {inf, 100, 10, 1}, a ridge pulling the learned
component means toward the control type mean and log-concentration toward log(kappa).
lambda = inf reproduces the control exactly, so "no change" is an available outcome.

Conditional exploratory development: the base recipe and this hypothesis were both shaped
by these 96 repeatedly inspected episodes. Nested fitting creates no independent holdout.
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
from scipy.special import betaln

from slot_model import pair_dict as PD
from slot_model import parity_prior as PP
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.win_head_train import build

TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
LAMBDAS = (np.inf, 100.0, 10.0, 1.0)
TYPES = ('balanced', 'jackpot', 'steady')
INITS = (0.0, 0.6, -0.6)          # deterministic predeclared offsets for mu_L/mu_H spread


def build_cache(data, package, history=None):
    """Per-query frozen selection identities and the two parity groups' decayed counts."""
    model = package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta_by_h, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    vq, vh = load_inputs(data / 'validation')
    index = causal_index(vq, vh if history is None else history)
    ent = build(vq, index)
    cache = {}
    for q in vq:
        e = ent[q.query_id]
        rec = {'type': q.machine_type, 'template': 'fallback'}
        if e['star'] is not None and len(e['y']):
            mu = float(prior_mean.get(q.machine_type, 0.44))
            a0, b0 = kappa * mu, kappa * (1 - mu)
            _, mix = W.predict_q_averaged(
                {h: beta_by_h[h].get(q.machine_type, np.zeros(W.DIM)) for h in beta_by_h},
                e['phi'], e['y'], e['used'], q.cutoff, e['star'], lam, adapt='full')
            p = mix['selected_half_life']
            h = None if p == 0.0 else p
            w = np.ones(len(e['y'])) if not h else 2.0 ** (-(q.cutoff - 1 - e['used']) / h)
            signs = PD.signs_from(e['phi'])
            star_signs = e['star'][1:1 + W.HORIZON]
            best = (-np.inf, 0)
            for ix, (subset, template) in enumerate(TEMPLATES):
                ci, nc = WI._template_cells(signs, subset, template)
                wn = np.bincount(ci, weights=w * e['y'], minlength=nc)
                ls = np.bincount(ci, weights=w * (1 - e['y']), minlength=nc)
                ev = float(np.sum(betaln(a0 + wn, b0 + ls) - betaln(a0, b0))) + float(log_prior[ix])
                if np.isfinite(ev) and ev > best[0]:
                    best = (ev, ix)
            subset, template = TEMPLATES[best[1]]
            rec.update({'template': template, 'subset': subset, 'half_life': h, 'mu': mu})
            if template == 'parity':
                ci, _ = WI._template_cells(signs, subset, 'parity')
                si, _ = WI._template_cells(star_signs[None, :], subset, 'parity')
                wn = np.bincount(ci, weights=w * e['y'], minlength=2)
                ls = np.bincount(ci, weights=w * (1 - e['y']), minlength=2)
                rec['counts'] = ((float(wn[0]), float(ls[0])), (float(wn[1]), float(ls[1])))
                rec['group'] = int(si[0])
        cache[q.query_id] = rec
    return cache, float(kappa), dict(prior_mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_exact')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/parity_prior')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
    types = np.array([cache[i]['type'] for i in ids])
    idxr = np.flatnonzero(routed)
    W0 = np.array([cache[ids[k]]['counts'][0][0] for k in idxr])
    L0 = np.array([cache[ids[k]]['counts'][0][1] for k in idxr])
    W1 = np.array([cache[ids[k]]['counts'][1][0] for k in idxr])
    L1 = np.array([cache[ids[k]]['counts'][1][1] for k in idxr])
    GRP = np.array([cache[ids[k]]['group'] for k in idxr])
    TIDX = np.array([TYPES.index(cache[ids[k]]['type']) for k in idxr])
    print(f'parity-routed {routed.sum()} of {len(ids)}; other routes keep the control q',
          flush=True)

    # exact-loss coefficients at frozen r
    u = 1.0 - truth[:, 0]
    A = 1.0 + np.sum(r_frozen ** 2, axis=1)
    Bc = u + np.sum(r_frozen * truth[:, 1:], axis=1)

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

    def weighted_loss(q, w):
        return float(np.sum(w * (A * q ** 2 - 2 * Bc * q)))

    # ---- parameterisation: 3 types x (mu_L, mu_H, pi) + shared log-conc + shared rho ----
    def unpack(theta):
        out = {}
        for i, t in enumerate(TYPES):
            base = np.log(prior_mean[t] / (1 - prior_mean[t]))
            dl, dh, zp = theta[3 * i:3 * i + 3]
            lo = 1 / (1 + np.exp(-(base - abs(dl))))
            hi = 1 / (1 + np.exp(-(base + abs(dh))))
            out[t] = (min(lo, hi), max(lo, hi), 1 / (1 + np.exp(-zp)))
        conc = float(np.exp(theta[9]))
        rho = float(np.tanh(theta[10]))
        return out, conc, rho

    def predict_all(theta, use_rho):
        per_type, conc, rho = unpack(theta)
        if not use_rho:
            rho = 0.0
        lo = np.array([per_type[t][0] for t in TYPES])[TIDX]
        hi = np.array([per_type[t][1] for t in TYPES])[TIDX]
        pi = np.array([per_type[t][2] for t in TYPES])[TIDX]
        q = q_dep.copy()
        q[idxr] = PP.predict_batch(W0, L0, W1, L1, GRP, lo, hi, conc, pi, rho)
        return q

    def penalty(theta, lam):
        if not np.isfinite(lam):
            return 0.0
        per_type, conc, _ = unpack(theta)
        pen = 0.0
        for t in TYPES:
            lo, hi, _ = per_type[t]
            pen += (lo - prior_mean[t]) ** 2 + (hi - prior_mean[t]) ** 2
        pen += (np.log(conc) - np.log(kappa)) ** 2
        return pen / lam

    def fit(mask, lam, use_rho, start_theta=None, fix_marginal=None):
        if not np.isfinite(lam) and fix_marginal is None:
            return None                      # control arm, no parameters
        w = row_weights(mask)
        if fix_marginal is not None:
            base = fix_marginal.copy()
            def obj(z):
                th = base.copy()
                th[10] = z[0]
                return weighted_loss(predict_all(th, True), w)
            best = None
            for z0 in (0.0, 1.0, -1.0):
                res = minimize(obj, np.array([z0]), method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 400})
                if best is None or res.fun < best[0]:
                    best = (res.fun, res.x)
            th = base.copy(); th[10] = float(best[1][0])
            return th
        best = None
        for s in INITS:
            th0 = np.zeros(11)
            for i in range(3):
                th0[3 * i] = s
                th0[3 * i + 1] = s
            th0[9] = np.log(kappa)
            th0[10] = 0.0
            res = minimize(lambda z: weighted_loss(predict_all(z, use_rho), w) + penalty(z, lam),
                           th0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 6000,
                                    'maxfev': 6000})
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x)
        return best[1]

    # ---- outer folds ----
    held = {'control': q_dep.copy(), 'marginal': q_dep.copy(), 'joint': q_dep.copy()}
    provenance = []
    for fold in range(4):
        outer = np.array([folds[x] == fold for x in ep])
        fit_mask = ~outer
        fit_eps = sorted({x for i, x in enumerate(ep) if fit_mask[i]})
        inner = {e: i % 3 for i, e in enumerate(fit_eps)}
        scores = {}
        for lam in LAMBDAS:
            tot = 0.0
            for ib in range(3):
                itr = np.array([fit_mask[i] and inner.get(ep[i], -1) != ib for i in range(len(ids))])
                ihe = np.array([fit_mask[i] and inner.get(ep[i], -1) == ib for i in range(len(ids))])
                th = fit(itr, lam, False)
                qq = q_dep if th is None else predict_all(th, False)
                tot += weighted_loss(qq, row_weights(ihe))
            scores[lam] = tot
        lam = min(scores, key=lambda L: scores[L])
        th_marg = fit(fit_mask, lam, False)
        th_joint = None if th_marg is None else fit(fit_mask, lam, True, fix_marginal=th_marg)
        if th_marg is not None:
            held['marginal'][outer] = predict_all(th_marg, False)[outer]
            held['joint'][outer] = predict_all(th_joint, True)[outer]
        pt, conc, rho = (unpack(th_joint) if th_joint is not None else ({}, kappa, 0.0))
        provenance.append({'fold': fold, 'lambda': 'inf' if not np.isfinite(lam) else lam,
                           'inner_scores': {('inf' if not np.isfinite(k) else str(k)): v
                                            for k, v in scores.items()},
                           'concentration': conc, 'rho': rho,
                           'per_type': {t: list(v) for t, v in pt.items()}})
        print(f"  fold {fold}: lambda={'inf' if not np.isfinite(lam) else lam}  "
              f"conc={conc:.4f}  rho={rho:+.4f}", flush=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def assemble(q):
        out = np.column_stack((1 - q, q[:, None] * r_frozen))
        return out / out.sum(axis=1, keepdims=True)

    def official(q):
        return score(mapping(assemble(q)), scoring, episodes)

    res = {k: official(v) for k, v in held.items()}
    ctrl_match = float(np.max(abs(assemble(held['control']) - dep / dep.sum(axis=1, keepdims=True))))
    pairs = {
        'marginal_vs_control': paired_bootstrap(mapping(assemble(held['marginal'])),
                                                mapping(assemble(held['control'])), scoring, episodes),
        'joint_vs_marginal': paired_bootstrap(mapping(assemble(held['joint'])),
                                              mapping(assemble(held['marginal'])), scoring, episodes),
        'joint_vs_control': paired_bootstrap(mapping(assemble(held['joint'])),
                                             mapping(assemble(held['control'])), scoring, episodes)}

    report = {'protocol': __doc__,
              'permission': 'organizer/RULES.md line 9 "Learn from the main training and '
                            'validation data"; line 39 "Public validation is available for model '
                            'selection." Released probability supervision is permitted.',
              'parity_routed': int(routed.sum()), 'total_queries': len(ids),
              'control_reproduces_package_to': ctrl_match,
              'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                              'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                          for k, v in res.items()},
              'gains': {'marginal_minus_control': res['marginal']['technical_points'] - res['control']['technical_points'],
                        'joint_minus_marginal': res['joint']['technical_points'] - res['marginal']['technical_points'],
                        'joint_minus_control': res['joint']['technical_points'] - res['control']['technical_points']},
              'paired': {k: {'ci95_delta_R': v['delta_loss_ratio_ci95'],
                             'units': 'difference in normalized error R; negative favours the '
                                      'FIRST named forecast'} for k, v in pairs.items()},
              'fold_provenance': provenance,
              'exposure': 'Conditional exploratory development on 96 repeatedly inspected episodes.',
              'seconds': time.perf_counter() - start}
    for k, v in held.items():
        write_submission(args.out / 'predictions' / f'oof_{k}.csv', mapping(assemble(v)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(res['control']['by_condition'])
    print(f"\n{'arm':<28}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('control', 'marginal', 'joint'):
        v = res[k]
        print(f"{k:<28}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print(f"\ncontrol arm reproduces the packaged forecast to {ctrl_match:.2e}")
    g = report['gains']
    print(f"marginal - control {g['marginal_minus_control']:+.6f}   "
          f"joint - marginal {g['joint_minus_marginal']:+.6f}   "
          f"joint - control {g['joint_minus_control']:+.6f}")
    for k, v in report['paired'].items():
        ci = v['ci95_delta_R']
        print(f"   {k:<22} dR 95% [{ci[0]:+.5f}, {ci[1]:+.5f}]"
              + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
