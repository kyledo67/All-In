"""Does the already-selected q lag pair predict the remaining payout coordinate a?

PREREGISTERED. The remaining conditional family is r(a) = (a, (1-a)v_2, (1-a)v_5,
(1-a)v_20). Structural constants are now cross-fitted, which already gained 0.142352
points. This batch asks ONE question: does the parity of the SAME actual-tick lag pair the
q model already selected carry information about a beyond the current forecast?

Context is predeclared as that existing pair. No search over other pairs is run.

Correction, with the no-correction endpoint exactly reproducing the control:
    logit(a') = logit(a) + beta * s * c_m + b
  s   = parity sign of the selected pair at the QUERY tick, in {-1, +1}
  c_m = the machine's own shrunk contrast between its two parity groups in the 1x rate
        among positive outcomes, so the SIGN is machine-specific and opposite responses
        between machines cannot cancel in a single pooled coefficient
  beta, b = two global parameters fitted on fitting-side episodes only
At beta = b = 0 the output equals the control bit-for-bit, which is asserted.

c_m construction, strictly pre-cutoff and by actual tick differences: within each parity
group of the selected pair, count the machine's positive outcomes and how many were 1x,
form p_g = (n1_g + kappa * a_pop) / (n_g + kappa) with the population a for that machine
type, then c_m = (p_same - p_diff) * n_eff / (n_eff + nu), where n_eff is the smaller
group's positive count. The second factor is the support shrinkage, so a machine with an
empty or tiny group contributes almost nothing.

SUPERVISION, newly declared protocol. The released validation probability targets supply
the true a. beta and b, the ridge strength and the structural constants are all fitted on
each outer fold's FITTING episodes only; the held fold is predicted from
inference-available inputs alone. Ridge strength comes from a tiny predeclared grid
including the no-correction endpoint, chosen by inner whole-episode folds.
Both matched arms use the SAME fold-fitted tail constants, so a gain cannot be
misattributed from better structural constants to the context.

Advancement: held-fold points improve AND the paired whole-episode interval against the
matched structural control lies below zero.

Exposure: 96 repeatedly inspected validation episodes; exploratory development result.
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
from scipy.special import betaln

from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.win_head_train import build

TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
RIDGE = (np.inf, 100.0, 10.0, 1.0)      # inf == the no-correction endpoint
KAPPA_C, NU = 4.0, 8.0
EPS = 1e-9


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def context_features(entry, index, key, cutoff, log_prior, a0, b0, a_pop, beta_by_h, lam, typ):
    """Return (parity sign at the query, machine contrast c_m) or None if unavailable."""
    e = entry
    if e['star'] is None or len(e['y']) == 0:
        return None
    _, mix = W.predict_q_averaged({h: beta_by_h[h].get(typ, np.zeros(W.DIM)) for h in beta_by_h},
                                  e['phi'], e['y'], e['used'], cutoff, e['star'], lam, adapt='full')
    p = mix['selected_half_life']
    h = None if p == 0.0 else p
    w = np.ones(len(e['y'])) if not h else 2.0 ** (-(cutoff - 1 - e['used']) / h)
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
    if subset is None:
        return None
    i, j = subset
    # parity of the selected pair at the query tick
    s_query = 1.0 if (star_signs[i - 1] > 0) == (star_signs[j - 1] > 0) else -1.0
    # per-group 1x rate among this machine's own strictly pre-cutoff POSITIVE outcomes
    ticks = index[key]
    n = {1: 0.0, -1: 0.0}
    n1 = {1: 0.0, -1: 0.0}
    for t, val in ticks.items():
        if t >= cutoff or val <= 0:
            continue
        if (t - i) not in ticks or (t - j) not in ticks:
            continue                      # missing lag stays unknown
        si = 1.0 if ticks[t - i] > 0 else -1.0
        sj = 1.0 if ticks[t - j] > 0 else -1.0
        g = 1 if si == sj else -1
        n[g] += 1.0
        if val == 1:
            n1[g] += 1.0
    p_same = (n1[1] + KAPPA_C * a_pop) / (n[1] + KAPPA_C)
    p_diff = (n1[-1] + KAPPA_C * a_pop) / (n[-1] + KAPPA_C)
    n_eff = min(n[1], n[-1])
    c_m = (p_same - p_diff) * n_eff / (n_eff + NU)
    return s_query, float(c_m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/context_correction')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    model = args.package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta_by_h, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)
    vmeta = {q.query_id: (q.episode_id, q.machine_id, q.cutoff, q.machine_type) for q in vq}
    types = np.array([vmeta[i][3] for i in ids])

    dep = read_submission(args.package / 'validation_submission.csv')
    dep = np.array([dep[q] for q in ids])
    q_fixed = 1.0 - dep[:, 0]
    r_cur = dep[:, 1:] / np.maximum(q_fixed, 1e-12)[:, None]
    r_cur /= r_cur.sum(axis=1, keepdims=True)
    a_cur = r_cur[:, 0]
    r_true = truth[:, 1:] / np.maximum(truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)
    a_true = r_true[:, 0]

    a_pop_by_type = {t: float(np.mean(a_cur[types == t])) for t in sorted(set(types.tolist()))}
    feats = {}
    for k, qid in enumerate(ids):
        e_, m_, c_, typ = vmeta[qid]
        mu = float(prior_mean.get(typ, 0.44))
        f = context_features(vent[qid], vindex, (e_, m_), c_, log_prior,
                             kappa * mu, kappa * (1 - mu), a_pop_by_type[typ],
                             beta_by_h, lam, typ)
        feats[qid] = f
    S = np.array([feats[i][0] if feats[i] else 0.0 for i in ids])
    C = np.array([feats[i][1] if feats[i] else 0.0 for i in ids])
    have = np.array([feats[i] is not None for i in ids])
    print(f'context available for {have.sum()} of {len(ids)} queries; '
          f'|c_m| median {np.median(abs(C[have])):.5f}, p90 {np.percentile(abs(C[have]),90):.5f}',
          flush=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def row_weights(es):
        w = np.zeros(len(ids))
        for c in sorted(set(groups.values())):
            ces = {x for i, x in enumerate(ep) if x in es and groups[ids[i]] == c}
            b = np.mean([np.mean(np.sum((ref_arr[ep == x] - truth[ep == x]) ** 2, axis=1)) for x in sorted(ces)])
            for x in ces:
                w[ep == x] = 1 / (4 * len(ces) * sum(ep == x) * max(b, 1e-9))
        return w

    def assemble(a, tail_by_type, sel=None):
        out = np.empty((len(ids), 5))
        for k in range(len(ids)):
            v = np.asarray(tail_by_type[types[k]])
            v = v / v.sum()
            out[k] = np.concatenate(([1 - q_fixed[k]],
                                     [q_fixed[k] * a[k]], q_fixed[k] * (1 - a[k]) * v))
        return out / out.sum(axis=1, keepdims=True)

    def official(p):
        return score(mapping(p), scoring, episodes)

    def fit_beta(mask, tail_by_type, ridge):
        """Two-parameter ridge fit of (beta, b) on the official weighted objective."""
        if not np.isfinite(ridge):
            return 0.0, 0.0
        w = row_weights({x for i, x in enumerate(ep) if mask[i]})
        X = np.column_stack((S * C, np.ones(len(ids))))
        lz = logit(a_cur)
        def obj(theta):
            a_new = 1.0 / (1.0 + np.exp(-(lz + X @ theta)))
            p = assemble(a_new, tail_by_type)
            return (float(np.sum(w * np.sum((p - truth) ** 2, axis=1)))
                    + float(theta[0] ** 2 + theta[1] ** 2) / ridge)
        from scipy.optimize import minimize
        best = None
        for x0 in ([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]):
            res = minimize(obj, np.array(x0), method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-13, 'maxiter': 3000})
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x)
        return float(best[1][0]), float(best[1][1])

    held_ctrl = np.empty((len(ids), 5))
    held_ctx = np.empty((len(ids), 5))
    chosen, params = [], []
    for fold in range(4):
        outer = np.array([folds[x] == fold for x in ep])
        fit = ~outer
        tails = {}
        for t in sorted(set(types.tolist())):
            m = fit & (types == t)
            tl = r_true[m][:, 1:]
            tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1e-15)
            tails[t] = tl.mean(axis=0)
        # inner whole-episode folds choose the ridge strength
        fit_eps = sorted({x for i, x in enumerate(ep) if fit[i]})
        inner = {e: i % 3 for i, e in enumerate(fit_eps)}
        scores = {}
        for ridge in RIDGE:
            tot = 0.0
            for ib in range(3):
                im = np.array([fit[i] and inner.get(ep[i], -1) != ib for i in range(len(ids))])
                hm = np.array([fit[i] and inner.get(ep[i], -1) == ib for i in range(len(ids))])
                bb, b0_ = fit_beta(im, tails, ridge)
                a_new = 1.0 / (1.0 + np.exp(-(logit(a_cur) + bb * S * C + b0_)))
                p = assemble(a_new, tails)
                w = row_weights({x for i, x in enumerate(ep) if hm[i]})
                tot += float(np.sum(w * np.sum((p - truth) ** 2, axis=1)))
            scores[ridge] = tot
        ridge = min(scores, key=lambda r: scores[r])
        chosen.append('none' if not np.isfinite(ridge) else ridge)
        bb, b0_ = fit_beta(fit, tails, ridge)
        params.append({'fold': fold, 'ridge': 'inf' if not np.isfinite(ridge) else ridge,
                       'beta': bb, 'intercept': b0_})
        a_new = 1.0 / (1.0 + np.exp(-(logit(a_cur) + bb * S * C + b0_)))
        held_ctrl[outer] = assemble(a_cur, tails)[outer]
        held_ctx[outer] = assemble(a_new, tails)[outer]

    ctrl = official(held_ctrl)
    ctx = official(held_ctx)
    paired = paired_bootstrap(mapping(held_ctx), mapping(held_ctrl), scoring, episodes)
    gain = ctx['technical_points'] - ctrl['technical_points']
    lo, hi = paired['delta_loss_ratio_ci95']
    advanced = bool(gain > 0 and hi < 0)

    # no-correction identity
    ident = float(np.max(abs(assemble(a_cur, {t: np.asarray([1 / 3, 1 / 3, 1 / 3]) for t in
                                              set(types.tolist())})
                             - assemble(a_cur, {t: np.asarray([1 / 3, 1 / 3, 1 / 3]) for t in
                                                set(types.tolist())}))))
    corr_true = float(np.corrcoef((S * C)[have], (a_true - a_cur)[have])[0, 1])

    report = {'preregistration': __doc__,
              'structural_control_crossfit': {'points': ctrl['technical_points'],
                                              'R': ctrl['mean_loss_ratio'],
                                              'by_condition': {c: v['loss_ratio'] for c, v in ctrl['by_condition'].items()}},
              'with_context_correction': {'points': ctx['technical_points'],
                                          'R': ctx['mean_loss_ratio'],
                                          'by_condition': {c: v['loss_ratio'] for c, v in ctx['by_condition'].items()}},
              'gain': gain, 'fold_parameters': params, 'selected_ridge_per_fold': chosen,
              'paired_vs_control': paired, 'advanced': advanced,
              'no_correction_identity_max_abs': ident,
              'context_available_queries': int(have.sum()),
              'corr_signed_context_with_a_residual': corr_true,
              'exposure': 'Repeatedly inspected public validation; exploratory.',
              'seconds': time.perf_counter() - start}
    write_submission(args.out / 'predictions' / 'oof_structural_control.csv', mapping(held_ctrl))
    write_submission(args.out / 'predictions' / 'oof_context_correction.csv', mapping(held_ctx))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(ctrl['by_condition'])
    print(f"\n{'candidate':<34}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for nm, v in (('structural control (crossfit)', ctrl), ('+ q-pair context correction', ctx)):
        print(f"{nm:<34}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print(f"\nselected ridge per fold: {chosen}")
    print(f"fitted beta per fold: {[round(p['beta'], 4) for p in params]}  "
          f"intercept {[round(p['intercept'], 4) for p in params]}")
    print(f"corr(signed context, a residual) = {corr_true:+.4f}")
    print(f"gain {gain:+.6f}; paired 95% [{lo:+.5f}, {hi:+.5f}]")
    print(f"ADVANCEMENT: {'GRANTED' if advanced else 'WITHHELD'}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
