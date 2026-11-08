"""Bounded full-vector residual correction on top of the protected release, gated on the
fitting side only.

Everything measured so far corrects the POSITIVE PROBABILITY q at a frozen conditional
vector r. This asks a different question: does a small, bounded correction to the whole
five-vector help, when the correction is allowed to move the zero mass and the 1x mass
independently rather than through a single q?

FORM. The five categories are collapsed to three masses
    m = (P(0), P(1), P({2,5,20}))
because the two-endpoint study found the within-tail split among {2,5,20} constant to 1e-9
per machine type. That split is taken from the protected release and FROZEN, so the tail is
reallocated proportionally and never reshaped.

The correction is a residual in log space against the protected release's own masses b:
    m = softmax(log b + X C),  C's reference row (category 0) fixed at zero.
At C = 0 this reproduces the protected forecast EXACTLY, so "no change" is representable
and is also the lambda = inf endpoint of the grid.

INPUTS, six, hence 2 free rows x 6 = 12 free coefficients:
  1 intercept
  2 indicator machine_type == 'balanced'
  3 indicator machine_type == 'jackpot'
  4 query signal: the queried row's lag-1 outcome coded +1 positive / -1 zero, and 0 when
    tick cutoff-1 is ABSENT. Missing is its own state, not a loss.
  5 queried-group evidence fraction: the share of the machine's decay-weighted evidence
    that sits in the queried parity group, 1.0 for null/full routes where all evidence is
    one cell.
  6 opposite-group posterior mean: the other parity group's Beta posterior mean under the
    shipped prior; for null/full routes the machine's overall posterior mean.
Inputs 5 and 6 are centred and scaled using FITTING-side statistics only.

EXTENDED variant, eight inputs hence 16 coefficients, adds:
  7 signed current-run length over ACTUAL ticks, capped at 8: +k if the k most recent
    observed outcomes were all positive, -k if all zero. Actual tick differences, never
    row offsets.
  8 missingness flag: 1 if any of ticks cutoff-1 .. cutoff-8 is absent.

OBJECTIVE. The official weighted full five-vector squared error with episode and condition
weighting and fitting-side reference normalization. Not a surrogate.

GATING. For each outer fold the arm is ADOPTED only if it beats the protected baseline on
inner complete-episode folds carved from that fold's fitting episodes. If it does not, the
outer held fold receives the protected forecast unchanged. The outer held targets never
influence whether the correction is used, how strongly it is regularized, or which input
set is chosen. The gate decision per fold is reported.

NESTING. Complete-episode outer folds frozen in advance; all coefficients, the ridge
strength, the input-set choice, the feature scaling and the gate are fitted on outer
fitting episodes only. Held-fold forecasts are concatenated and scored once.

Conditional exploratory development: shaped by the same 96 repeatedly inspected episodes;
nested fitting creates no independent holdout.
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

from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.parity_prior_run import build_cache

LAMBDAS = (np.inf, 100.0, 10.0, 1.0, 0.1)
SETS = {'base6': 6, 'extended8': 8}
HORIZON = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_exact')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/residual_fallback')
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
    dep = dep / dep.sum(axis=1, keepdims=True)
    # three masses and the FROZEN within-tail split
    base = np.column_stack((dep[:, 0], dep[:, 1], dep[:, 2:].sum(axis=1)))
    tail_split = dep[:, 2:] / np.maximum(base[:, 2:3], 1e-15)
    tail_split = tail_split / np.maximum(tail_split.sum(axis=1, keepdims=True), 1e-15)
    logb = np.log(np.clip(base, 1e-12, None))

    # ---- causal features ----
    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    by_id = {q.query_id: q for q in vq}
    sig, runs, miss, qfrac, omean = [], [], [], [], []
    for qid in ids:
        q = by_id[qid]
        obs = index[(q.episode_id, q.machine_id)]
        last = obs.get(q.cutoff - 1)
        sig.append(0.0 if last is None else (1.0 if last > 0 else -1.0))
        r, anymiss = 0, False
        for k in range(1, HORIZON + 1):
            v = obs.get(q.cutoff - k)
            if v is None:
                anymiss = True
                break
            s = 1.0 if v > 0 else -1.0
            if k == 1:
                r = s
            elif s == np.sign(r):
                r += s
            else:
                break
        for k in range(1, HORIZON + 1):
            if obs.get(q.cutoff - k) is None:
                anymiss = True
        runs.append(float(np.clip(r, -HORIZON, HORIZON)))
        miss.append(1.0 if anymiss else 0.0)
        rec = cache[qid]
        mu = float(prior_mean.get(q.machine_type, 0.44))
        if rec['template'] == 'parity':
            (w0, l0), (w1, l1) = rec['counts']
            g = rec['group']
            tot = w0 + l0 + w1 + l1
            own = (w0 + l0) if g == 0 else (w1 + l1)
            qfrac.append(own / tot if tot > 0 else 1.0)
            ow, ol = (w1, l1) if g == 0 else (w0, l0)
            omean.append((kappa * mu + ow) / (kappa + ow + ol))
        else:
            qfrac.append(1.0)
            omean.append(mu)
    types = np.array([by_id[i].machine_type for i in ids])
    RAWX = np.column_stack((
        np.ones(len(ids)),
        (types == 'balanced').astype(float),
        (types == 'jackpot').astype(float),
        np.array(sig), np.array(qfrac), np.array(omean),
        np.array(runs), np.array(miss)))
    SCALE_COLS = (4, 5, 6)

    def design(mask, name):
        """Feature matrix with columns 5,6,7 centred/scaled on FITTING rows only."""
        X = RAWX[:, :SETS[name]].copy()
        stats = {}
        for c in SCALE_COLS:
            if c >= X.shape[1]:
                continue
            m = float(RAWX[mask, c].mean())
            s = float(RAWX[mask, c].std()) or 1.0
            X[:, c] = (RAWX[:, c] - m) / s
            stats[c] = (m, s)
        return X, stats

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

    def forecast(C, X):
        """C has shape (2, p): rows for categories 1 and 2; category 0 is the reference."""
        eta = logb.copy()
        eta[:, 1:] += X @ C.T
        eta -= eta.max(axis=1, keepdims=True)
        m = np.exp(eta)
        m /= m.sum(axis=1, keepdims=True)
        return np.column_stack((m[:, 0], m[:, 1], m[:, 2:3] * tail_split))

    def wloss(p, w):
        return float(np.sum(w * np.sum((p - truth) ** 2, axis=1)))

    def fit(mask, name, lam):
        if not np.isfinite(lam):
            return None, None
        X, stats = design(mask, name)
        w = row_weights(mask)
        p = X.shape[1]

        def obj(z):
            C = z.reshape(2, p)
            return wloss(forecast(C, X), w) + float(np.sum(C[:, 1:] ** 2)) / lam

        r = minimize(obj, np.zeros(2 * p), method='L-BFGS-B',
                     options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        return r.x.reshape(2, p), stats

    def apply_fit(C, stats, name):
        X = RAWX[:, :SETS[name]].copy()
        for c, (m, s) in stats.items():
            X[:, c] = (RAWX[:, c] - m) / s
        return forecast(C, X)

    held = dep.copy()
    prov = []
    for fold in range(4):
        outer = np.array([folds[x] == fold for x in ep])
        fm = ~outer
        fit_eps = sorted({x for i, x in enumerate(ep) if fm[i]})
        inner = {e: i % 3 for i, e in enumerate(fit_eps)}
        parts = []
        for ib in range(3):
            itr = np.array([fm[i] and inner.get(ep[i], -1) != ib for i in range(len(ids))])
            ihe = np.array([fm[i] and inner.get(ep[i], -1) == ib for i in range(len(ids))])
            parts.append((itr, row_weights(ihe)))
        base_inner = sum(wloss(dep, wh) for _, wh in parts)
        inner_scores = {}
        for name in SETS:
            for lam in LAMBDAS:
                tot = 0.0
                for itr, wh in parts:
                    C, st = fit(itr, name, lam)
                    tot += base_inner / 3 if C is None else wloss(apply_fit(C, st, name), wh)
                inner_scores[f'{name}|{"inf" if not np.isfinite(lam) else lam}'] = tot
        # lambda=inf is the baseline, so recompute its inner score exactly
        for name in SETS:
            inner_scores[f'{name}|inf'] = base_inner
        key = min(inner_scores, key=lambda k: inner_scores[k])
        name, ls = key.split('|')
        lam = np.inf if ls == 'inf' else float(ls)
        adopted = bool(np.isfinite(lam) and inner_scores[key] < base_inner - 1e-15)
        if adopted:
            C, st = fit(fm, name, lam)
            held[outer] = apply_fit(C, st, name)[outer]
            coef = C.tolist()
        else:
            coef = None
        prov.append({'fold': fold, 'selected': key, 'adopted': adopted,
                     'inner_baseline': base_inner,
                     'inner_selected': inner_scores[key],
                     'inner_scores': inner_scores,
                     'coefficients_cat1_cat2': coef})
        print(f"  fold {fold}: inner pick {key}  adopted={adopted}  "
              f"inner {inner_scores[key]:.6f} vs baseline {base_inner:.6f}", flush=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    res_base = score(mapping(dep), scoring, episodes)
    res_res = score(mapping(held / held.sum(axis=1, keepdims=True)), scoring, episodes)
    pb = paired_bootstrap(mapping(held / held.sum(axis=1, keepdims=True)),
                          mapping(dep), scoring, episodes)
    rep = {'protocol': __doc__,
           'results': {'protected': {'points': res_base['technical_points'],
                                     'R': res_base['mean_loss_ratio'],
                                     'by_condition': {c: x['loss_ratio'] for c, x in res_base['by_condition'].items()}},
                       'residual_fallback': {'points': res_res['technical_points'],
                                             'R': res_res['mean_loss_ratio'],
                                             'by_condition': {c: x['loss_ratio'] for c, x in res_res['by_condition'].items()}}},
           'gain': res_res['technical_points'] - res_base['technical_points'],
           'paired_vs_protected': {'ci95_delta_R': pb['delta_loss_ratio_ci95'],
                                   'units': 'difference in normalized error R; negative '
                                            'favours the residual fallback'},
           'folds_adopted': sum(1 for p in prov if p['adopted']),
           'fold_provenance': prov,
           'tail_split_frozen': True,
           'max_abs_change_vs_protected': float(np.max(np.abs(held - dep))),
           'exposure': 'Conditional exploratory development on 96 repeatedly inspected '
                       'episodes; nested fitting creates no independent holdout.',
           'seconds': time.perf_counter() - start}
    write_submission(args.out / 'predictions' / 'oof_residual_fallback.csv',
                     mapping(held / held.sum(axis=1, keepdims=True)))
    (args.out / 'evaluation.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')

    conds = sorted(res_base['by_condition'])
    print(f"\n{'arm':<22}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k, v in (('protected', res_base), ('residual_fallback', res_res)):
        print(f"{k:<22}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    ci = pb['delta_loss_ratio_ci95']
    print(f"\ngain {rep['gain']:+.6f}   dR 95% [{ci[0]:+.5f}, {ci[1]:+.5f}]"
          + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    print(f"folds adopted {rep['folds_adopted']}/4   max abs change "
          f"{rep['max_abs_change_vs_protected']:.6f}")
    print(f"wrote {args.out / 'evaluation.json'} in {rep['seconds']:.1f}s")


if __name__ == '__main__':
    main()
