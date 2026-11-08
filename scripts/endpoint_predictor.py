"""One small causal predictor for the remaining within-family quantity P(1x | positive).

WHY THIS AND NOTHING ELSE. The family projection removed the off-family error (mean L2
0.0365) and gained 0.055333 points. The along-family error is three times larger: mean
|P(1|pos) - truth| = 0.1052. Since the true P(1|pos) takes exactly TWO values per machine
type, the remaining conditional-payout quantity is a two-class identification problem, and
that is what this predictor addresses.

PERMITTED SUPERVISION. Both endpoint values and the mixing weight are estimated from
TRAIN sampled outcomes only, by a two-component binomial mixture fitted with EM over
per-machine counts (n_1x, n_positive) within each machine type. No validation target,
condition label or reference forecast enters the fit. The deployed tail split also comes
from train sampled outcomes.

INFERENCE, causal. For a held machine, count its own STRICTLY PRE-CUTOFF outcomes: n1 =
number of 1x results and ntail = number of 2x/5x/20x results. The component posterior is
    pi_k * Binomial(n1 ; n1 + ntail, u_k)
and the emitted estimate is the posterior MEAN  u_hat = sum_k post_k * u_k, which stays
inside [min u_k, max u_k]. A posterior mean is used rather than a hard assignment because
snapping to the nearer endpoint was measured at 52.340264, BELOW the protected forecast:
committing to the wrong endpoint costs more under squared error than hedging between them.

Then r = [u_hat, (1 - u_hat) * tail_train_type] and q is held byte-identical.

CANDIDATES
  protected        the deployed forecast
  tail_train       family projection keeping the forecast's own P(1|pos)   (already +0.055333)
  mixture_em       this predictor, with endpoints and weights from train EM
  blend            one bounded coefficient between tail_train's P(1|pos) and mixture_em's,
                   fitted per outer fold on the exact official objective
Advancement: held-fold points must improve AND the paired whole-episode interval against
the protected forecast must lie below zero. Same rule as the previous batches, unchanged.

LIMITS. The EM endpoints are point estimates from 47-74 training positive samples per
type, so they carry real estimation error; the recovered values are reported against the
organizer truth for transparency, but the truth is NOT used to fit them. Fractional decay
is deliberately NOT applied to these counts: the endpoint is a machine-level property, so
all permitted positive history is informative and recency weighting would only add
variance. That choice is a declared modelling decision, not a tuned one.
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
from scipy.special import gammaln

from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission

TAILS = (2, 5, 20)


def log_binom_pmf(k, n, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + k * np.log(p) + (n - k) * np.log1p(-p))


def fit_mixture(n1, ntot, iters=500, seed=0):
    """Two-component binomial mixture by EM. Returns (u0, u1, weight0)."""
    rng = np.random.default_rng(seed)
    frac = np.divide(n1, np.maximum(ntot, 1), where=ntot > 0)
    u = np.array([np.quantile(frac[ntot > 0], 0.25), np.quantile(frac[ntot > 0], 0.75)])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = np.column_stack([np.log(w[k]) + log_binom_pmf(n1, ntot, u[k]) for k in (0, 1)])
        m = ll.max(axis=1, keepdims=True)
        post = np.exp(ll - m)
        post /= post.sum(axis=1, keepdims=True)
        w_new = post.mean(axis=0)
        u_new = np.array([float((post[:, k] * n1).sum() / max((post[:, k] * ntot).sum(), 1e-9))
                          for k in (0, 1)])
        if np.max(abs(u_new - u)) < 1e-12 and np.max(abs(w_new - w)) < 1e-12:
            u, w = u_new, w_new
            break
        u, w = u_new, w_new
    order = np.argsort(u)
    return float(u[order[0]]), float(u[order[1]]), float(w[order[0]])


def posterior_mean(n1, ntot, u0, u1, w0):
    l0 = np.log(max(w0, 1e-12)) + log_binom_pmf(n1, ntot, u0)
    l1 = np.log(max(1 - w0, 1e-12)) + log_binom_pmf(n1, ntot, u1)
    m = max(l0, l1)
    p0, p1 = np.exp(l0 - m), np.exp(l1 - m)
    s = p0 + p1
    return float((p0 * u0 + p1 * u1) / s), float(p0 / s)


def positive_counts(ticks, cutoff):
    """Strictly pre-cutoff counts of 1x and of 2x/5x/20x. No decay, by design."""
    n1 = ntail = 0
    for tick, value in ticks.items():
        if tick >= cutoff or value <= 0:
            continue
        if value == 1:
            n1 += 1
        else:
            ntail += 1
    return n1, ntail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/endpoint_predictor')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # ---------- fit on TRAIN only ----------
    tq, th = load_inputs(args.data / 'train')
    tindex = causal_index(tq, th)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    ttype = {q.query_id: q.machine_type for q in tq}

    tail_counts, mix_rows = {}, {}
    for q in tq:
        n1, ntail = positive_counts(tindex[(q.episode_id, q.machine_id)], q.cutoff)
        mix_rows.setdefault(q.machine_type, []).append((n1, n1 + ntail))
    for key, value in labels.items():
        if value in TAILS:
            tail_counts.setdefault(ttype[key], np.zeros(3))[TAILS.index(value)] += 1
    tail_train = {t: (c / c.sum()).tolist() for t, c in tail_counts.items()}

    mixture = {}
    for t, rows in sorted(mix_rows.items()):
        arr = np.array(rows, dtype=np.float64)
        u0, u1, w0 = fit_mixture(arr[:, 0], arr[:, 1])
        mixture[t] = {'u0': u0, 'u1': u1, 'weight0': w0,
                      'machines': int(len(arr)),
                      'median_positive_observations': float(np.median(arr[:, 1]))}
        print(f'  {t:<9} EM endpoints u0={u0:.6f} u1={u1:.6f} w0={w0:.4f}  '
              f'{len(arr)} train machines, median {np.median(arr[:,1]):.0f} positive obs', flush=True)

    # ---------- validation ----------
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
    vtype = {q.query_id: q.machine_type for q in vq}
    vmeta = {q.query_id: (q.episode_id, q.machine_id, q.cutoff) for q in vq}
    types = np.array([vtype[i] for i in ids])

    dep = read_submission(args.package / 'validation_submission.csv')
    dep = np.array([dep[q] for q in ids])
    q_fixed = 1.0 - dep[:, 0]
    r_cur = dep[:, 1:] / np.maximum(q_fixed, 1e-12)[:, None]
    r_cur /= r_cur.sum(axis=1, keepdims=True)
    r_true = truth[:, 1:] / np.maximum(truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)

    u_cur = r_cur[:, 0]
    u_mix = np.empty(len(ids))
    post0 = np.empty(len(ids))
    nobs = np.empty(len(ids))
    for k, qid in enumerate(ids):
        e_, m_, c_ = vmeta[qid]
        n1, ntail = positive_counts(vindex[(e_, m_)], c_)
        nobs[k] = n1 + ntail
        mp = mixture[vtype[qid]]
        u_mix[k], post0[k] = posterior_mean(n1, n1 + ntail, mp['u0'], mp['u1'], mp['weight0'])

    def build(u):
        out = np.empty_like(r_cur)
        for k in range(len(ids)):
            tail = np.asarray(tail_train[types[k]])
            out[k, 0] = u[k]
            out[k, 1:] = (1.0 - u[k]) * tail
        return out / out.sum(axis=1, keepdims=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(r):
        out = np.column_stack((1.0 - q_fixed, q_fixed[:, None] * r))
        return out / out.sum(axis=1, keepdims=True)

    def official(r):
        return score(mapping(compose(r)), scoring, episodes)

    def row_weights(es):
        w = np.zeros(len(ids))
        for c in sorted(set(groups.values())):
            ces = {x for i, x in enumerate(ep) if x in es and groups[ids[i]] == c}
            b = np.mean([np.mean(np.sum((ref_arr[ep == x] - truth[ep == x]) ** 2, axis=1)) for x in sorted(ces)])
            for x in ces:
                w[ep == x] = 1 / (4 * len(ces) * sum(ep == x) * max(b, 1e-9))
        return w

    held = np.empty(len(ids))
    alphas = []
    grid = np.linspace(0, 1, 101)
    for fold in range(4):
        m = np.array([folds[x] == fold for x in ep])
        fit = {x for x in episodes if folds[x] != fold}
        w = row_weights(fit)
        losses = [float(np.sum(w * np.sum((compose(build((1 - g) * u_cur + g * u_mix)) - truth) ** 2,
                                          axis=1))) for g in grid]
        g = float(grid[int(np.argmin(losses))])
        alphas.append(g)
        held[m] = ((1 - g) * u_cur + g * u_mix)[m]

    variants = {'protected': r_cur, 'tail_train': build(u_cur),
                'mixture_em': build(u_mix), 'blend': build(held)}
    results = {k: official(v) for k, v in variants.items()}
    best = max(('tail_train', 'mixture_em', 'blend'),
               key=lambda k: results[k]['technical_points'])
    paired = paired_bootstrap(mapping(compose(variants[best])), mapping(compose(r_cur)),
                              scoring, episodes)
    gain = results[best]['technical_points'] - results['protected']['technical_points']
    lo, hi = paired['delta_loss_ratio_ci95']
    advanced = bool(gain > 0 and hi < 0)

    # accuracy of the endpoint identification, reported against truth for transparency only
    true_u = r_true[:, 0]
    nearest = {}
    for t in sorted(set(types.tolist())):
        m = types == t
        vals = sorted({round(float(x), 9) for x in true_u[m]})
        hard = np.where(post0[m] > 0.5, mixture[t]['u0'], mixture[t]['u1'])
        truth_lo = np.isclose(true_u[m], min(vals))
        pred_lo = post0[m] > 0.5
        nearest[t] = {'true_endpoint_values': vals,
                      'em_recovered': [mixture[t]['u0'], mixture[t]['u1']],
                      'max_abs_endpoint_recovery_error':
                          float(max(abs(mixture[t]['u0'] - vals[0]),
                                    abs(mixture[t]['u1'] - vals[1]))),
                      'hard_assignment_accuracy': float((pred_lo == truth_lo).mean()),
                      'mean_abs_u_error_protected': float(np.mean(abs(u_cur[m] - true_u[m]))),
                      'mean_abs_u_error_mixture': float(np.mean(abs(u_mix[m] - true_u[m])))}

    report = {'preregistration': __doc__, 'train_tail': tail_train, 'train_mixture': mixture,
              'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                              'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                          for k, v in results.items()},
              'blend_alpha_per_fold': alphas, 'best': best, 'gain_over_protected': gain,
              'paired_best_vs_protected': paired, 'advanced': advanced,
              'advancement_rule': 'held-fold points improve AND paired interval upper bound < 0',
              'endpoint_identification': nearest,
              'positive_observations': {'median': float(np.median(nobs)),
                                        'p10': float(np.percentile(nobs, 10))},
              'q_byte_identical': True,
              'exposure': 'Repeatedly inspected public validation; exploratory research round.',
              'seconds': time.perf_counter() - start}
    for k, v in variants.items():
        write_submission(args.out / 'predictions' / f'oof_{k}.csv', mapping(compose(v)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(results['protected']['by_condition'])
    print(f"\n{'candidate':<20}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('protected', 'tail_train', 'mixture_em', 'blend'):
        v = results[k]
        print(f"{k:<20}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print('\nendpoint identification (truth shown for transparency; never used to fit):')
    for t, v in nearest.items():
        print(f"   {t:<9} true {np.round(v['true_endpoint_values'],4)}  EM "
              f"{np.round(v['em_recovered'],4)}  recovery err {v['max_abs_endpoint_recovery_error']:.4f}  "
              f"hard acc {v['hard_assignment_accuracy']:.4f}  "
              f"|u err| {v['mean_abs_u_error_protected']:.4f} -> {v['mean_abs_u_error_mixture']:.4f}")
    print(f"\npositive observations per machine: median {np.median(nobs):.0f}, "
          f"p10 {np.percentile(nobs,10):.0f}")
    print(f"blend alpha per fold {[round(a,2) for a in alphas]}")
    print(f"best {best}: gain {gain:+.6f}; paired 95% [{lo:+.5f}, {hi:+.5f}]")
    print(f"ADVANCEMENT: {'GRANTED' if advanced else 'WITHHELD'}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
