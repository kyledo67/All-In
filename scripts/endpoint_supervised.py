"""FAMILY E -- supervised prediction of the CURRENT positive-payout endpoint.

ESTABLISHED STRUCTURE (independently verified in scripts/endpoint_family.py and re-asserted
here from the fitting side only). Within each machine type the TRUE conditional positive
payout vector takes exactly two values, sharing an identical split among {2x,5x,20x} and
differing only in u = P(1x | positive). So the conditional head is one-parameter per type:

    r(u) = [u, (1-u) * tail_type]

The protected release is EXACTLY this family: its q, its own u, and per-outer-fold tail
constants refit on that fold's fitting episodes. The script asserts that reconstruction
reproduces the release to < 1e-12, so every arm here is nested around it and differs ONLY
in u. That makes the zero-correction arm a byte-matched control.

WHY THE PRIOR WORK FAILED. scripts/endpoint_predictor.py estimated the two endpoints by
UNSUPERVISED two-component binomial EM over per-machine (n_1x, n_positive) counts and then
formed a Binomial posterior mean. Two things break:
  (1) the per-machine 1x fraction among positives is strongly COMPRESSED toward the midpoint
      relative to the endpoint values (measured on fold-0 fitting episodes: steady endpoints
      0.455/0.754 but observed fractions 0.565/0.659; balanced 0.214/0.500 -> 0.312/0.374;
      jackpot 0.121/0.335 -> 0.174/0.241). A machine spends only about 65% of its history in
      its current endpoint, so the time average is a blurred mixture, not the endpoint.
      Unsupervised EM therefore recovers the COMPRESSED means, not the endpoints.
  (2) with a median of 74 positive observations, a Binomial likelihood at those means is
      almost a hard assignment, and the empirical overdispersion is large (effective sample
      size about 33 of 120). An overconfident posterior at 56-72% accuracy is worse than
      hedging, which is why snap_nearest measured 52.340264, below the release.
The fix is proper supervision: endpoint VALUES from the fitting partition's probability
targets, and the evidence->endpoint map CALIBRATED on the fitting partition rather than
assumed binomial.

A 2-state HMM filter over ticks (symmetric per-tick switch probability s, emissions only at
positive ticks) was tried during design on fold-0 fitting episodes: the fitting objective is
monotone in s and optimal as s -> 0, i.e. no usable slow switching is identifiable. DATA.md
permits winning amounts to change, and they clearly do, but not as a slowly drifting
machine-level level. What IS identifiable is STATE-matched evidence: the machine's own 1x
rate among positives whose actual-tick lag pattern matches the pattern at the query tick.

ARMS. All share q (byte-identical to the protected release) and the per-fold tail constants;
they differ only in u. Every arm's final answer is a posterior-weighted endpoint MEAN,
reconstructed into the whole five-vector and selected on the official objective.
  E0a zero_correction        u = release u. Reproduces the protected release exactly.
  E0b0 train_em_pure         UNSUPERVISED two-component binomial EM on TRAIN counts only,
                             Binomial posterior mean, full-history causal counts. The old
                             endpoint_predictor recipe, reimplemented on these folds.
  E0b  train_em_blend        the same with a nested blend weight toward the release u
                             (the original chose that weight on outer fitting episodes; here
                             it is chosen on inner partitions).
  E0c0 valfit_em_pure        the same unsupervised EM refit on each fitting partition's
  E0c  valfit_em_blend       VALIDATION histories (counts only, no targets at all).
  E1a  endpoint_binomial     endpoint VALUES and prior from the fitting partition's targets,
                             Binomial emission on full-history causal counts.
  E1b  endpoint_betabinom    the same endpoint values, but the emission (mean and
                             concentration per type and per endpoint) fitted by maximum
                             likelihood on the fitting partition's labelled machines. This
                             is E1 done with proper supervision.
  E1c  endpoint_bb_window    E1b with the evidence window chosen by inner partitions from
                             {40, 80, 120, full ticks}: the full-history versus recent /
                             change-sensitive comparison the brief asks for.
  E2   calibrator            a small REGULARIZED logistic endpoint classifier on fitting
                             partition endpoint labels, posterior-weighted endpoint mean.
                             Feature set, ridge strength and blend weight are all chosen on
                             inner partitions from a declared ladder that includes the
                             no-correction rung.
  E2r  calibrator_direct     the same features by ridge LINEAR regression on the fitting
                             partition's u, clipped to that partition's endpoint range.

NESTING. Four frozen complete-episode outer folds. For every arm and every outer fold, all
target-dependent quantities -- endpoint values, endpoint labels, the prior, emission
parameters, classifier coefficients, feature standardisation, feature-set choice, ridge
strength, evidence window, blend weight, and the decision to correct at all -- are fitted or
chosen using only that fold's fitting episodes through b.inner_folds. Tail constants are
refit on each inner TRAINING partition when scoring an inner-held partition, and on the
outer fitting episodes for the held fold. Held-fold forecasts are concatenated and scored
ONCE. The blend grid starts at 0 and improvement must be strict, so an arm that finds
nothing returns the release exactly.

CAUSALITY. Evidence uses only observed ticks strictly before each query's cutoff, addressed
by ACTUAL tick number. Lag patterns require all eight actual lags present. Missing ticks are
simply absent, never zero. No historical query signal is fabricated; query.signal is used
only as supplied per query. Nothing from another machine at the target tick is used.

DETERMINISM. Every reduction iterates a sorted sequence. Verified by two runs, one with
PYTHONHASHSEED=777, compared byte for byte.

EXPOSURE. 96 repeatedly inspected public validation episodes. The candidate feature ladder
was designed by inspecting fold-0 FITTING episodes only; per-fold selection among the rungs
is nested. Exploratory research round.
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
from scipy.special import gammaln

from slot_model import harness as H
from slot_model.features import causal_index
from slot_forecasting.io import load_inputs, write_submission

LAGS = (1, 2, 3, 4, 5, 6, 7, 8)
PAIRS = ((1, 2), (2, 4), (4, 8), (1, 8))
TAU = 2.0                 # Hamming kernel width, declared, not tuned
KAPPA, NU = 4.0, 8.0      # contrast smoothing and support damping, declared
SHRINK = 8.0              # fraction smoothing toward the type prior, declared
BLEND = tuple(round(x, 3) for x in np.arange(0.0, 1.0001, 0.125))
RIDGE = (0.3, 1.0, 3.0, 10.0, 30.0)
WINDOWS = (40, 80, 120, 10 ** 9)
EPS = 1e-6


# --------------------------------------------------------------------------- evidence
def lagvec(obs, tick):
    if any((tick - lag) not in obs for lag in LAGS):
        return None
    return np.array([1.0 if obs[tick - lag] > 0 else -1.0 for lag in LAGS])


def gather(bundle):
    """Per-query causal evidence. Target independent; strictly pre-cutoff; actual ticks."""
    queries, _ = load_inputs(bundle.data / 'validation')
    meta = {q.query_id: (q.episode_id, q.machine_id, q.cutoff) for q in queries}
    out = []
    for qid in bundle.ids:
        episode, machine, cutoff = meta[qid]
        obs = bundle.index[(episode, machine)]
        qv = lagvec(obs, cutoff)
        rec = {'cutoff': cutoff, 'has_query_state': qv is not None}
        counts = {w: [0.0, 0.0] for w in WINDOWS}
        older = [0.0, 0.0]
        rows = []
        for tick in sorted(obs):
            if tick >= cutoff or obs[tick] <= 0:
                continue
            y = 1.0 if obs[tick] == 1 else 0.0
            for w in WINDOWS:
                if tick >= cutoff - w:
                    counts[w][0] += y
                    counts[w][1] += 1.0
            if tick < cutoff - 40:
                older[0] += y
                older[1] += 1.0
            lv = lagvec(obs, tick)
            if lv is not None:
                rows.append((lv, y))
        rec['counts'] = counts
        rec['older'] = older
        rec['n_state_rows'] = len(rows)
        lagm = {lag: [[0.0, 0.0], [0.0, 0.0]] for lag in LAGS}
        pairm = {p: [[0.0, 0.0], [0.0, 0.0]] for p in PAIRS}
        ham = [0.0, 0.0]
        if qv is not None:
            qpar = {(i, j): qv[i - 1] * qv[j - 1] for i, j in PAIRS}
            for lv, y in rows:
                weight = float(np.exp(-float(np.sum(lv != qv)) / TAU))
                ham[0] += weight * y
                ham[1] += weight
                for lag in LAGS:
                    cell = lagm[lag][0 if lv[lag - 1] == qv[lag - 1] else 1]
                    cell[0] += y
                    cell[1] += 1.0
                for i, j in PAIRS:
                    cell = pairm[(i, j)][0 if lv[i - 1] * lv[j - 1] == qpar[(i, j)] else 1]
                    cell[0] += y
                    cell[1] += 1.0
        rec['lagm'], rec['pairm'], rec['ham'] = lagm, pairm, ham
        out.append(rec)
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def smoothed(num, den, prior, k=SHRINK):
    return (num + k * prior) / (den + k)


def contrast(cell, prior):
    (ys, ns), (yd, nd) = cell
    neff = min(ns, nd)
    return (logit(smoothed(ys, ns, prior, KAPPA)) - logit(smoothed(yd, nd, prior, KAPPA))) \
        * neff / (neff + NU)


NAMES = (['uprot', 'full', 'sup_full', 'sup_state', 'ham', 'r40', 'older', 'rec_minus_old',
          'sup_r40', 'sig_m1', 'sig_p1']
         + [f'lagc{lag}' for lag in LAGS] + [f'pairc{i}{j}' for i, j in PAIRS])
IDX = {name: i for i, name in enumerate(NAMES)}
LADDER = [
    ('F0_prior_only', []),
    ('F1_uprot', ['uprot']),
    ('F2_counts', ['uprot', 'full', 'sup_full']),
    ('F3_change', ['uprot', 'full', 'r40', 'older', 'rec_minus_old', 'sup_full', 'sup_r40']),
    ('F4_hamming', ['uprot', 'full', 'sup_full', 'ham', 'sup_state']),
    ('F5_lag1', ['uprot', 'full', 'sup_full', 'sup_state', 'lagc1']),
    ('F6_lagall', ['uprot', 'full', 'sup_full', 'sup_state'] + [f'lagc{l}' for l in LAGS]),
    ('F7_lagpair', ['uprot', 'full', 'sup_full', 'sup_state'] + [f'lagc{l}' for l in LAGS]
     + [f'pairc{i}{j}' for i, j in PAIRS]),
    ('F8_wide', ['uprot', 'full', 'sup_full', 'sup_state', 'ham', 'r40', 'older',
                 'rec_minus_old', 'sup_r40', 'sig_m1', 'sig_p1']
     + [f'lagc{l}' for l in LAGS] + [f'pairc{i}{j}' for i, j in PAIRS]),
]


_DESIGN_CACHE = {}


def design(bundle, raw, mask, u_prot):
    """Feature matrix. Type priors come from the masked episodes' HISTORIES only."""
    key = np.asarray(mask, dtype=bool).tobytes()
    if key in _DESIGN_CACHE:
        return _DESIGN_CACHE[key]
    n = len(bundle.ids)
    X = np.zeros((n, len(NAMES)))
    priors = {}
    for t in H.TYPES:
        num = sum(raw[k]['counts'][WINDOWS[-1]][0] for k in range(n)
                  if mask[k] and bundle.types[k] == t)
        den = sum(raw[k]['counts'][WINDOWS[-1]][1] for k in range(n)
                  if mask[k] and bundle.types[k] == t)
        priors[t] = float(num / max(den, 1.0))
    for k in range(n):
        r, pr = raw[k], priors[bundle.types[k]]
        f = r['counts'][WINDOWS[-1]]
        r40 = r['counts'][40]
        X[k, IDX['uprot']] = logit(u_prot[k])
        X[k, IDX['full']] = logit(smoothed(f[0], f[1], pr))
        X[k, IDX['sup_full']] = np.log1p(f[1])
        X[k, IDX['sup_state']] = np.log1p(r['n_state_rows'])
        X[k, IDX['ham']] = logit(smoothed(r['ham'][0], r['ham'][1], pr))
        X[k, IDX['r40']] = logit(smoothed(r40[0], r40[1], pr))
        X[k, IDX['older']] = logit(smoothed(r['older'][0], r['older'][1], pr))
        X[k, IDX['rec_minus_old']] = X[k, IDX['r40']] - X[k, IDX['older']]
        X[k, IDX['sup_r40']] = np.log1p(r40[1])
        X[k, IDX['sig_m1']] = 1.0 * (int(bundle.signals[k]) == -1)
        X[k, IDX['sig_p1']] = 1.0 * (int(bundle.signals[k]) == 1)
        for lag in LAGS:
            X[k, IDX[f'lagc{lag}']] = contrast(r['lagm'][lag], pr) if r['has_query_state'] else 0.0
        for i, j in PAIRS:
            X[k, IDX[f'pairc{i}{j}']] = contrast(r['pairm'][(i, j)], pr) \
                if r['has_query_state'] else 0.0
    _DESIGN_CACHE[key] = (X, priors)
    return X, priors


# --------------------------------------------------------------------------- fitting side
def endpoints(bundle, mask):
    """Per type (u_lo, u_hi, prior_hi) from the masked episodes' probability TARGETS only.

    Declared fallback: a type with fewer than two distinct values in this partition, or none
    at all, borrows the partition's pooled two extreme values across all types that do have
    two; if even that is unavailable it degenerates to a single value, i.e. no correction.
    """
    rt = bundle.truth[:, 1:] / np.maximum(bundle.truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)
    u = rt[:, 0]
    out, pooled, note = {}, [], {}
    for t in H.TYPES:
        m = np.asarray(mask, dtype=bool) & (bundle.types == t)
        vals = sorted({round(float(x), 9) for x in u[m]}) if m.any() else []
        if len(vals) >= 2:
            lo, hi = vals[0], vals[-1]
            prior = float((u[m] > 0.5 * (lo + hi)).mean())
            out[t] = (lo, hi, prior)
            pooled.append((lo, hi))
            note[t] = 'partition'
        else:
            out[t] = None
            note[t] = 'fallback'
    for t in H.TYPES:
        if out[t] is None:
            m = np.asarray(mask, dtype=bool) & (bundle.types == t)
            if m.any():
                mean = float(u[m].mean())
                out[t] = (mean, mean, 0.5)
            elif pooled:
                out[t] = (pooled[0][0], pooled[0][1], 0.5)
            else:
                out[t] = (0.5, 0.5, 0.5)
    return out, note


def labels(bundle, ends):
    rt = bundle.truth[:, 1:] / np.maximum(bundle.truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)
    u = rt[:, 0]
    y = np.zeros(len(bundle.ids))
    for k in range(len(bundle.ids)):
        lo, hi, _ = ends[bundle.types[k]]
        y[k] = 1.0 if u[k] > 0.5 * (lo + hi) else 0.0
    return y, u


# --------------------------------------------------------------------------- small models
def log_binom(k, n, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + k * np.log(p) + (n - k) * np.log1p(-p))


def log_betabinom(k, n, mean, conc):
    a = max(mean, 1e-6) * conc
    bb = max(1 - mean, 1e-6) * conc
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + gammaln(k + a) + gammaln(n - k + bb) - gammaln(n + a + bb)
            + gammaln(a + bb) - gammaln(a) - gammaln(bb))


def fit_mixture(n1, ntot, iters=500):
    """Unsupervised two-component binomial mixture by EM. Returns (u0, u1, weight0)."""
    use = ntot > 0
    if use.sum() < 4:
        frac = float(n1.sum() / max(ntot.sum(), 1.0))
        return frac, frac, 0.5
    frac = np.divide(n1, np.maximum(ntot, 1.0), where=use, out=np.zeros_like(n1))
    u = np.array([np.quantile(frac[use], 0.25), np.quantile(frac[use], 0.75)])
    if u[1] - u[0] < 1e-6:
        u = np.array([max(u[0] - 0.05, 1e-3), min(u[1] + 0.05, 1 - 1e-3)])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = np.column_stack([np.log(max(w[k], 1e-12)) + log_binom(n1, ntot, u[k]) for k in (0, 1)])
        post = np.exp(ll - ll.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        wn = post.mean(axis=0)
        un = np.array([float((post[:, k] * n1).sum() / max((post[:, k] * ntot).sum(), 1e-9))
                       for k in (0, 1)])
        done = np.max(np.abs(un - u)) < 1e-12 and np.max(np.abs(wn - w)) < 1e-12
        u, w = un, wn
        if done:
            break
    order = np.argsort(u)
    return float(u[order[0]]), float(u[order[1]]), float(w[order[0]])


def ridge_logistic(X, y, lam, iters=200):
    """Newton with a guard: a non-finite or diverging step keeps the last finite iterate."""
    Xb = np.column_stack((np.ones(len(X)), X))
    w = np.zeros(Xb.shape[1])
    P = np.eye(Xb.shape[1]) * lam
    P[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + P @ w
        W = np.maximum(p * (1 - p), 1e-6)
        try:
            step = np.linalg.solve(Xb.T @ (Xb * W[:, None]) + P, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)) or np.max(np.abs(step)) > 1e6:
            break
        w = w - step
        if np.max(np.abs(step)) < 1e-11:
            break
    return w if np.all(np.isfinite(w)) else None


def apply_logistic(w, X):
    return 1.0 / (1.0 + np.exp(-np.clip(np.column_stack((np.ones(len(X)), X)) @ w, -30, 30)))


def ridge_linear(X, y, lam):
    Xb = np.column_stack((np.ones(len(X)), X))
    P = np.eye(Xb.shape[1]) * lam
    P[0, 0] = 0.0
    try:
        w = np.linalg.solve(Xb.T @ Xb + P, Xb.T @ y)
    except np.linalg.LinAlgError:
        return None
    return w if np.all(np.isfinite(w)) else None


def fit_emission(n1, ntot, mean0):
    """ML (mean, concentration) of a beta-binomial on one labelled endpoint group."""
    if len(n1) < 4 or ntot.sum() <= 0:
        return float(mean0), 50.0

    def neg(theta):
        mean = 1.0 / (1.0 + np.exp(-theta[0]))
        conc = float(np.exp(theta[1]))
        return -float(np.sum(log_betabinom(n1, ntot, mean, conc)))

    start = np.array([logit(np.clip(n1.sum() / max(ntot.sum(), 1.0), 1e-3, 1 - 1e-3)),
                      np.log(50.0)])
    res = minimize(neg, start, method='L-BFGS-B',
                   bounds=[(-8.0, 8.0), (np.log(1.0), np.log(1e5))])
    theta = res.x if res.success or np.all(np.isfinite(res.x)) else start
    return float(1.0 / (1.0 + np.exp(-theta[0]))), float(np.exp(theta[1]))


# --------------------------------------------------------------------------- arms
class Arm:
    blend = BLEND

    def fit_grid(self):
        return [()]

    def fit(self, ctx, mask, cfg):
        return None

    def predict(self, ctx, model, cfg):
        raise NotImplementedError


class Zero(Arm):
    blend = (0.0,)

    def predict(self, ctx, model, cfg):
        return ctx['u_prot'].copy()


class TrainEM(Arm):
    """Unsupervised EM on TRAIN counts; Binomial posterior mean on full-history counts."""

    def __init__(self, blend=BLEND):
        self.blend = blend

    def fit(self, ctx, mask, cfg):
        return ctx['train_mixture']

    def predict(self, ctx, model, cfg):
        return binomial_posterior_mean(ctx, model, WINDOWS[-1],
                                       {t: (model[t][0], model[t][1]) for t in H.TYPES})


class ValEM(Arm):
    """The same unsupervised EM refit on the fitting partition's VALIDATION histories."""

    def __init__(self, blend=BLEND):
        self.blend = blend

    def fit(self, ctx, mask, cfg):
        out = {}
        for t in H.TYPES:
            rows = [ctx['raw'][k]['counts'][WINDOWS[-1]] for k in range(ctx['n'])
                    if mask[k] and ctx['bundle'].types[k] == t]
            a = np.array(rows, dtype=float) if rows else np.zeros((0, 2))
            if len(a) == 0:
                out[t] = (0.5, 0.5, 0.5)
            else:
                out[t] = fit_mixture(a[:, 0], a[:, 1])
        return out

    def predict(self, ctx, model, cfg):
        return binomial_posterior_mean(ctx, model, WINDOWS[-1],
                                       {t: (model[t][0], model[t][1]) for t in H.TYPES})


def binomial_posterior_mean(ctx, model, window, values):
    """Posterior mean over two components with a Binomial emission at the component rates."""
    b, raw, n = ctx['bundle'], ctx['raw'], ctx['n']
    out = np.zeros(n)
    for k in range(n):
        t = b.types[k]
        u0, u1, w0 = model[t]
        n1, ntot = raw[k]['counts'][window]
        l0 = np.log(max(w0, 1e-12)) + log_binom(n1, ntot, u0)
        l1 = np.log(max(1 - w0, 1e-12)) + log_binom(n1, ntot, u1)
        m = max(l0, l1)
        p0, p1 = np.exp(l0 - m), np.exp(l1 - m)
        post = p1 / (p0 + p1)
        lo, hi = values[t]
        out[k] = (1 - post) * lo + post * hi
    return out


class EndpointBinomial(Arm):
    """Fitting-side endpoint VALUES and prior; Binomial emission at those values."""

    def fit(self, ctx, mask, cfg):
        ends, _ = endpoints(ctx['bundle'], mask)
        return {t: (ends[t][0], ends[t][1], 1.0 - ends[t][2]) for t in H.TYPES}

    def predict(self, ctx, model, cfg):
        return binomial_posterior_mean(ctx, model, WINDOWS[-1],
                                       {t: (model[t][0], model[t][1]) for t in H.TYPES})


class EndpointBetaBinomial(Arm):
    """Endpoint values from targets; emission mean and concentration fitted per group."""

    def __init__(self, windows=(WINDOWS[-1],)):
        self.windows = windows

    def fit_grid(self):
        return [(w,) for w in self.windows]

    def fit(self, ctx, mask, cfg):
        window = cfg[0]
        b = ctx['bundle']
        ends, _ = endpoints(b, mask)
        lev, _ = labels(b, ends)
        model = {}
        for t in H.TYPES:
            lo, hi, prior = ends[t]
            per = {}
            for z, mean0 in ((0, lo), (1, hi)):
                sel = [k for k in range(ctx['n'])
                       if mask[k] and b.types[k] == t and lev[k] == z]
                a = np.array([ctx['raw'][k]['counts'][window] for k in sel], dtype=float) \
                    if sel else np.zeros((0, 2))
                per[z] = fit_emission(a[:, 0], a[:, 1], mean0) if len(a) else (mean0, 50.0)
            model[t] = {'lo': lo, 'hi': hi, 'prior_hi': prior, 'em': per}
        return model

    def predict(self, ctx, model, cfg):
        window = cfg[0]
        b, raw, n = ctx['bundle'], ctx['raw'], ctx['n']
        out = np.zeros(n)
        for k in range(n):
            m = model[b.types[k]]
            n1, ntot = raw[k]['counts'][window]
            l0 = np.log(max(1 - m['prior_hi'], 1e-12)) + log_betabinom(n1, ntot, *m['em'][0])
            l1 = np.log(max(m['prior_hi'], 1e-12)) + log_betabinom(n1, ntot, *m['em'][1])
            mx = max(l0, l1)
            p0, p1 = np.exp(l0 - mx), np.exp(l1 - mx)
            post = p1 / (p0 + p1)
            out[k] = (1 - post) * m['lo'] + post * m['hi']
        return out


class Calibrator(Arm):
    """Regularised logistic endpoint classifier -> posterior-weighted endpoint mean.

    `scramble` is an integrity control, not a model: the fitting partition's endpoint
    labels are permuted within (partition, machine type) with a fixed seed, so the
    classifier has no information. The whole nested machinery is otherwise untouched, so a
    held-fold score materially above the release would mean the pipeline can manufacture a
    gain from noise.
    """

    def __init__(self, regression=False, scramble=None, drop_uprot=False):
        self.regression = regression
        self.scramble = scramble
        self.drop_uprot = drop_uprot

    def fit_grid(self):
        return [(name, lam) for name, _ in LADDER for lam in RIDGE]

    def fit(self, ctx, mask, cfg):
        name, lam = cfg
        want = [c for c in dict(LADDER)[name] if not (self.drop_uprot and c == 'uprot')]
        cols = [IDX[c] for c in want]
        b = ctx['bundle']
        ends, _ = endpoints(b, mask)
        lev, utrue = labels(b, ends)
        if self.scramble is not None:
            rng = np.random.default_rng(self.scramble)
            for t in H.TYPES:
                rows = sorted(int(k) for k in np.where(mask & (b.types == t))[0])
                if rows:
                    perm = rng.permutation(len(rows))
                    lev[rows] = lev[[rows[i] for i in perm]]
                    utrue[rows] = utrue[[rows[i] for i in perm]]
        X, _ = design(b, ctx['raw'], mask, ctx['u_prot'])
        per = {}
        for t in H.TYPES:
            m = mask & (b.types == t)
            if not m.any() or not cols:
                per[t] = None
                continue
            mu = X[m][:, cols].mean(axis=0)
            sd = np.maximum(X[m][:, cols].std(axis=0), 1e-6)
            Z = (X[m][:, cols] - mu) / sd
            if self.regression:
                coef = ridge_linear(Z, utrue[m], lam)
            elif len(np.unique(lev[m])) < 2:
                coef = None
            else:
                coef = ridge_logistic(Z, lev[m], lam)
            if coef is not None and not np.all(np.isfinite(coef)):
                coef = None
            per[t] = {'mu': mu, 'sd': sd, 'coef': coef}
        return {'per': per, 'ends': ends, 'X': X, 'cols': cols}

    def predict(self, ctx, model, cfg):
        b = ctx['bundle']
        out = ctx['u_prot'].copy()
        cols, X = model['cols'], model['X']
        for t in H.TYPES:
            m = b.types == t
            p = model['per'][t]
            lo, hi, prior = model['ends'][t]
            if p is None or p['coef'] is None:
                out[m] = (1 - prior) * lo + prior * hi
                continue
            Z = (X[m][:, cols] - p['mu']) / p['sd']
            if self.regression:
                pred = np.column_stack((np.ones(len(Z)), Z)) @ p['coef']
                out[m] = np.clip(pred, min(lo, hi), max(lo, hi))
            else:
                post = apply_logistic(p['coef'], Z)
                out[m] = (1 - post) * lo + post * hi
        return out


# --------------------------------------------------------------------------- nested runner
def compose(bundle, qh, u, V):
    r = np.column_stack((u, (1.0 - u)[:, None] * V))
    return bundle.compose(qh, r / r.sum(axis=1, keepdims=True))


def tail_matrix(bundle, mask):
    tails = bundle.tail_constants(mask)
    return np.array([tails[t] for t in bundle.types])


def run_arm(bundle, arm, ctx, label):
    n = ctx['n']
    qh, u_prot = ctx['q'], ctx['u_prot']
    held_u = np.zeros(n)
    held_V = np.zeros((n, 3))
    chosen, table = [], []
    for fold in range(4):
        held = bundle.outer_mask(fold)
        fit = ~held
        w = bundle.row_weights(fit)
        inner = bundle.inner_folds(fit, 3)
        cache = {}
        for cfg in arm.fit_grid():
            u_in = u_prot.copy()
            V_in = np.zeros((n, 3))
            for tr, he in inner:
                V_in[he] = tail_matrix(bundle, tr)[he]
                model = arm.fit(ctx, tr, cfg)
                u_in[he] = arm.predict(ctx, model, cfg)[he]
            cache[cfg] = (u_in, V_in)
        best, per_cfg = None, {}
        for cfg in arm.fit_grid():
            u_in, V_in = cache[cfg]
            for beta in arm.blend:
                u = np.clip(u_prot + beta * (u_in - u_prot), EPS, 1 - EPS)
                value = bundle.wloss(compose(bundle, qh, u, V_in), w)
                if not np.isfinite(value):
                    continue
                key = '|'.join(str(x) for x in cfg)
                if key not in per_cfg or value < per_cfg[key][0]:
                    per_cfg[key] = (value, beta)
                if best is None or value < best[0] - 1e-13:
                    best = (value, cfg, beta)
        table.append({'fold': fold,
                      'best_inner_per_config': {k: {'objective': v[0], 'blend': v[1]}
                                                for k, v in sorted(per_cfg.items())}})
        _, cfg, beta = best
        V = tail_matrix(bundle, fit)
        model = arm.fit(ctx, fit, cfg)
        u = np.clip(u_prot + beta * (arm.predict(ctx, model, cfg) - u_prot), EPS, 1 - EPS)
        held_u[held] = u[held]
        held_V[held] = V[held]
        chosen.append({'fold': fold, 'config': [str(x) for x in cfg], 'blend': beta,
                       'inner_objective': best[0],
                       'inner_zero_correction': bundle.wloss(
                           compose(bundle, qh, u_prot, cache[arm.fit_grid()[0]][1]), w)})
    assert np.all(np.isfinite(held_u)) and np.all(np.isfinite(held_V)), label
    P = compose(bundle, qh, held_u, held_V)
    assert np.all(np.isfinite(P)), label
    return {'name': label, 'P': P, 'u': held_u, 'chosen': chosen, 'inner_table': table}


def run_q_combination(bundle, ctx, arm, e2_chosen):
    """E3. The E2 u recipe composed with a nested-selected q.

    For each outer fold the u configuration is the one E2 already selected on that fold's
    inner partitions. The q candidate and its convex weight are then chosen on the SAME
    inner partitions, with the identity q = release q first in the grid and strict
    improvement required, so the arm returns the E2 forecast when no q helps.
    """
    n = ctx['n']
    qh, u_prot = ctx['q'], ctx['u_prot']
    names, _, Q, _, info = H.raw_pool(bundle)
    grid = [(-1, 0.0)] + [(j, g) for j in range(len(Q)) for g in (0.25, 0.5, 0.75, 1.0)]
    held_P = np.zeros((n, 5))
    chosen = []
    for fold in range(4):
        held = bundle.outer_mask(fold)
        fit = ~held
        w = bundle.row_weights(fit)
        inner = bundle.inner_folds(fit, 3)
        cfg, beta = e2_chosen[fold]
        u_in = u_prot.copy()
        V_in = np.zeros((n, 3))
        for tr, he in inner:
            V_in[he] = tail_matrix(bundle, tr)[he]
            model = arm.fit(ctx, tr, cfg)
            u_in[he] = np.clip(u_prot + beta * (arm.predict(ctx, model, cfg) - u_prot),
                               EPS, 1 - EPS)[he]
        best = None
        for j, g in grid:
            qq = qh if j < 0 else (1.0 - g) * qh + g * Q[j]
            r = np.column_stack((u_in, (1.0 - u_in)[:, None] * V_in))
            value = bundle.wloss(bundle.compose(qq, r / r.sum(axis=1, keepdims=True)), w)
            if np.isfinite(value) and (best is None or value < best[0] - 1e-13):
                best = (value, j, g)
        _, j, g = best
        V = tail_matrix(bundle, fit)
        model = arm.fit(ctx, fit, cfg)
        u = np.clip(u_prot + beta * (arm.predict(ctx, model, cfg) - u_prot), EPS, 1 - EPS)
        qq = qh if j < 0 else (1.0 - g) * qh + g * Q[j]
        P = compose(bundle, qq, u, V)
        held_P[held] = P[held]
        chosen.append({'fold': fold, 'u_config': [str(x) for x in cfg], 'u_blend': beta,
                       'q_candidate': 'release_q' if j < 0 else names[j], 'q_weight': g,
                       'inner_objective': best[0]})
    return {'name': 'E3_q_combination', 'P': held_P, 'u': np.zeros(n), 'chosen': chosen,
            'inner_table': [], 'pool': {'members': names, 'info': info}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('main-participants-60-40/public'))
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/endpoint_supervised')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    bundle = H.load(args.data)
    n = len(bundle.ids)
    qh, r_prot = bundle.split_heads(bundle.protected)
    u_prot = r_prot[:, 0]
    raw = gather(bundle)

    # the family must contain the protected release exactly
    V_check = np.zeros((n, 3))
    for fold in range(4):
        held = bundle.outer_mask(fold)
        V_check[held] = tail_matrix(bundle, ~held)[held]
    recon = compose(bundle, qh, u_prot, V_check)
    nest_dev = float(np.max(np.abs(recon - bundle.protected)))
    assert nest_dev < 1e-12, nest_dev

    # TRAIN-only unsupervised mixture, for the E0b control
    tq, th = load_inputs(args.data / 'train')
    tindex = causal_index(tq, th)
    rows = {}
    for q in sorted(tq, key=lambda q: q.query_id):
        obs = tindex[(q.episode_id, q.machine_id)]
        n1 = ntail = 0
        for tick in sorted(obs):
            if tick >= q.cutoff or obs[tick] <= 0:
                continue
            if obs[tick] == 1:
                n1 += 1
            else:
                ntail += 1
        rows.setdefault(q.machine_type, []).append((n1, n1 + ntail))
    train_mixture = {}
    for t in H.TYPES:
        a = np.array(sorted(rows.get(t, [])), dtype=float)
        train_mixture[t] = fit_mixture(a[:, 0], a[:, 1]) if len(a) else (0.5, 0.5, 0.5)

    ctx = {'bundle': bundle, 'raw': raw, 'n': n, 'q': qh, 'u_prot': u_prot,
           'train_mixture': train_mixture}

    arms = [
        ('E0a_zero_correction', Zero()),
        ('E0b0_train_em_pure', TrainEM(blend=(1.0,))),
        ('E0b_train_em_blend', TrainEM()),
        ('E0c0_valfit_em_pure', ValEM(blend=(1.0,))),
        ('E0c_valfit_em_blend', ValEM()),
        ('E1a_endpoint_binomial', EndpointBinomial()),
        ('E1b_endpoint_betabinom', EndpointBetaBinomial()),
        ('E1c_endpoint_bb_window', EndpointBetaBinomial(windows=WINDOWS)),
        ('E2_calibrator', Calibrator()),
        ('E2r_calibrator_direct', Calibrator(regression=True)),
        ('E2n_label_scrambled', Calibrator(scramble=20261108)),
        ('E2x_calibrator_no_uprot', Calibrator(drop_uprot=True)),
    ]
    results, timings = {}, {}
    for label, arm in arms:
        t0 = time.perf_counter()
        results[label] = run_arm(bundle, arm, ctx, label)
        timings[label] = time.perf_counter() - t0
        o = bundle.official(results[label]['P'])
        results[label]['official'] = o
        print(f"{label:<26}{o['technical_points']:>12.6f}  R={o['mean_loss_ratio']:.9f}  "
              f"{timings[label]:6.1f}s", flush=True)

    e2_chosen = {c['fold']: (tuple(Calibrator().fit_grid()[0]) if False else None, None)
                 for c in results['E2_calibrator']['chosen']}
    grid2 = Calibrator().fit_grid()
    for c in results['E2_calibrator']['chosen']:
        match = [g for g in grid2 if [str(x) for x in g] == c['config']]
        e2_chosen[c['fold']] = (match[0], c['blend'])
    t0 = time.perf_counter()
    results['E3_q_combination'] = run_q_combination(bundle, ctx, Calibrator(), e2_chosen)
    timings['E3_q_combination'] = time.perf_counter() - t0
    o3 = bundle.official(results['E3_q_combination']['P'])
    results['E3_q_combination']['official'] = o3
    arms.append(('E3_q_combination', None))
    print(f"{'E3_q_combination':<26}{o3['technical_points']:>12.6f}  "
          f"R={o3['mean_loss_ratio']:.9f}  {timings['E3_q_combination']:6.1f}s", flush=True)

    rt = bundle.truth[:, 1:] / np.maximum(bundle.truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)
    u_true = rt[:, 0]
    prot = bundle.official(bundle.protected)
    control = results['E0a_zero_correction']['P']
    report = {'preregistration': __doc__,
              'nesting_identity_max_abs_dev_from_protected': nest_dev,
              'protected': {'points': prot['technical_points'], 'R': prot['mean_loss_ratio'],
                            'by_condition': {c: v['loss_ratio']
                                             for c, v in prot['by_condition'].items()}},
              'train_mixture_unsupervised': {t: list(train_mixture[t]) for t in H.TYPES},
              'fold_endpoints': {}, 'arms': {}}
    for fold in range(4):
        ends, note = endpoints(bundle, ~bundle.outer_mask(fold))
        report['fold_endpoints'][str(fold)] = {t: {'u_lo': ends[t][0], 'u_hi': ends[t][1],
                                                   'prior_hi': ends[t][2], 'source': note[t]}
                                               for t in H.TYPES}
    for label, _ in arms:
        res = results[label]
        o = res['official']
        pv = bundle.paired(res['P'], bundle.protected)
        pc = pv if label == 'E0a_zero_correction' else bundle.paired(res['P'], control)
        report['arms'][label] = {
            'points': o['technical_points'], 'R': o['mean_loss_ratio'],
            'by_condition': {c: v['loss_ratio'] for c, v in o['by_condition'].items()},
            'gain_over_protected': o['technical_points'] - prot['technical_points'],
            'paired_vs_protected': pv, 'paired_vs_family_control': pc,
            'significant_vs_protected': bool(pv['delta_loss_ratio_ci95'][1] < 0),
            'mean_abs_u_error': float(np.mean(np.abs(res['u'] - u_true))),
            'held_selection_per_fold': res['chosen'], 'seconds': timings[label],
            'per_fold_held_points': [
                {'fold': f,
                 'arm': bundle.official(res['P'], episodes={
                     e for i, e in enumerate(bundle.ep) if bundle.outer_mask(f)[i]}
                 )['technical_points'],
                 'protected': bundle.official(bundle.protected, episodes={
                     e for i, e in enumerate(bundle.ep) if bundle.outer_mask(f)[i]}
                 )['technical_points']} for f in range(4)],
            'inner_ladder_table': res.get('inner_table', [])}
    # within-family contrast: does the calibrator beat the supervised generative rung?
    report['E2_vs_E1b'] = bundle.paired(results['E2_calibrator']['P'],
                                        results['E1b_endpoint_betabinom']['P'])
    report['E3_vs_E2'] = bundle.paired(results['E3_q_combination']['P'],
                                       results['E2_calibrator']['P'])
    report['E2r_vs_E2'] = bundle.paired(results['E2r_calibrator_direct']['P'],
                                        results['E2_calibrator']['P'])
    report['E2x_vs_E2'] = bundle.paired(results['E2x_calibrator_no_uprot']['P'],
                                        results['E2_calibrator']['P'])
    report['multiplicity_note'] = (
        'Thirteen declared arms were scored. E2_calibrator is the brief\'s primary arm and was '
        'designated before the run; E2r and E2n are declared variants of the same mechanism '
        'and E3 is the declared q-combination extension. The paired intervals are nominal '
        '95% and are NOT corrected for having run thirteen arms: E2 would not survive a '
        'Bonferroni correction across them, E2r and E3 are wider margins. The per-fold '
        'held-fold table and the inner ladder table are the independent corroboration.')
    report['mean_abs_u_error_protected'] = float(np.mean(np.abs(u_prot - u_true)))
    report['seconds'] = time.perf_counter() - start

    (args.out / 'predictions').mkdir(parents=True, exist_ok=True)
    for label, _ in arms:
        write_submission(args.out / 'predictions' / f'oof_{label}.csv',
                         bundle.mapping(results[label]['P']))
    lines = ['arm,points,R,stable,drift,shared_change,short_history,gain,ci_lo,ci_hi']
    for label, _ in arms:
        a = report['arms'][label]
        lo, hi = a['paired_vs_protected']['delta_loss_ratio_ci95']
        lines.append(','.join([label, f"{a['points']:.9f}", f"{a['R']:.12f}"]
                              + [f"{a['by_condition'][c]:.9f}" for c in
                                 ('stable', 'drift', 'shared_change', 'short_history')]
                              + [f"{a['gain_over_protected']:+.9f}", f'{lo:+.9f}', f'{hi:+.9f}']))
    (args.out / 'results.csv').write_text('\n'.join(lines) + '\n')
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True,
                                                         default=float) + '\n')
    conds = ('stable', 'drift', 'shared_change', 'short_history')
    print(f"\n{'arm':<26}{'points':>12}{'gain':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds)
          + '   paired95 vs protected')
    for label, _ in arms:
        a = report['arms'][label]
        lo, hi = a['paired_vs_protected']['delta_loss_ratio_ci95']
        print(f"{label:<26}{a['points']:>12.6f}{a['gain_over_protected']:>+10.6f}  "
              + '  '.join(f"{a['by_condition'][c]:>9.5f}" for c in conds)
              + f'   [{lo:+.5f}, {hi:+.5f}]' + ('  SIG' if hi < 0 else ''))
    print(f"\nwrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
