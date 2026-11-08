"""Hierarchical logistic autoregressive win-probability head.

Motivation, measured rather than assumed: on the three-component candidate, the
p_0 coordinate carries 67.6% of all squared probability error, and substituting the
true win mass while keeping the candidate's positive-payout proportions scores
57.292704/60 against its own 32.260531. Weight search is exhausted (the best fixed
global convex blend of the four saved columns buys 0.08 points). So all remaining
headroom is in the total win probability q = 1 - p_0.

Model. For machine m at tick t, with s_l = 2*1(Y_{t-l} > 0) - 1 over ACTUAL ticks
l = 1..8, the predeclared local basis is

    phi_t = [1, s_1..s_8, {s_i s_j : 1 <= i < j <= 8}]        (1 + 8 + 28 = 37 dims)

and

    logit q_{m,t} = beta_{type(m)}^T phi_{m,t} + u_m^T phi_{m,t}

beta is pooled within machine type and fitted on permitted training observations.
u_m are regularized machine-specific deviations, free in sign, fitted using only that
machine's own pre-cutoff outcomes, with separate group shrinkage on the intercept,
the eight main effects and the 28 interactions.

The pair products can represent XOR-type binary responses between two lags, which
neither additive MTD nor a heavily shrunk pair-cell correction can. LIMITATION: this
basis covers only sign-coded win/loss main effects and pairwise products. It cannot
represent interactions of three or more lags, any dependence on payout magnitude
(2x vs 5x vs 20x are all simply "win"), or lags beyond 8.

Fitting is MAP (penalized maximum likelihood), not posterior inference. Predictive
uncertainty uses a Laplace approximation at the local mode, integrated over the
one-dimensional logit uncertainty only; see `predict_q`. Point estimates and that
one-dimensional integral are NOT full Bayesian inference and are not described as such.
"""
import math

import numpy as np

HORIZON = 8
PAIRS = tuple((i, j) for i in range(1, HORIZON + 1) for j in range(i + 1, HORIZON + 1))
DIM = 1 + HORIZON + len(PAIRS)
INTERCEPT = np.array([0])
MAIN = np.arange(1, 1 + HORIZON)
INTER = np.arange(1 + HORIZON, DIM)
ETA_CLIP = 30.0


def basis_names(magnitude=False, neighbour_lags=0):
    names = (['bias'] + [f's{l}' for l in range(1, HORIZON + 1)]
             + [f's{i}xs{j}' for i, j in PAIRS])
    if magnitude:
        names += [f'big{l}' for l in range(1, HORIZON + 1)]
    names += [f'nb{l}' for l in range(1, neighbour_lags + 1)]
    return names


MAG = np.arange(DIM, DIM + HORIZON)
DIM_MAGNITUDE = DIM + HORIZON


def _row(signs, bigs=None, nbs=None):
    """signs[l-1] is s_l in {-1,+1}. Optional blocks are appended after the core 37.

    `bigs[l-1]` is +1 when the outcome at lag l paid 5x or 20x and -1 otherwise, so a
    loss and a 1x/2x win are distinguished from a large win. MEASURED RESULT: this block
    made the train surrogate worse (0.151515 vs 0.147693), so it is not used.

    `nbs[l-1]` is the mean win sign of the OTHER machines in the same episode at tick
    t-l, excluding this machine, and 0.0 when no neighbour observed that tick. Episodes
    share one cutoff, so every neighbour tick t-l with l >= 1 is strictly below it.
    """
    extra = [np.asarray(b, dtype=np.float64) for b in (bigs, nbs) if b is not None]
    width = DIM + sum(len(b) for b in extra)
    row = np.empty(width)
    row[0] = 1.0
    row[MAIN] = signs
    row[INTER] = [signs[i - 1] * signs[j - 1] for i, j in PAIRS]
    cursor = DIM
    for block in extra:
        row[cursor:cursor + len(block)] = block
        cursor += len(block)
    return row


def neighbour_signs(episode_history, machine_id):
    """tick -> mean win sign of the other machines in the episode at that tick."""
    totals = {}
    for machine, history in episode_history.items():
        if machine == machine_id:
            continue
        for tick, value in history.items():
            entry = totals.setdefault(tick, [0.0, 0])
            entry[0] += 1.0 if value > 0 else -1.0
            entry[1] += 1
    return {tick: entry[0] / entry[1] for tick, entry in totals.items() if entry[1] > 0}


def design(ticks, cutoff, *, half_life=128.0, horizon=HORIZON, magnitude=False,
           neighbours=None, neighbour_lags=3):
    """Complete-lag transitions strictly below `cutoff`, by actual tick arithmetic.

    `ticks` maps actual tick -> payout index. A tick is usable only when the target
    tick and all of t-1..t-horizon are observed. A missing lag never becomes an
    observed loss: the transition is dropped instead. Recency weight is
    2**(-(cutoff-1-t)/half_life), frozen across every matched control.
    """
    rows, labels, weights, used = [], [], [], []
    for t in sorted(ticks):
        if t >= cutoff:
            continue
        if any((t - l) not in ticks for l in range(1, horizon + 1)):
            continue
        signs = np.array([1.0 if ticks[t - l] > 0 else -1.0 for l in range(1, horizon + 1)])
        bigs = (np.array([1.0 if ticks[t - l] >= 3 else -1.0 for l in range(1, horizon + 1)])
                if magnitude else None)
        nbs = (np.array([neighbours.get(t - l, 0.0) for l in range(1, neighbour_lags + 1)])
               if neighbours is not None else None)
        rows.append(_row(signs, bigs, nbs))
        labels.append(1.0 if ticks[t] > 0 else 0.0)
        weights.append(2.0 ** (-(cutoff - 1 - t) / half_life) if half_life else 1.0)
        used.append(t)
    if not rows:
        width = DIM + (HORIZON if magnitude else 0) + (neighbour_lags if neighbours is not None else 0)
        return np.zeros((0, width)), np.zeros(0), np.zeros(0), []
    return np.asarray(rows), np.asarray(labels), np.asarray(weights), used


def query_row(ticks, cutoff, *, horizon=HORIZON, magnitude=False,
              neighbours=None, neighbour_lags=3):
    """Basis row for the prediction tick itself, or None when a lag is unobserved."""
    if any((cutoff - l) not in ticks for l in range(1, horizon + 1)):
        return None
    signs = np.array([1.0 if ticks[cutoff - l] > 0 else -1.0 for l in range(1, horizon + 1)])
    bigs = (np.array([1.0 if ticks[cutoff - l] >= 3 else -1.0 for l in range(1, horizon + 1)])
            if magnitude else None)
    nbs = (np.array([neighbours.get(cutoff - l, 0.0) for l in range(1, neighbour_lags + 1)])
           if neighbours is not None else None)
    return _row(signs, bigs, nbs)


def penalty(lam_intercept, lam_main, lam_inter, lam_magnitude=None, lam_neighbour=None,
            neighbour_lags=3):
    blocks = []
    if lam_magnitude is not None:
        blocks.append(np.full(HORIZON, lam_magnitude))
    if lam_neighbour is not None:
        blocks.append(np.full(neighbour_lags, lam_neighbour))
    diag = np.empty(DIM + sum(len(b) for b in blocks))
    diag[INTERCEPT] = lam_intercept
    diag[MAIN] = lam_main
    diag[INTER] = lam_inter
    cursor = DIM
    for block in blocks:
        diag[cursor:cursor + len(block)] = block
        cursor += len(block)
    return diag


def _newton(phi, y, w, offset, lam_diag, *, start=None, iters=60, tol=1e-10):
    """Penalized IRLS/Newton for 0.5*sum(lam*theta^2) - weighted log likelihood.

    Convex in theta, so the mode is unique for positive penalties. Returns the mode
    and the Hessian at the mode (which is the Laplace precision).
    """
    theta = np.zeros(phi.shape[1]) if start is None else start.copy()
    lam = np.asarray(lam_diag, dtype=np.float64)
    for _ in range(iters):
        eta = np.clip(offset + phi @ theta, -ETA_CLIP, ETA_CLIP)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = phi.T @ (w * (p - y)) + lam * theta
        hess = phi.T @ (phi * (w * p * (1.0 - p))[:, None]) + np.diag(lam)
        # Jitter keeps the solve stable when a basis column is constant on this machine.
        try:
            step = np.linalg.solve(hess + 1e-10 * np.eye(phi.shape[1]), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess + 1e-6 * np.eye(phi.shape[1]), grad, rcond=None)[0]
        # Damped update: never let one step move the logit scale absurdly far.
        scale = 1.0
        norm = float(np.linalg.norm(step))
        if norm > 10.0:
            scale = 10.0 / norm
        theta = theta - scale * step
        if norm * scale < tol:
            break
    eta = np.clip(offset + phi @ theta, -ETA_CLIP, ETA_CLIP)
    p = 1.0 / (1.0 + np.exp(-eta))
    hess = phi.T @ (phi * (w * p * (1.0 - p))[:, None]) + np.diag(lam)
    return theta, hess


def objective(theta, phi, y, w, offset, lam_diag):
    """The exact MAP objective the Newton solver minimizes. Used by gradient checks."""
    eta = np.clip(offset + phi @ theta, -ETA_CLIP, ETA_CLIP)
    # log(1+exp(eta)) computed stably.
    ll = np.sum(w * (y * eta - np.logaddexp(0.0, eta)))
    return float(0.5 * np.sum(np.asarray(lam_diag) * theta ** 2) - ll)


def gradient(theta, phi, y, w, offset, lam_diag):
    eta = np.clip(offset + phi @ theta, -ETA_CLIP, ETA_CLIP)
    p = 1.0 / (1.0 + np.exp(-eta))
    return phi.T @ (w * (p - y)) + np.asarray(lam_diag) * theta


def fit_population(groups, lam_diag):
    """Pooled effects per machine type. `groups` maps type -> (phi, y, w)."""
    beta = {}
    report = {}
    for name, (phi, y, w) in groups.items():
        if len(y) == 0:
            beta[name] = np.zeros(len(lam_diag))
            report[name] = {'rows': 0, 'win_rate': None}
            continue
        theta, hess = _newton(phi, y, w, np.zeros(len(y)), lam_diag)
        beta[name] = theta
        report[name] = {'rows': int(len(y)), 'win_rate': float(np.average(y, weights=w)),
                        'grad_norm': float(np.linalg.norm(gradient(theta, phi, y, w, np.zeros(len(y)), lam_diag))),
                        'intercept': float(theta[0]),
                        'main_abs_mean': float(np.mean(abs(theta[MAIN]))),
                        'inter_abs_mean': float(np.mean(abs(theta[INTER])))}
    return beta, report


def predict_q(beta, phi, y, w, phi_star, lam_diag, *, adapt=None, laplace=True):
    """Return (q, detail) for one machine.

    `adapt` selects which coefficient groups may deviate locally:
      None            -> pooled only, u = 0                       (population control)
      'main'          -> intercept + the eight main lag effects    (main-effects control)
      'full'          -> intercept + main effects + 28 interactions
    Shrinkage is applied by setting the penalty of non-adapted groups to a large value,
    so every control solves the same objective with the same solver.

    With `laplace`, the predictive probability integrates the sigmoid over the
    one-dimensional Gaussian logit uncertainty implied by the Laplace precision at the
    mode, using the standard probit scaling. Assumptions: the MAP objective is
    approximately quadratic near the mode, the prior is the Gaussian implied by the
    ridge penalty, and only the logit's marginal variance is propagated. This widens
    short-history predictions toward 0.5 instead of letting them saturate.
    """
    offset_star = float(beta @ phi_star)
    if adapt is None or len(y) == 0:
        return _sigmoid(offset_star), {'transitions': int(len(y)), 'adapted': False,
                                       'logit': offset_star, 'logit_var': 0.0}
    lam = np.array(lam_diag, dtype=np.float64)
    frozen = 1e12
    if adapt == 'main':
        lam[INTER] = frozen
    elif adapt != 'full':
        raise ValueError(f'Unknown adaptation scope {adapt!r}')
    offset = phi @ beta
    u, hess = _newton(phi, y, w, offset, lam)
    logit = offset_star + float(u @ phi_star)
    variance = 0.0
    if laplace:
        try:
            variance = float(phi_star @ np.linalg.solve(hess, phi_star))
        except np.linalg.LinAlgError:
            variance = 0.0
        variance = max(variance, 0.0)
    q = _sigmoid(logit / math.sqrt(1.0 + math.pi * variance / 8.0)) if variance > 0 else _sigmoid(logit)
    return q, {'transitions': int(len(y)), 'adapted': True, 'logit': logit,
               'logit_var': variance, 'u_norm': float(np.linalg.norm(u)),
               'u_intercept': float(u[0]), 'u_main_abs_mean': float(np.mean(abs(u[MAIN]))),
               'u_inter_abs_mean': float(np.mean(abs(u[INTER])))}


def predict_q_averaged(beta_by_half_life, phi, y, used, cutoff, phi_star, lam_diag, *,
                       adapt='full', min_holdout=8, max_holdout=40, laplace=True):
    """Average over the recency half-life using each machine's OWN pre-cutoff evidence.

    The single frozen half-life is a compromise: a short memory tracks drifting
    machines but starves the 37-parameter fit, while a long memory recovers stationary
    machines precisely and is stale when coefficients move. Which regime a machine is
    in is observable from its own history, so the choice does not need a condition
    label (which does not exist at test time) and does not need a change detector.

    For each candidate half-life H we fit u on all transitions EXCEPT the most recent
    block, score the unweighted predictive log likelihood on that held-out recent
    block, and weight H by a softmax of those scores. The prediction for each H is the
    full-history fit. The held-out block is the transitions closest to the cutoff,
    which is exactly the regime the forecast must match.

    Everything here reads only ticks strictly below the cutoff.
    """
    half_lives = sorted(beta_by_half_life, key=lambda h: (h is None, h))
    n = len(y)
    order = np.argsort(used)
    holdout = min(max_holdout, max(min_holdout, n // 3))
    enough = n - holdout >= max(2 * min_holdout, 12)
    ticks = np.asarray(used, dtype=np.float64)
    scores, qs, variances = [], [], []
    for half_life in half_lives:
        beta = beta_by_half_life[half_life]
        w = np.ones(n) if not half_life else 2.0 ** (-(cutoff - 1 - ticks) / half_life)
        if enough:
            fit_idx, val_idx = order[:-holdout], order[-holdout:]
            lam = np.array(lam_diag, dtype=np.float64)
            if adapt == 'main':
                lam[INTER] = 1e12
            u_fit, _ = _newton(phi[fit_idx], y[fit_idx], w[fit_idx], phi[fit_idx] @ beta, lam)
            eta = np.clip(phi[val_idx] @ (beta + u_fit), -ETA_CLIP, ETA_CLIP)
            scores.append(float(np.sum(y[val_idx] * eta - np.logaddexp(0.0, eta))))
        else:
            scores.append(0.0)
        q, detail = predict_q(beta, phi, y, w, phi_star, lam_diag, adapt=adapt, laplace=laplace)
        qs.append(q)
        variances.append(detail['logit_var'])
    scores = np.asarray(scores, dtype=np.float64)
    mix = np.exp(scores - scores.max())
    mix /= mix.sum()
    q = float(np.dot(mix, qs))
    return q, {'transitions': int(n), 'adapted': adapt is not None and n > 0,
               'holdout': int(holdout if enough else 0),
               'half_lives': [h if h else 0.0 for h in half_lives],
               'mix': mix.tolist(), 'per_half_life_q': [float(v) for v in qs],
               'selected_half_life': float(half_lives[int(np.argmax(mix))] or 0.0),
               'mix_entropy': float(-np.sum(mix * np.log(np.maximum(mix, 1e-15)))),
               'logit_var': float(np.dot(mix, variances)), 'fallback': False}


def _sigmoid(z):
    z = float(np.clip(z, -ETA_CLIP, ETA_CLIP))
    return 1.0 / (1.0 + math.exp(-z))


def compose(q, proportions):
    """Rebuild the five-vector: [1-q, q*r_1, q*r_2, q*r_5, q*r_20].

    `proportions` are the reference predictor's positive-payout proportions and must
    sum to 1. The result is a valid simplex point by construction.
    """
    r = np.asarray(proportions, dtype=np.float64)
    out = np.empty(5)
    out[0] = 1.0 - q
    out[1:] = q * r
    total = out.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('compose produced a degenerate distribution')
    return out / total
