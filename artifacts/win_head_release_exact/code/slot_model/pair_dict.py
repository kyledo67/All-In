"""Compact pair-model dictionary with predictive averaging over pair identity.

Hypothesis. The dense 37-coefficient local model learns useful pair interactions, but
estimating all 28 interaction coefficients jointly is inefficient when recent support is
limited (282 of 768 machines choose half-life 8, about 11 effective transitions). Small
competing pair models, with learned population priors and predictive averaging over
which pair is active, may estimate the win probability more accurately.

Dictionary: one null model (local intercept only) plus one model per lag pair (i,j),
1 <= i < j <= 8, so 29 candidates. For pair (i,j):

    eta = eta_population(x_{m,t}) + a_m + b_m s_i + c_m s_j + d_m s_i s_j

with s_l = 2*1(Y_{t-l} > 0) - 1 on ACTUAL ticks. Four local coefficients, free in sign.
eta_population is the UNCHANGED existing pooled offset beta_type^T phi over the full
37-dim basis, so this experiment isolates local complexity and pair uncertainty.

WEIGHTING METHOD, stated explicitly: a Laplace approximation to the marginal likelihood
under a proper Gaussian prior theta ~ N(0, Lambda^-1),

    log Z_M  ~=  l(theta_hat) - 0.5 * theta_hat^T Lambda theta_hat
                 + 0.5 log|Lambda| - 0.5 log|H|

where l is the observation-weighted log likelihood and H = Z^T W Z + Lambda is the
Hessian at the mode. The (d/2)log(2*pi) terms of the prior normalizer and the Gaussian
integral cancel exactly, which is why they do not appear. Model weights are

    omega_M  proportional to  pi_M * exp(log Z_M)

with pi_M a smoothed prior over pair identity learned from training machines. This is
NOT a MAP objective reinterpreted as a model probability.

Because the likelihood carries recency decay weights, l is a TEMPERED (weighted)
likelihood rather than the true joint density of the observations. The resulting
quantity is therefore a generalized / approximate Bayesian model weight, not an exact
Bayes factor; it is reported as such and never called a posterior probability.

Prediction averages PROBABILITIES, not coefficients or logits:

    q = sum_M omega_M * E_theta[ sigma(eta*) ]

with the inner expectation taken over the Laplace Gaussian for theta via the standard
probit scaling, so each candidate's own coefficient uncertainty widens its contribution.
"""
import math

import numpy as np

from .win_head import HORIZON, PAIRS, _sigmoid, ETA_CLIP

# 29 candidates: the null model, then one per lag pair.
CANDIDATES = [None] + list(PAIRS)


def candidate_names():
    return ['null'] + [f'pair{i}_{j}' for i, j in PAIRS]


def local_design(signs_matrix, candidate):
    """Columns for one candidate. `signs_matrix` is (n, HORIZON) of +/-1 lag signs."""
    n = signs_matrix.shape[0]
    if candidate is None:
        return np.ones((n, 1))
    i, j = candidate
    si = signs_matrix[:, i - 1]
    sj = signs_matrix[:, j - 1]
    return np.column_stack((np.ones(n), si, sj, si * sj))


def signs_from(phi):
    """Recover the (n, HORIZON) lag-sign block from the dense 37-dim basis rows."""
    return phi[:, 1:1 + HORIZON]


def _fit(z, y, w, offset, lam, iters=50, tol=1e-11):
    d = z.shape[1]
    theta = np.zeros(d)
    for _ in range(iters):
        eta = np.clip(offset + z @ theta, -ETA_CLIP, ETA_CLIP)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = z.T @ (w * (p - y)) + lam * theta
        hess = z.T @ (z * (w * p * (1.0 - p))[:, None]) + np.diag(lam)
        step = np.linalg.solve(hess + 1e-12 * np.eye(d), grad)
        norm = float(np.linalg.norm(step))
        scale = min(1.0, 10.0 / norm) if norm > 10.0 else 1.0
        theta = theta - scale * step
        if norm * scale < tol:
            break
    eta = np.clip(offset + z @ theta, -ETA_CLIP, ETA_CLIP)
    p = 1.0 / (1.0 + np.exp(-eta))
    hess = z.T @ (z * (w * p * (1.0 - p))[:, None]) + np.diag(lam)
    loglik = float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))
    return theta, hess, loglik


def log_evidence(theta, hess, loglik, lam):
    """Laplace log marginal likelihood under theta ~ N(0, diag(lam)^-1)."""
    sign_h, logdet_h = np.linalg.slogdet(hess)
    if sign_h <= 0:
        return -np.inf
    return float(loglik - 0.5 * float(theta @ (lam * theta))
                 + 0.5 * float(np.sum(np.log(lam))) - 0.5 * logdet_h)


def machine_posterior(phi, y, w, phi_star, population, lam_intercept, lam_slope,
                      log_prior=None):
    """Return (weights, per-candidate q, log evidences) for one machine.

    `population` supplies the offset: a vector of length n for the history rows and a
    scalar for the prediction tick, both computed from the UNCHANGED pooled model.
    """
    offset_rows, offset_star = population
    signs = signs_from(phi)
    star_signs = phi_star[1:1 + HORIZON]
    evidences = np.empty(len(CANDIDATES))
    qs = np.empty(len(CANDIDATES))
    for index, candidate in enumerate(CANDIDATES):
        z = local_design(signs, candidate)
        z_star = local_design(star_signs[None, :], candidate)[0]
        lam = np.full(z.shape[1], lam_slope)
        lam[0] = lam_intercept
        theta, hess, loglik = _fit(z, y, w, offset_rows, lam)
        evidences[index] = log_evidence(theta, hess, loglik, lam)
        mu = offset_star + float(z_star @ theta)
        try:
            var = max(float(z_star @ np.linalg.solve(hess, z_star)), 0.0)
        except np.linalg.LinAlgError:
            var = 0.0
        # Integrate sigma over the Laplace Gaussian for theta (probit scaling).
        qs[index] = _sigmoid(mu / math.sqrt(1.0 + math.pi * var / 8.0))
    scores = evidences + (0.0 if log_prior is None else log_prior)
    finite = np.isfinite(scores)
    if not finite.any():
        weights = np.zeros(len(CANDIDATES))
        weights[0] = 1.0
        return weights, qs, evidences
    scores = np.where(finite, scores, -np.inf)
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    return weights, qs, evidences


def average(weights, qs):
    """Average PROBABILITIES across candidates, never logits or coefficients."""
    q = float(np.dot(weights, qs))
    return min(max(q, 1e-12), 1.0 - 1e-12)
