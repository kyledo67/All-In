"""Finite Beta-mixture population prior over the two parity-group win probabilities.

CONTROL. The deployed model gives BOTH parity groups the same prior,
Beta(kappa*mu_type, kappa*(1-mu_type)) with kappa = 0.5, and predicts each group
independently as (kappa*mu + W_g) / (kappa + W_g + L_g). There is no group-specific prior
and no feature dependence beyond machine type, so the new component means are
parameterised as adjustments to those existing type means rather than replacing them.

MODEL. Two shared Beta components per machine type,
    B_L = Beta(k*mu_L, k*(1-mu_L)),   B_H = Beta(k*mu_H, k*(1-mu_H)),   mu_L <= mu_H,
and the four Cartesian assignments (j, k) of those two distributions to the two groups:
group 0 draws from B_j, group 1 from B_k. The marginal for either group is
    (1 - pi) B_L + pi B_H
and the joint weights are
    w_HH = pi^2 + delta,  w_LL = (1-pi)^2 + delta,  w_HL = w_LH = pi(1-pi) - delta,
with -min(pi^2, (1-pi)^2) <= delta <= pi(1-pi). delta = 0 is the exact
independent-marginal control and leaves the marginals unchanged for ANY delta in range.
Nothing forces mu_L + mu_H = 1, nor mu_L < mu_H strictly, nor a persistent binary state.

WHY delta IS THE ONLY BORROWING CHANNEL, shown analytically rather than assumed. With
delta = 0 the component log weight separates,
    log w~_jk = [log w_j + E_0(j)] + [log w_k + E_1(k)],
so the posterior over components factorises and the queried group's predictive mean
depends only on its OWN sufficient statistics. Cross-group borrowing therefore arises
only when delta != 0. This is asserted numerically in the verification suite.

INFERENCE.
    log w~_jk = log w_jk + sum_g [ betaln(a_gjk + W_g, b_gjk + L_g) - betaln(a_gjk, b_gjk) ]
    omega     = softmax over the four components
    q_g       = sum_jk omega_jk * (a_gjk + W_g) / (a_gjk + b_gjk + W_g + L_g)
The component posterior is averaged over, not hard-selected.

INTERPRETATION LIMITS. W_g and L_g are fractional decay-weighted counts, so these are
power-likelihood / generalized-posterior quantities, not exact Bayes factors for an
ordered sequence. The global hyperparameters are point estimates fitted by empirical
predictive calibration (Efron's second strategy), so this is NOT full Bayesian inference
over every parameter. The population spread being estimated is also conditional on the
lag pair each machine's own history selected, so it partly reflects selection and
estimation noise in that upstream step; that is a known limitation of transferring
empirical-Bayes reasoning to a selected-context setting.
"""
import numpy as np
from scipy.special import betaln, logsumexp

EPS = 1e-9


def components(mu_low, mu_high, conc):
    """Return (a, b) for the two shared Beta components."""
    lo = float(np.clip(mu_low, 1e-4, 1 - 1e-4))
    hi = float(np.clip(mu_high, 1e-4, 1 - 1e-4))
    k = float(max(conc, 1e-4))
    return np.array([[k * lo, k * (1 - lo)], [k * hi, k * (1 - hi)]])


def joint_weights(pi, delta):
    """2x2 weights with the stated marginals; delta = 0 gives exact independence."""
    p = float(np.clip(pi, 1e-6, 1 - 1e-6))
    hi_lim = p * (1 - p)
    lo_lim = -min(p * p, (1 - p) * (1 - p))
    d = float(np.clip(delta, lo_lim, hi_lim))
    w = np.array([[(1 - p) ** 2 + d, p * (1 - p) - d],
                  [p * (1 - p) - d, p * p + d]])
    w = np.maximum(w, 1e-12)
    return w / w.sum()


def delta_from_rho(pi, rho):
    """Shared normalized coupling: rho in [-1, 1], rho = 0 is independence."""
    p = float(np.clip(pi, 1e-6, 1 - 1e-6))
    if rho >= 0:
        return rho * p * (1 - p)
    return rho * min(p * p, (1 - p) * (1 - p))


def predict(counts, mu_low, mu_high, conc, pi, rho, queried_group):
    """Predictive win probability for `queried_group` given both groups' counts.

    `counts` is ((W_0, L_0), (W_1, L_1)) with fractional decay-weighted entries.
    Returns (q_hat, omega) with omega the 2x2 component posterior.
    """
    ab = components(mu_low, mu_high, conc)
    w = joint_weights(pi, delta_from_rho(pi, rho))
    (w0, l0), (w1, l1) = counts
    # evidence of each shared component for each group, computed once
    ev = np.empty((2, 2))          # ev[g, j]
    for j in range(2):
        a, b = ab[j]
        ev[0, j] = betaln(a + w0, b + l0) - betaln(a, b)
        ev[1, j] = betaln(a + w1, b + l1) - betaln(a, b)
    logw = np.log(w) + ev[0][:, None] + ev[1][None, :]
    flat = logw.ravel()
    omega = np.exp(flat - logsumexp(flat)).reshape(2, 2)
    gw, gl = counts[queried_group]
    means = np.array([(ab[j, 0] + gw) / (ab[j, 0] + ab[j, 1] + gw + gl) for j in range(2)])
    # marginalise the OTHER group's component out of the queried group's mean
    if queried_group == 0:
        weight_by_j = omega.sum(axis=1)
    else:
        weight_by_j = omega.sum(axis=0)
    return float(np.dot(weight_by_j, means)), omega


def predict_control(counts, mu, conc, queried_group):
    """The deployed single-Beta prediction, for the exact-control arm."""
    a, b = conc * mu, conc * (1 - mu)
    gw, gl = counts[queried_group]
    return float((a + gw) / (a + b + gw + gl))


def predict_batch(W0, L0, W1, L1, group, mu_low, mu_high, conc, pi, rho):
    """Vectorised form of `predict` over many queries sharing one parameter set.

    All arguments except the parameters are arrays of length n. `mu_low`, `mu_high` and
    `pi` are per-query arrays (so they can carry a per-type value); `conc` and `rho` are
    scalars. Returns the queried-group predictive means.

    This is numerically identical to looping `predict`, which the verification suite
    asserts; it exists because the fitting objective needs thousands of evaluations.
    """
    lo = np.clip(np.asarray(mu_low, dtype=np.float64), 1e-4, 1 - 1e-4)
    hi = np.clip(np.asarray(mu_high, dtype=np.float64), 1e-4, 1 - 1e-4)
    k = float(max(conc, 1e-4))
    a = np.stack((k * lo, k * hi), axis=1)              # (n, 2) component alphas
    b = np.stack((k * (1 - lo), k * (1 - hi)), axis=1)  # (n, 2) component betas

    W = np.stack((np.asarray(W0, float), np.asarray(W1, float)), axis=1)   # (n, 2) by group
    L = np.stack((np.asarray(L0, float), np.asarray(L1, float)), axis=1)

    # ev[:, g, j] = log B(a_j + W_g, b_j + L_g) - log B(a_j, b_j)
    ev = (betaln(a[:, None, :] + W[:, :, None], b[:, None, :] + L[:, :, None])
          - betaln(a, b)[:, None, :])

    p = np.clip(np.asarray(pi, dtype=np.float64), 1e-6, 1 - 1e-6)
    if rho >= 0:
        d = rho * p * (1 - p)
    else:
        d = rho * np.minimum(p * p, (1 - p) * (1 - p))
    w = np.stack((np.stack(((1 - p) ** 2 + d, p * (1 - p) - d), axis=1),
                  np.stack((p * (1 - p) - d, p * p + d), axis=1)), axis=1)   # (n, 2, 2) [j, k]
    w = np.maximum(w, 1e-12)
    w /= w.sum(axis=(1, 2), keepdims=True)

    logw = np.log(w) + ev[:, 0, :][:, :, None] + ev[:, 1, :][:, None, :]
    flat = logw.reshape(len(lo), 4)
    omega = np.exp(flat - logsumexp(flat, axis=1, keepdims=True)).reshape(len(lo), 2, 2)

    gi = np.asarray(group, dtype=np.int64)
    gw = W[np.arange(len(gi)), gi]
    gl = L[np.arange(len(gi)), gi]
    means = (a + gw[:, None]) / (a + b + gw[:, None] + gl[:, None])          # (n, 2)
    # marginalise the OTHER group's component index out
    by_j = np.where(gi[:, None] == 0, omega.sum(axis=2), omega.sum(axis=1))  # (n, 2)
    return np.einsum('nj,nj->n', by_j, means)
