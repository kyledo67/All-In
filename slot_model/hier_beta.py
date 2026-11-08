"""Hierarchical Beta partial pooling within a parity group, with an exact pooled endpoint.

The deployed parity estimator pools the two original lag-pair cells inside each parity
group: ++ with --, and +- with -+. That suppresses estimation variance but erases any
machine-specific difference between the cells it merges. This module keeps the parity
group as the parent and lets its two cells depart from it by an amount the data has to
earn.

For a selected parity group g holding original cells c:

    theta_g  ~ Beta(a_g, b_g)                       the existing parity-group prior
    p_c | theta_g, tau ~ Beta(tau*theta_g, tau*(1-theta_g))

so tau is the pooling strength. The parent posterior is proportional to

    Beta(theta; a_g, b_g) * prod_c  B(tau*theta + W_c, tau*(1-theta) + L_c)
                                  / B(tau*theta,       tau*(1-theta))

which is integrated in one dimension numerically. Because E[p_c | theta, data] is linear
in theta, the queried cell's predictive mean marginalises exactly:

    q_c = (W_c + tau * E[theta_g | data]) / (W_c + L_c + tau)

POOLED ENDPOINT. As tau -> infinity the cell ratio tends to theta^{W_c}(1-theta)^{L_c},
the parent posterior collapses to Beta(a_g + sum_c W_c, b_g + sum_c L_c) and q_c tends to
its mean - exactly the deployed parity model. That endpoint is implemented in CLOSED FORM
rather than approximated with a large tau, so the control is reproduced bit-for-bit.

LIMITS, stated rather than implied.
  - This is an adaptation of a hierarchical/twofold Beta construction, not an exact
    implementation of any published model.
  - W_c and L_c are FRACTIONAL decayed counts, so every quantity here is a generalized
    posterior, not an exact Bayes posterior for a specific ordered sequence.
  - Small tau is NOT the same object as an independent four-cell Beta prior. That remains
    a separately specified control.
  - Each observation enters the joint likelihood exactly once. The parent is never fitted
    from the local observations and then updated again with those same observations.
  - Predictive MEANS are returned throughout; no mode or plug-in logit is substituted.
"""
import numpy as np
from scipy.special import betaln, roots_jacobi

# Original four-cell index is 2*(s_i > 0) + (s_j > 0).
# Parity group "same" merges 00 and 11; "different" merges 01 and 10.
SAME = (0, 3)
DIFFERENT = (1, 2)
EPS = 1e-12


def group_of(cell):
    return SAME if cell in SAME else DIFFERENT


_NODE_CACHE = {}


def _nodes(count, a_g=1.0, b_g=1.0):
    """Gauss-Jacobi nodes for integration against the Beta(a_g, b_g) prior weight.

    Gauss-Legendre fails here: with a weak prior such as Beta(0.05, 0.05) the parent
    density carries integrable endpoint singularities theta^(a-1) (1-theta)^(b-1), and a
    polynomial rule on a smooth weight mis-integrates them (measured error 3.8e-03).
    Gauss-Jacobi absorbs exactly that weight into the rule, so the remaining integrand is
    the smooth cell-ratio product. Nodes depend only on (a_g, b_g), of which there are a
    handful, so they are cached.
    """
    key = (count, float(a_g), float(b_g))
    if key not in _NODE_CACHE:
        # integral of f(th) th^(a-1) (1-th)^(b-1) dth over (0,1), th = (1+x)/2
        x, w = roots_jacobi(count, b_g - 1.0, a_g - 1.0)
        theta = 0.5 * (1.0 + x)
        scale = 0.5 ** (a_g + b_g - 1.0)
        _NODE_CACHE[key] = (theta, w * scale)
    return _NODE_CACHE[key]


def pooled_mean(a_g, b_g, wins, losses, cells):
    """Exact tau -> infinity endpoint: the deployed parity posterior mean."""
    w = float(sum(wins[c] for c in cells))
    l = float(sum(losses[c] for c in cells))
    return (a_g + w) / (a_g + b_g + w + l)


def cell_log_ratio(theta, wins, losses, cells, tau):
    """Log of the product of cell Beta-function ratios; the PRIOR is not included.

    The prior weight theta^(a-1)(1-theta)^(b-1) is supplied by the Gauss-Jacobi rule, so
    including it here too would double count it.
    """
    th = np.clip(theta, EPS, 1.0 - EPS)
    alpha = tau * th
    beta_ = tau * (1.0 - th)
    out = np.zeros_like(th)
    for c in cells:
        out = out + (betaln(alpha + wins[c], beta_ + losses[c]) - betaln(alpha, beta_))
    return out


def parent_log_weight(theta, a_g, b_g, wins, losses, cells, tau):
    """Unnormalised log parent posterior INCLUDING the prior, for reference checks."""
    th = np.clip(theta, EPS, 1.0 - EPS)
    return ((a_g - 1.0) * np.log(th) + (b_g - 1.0) * np.log1p(-th)
            + cell_log_ratio(th, wins, losses, cells, tau))


def predictive(a_g, b_g, wins, losses, cell, tau, nodes=None):
    """Predictive mean for `cell` under pooling strength `tau`.

    `tau=None` selects the exact pooled endpoint. Returns (q, parent_mean).
    """
    cells = group_of(cell)
    if tau is None or not np.isfinite(tau):
        m = pooled_mean(a_g, b_g, wins, losses, cells)
        return m, m
    x, w = nodes if nodes is not None else _nodes(128, a_g, b_g)
    logw = cell_log_ratio(x, wins, losses, cells, tau)
    logw = logw - logw.max()
    dens = np.exp(logw) * w
    total = dens.sum()
    if not np.isfinite(total) or total <= 0:
        m = pooled_mean(a_g, b_g, wins, losses, cells)
        return m, m
    parent = float(np.dot(dens, x) / total)
    wc, lc = float(wins[cell]), float(losses[cell])
    q = (wc + tau * parent) / (wc + lc + tau)
    return float(q), parent


def independent_mean(a_g, b_g, wins, losses, cell):
    """Independent four-cell control: each cell gets the parity prior on its own."""
    return (a_g + float(wins[cell])) / (a_g + b_g + float(wins[cell]) + float(losses[cell]))


def departure_sd(theta, tau):
    """Interpretable scale of tau: prior sd of a cell probability around the parent."""
    return float(np.sqrt(theta * (1.0 - theta) / (tau + 1.0)))
