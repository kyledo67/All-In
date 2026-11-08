"""Exact Euclidean projection of a probability vector onto a convex hull of vertices.

Needed because the deployed `tail_train` transformation is NOT a projection. It preserves
a = P(1x | positive) - hence p_0 and p_1 - and rescales the remaining mass to the fixed
tail ratio. That lands inside the declared family but is not the nearest point in it, so it
does not test the projection hypothesis.

For vertices V (m by 5) the family is conv(V) = {V^T lambda : lambda in the simplex}, and

    project(p) = V^T argmin_lambda  lambda^T G lambda - 2 b^T lambda,
        G = V V^T,  b = V p,  sum(lambda) = 1,  lambda >= 0.

With m = 3 the exact solution is found by enumerating all 7 nonempty active sets and
solving the equality-constrained subproblem on each, so no iterative solver is involved
and the result is exact up to the linear solve. Predd et al. (2009) motivate projecting a
forecast onto a constraint set the target is known to satisfy; note the guarantee that
projection cannot increase squared error holds only when the TRUE target lies in the set,
and here the set is ESTIMATED, so no unconditional improvement follows on unseen data.
"""
from itertools import combinations

import numpy as np


def _solve_equality(G, b, index):
    """Minimise over lambda supported on `index` with the entries summing to one."""
    k = len(index)
    sub = G[np.ix_(index, index)]
    # [2G 1; 1^T 0] [lambda; mu] = [2b; 1]
    A = np.zeros((k + 1, k + 1))
    A[:k, :k] = 2.0 * sub
    A[:k, k] = 1.0
    A[k, :k] = 1.0
    rhs = np.zeros(k + 1)
    rhs[:k] = 2.0 * b[index]
    rhs[k] = 1.0
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:k]


def project_to_hull(p, vertices, tol=1e-12):
    """Exact nearest point of conv(vertices) to p, plus its barycentric weights."""
    V = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = V.shape[0]
    G = V @ V.T
    b = V @ p
    best = None
    for size in range(1, m + 1):
        for index in combinations(range(m), size):
            lam_sub = _solve_equality(G, b, list(index))
            if lam_sub is None or np.any(lam_sub < -tol):
                continue
            lam = np.zeros(m)
            lam[list(index)] = np.maximum(lam_sub, 0.0)
            s = lam.sum()
            if s <= 0:
                continue
            lam = lam / s
            value = float(lam @ G @ lam - 2.0 * b @ lam)
            if best is None or value < best[0] - 1e-15:
                best = (value, lam)
    if best is None:                      # fall back to the nearest vertex
        d = [float(np.sum((V[i] - p) ** 2)) for i in range(m)]
        lam = np.zeros(m)
        lam[int(np.argmin(d))] = 1.0
        best = (0.0, lam)
    lam = best[1]
    out = V.T @ lam
    total = out.sum()
    if np.isfinite(total) and total > 0:
        out = out / total
    return out, lam


def tail_family_vertices(tail):
    """conv{(1,0,0,0,0), (0,1,0,0,0), (0,0,v_2,v_5,v_20)} for a type's fixed tail split."""
    v = np.asarray(tail, dtype=np.float64)
    v = v / v.sum()
    return np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, v[0], v[1], v[2]]])


def endpoint_family_vertices(r_low, r_high):
    """conv{e_0, (0, r_L), (0, r_H)}: all-loss plus the two pure conditional endpoints.

    This permits continuous uncertainty BETWEEN the endpoints and in the win mass, which
    hard snapping to the nearer endpoint does not.
    """
    lo = np.asarray(r_low, dtype=np.float64)
    hi = np.asarray(r_high, dtype=np.float64)
    return np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                     np.concatenate(([0.0], lo / lo.sum())),
                     np.concatenate(([0.0], hi / hi.sum()))])
