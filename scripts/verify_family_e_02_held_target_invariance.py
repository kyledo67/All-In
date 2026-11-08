"""Empirical leakage test: corrupt the OUTER HELD targets of one fold and check that
the arm's forecast on that fold is unchanged."""
import sys, copy
sys.path.insert(0,'/Users/ayan/Downloads/All-In'); sys.path.insert(0,'/Users/ayan/Downloads/All-In/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
import scripts.endpoint_supervised as E

b = H.load(Path('main-participants-60-40/public'))
qh, r_prot = b.split_heads(b.protected)
u_prot = r_prot[:,0]
raw = E.gather(b)
n = len(b.ids)
ctx = {'bundle': b, 'raw': raw, 'n': n, 'q': qh, 'u_prot': u_prot, 'train_mixture': None}

def one_fold(bundle, arm, ctx, fold):
    """Exact copy of run_arm's per-fold body."""
    n = ctx['n']; qh, u_prot = ctx['q'], ctx['u_prot']
    held = bundle.outer_mask(fold); fit = ~held
    w = bundle.row_weights(fit)
    inner = bundle.inner_folds(fit, 3)
    cache = {}
    for cfg in arm.fit_grid():
        u_in = u_prot.copy(); V_in = np.zeros((n,3))
        for tr, he in inner:
            V_in[he] = E.tail_matrix(bundle, tr)[he]
            model = arm.fit(ctx, tr, cfg)
            u_in[he] = arm.predict(ctx, model, cfg)[he]
        cache[cfg] = (u_in, V_in)
    best = None
    for cfg in arm.fit_grid():
        u_in, V_in = cache[cfg]
        for beta in arm.blend:
            u = np.clip(u_prot + beta*(u_in-u_prot), E.EPS, 1-E.EPS)
            value = bundle.wloss(E.compose(bundle, qh, u, V_in), w)
            if not np.isfinite(value): continue
            if best is None or value < best[0]-1e-13: best = (value, cfg, beta)
    _, cfg, beta = best
    V = E.tail_matrix(bundle, fit)
    model = arm.fit(ctx, fit, cfg)
    u = np.clip(u_prot + beta*(arm.predict(ctx, model, cfg)-u_prot), E.EPS, 1-E.EPS)
    return cfg, beta, u[held], V[held]

arm = E.Calibrator(drop_uprot=True)
base = {}
for f in range(4):
    base[f] = one_fold(b, arm, ctx, f)
    print('fold', f, 'cfg', base[f][0], 'beta', base[f][1])

rng = np.random.default_rng(99)
for f in range(4):
    held = b.outer_mask(f)
    b2 = copy.copy(b)
    b2._wcache = {}
    t2 = b.truth.copy()
    # replace held-fold targets with a fixed wrong simplex vector (destroys all held info)
    bad = np.array([0.5,0.2,0.15,0.1,0.05])
    t2[held] = bad
    b2.truth = t2
    E._DESIGN_CACHE.clear()
    ctx2 = dict(ctx); ctx2['bundle'] = b2
    cfg, beta, uh, Vh = one_fold(b2, arm, ctx2, f)
    du = float(np.abs(uh - base[f][2]).max()); dV = float(np.abs(Vh - base[f][3]).max())
    print('fold %d corrupted-held: cfg=%s beta=%s  max|du|=%.3e max|dV|=%.3e  same_choice=%s'
          % (f, cfg, beta, du, dV, (cfg==base[f][0] and beta==base[f][1])))

# also: shuffle held targets within the held fold (another corruption)
for f in range(4):
    held = b.outer_mask(f)
    b2 = copy.copy(b); b2._wcache = {}
    t2 = b.truth.copy()
    idx = np.where(held)[0]
    t2[idx] = b.truth[rng.permutation(idx)]
    b2.truth = t2
    E._DESIGN_CACHE.clear()
    ctx2 = dict(ctx); ctx2['bundle'] = b2
    cfg, beta, uh, Vh = one_fold(b2, arm, ctx2, f)
    du = float(np.abs(uh - base[f][2]).max()); dV = float(np.abs(Vh - base[f][3]).max())
    print('fold %d shuffled-held: max|du|=%.3e max|dV|=%.3e same_choice=%s'
          % (f, du, dV, (cfg==base[f][0] and beta==base[f][1])))
