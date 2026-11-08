"""Mechanism probes for Family X arm X2full: native-tail variant and a null calibration."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json, sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)
from slot_model import harness as H
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_family_x_adv import my_qp, build

OUT = ROOT / 'artifacts/verify_x2full_audit'
DATA = ROOT / 'main-participants-60-40/public'


def run(b, Q, A, qk, ak, tail_mode, fit_truth=None):
    """tail_mode: 'fitted' (per-type v from fitting truth) or 'native' (member's own tail)."""
    names, P, Q0, R0, _ = H.raw_pool(b)
    tr = b.truth if fit_truth is None else fit_truth
    n = len(b.ids)
    NQ, NR = len(qk), len(ak)
    held = np.empty((n, 5))
    for f in range(4):
        outer = b.outer_mask(f); fitm = ~outer
        if tail_mode == 'fitted':
            rt = tr[:, 1:] / np.maximum(tr[:, 1:].sum(axis=1, keepdims=True), 1e-15)
            tails = {}
            for t in sorted(set(b.types.tolist())):
                m = fitm & (b.types == t)
                tl = rt[m][:, 1:]
                tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1e-15)
                v = tl.mean(axis=0)
                tails[t] = v / v.sum()
            V = np.stack([tails[t] for t in b.types])
            bank = np.empty((NQ * NR, n, 5))
            for jj, j in enumerate(qk):
                for kk, k in enumerate(ak):
                    bank[jj * NR + kk] = b.compose_qav(Q[j], A[k], V)
        else:   # native: cross q head j with r head k verbatim (no tail replacement)
            bank = np.empty((NQ * NR, n, 5))
            for jj, j in enumerate(qk):
                for kk, k in enumerate(ak):
                    bank[jj * NR + kk] = b.compose(Q[j], R0[k])
        w = b.row_weights(fitm) if fit_truth is None else rw(b, fitm, tr)
        s = np.sqrt(w)
        D = ((bank - tr[None]) * s[None, :, None]).reshape(NQ * NR, -1)
        W, _ = my_qp(D @ D.T)
        p = np.einsum('i,ind->nd', W, bank)
        held[outer] = (p / p.sum(axis=1, keepdims=True))[outer]
    return held


def rw(b, mask, tr):
    w = np.zeros(len(b.ids))
    inm = sorted({e for i, e in enumerate(b.ep) if mask[i]})
    for c in sorted(set(b.groups.values())):
        ces = sorted({e for i, e in enumerate(b.ep)
                      if e in set(inm) and b.groups[b.ids[i]] == c})
        if not ces:
            continue
        bb = float(np.mean([np.mean(np.sum((b.ref[b.ep == e] - tr[b.ep == e]) ** 2, axis=1))
                            for e in ces]))
        for e in ces:
            nn = int((b.ep == e).sum())
            w[b.ep == e] = 1.0 / (4 * len(ces) * nn * max(bb, 1e-9))
    return w


def main():
    t0 = time.perf_counter()
    b = H.load(DATA)
    names, Q, A, qk, ak, _ = build(b)
    R0 = H.raw_pool(b)[3]
    out = {}
    # A. members' native tails, everything else identical  -> isolates the tail refit
    nat = run(b, Q, A, qk, ak, 'native')
    o = b.official(nat)
    out['native_tails'] = {'points': o['technical_points'], 'R': o['mean_loss_ratio'],
                           'paired_vs_protected': b.paired(nat, b.protected)['delta_loss_ratio_ci95']}
    # B. fitted tails (reproduces X2full) for reference
    fit = run(b, Q, A, qk, ak, 'fitted')
    o = b.official(fit)
    out['fitted_tails'] = {'points': o['technical_points'], 'R': o['mean_loss_ratio'],
                           'paired_vs_protected': b.paired(fit, b.protected)['delta_loss_ratio_ci95'],
                           'paired_vs_native': b.paired(fit, nat)['delta_loss_ratio_ci95']}
    # C. null calibration: fit the whole nested recipe against a GLOBAL permutation of the
    #    truth, then score against the REAL targets. A gain here would be machinery, not signal.
    rng = np.random.default_rng(20261108)
    perm = rng.permutation(len(b.ids))
    fake = b.truth[perm]
    nul = run(b, Q, A, qk, ak, 'fitted', fit_truth=fake)
    o = b.official(nul)
    out['null_calibration_permuted_truth'] = {
        'points': o['technical_points'], 'R': o['mean_loss_ratio'],
        'paired_vs_protected': b.paired(nul, b.protected)['delta_loss_ratio_ci95']}
    out['seconds'] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'probe2.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
