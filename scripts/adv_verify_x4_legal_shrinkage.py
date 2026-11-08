"""Second counterfactual for the Family X / X4 verification.

X4's shrinkage baseline is artifacts/win_head_release_exact/validation_submission.csv,
which scripts/stack_manifest.py:117 classifies DEPLOY_FIT. A1 in
scripts/adv_verify_x4_mixture.py removed the baseline entirely (lam == 0). This script
instead keeps the shrinkage ESTIMATOR and swaps the inadmissible occupant of the
baseline slot for a legal, nested one: the fitting-side best single crossed bank member
of that fold, which is built only from the 15 audited RAW sources and the fitting-side
tail constants.

Arm X4legal: general convex mixture over the complete crossed bank, shrinkage toward the
nested baseline, lam chosen by 3 complete-episode inner folds of the fitting episodes
together with the no-change-relative-to-that-baseline option. Tail constants refit on
each inner-TRAINING partition. Held folds concatenated, scored once.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)
from slot_forecasting.io import read_submission, write_submission
from slot_model import harness as H
from scripts.general_convex_crossed import solve_simplex_qp, LAM_GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/adv_verify_x4_mixture')
    args = ap.parse_args()
    (args.out / 'predictions').mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    b = H.load(args.data)
    names, P, Q, R, _ = H.raw_pool(b)
    n, truth, base = len(b.ids), b.truth, b.protected
    A_all = R[:, :, 0]
    type_order = sorted(set(b.types.tolist()))

    def dedup(arr):
        keep, seen = [], set()
        for nm in sorted(names):
            j = names.index(nm)
            h = hashlib.sha256(arr[j].round(12).tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            keep.append(j)
        return sorted(keep)

    qkeep, akeep = dedup(Q), dedup(A_all)
    NQ, NR = len(qkeep), len(akeep)

    def state(mask):
        tails = b.tail_constants(mask)
        tails = {t: tails[t] / tails[t].sum() for t in type_order}
        V = np.stack([tails[t] for t in b.types])
        bank = np.empty((NQ * NR, n, 5))
        for jj, j in enumerate(qkeep):
            for kk, k in enumerate(akeep):
                bank[jj * NR + kk] = b.compose_qav(Q[j], A_all[k], V)
        w = b.row_weights(mask)
        s = np.sqrt(w)
        D = ((bank - truth[None, :, :]) * s[None, :, None]).reshape(NQ * NR, -1)
        G = D @ D.T
        # LEGAL baseline: the fitting-side best single crossed member of this very mask.
        best = int(np.argmin(np.diag(G)))
        nested_base = bank[best]
        E = ((bank - nested_base[None, :, :]) * s[None, :, None]).reshape(NQ * NR, -1)
        return {'bank': bank, 'G': G, 'H': E @ E.T, 'w': w, 'base': nested_base,
                'best': best}

    def fit(st, lam):
        Gs = st['G'] + (lam * st['H'] if lam else 0.0)
        W, info = solve_simplex_qp(Gs)
        p = np.einsum('i,ind->nd', W, st['bank'])
        return p / p.sum(axis=1, keepdims=True), W, info

    held = np.empty_like(base)
    prov = []
    for f in range(4):
        outer = b.outer_mask(f)
        fitm = ~outer
        inner = b.inner_folds(fitm, 3)
        ist = [(state(itr), b.row_weights(ihe)) for itr, ihe in inner]
        fs = state(fitm)
        tbl = {}
        for lam in LAM_GRID:
            tbl[lam] = sum(b.wloss(fit(s, lam)[0], whe) for s, whe in ist)
        # no-change option, measured against each inner partition's OWN nested baseline
        no_change = sum(b.wloss(s['base'], whe) for s, whe in ist)
        cfg = min(sorted(tbl), key=lambda c: (tbl[c], c))
        rec = {'fold': f, 'inner_table': {str(k): v for k, v in sorted(tbl.items())},
               'inner_no_change': no_change,
               'nested_baseline_member': [names[qkeep[fs['best'] // NR]],
                                          names[akeep[fs['best'] % NR]]]}
        if no_change <= tbl[cfg]:
            held[outer] = fs['base'][outer]
            rec['choice'] = 'no_change'
        else:
            p, W, info = fit(fs, cfg)
            held[outer] = p[outer]
            rec.update({'choice': 'mixture', 'lam': cfg, 'support': int((W > 1e-9).sum()),
                        'kkt': info['kkt_residual']})
        prov.append(rec)
        print(f"  fold {f}: {rec['choice']} lam={rec.get('lam','-')}", flush=True)

    o = b.official(held)
    sp = ROOT / 'artifacts/nested_audited/predictions/oof_separable_rescaled.csv'
    d = read_submission(sp)
    S = np.array([d[i] for i in b.ids]); S = S / S.sum(axis=1, keepdims=True)
    dx = read_submission(ROOT / 'artifacts/general_convex_crossed/predictions/oof_X4.csv')
    X4 = np.array([dx[i] for i in b.ids]); X4 = X4 / X4.sum(axis=1, keepdims=True)
    out = {'arm': 'X4legal',
           'description': 'general mixture, complete crossed bank, shrinkage toward the '
                          'fitting-side best single crossed member (nested, RAW-only), '
                          'lam and the no-change option by inner folds',
           'points': o['technical_points'], 'R': o['mean_loss_ratio'],
           'by_condition': {c: v['loss_ratio'] for c, v in sorted(o['by_condition'].items())},
           'paired_vs_protected': b.paired(held, base)['delta_loss_ratio_ci95'],
           'paired_vs_their_X4': b.paired(held, X4)['delta_loss_ratio_ci95'],
           'paired_vs_separable_rescaled': b.paired(held, S)['delta_loss_ratio_ci95'],
           'fold_provenance': prov, 'seconds': time.perf_counter() - t0}
    write_submission(args.out / 'predictions' / 'oof_X4legal.csv', b.mapping(held))
    (args.out / 'x4legal.json').write_text(json.dumps(out, indent=2, sort_keys=True,
                                                      default=float) + '\n')
    print(f"\nX4legal {out['points']:.6f}  R {out['R']:.6f}")
    print(f"  vs protected {out['paired_vs_protected']}")
    print(f"  vs their X4  {out['paired_vs_their_X4']}")
    print(f"  vs separable {out['paired_vs_separable_rescaled']}")
    print(f"wrote in {out['seconds']:.1f}s")


if __name__ == '__main__':
    main()
