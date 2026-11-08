"""Mechanism and implementation checks for the parity-prior Beta mixture.

Each check targets a specific risk rather than restating the model. Sufficient-statistic
checks are kept DISTINCT from full-history mutation checks, because mutating a history can
also change lag assignments and the current context, which would confound a claim about
the prior.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import parity_prior as PP
from slot_model.features import PAYOUTS, causal_index
from slot_forecasting.io import load_inputs, read_submission
from slot_forecasting.schema import Play
from scripts.parity_prior_run import build_cache

CHECKS = []


def record(name, ok, detail):
    CHECKS.append({'check': name, 'passed': bool(ok), 'detail': detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_exact')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'artifacts/parity_prior/verification.json')
    args = parser.parse_args()

    counts = ((6.0, 4.0), (2.0, 8.0))
    lo, hi, conc, pi = 0.30, 0.70, 4.0, 0.5

    # 1. one-component case equals the ordinary Beta posterior mean
    worst = 0.0
    for g in (0, 1):
        q, om = PP.predict(counts, 0.45, 0.45, conc, pi, 0.0, g)
        gw, gl = counts[g]
        ref = (conc * 0.45 + gw) / (conc + gw + gl)
        worst = max(worst, abs(q - ref))
    record('one_component_identity', worst < 1e-12,
           f'with mu_L = mu_H the mixture collapses to the ordinary Beta posterior mean to '
           f'{worst:.2e}, for both queried groups')

    # 2. zero coupling: the queried group ignores the OTHER group entirely
    deltas = []
    for g in (0, 1):
        base, _ = PP.predict(counts, lo, hi, conc, pi, 0.0, g)
        other = 1 - g
        for alt in ((0.0, 0.0), (30.0, 1.0), (1.0, 30.0)):
            c2 = list(counts)
            c2[other] = alt
            q2, _ = PP.predict(tuple(c2), lo, hi, conc, pi, 0.0, g)
            deltas.append(abs(q2 - base))
    record('independence_blocks_borrowing', max(deltas) < 1e-12,
           f'at rho = 0, replacing the opposite group counts with (0,0), (30,1) and (1,30) '
           f'moves the queried estimate by at most {max(deltas):.2e} - the component '
           f'posterior factorises exactly, so borrowing can only come from the coupling')

    # 3. with dependence, an informative opposite group moves the estimate, and the two
    #    signs of dependence move it in opposite directions
    g = 0
    base, _ = PP.predict(counts, lo, hi, conc, pi, 0.0, g)
    hi_other = ((6.0, 4.0), (30.0, 1.0))      # other group looks HIGH
    pos, _ = PP.predict(hi_other, lo, hi, conc, pi, +0.9, g)
    neg, _ = PP.predict(hi_other, lo, hi, conc, pi, -0.9, g)
    ind, _ = PP.predict(hi_other, lo, hi, conc, pi, 0.0, g)
    record('dependence_borrows_with_sign',
           pos > ind + 1e-4 and neg < ind - 1e-4,
           f'with the opposite group strongly HIGH: independence {ind:.5f}, positive '
           f'dependence {pos:.5f} (pulled up), negative dependence {neg:.5f} (pushed down) - '
           f'both signs are representable and act in opposite directions')

    # 4. marginals are preserved for any admissible delta
    worst_marg = 0.0
    for p in (0.2, 0.5, 0.8):
        for rho in (-1.0, -0.5, 0.0, 0.5, 1.0):
            w = PP.joint_weights(p, PP.delta_from_rho(p, rho))
            worst_marg = max(worst_marg, abs(w.sum(axis=1)[1] - p), abs(w.sum(axis=0)[1] - p))
    record('marginals_preserved', worst_marg < 1e-9,
           f'across pi in {{0.2,0.5,0.8}} and rho in [-1,1] both group marginals stay at pi to '
           f'{worst_marg:.2e}, so the coupling changes dependence only')

    # 5. strong queried-group evidence overrides the prior
    weak, _ = PP.predict(((1.0, 0.0), (0.0, 0.0)), lo, hi, conc, pi, 0.0, 0)
    strong, _ = PP.predict(((200.0, 20.0), (0.0, 0.0)), lo, hi, conc, pi, 0.0, 0)
    record('evidence_dominates_prior', strong > 0.85 and abs(strong - 200 / 220) < 0.03,
           f'1 win alone gives {weak:.4f}; 200 wins against 20 losses gives {strong:.4f}, '
           f'within 0.03 of the empirical rate {200/220:.4f}')

    # 6. asymmetric magnitudes and opposite machine orientations representable
    a_hi, _ = PP.predict(((9.0, 1.0), (1.0, 9.0)), lo, hi, conc, pi, 0.0, 0)
    a_lo, _ = PP.predict(((1.0, 9.0), (9.0, 1.0)), lo, hi, conc, pi, 0.0, 0)
    record('opposite_orientations', a_hi > 0.6 and a_lo < 0.4,
           f'a machine whose queried group is high gives {a_hi:.4f} and the mirrored machine '
           f'{a_lo:.4f}, so opposite orientations are representable per machine')

    # 7. finiteness and normalisation over a sweep
    rng = np.random.default_rng(0)
    bad = 0
    wsum = 0.0
    for _ in range(5000):
        c = ((float(rng.gamma(1, 5)), float(rng.gamma(1, 5))),
             (float(rng.gamma(1, 5)), float(rng.gamma(1, 5))))
        q, om = PP.predict(c, float(rng.uniform(0.05, 0.5)), float(rng.uniform(0.5, 0.95)),
                           float(rng.uniform(0.1, 50)), float(rng.uniform(0.05, 0.95)),
                           float(rng.uniform(-1, 1)), int(rng.integers(2)))
        wsum = max(wsum, abs(om.sum() - 1.0))
        if not (np.isfinite(q) and 0.0 < q < 1.0):
            bad += 1
    record('finite_normalised', bad == 0 and wsum < 1e-12,
           f'{bad} of 5000 random configurations left (0,1); max component-weight sum error '
           f'{wsum:.2e}')

    # ---- real-data checks: frozen identities, post-cutoff, positive control ----
    cache, kappa, prior_mean = build_cache(args.data, args.package)
    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    routed = [q.query_id for q in vq if cache[q.query_id]['template'] == 'parity']
    record('routes_cached', len(routed) > 0,
           f'{len(routed)} of {len(vq)} queries are parity-routed and carry frozen '
           f'(pair, template, half-life) identities reused across every arm')

    cut = {(q.episode_id, q.machine_id): q.cutoff for q in vq}
    extra = []
    for (e, m), c in sorted(cut.items()):
        obs = index[(e, m)]
        for off in range(8):
            if c + off not in obs:
                extra.append(Play(e, m, c + off, PAYOUTS[(obs.get(c - 1, 0) + 3) % 5]))
    mutated, _, _ = build_cache(args.data, args.package, tuple(vh) + tuple(extra))
    same = sum(1 for k in cache
               if json.dumps(_norm(cache[k]), sort_keys=True) == json.dumps(_norm(mutated[k]), sort_keys=True))
    record('post_cutoff_invariant', same == len(cache),
           f'{len(extra)} adversarial plays appended at ticks cutoff..cutoff+7; '
           f'{len(cache) - same} of {len(cache)} cached identities/counts changed')

    flipped = []
    for p in vh:
        c = cut.get((p.episode_id, p.machine_id))
        flipped.append(Play(p.episode_id, p.machine_id, p.tick, 0 if p.payout > 0 else 5)
                       if c is not None and p.tick == c - 1 else p)
    pos_cache, _, _ = build_cache(args.data, args.package, tuple(flipped))
    moved = sum(1 for k in cache if _norm(cache[k]) != _norm(pos_cache[k]))
    record('positive_control', moved > 0.3 * len(cache),
           f'flipping each machine lag-1 pre-cutoff outcome changed {moved} of {len(cache)} '
           f'cached states, so post-cutoff invariance is not an inert pipeline')

    passed = all(c['passed'] for c in CHECKS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'checks': CHECKS, 'passed': passed}, indent=2) + '\n')
    print(f"\n{sum(c['passed'] for c in CHECKS)}/{len(CHECKS)} checks passed")
    print('PARITY PRIOR VERIFICATION', 'PASSED' if passed else 'FAILED')
    if not passed:
        raise SystemExit(1)


def _norm(rec):
    out = {k: v for k, v in rec.items() if k != 'mu'}
    if 'counts' in out:
        out['counts'] = [[round(x, 9) for x in pair] for pair in out['counts']]
    if out.get('subset') is not None:
        out['subset'] = list(out['subset'])
    return out


if __name__ == '__main__':
    main()
