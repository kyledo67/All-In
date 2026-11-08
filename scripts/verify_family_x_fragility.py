"""How fragile is X1full's 'significant vs protected' verdict?

The official paired_bootstrap is seeded (314159, 2000 repeats), so its interval is a fixed
number, not a random one. This script does not second-guess that number; it measures how
much slack the verdict has:

  1. the one-sided bootstrap mass at or above zero (the implied one-sided p)
  2. the upper bound at Bonferroni levels for the 4 and 9 arms the family reported
  3. the spread of the upper bound across 400 alternative bootstrap seeds and across
     repeat counts, i.e. how much of the verdict is Monte-Carlo noise in the resampler
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import harness as H
from slot_forecasting.io import read_submission
from slot_forecasting.schema import CONDITIONS

OUT = ROOT / 'artifacts/verify_family_x'


def rows_for(b, a, q):
    """Per-episode per-condition [arm, base, reference] mean squared error, as the official
    paired_bootstrap builds them."""
    targets, groups, reference = b.scoring
    am, qm = b.mapping(a), b.mapping(q)
    rows = {}
    for condition in CONDITIONS:
        es = sorted({t.episode_id for t in targets if groups[t.query_id] == condition})
        vals = []
        for e in es:
            ts = [t for t in targets if t.episode_id == e]
            vals.append([np.mean([np.sum((np.asarray(p[t.query_id]) - t.probabilities) ** 2)
                                  for t in ts]) for p in (am, qm, reference)])
        rows[condition] = np.asarray(vals)
    return rows


def totals(rows, seed, repeats):
    rng = np.random.default_rng(seed)
    tot = np.zeros(repeats)
    for condition in CONDITIONS:
        v = rows[condition]
        draws = rng.integers(0, len(v), (repeats, len(v)))
        m = v[draws].mean(1)
        tot += ((m[:, 0] - m[:, 1]) / np.maximum(m[:, 2], 1e-9)) / 4
    return tot


def main():
    b = H.load(ROOT / 'main-participants-60-40/public')
    arms = ('Xnull', 'X0', 'X1', 'X2', 'X3', 'X1full', 'X2full', 'X4', 'X5')
    pred = {}
    for k in arms:
        f = ROOT / f'artifacts/general_convex_crossed/predictions/oof_{k}.csv'
        d = read_submission(f)
        A = np.array([d[i] for i in b.ids])
        pred[k] = A / A.sum(axis=1, keepdims=True)

    rep = {}
    for k in ('X1full', 'X2full', 'X4', 'X5'):
        rows = rows_for(b, pred[k], b.protected)
        t = totals(rows, 314159, 2000)
        lo, hi = np.quantile(t, [.025, .975])
        n9 = 0.05 / 9          # 9 arms reported
        n4 = 0.05 / 4          # the 4 arms the family called significant
        rec = {
            'ci95': [float(lo), float(hi)],
            'point_estimate_mean': float(t.mean()),
            'one_sided_mass_at_or_above_zero': float((t >= 0).mean()),
            'upper_bound_bonferroni_9_arms': float(np.quantile(t, 1 - n9 / 2)),
            'upper_bound_bonferroni_4_arms': float(np.quantile(t, 1 - n4 / 2)),
            'survives_bonferroni_9': bool(np.quantile(t, 1 - n9 / 2) < 0),
            'survives_bonferroni_4': bool(np.quantile(t, 1 - n4 / 2) < 0),
        }
        his = []
        for s in range(400):
            tt = totals(rows, 1000 + s, 2000)
            his.append(float(np.quantile(tt, .975)))
        his = np.asarray(his)
        rec['reseed_400'] = {
            'upper_bound_min': float(his.min()), 'upper_bound_max': float(his.max()),
            'upper_bound_mean': float(his.mean()),
            'fraction_of_seeds_with_upper_bound_below_zero': float((his < 0).mean()),
        }
        for reps in (2000, 20000, 200000):
            tt = totals(rows, 314159, reps)
            rec[f'upper_bound_repeats_{reps}'] = float(np.quantile(tt, .975))
        rep[k] = rec

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fragility.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
