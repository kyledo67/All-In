"""ADVERSARIAL VERIFICATION of Family N's claimed gain `boost_enriched_nopreq`.

Six checks, each independent of Family N's own driver code:
  1. rescore the arm's written CSV with the organizer scorer (concatenated, once);
  2. feature-matrix truth-independence: rebuild X with ALL truth replaced by noise;
  3. independent re-implementation of the nopreq arm from the harness primitives;
  4. per-fold truth-corruption leaktest on ALL FOUR outer folds, not just fold 0;
  5. recompute b.paired(arm, protected) and b.paired(arm, control);
  6. multiplicity-aware paired bootstrap over the five fixed feature subsets, i.e.
     the honest null for "take the best of five rows in the OUTER table".
Writes artifacts/verify_family_n_adv/verification.json. Reads nothing it fits.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_forecasting.io import read_submission
from slot_forecasting.schema import CONDITIONS
from slot_model import harness as H
from scripts import nonlinear_diagnostic_residual as N

OUT = ROOT / 'artifacts/verify_family_n_adv'
SRC = ROOT / 'artifacts/nonlinear_diag_residual'
ARM = 'boost_enriched_nopreq'
FIXED_SUBSETS = ('all', 'nopreq', 'nopool', 'noother', 'diag_only')
ARM_FILE = {'all': 'boost_enriched', 'nopreq': 'boost_enriched_nopreq',
            'nopool': 'boost_enriched_nopool', 'noother': 'boost_enriched_noother',
            'diag_only': 'boost_diag_only'}


def load_csv(path, ids):
    d = read_submission(path)
    a = np.array([d[i] for i in ids])
    return a / a.sum(axis=1, keepdims=True)


def nopreq_cols():
    keep = [n for n in N.NAMES if n not in N.BLOCKS['preq']]
    return np.array([N.NAMES.index(n) for n in keep]), keep


def my_nopreq_arm(b, X, cols):
    """Independent re-implementation: inner-select depth and shrinkage (zero included),
    refit on fitting episodes, apply to the held fold. Nothing touches held truth."""
    XS = X[:, cols]
    Y = b.truth - b.protected
    held = b.protected.copy()
    picks = []
    for fold in range(4):
        outer = b.outer_mask(fold)
        fit = ~outer
        parts = [(tr, he, b.row_weights(he)) for tr, he in b.inner_folds(fit, k=3)]
        baseline = sum(b.wloss(b.protected, wh) for _, _, wh in parts)
        scores = {'zero': baseline}
        for depth in N.DEPTHS:
            cache = []
            for tr, he, wh in parts:
                wtr = b.row_weights(tr)
                cache.append(N.boost_delta(XS[tr], Y[tr], wtr[tr], XS, depth))
            for s in N.SHRINKS:
                if s == 0.0:
                    continue
                scores[f'd{depth}|s{s:g}'] = sum(
                    b.wloss(N.apply_delta(b.protected, D, s), wh)
                    for D, (_, _, wh) in zip(cache, parts))
        key = min(sorted(scores), key=lambda k: (scores[k], k))
        if key != 'zero' and scores[key] < baseline - 1e-15:
            ds, ss = key.split('|')
            wf = b.row_weights(fit)
            D = N.boost_delta(XS[fit], Y[fit], wf[fit], XS, int(ds[1:]))
            held[outer] = N.apply_delta(b.protected, D, float(ss[1:]))[outer]
        picks.append(key)
    return held, picks


def corrupt(b, mask, seed):
    b2 = copy.copy(b)
    b2.truth = b.truth.copy()
    b2._wcache = {}
    rng = np.random.default_rng(seed)
    bad = rng.random((int(mask.sum()), 5)) ** 3
    b2.truth[mask] = bad / bad.sum(axis=1, keepdims=True)
    return b2


def multiplicity_bootstrap(b, arms, base, repeats=20000, seed=314159):
    """Exactly paired_bootstrap's resampling, but tracking all five subsets jointly.

    For each replicate we record dR for every subset and the MINIMUM over subsets, which
    is the statistic actually reported when the best outer row is quoted. The upper 97.5%
    quantile of that minimum is the selection-aware bound.
    """
    targets, groups, reference = b.scoring
    names = sorted(arms)
    rows = {}
    for condition in CONDITIONS:
        es = sorted({t.episode_id for t in targets if groups[t.query_id] == condition})
        vals = []
        for episode in es:
            ts = [t for t in targets if t.episode_id == episode]
            series = [arms[n] for n in names] + [base, reference]
            vals.append([np.mean([np.sum((np.asarray(p[t.query_id]) - t.probabilities) ** 2)
                                  for t in ts]) for p in series])
        rows[condition] = np.asarray(vals)
    rng = np.random.default_rng(seed)
    totals = np.zeros((repeats, len(names)))
    for condition, vals in rows.items():
        draws = rng.integers(0, len(vals), (repeats, len(vals)))
        means = vals[draws].mean(1)
        den = np.maximum(means[:, -1], 1e-9)
        for j in range(len(names)):
            totals[:, j] += ((means[:, j] - means[:, -2]) / den) / 4
    best = totals.min(axis=1)
    out = {'per_subset_ci95': {n: np.quantile(totals[:, j], [.025, .975]).tolist()
                               for j, n in enumerate(names)},
           'best_of_five_ci95': np.quantile(best, [.025, .975]).tolist(),
           'best_of_five_upper_975': float(np.quantile(best, .975)),
           'p_best_of_five_negative': float(np.mean(best < 0)),
           'argmin_share': {n: float(np.mean(totals.argmin(axis=1) == j))
                            for j, n in enumerate(names)},
           'repeats': repeats, 'seed': seed}
    return out


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    b = H.load(ROOT / 'main-participants-60-40/public')
    prot = b.protected
    po = b.official(prot)
    rep['protected'] = {'points': po['technical_points'], 'R': po['mean_loss_ratio']}
    print(f"protected  pts={po['technical_points']:.9f} R={po['mean_loss_ratio']:.15f}")

    # ---- CHECK 1: rescore the arm's own written CSV, concatenated, once
    arm_csv = load_csv(SRC / f'predictions/oof_{ARM}.csv', b.ids)
    r = b.official(arm_csv)
    rep['check1_rescore_written_csv'] = {
        'points': r['technical_points'], 'R': r['mean_loss_ratio'],
        'by_condition': {c: v['loss_ratio'] for c, v in r['by_condition'].items()},
        'claimed_points': 53.861958780759, 'claimed_R': 0.11395873339532132,
        'abs_points_diff': abs(r['technical_points'] - 53.861958780759)}
    print(f"check1 CSV rescore pts={r['technical_points']:.9f} R={r['mean_loss_ratio']:.15f}")

    # ---- CHECK 2: feature matrix must not move when ALL truth is replaced by noise
    X, XO, _ = N.build_features(b, ROOT / 'artifacts/win_head_release_exact', verbose=False)
    ball = corrupt(b, np.ones(len(b.ids), dtype=bool), 12345)
    X2, XO2, _ = N.build_features(ball, ROOT / 'artifacts/win_head_release_exact', verbose=False)
    rep['check2_features_truth_independent'] = {
        'X_bit_identical': bool(np.array_equal(X, X2)),
        'XO_bit_identical': bool(np.array_equal(XO, XO2)),
        'max_abs_X_diff': float(np.max(np.abs(X - X2))),
        'n_features': int(X.shape[1])}
    print(f"check2 X identical under fully corrupted truth: {np.array_equal(X, X2)}")

    cols, keep = nopreq_cols()
    rep['nopreq_columns'] = keep

    # ---- CHECK 3: independent re-implementation of the arm
    mine, picks = my_nopreq_arm(b, X, cols)
    mine_n = mine / mine.sum(axis=1, keepdims=True)
    rm = b.official(mine_n)
    rep['check3_independent_reimpl'] = {
        'points': rm['technical_points'], 'R': rm['mean_loss_ratio'],
        'fold_picks': picks,
        'max_abs_diff_vs_family_csv': float(np.max(np.abs(mine_n - arm_csv))),
        'matches_family_points_to_1e-9': bool(
            abs(rm['technical_points'] - r['technical_points']) < 1e-9)}
    print(f"check3 reimpl pts={rm['technical_points']:.9f} picks={picks} "
          f"maxdiff={np.max(np.abs(mine_n - arm_csv)):.3e}")

    # ---- CHECK 4: per-fold truth corruption, ALL four folds
    folds_ok = {}
    for fold in range(4):
        mask = b.outer_mask(fold)
        bad = corrupt(b, mask, 900 + fold)
        dirty, dpicks = my_nopreq_arm(bad, X, cols)
        dn = dirty / dirty.sum(axis=1, keepdims=True)
        folds_ok[f'fold{fold}'] = {
            'held_rows_bit_identical': bool(np.array_equal(mine[mask], dirty[mask])),
            'max_abs_held_diff': float(np.max(np.abs(mine[mask] - dirty[mask]))),
            'other_folds_moved': bool(not np.array_equal(mine[~mask], dirty[~mask])),
            'picks_under_corruption': dpicks}
        print(f"check4 fold{fold} held identical="
              f"{folds_ok[f'fold{fold}']['held_rows_bit_identical']} "
              f"others moved={folds_ok[f'fold{fold}']['other_folds_moved']}")
    rep['check4_per_fold_leaktest'] = folds_ok

    # ---- CHECK 5: recompute paired intervals
    ctrl = load_csv(SRC / 'predictions/oof_boost_enriched.csv', b.ids)
    nested = load_csv(SRC / 'predictions/oof_boost_subset_nested.csv', b.ids)
    rep['check5_paired'] = {
        'arm_vs_protected': b.paired(arm_csv, prot)['delta_loss_ratio_ci95'],
        'arm_vs_control_boost_enriched': b.paired(arm_csv, ctrl)['delta_loss_ratio_ci95'],
        'nested_subset_vs_protected': b.paired(nested, prot)['delta_loss_ratio_ci95'],
        'control_vs_protected': b.paired(ctrl, prot)['delta_loss_ratio_ci95'],
        'claimed_vs_protected': [-0.011098, -0.002419]}
    for k, v in rep['check5_paired'].items():
        print(f"check5 {k:<32} [{v[0]:+.6f},{v[1]:+.6f}]")

    # ---- CHECK 6: selection-aware bootstrap over the five fixed outer-table subsets
    subs = {s: b.mapping(load_csv(SRC / f'predictions/oof_{ARM_FILE[s]}.csv', b.ids))
            for s in FIXED_SUBSETS}
    rep['check6_multiplicity'] = multiplicity_bootstrap(b, subs, b.mapping(prot))
    m = rep['check6_multiplicity']
    print(f"check6 best-of-five 95% [{m['best_of_five_ci95'][0]:+.6f},"
          f"{m['best_of_five_ci95'][1]:+.6f}]  P(min<0)={m['p_best_of_five_negative']:.4f}")
    print('       argmin share: ' + ', '.join(f'{k}={v:.3f}'
                                              for k, v in m['argmin_share'].items()))

    # outer table of the five fixed subsets, for the record
    rep['outer_table_five_fixed_subsets'] = {
        s: b.official(load_csv(SRC / f'predictions/oof_{ARM_FILE[s]}.csv',
                               b.ids))['technical_points'] for s in FIXED_SUBSETS}
    rep['seconds'] = time.perf_counter() - t0
    (OUT / 'verification.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
    print(f"\nwrote {OUT / 'verification.json'} in {rep['seconds']:.1f}s")


if __name__ == '__main__':
    main()
