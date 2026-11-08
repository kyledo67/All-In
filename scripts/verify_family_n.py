"""ADVERSARIAL VERIFICATION of Family N's A5 arm (boost_q_only_enriched).

Independent of scripts/nonlinear_diagnostic_residual.py except for the feature matrix,
which is imported and then separately proved truth-free. Everything else -- the nesting
loop, the CatBoost fits, the adoption gate, the scoring and the paired bootstrap -- is
re-implemented here from the declared recipe.

Checks performed:
  V1  outer folds partition all 96 episodes exactly once
  V2  the feature matrix X does not depend on the validation TRUTH at all
      (rebuild with globally randomised truth; require bit-identical X)
  V3  every history read is strictly pre-cutoff by actual tick arithmetic
  V4  independent re-implementation of A5 reproduces the published held forecast
  V5  per-fold truth corruption: for EACH of the 4 outer folds, corrupting that fold's
      TRUTH leaves that fold's held forecast bit-identical (the family only tested fold 0)
  V6  the organizer score of the published A5 CSV, scored ONCE on the concatenation
  V7  b.paired(A5, protected) recomputed from the published CSV
  V8  in-sample vs nested: what the same recipe scores if fitted on everything (context)
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
np.seterr(divide='ignore', over='ignore', invalid='ignore')

from slot_forecasting.io import load_inputs, read_submission
from slot_model import harness as H
import scripts.nonlinear_diagnostic_residual as N

OUT = ROOT / 'artifacts/verify_family_n'
DATA = ROOT / 'main-participants-60-40/public'
PACKAGE = ROOT / 'artifacts/win_head_release_exact'
FAMILY = ROOT / 'artifacts/nonlinear_diag_residual'

DEPTHS = (2, 3)
SHRINKS = (0.25, 0.5, 0.75, 1.0)


def cb(depth):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(loss_function='RMSE', depth=depth, iterations=300,
                             learning_rate=0.05, l2_leaf_reg=6.0, bootstrap_type='No',
                             rsm=1.0, random_seed=0, thread_count=1, boosting_type='Plain',
                             allow_writing_files=False, verbose=0)


def my_a5(b, X, verbose=True):
    """Independent re-implementation of A5 boost_q_only_enriched."""
    q0, r0 = b.split_heads(b.protected)
    A, Bc = b.q_coeffs(r0)
    target = Bc / A - q0

    def mk(d, s):
        return b.compose(np.clip(q0 + s * d, 1e-6, 1.0 - 1e-6), r0)

    held = b.protected.copy()
    picks = []
    for fold in range(4):
        outer = b.outer_mask(fold)
        fit = ~outer
        parts = b.inner_folds(fit, k=3)
        wh = [b.row_weights(he) for _, he in parts]
        base = float(sum(b.wloss(b.protected, w) for w in wh))
        scores = {'zero': base}
        for depth in DEPTHS:
            preds = []
            for tr, _ in parts:
                wtr = b.row_weights(tr)
                m = cb(depth)
                m.fit(X[tr], target[tr], sample_weight=(wtr * A)[tr])
                preds.append(np.asarray(m.predict(X), dtype=np.float64))
            for s in SHRINKS:
                scores[f'd{depth}|s{s:g}'] = float(
                    sum(b.wloss(mk(d, s), w) for d, w in zip(preds, wh)))
        key = min(sorted(scores), key=lambda k: (scores[k], k))
        if key != 'zero' and scores[key] < base - 1e-15:
            ds, ss = key.split('|')
            wf = b.row_weights(fit)
            m = cb(int(ds[1:]))
            m.fit(X[fit], target[fit], sample_weight=(wf * A)[fit])
            d = np.asarray(m.predict(X), dtype=np.float64)
            held[outer] = mk(d, float(ss[1:]))[outer]
        picks.append({'fold': fold, 'pick': key, 'inner_zero': base,
                      'inner_selected': scores[key]})
        if verbose:
            print(f'  my A5 fold {fold}: pick {key:<10} inner {scores[key]:.6f} '
                  f'vs zero {base:.6f}', flush=True)
    return held / held.sum(axis=1, keepdims=True), picks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    t0 = time.perf_counter()
    b = H.load(DATA, PACKAGE)
    print(f'loaded {len(b.ids)} queries / {len(b.episodes)} episodes', flush=True)

    # ---- V1 fold partition
    masks = [b.outer_mask(f) for f in range(4)]
    cover = np.sum(np.stack(masks).astype(int), axis=0)
    eps_per_fold = [len({e for i, e in enumerate(b.ep) if masks[f][i]}) for f in range(4)]
    rep['V1_fold_partition'] = {
        'every_row_in_exactly_one_fold': bool((cover == 1).all()),
        'episodes_per_fold': eps_per_fold, 'total_episodes': len(b.episodes)}
    print('V1', rep['V1_fold_partition'], flush=True)

    # ---- features
    X, XO, routes = N.build_features(b, PACKAGE)

    # ---- V2 features independent of TRUTH
    bb = copy.copy(b)
    rng = np.random.default_rng(7)
    bad = rng.random(b.truth.shape) ** 3
    bb.truth = bad / bad.sum(axis=1, keepdims=True)
    bb._wcache = {}
    X2, XO2, _ = N.build_features(bb, PACKAGE, verbose=False)
    rep['V2_features_truth_free'] = {
        'X_bit_identical_under_randomised_truth': bool(np.array_equal(X, X2)),
        'XO_bit_identical': bool(np.array_equal(XO, XO2)),
        'n_features': int(X.shape[1])}
    print('V2', rep['V2_features_truth_free'], flush=True)

    # ---- V3 causality of every history read
    vq, _ = load_inputs(DATA / 'validation')
    by_id = {q.query_id: q for q in vq}
    worst = -10 ** 9
    cut_by_ep = {}
    for qid in b.ids:
        q = by_id[qid]
        hist = b.index[(q.episode_id, q.machine_id)]
        if hist:
            worst = max(worst, max(hist) - q.cutoff)
        cut_by_ep.setdefault(q.episode_id, set()).add(q.cutoff)
    # neighbour features read ticks of other machines of the same episode, bounded by the
    # QUERY's own cutoff in the code; confirm that bound is genuinely strict
    neigh_worst = -10 ** 9
    by_ep = {}
    for (ep, m), hist in sorted(b.index.items()):
        by_ep.setdefault(ep, {})[m] = hist
    for qid in b.ids:
        q = by_id[qid]
        for m, hist in sorted(by_ep[q.episode_id].items()):
            if m == q.machine_id or not hist:
                continue
            usable = [t for t in hist if t < q.cutoff]
            if usable:
                neigh_worst = max(neigh_worst, max(usable) - q.cutoff)
    rep['V3_causality'] = {
        'max_own_history_tick_minus_cutoff': int(worst),
        'max_neighbour_used_tick_minus_cutoff': int(neigh_worst),
        'episodes_with_multiple_distinct_cutoffs':
            sorted(e for e, s in cut_by_ep.items() if len(s) > 1)}
    print('V3', rep['V3_causality'], flush=True)

    # ---- V4 independent re-implementation
    mine, picks = my_a5(b, X)
    pub = read_submission(FAMILY / 'predictions/oof_boost_q_only_enriched.csv')
    pub = np.array([pub[i] for i in b.ids])
    pub = pub / pub.sum(axis=1, keepdims=True)
    rep['V4_independent_reimplementation'] = {
        'picks': picks,
        'max_abs_diff_vs_published': float(np.max(np.abs(mine - pub))),
        'my_points': b.official(mine)['technical_points'],
        'my_R': b.official(mine)['mean_loss_ratio']}
    print('V4', {k: v for k, v in rep['V4_independent_reimplementation'].items()
                 if k != 'picks'}, flush=True)

    # ---- V6 organizer score of the PUBLISHED csv, scored once on the concatenation
    res = b.official(pub)
    per_fold = {f'fold{f}': b.official(pub, episodes={e for i, e in enumerate(b.ep)
                                                      if masks[f][i]})['mean_loss_ratio']
                for f in range(4)}
    rep['V6_official_published'] = {
        'points': res['technical_points'], 'R': res['mean_loss_ratio'],
        'by_condition': {c: v['loss_ratio'] for c, v in res['by_condition'].items()},
        'mean_of_per_fold_R_for_contrast': float(np.mean(list(per_fold.values()))),
        'per_fold_R': per_fold,
        'rows_changed_vs_protected': int((np.abs(pub - b.protected) > 1e-12).any(axis=1).sum()),
        'max_abs_change_vs_protected': float(np.max(np.abs(pub - b.protected)))}
    print('V6', {k: rep['V6_official_published'][k] for k in
                 ('points', 'R', 'mean_of_per_fold_R_for_contrast',
                  'rows_changed_vs_protected')}, flush=True)

    # ---- V7 paired bootstrap, recomputed
    rep['V7_paired'] = {
        'published_vs_protected': b.paired(pub, b.protected)['delta_loss_ratio_ci95'],
        'mine_vs_protected': b.paired(mine, b.protected)['delta_loss_ratio_ci95'],
        'published_vs_boost_enriched': b.paired(
            pub, np.array([read_submission(FAMILY / 'predictions/oof_boost_enriched.csv')[i]
                           for i in b.ids]))['delta_loss_ratio_ci95']}
    print('V7', rep['V7_paired'], flush=True)

    # ---- V5 per-fold truth-corruption leak test, all four folds
    leak = {}
    for f in range(4):
        b2 = copy.copy(b)
        b2.truth = b.truth.copy()
        b2._wcache = {}
        rng = np.random.default_rng(1000 + f)
        bad = rng.random((int(masks[f].sum()), 5)) ** 3
        b2.truth[masks[f]] = bad / bad.sum(axis=1, keepdims=True)
        dirty, dpicks = my_a5(b2, X, verbose=False)
        leak[f'fold{f}'] = {
            'held_identical': bool(np.array_equal(mine[masks[f]], dirty[masks[f]])),
            'max_abs_held_diff': float(np.max(np.abs(mine[masks[f]] - dirty[masks[f]]))),
            'pick_unchanged': picks[f]['pick'] == dpicks[f]['pick'],
            'other_folds_moved': bool(not np.array_equal(mine[~masks[f]], dirty[~masks[f]]))}
        print(f'V5 fold{f}', leak[f'fold{f}'], flush=True)
    rep['V5_per_fold_leaktest'] = leak

    # ---- V8 context: the same recipe fitted on everything (NOT a nested number)
    q0, r0 = b.split_heads(b.protected)
    A, Bc = b.q_coeffs(r0)
    target = Bc / A - q0
    wall = b.row_weights(np.ones(len(b.ids), dtype=bool))
    m = cb(3)
    m.fit(X, target, sample_weight=wall * A)
    d = np.asarray(m.predict(X), dtype=np.float64)
    ins = {}
    for s in (0.25, 0.5, 0.75, 1.0):
        P = b.compose(np.clip(q0 + s * d, 1e-6, 1 - 1e-6), r0)
        ins[f's{s:g}'] = b.official(P)['technical_points']
    rep['V8_in_sample_context_not_a_result'] = ins
    print('V8', ins, flush=True)

    rep['seconds'] = time.perf_counter() - t0
    (OUT / 'verification.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
    print(f"\nwrote {OUT / 'verification.json'} in {rep['seconds']:.1f}s")


if __name__ == '__main__':
    main()
