"""Does restricting r to the observed type-specific family improve the official score?

ESTABLISHED STRUCTURE, verified independently from organizer data (validation/targets.csv).
Within each machine type the TRUE conditional positive-payout vector r takes exactly two
distinct values (2 of 269 steady, 2 of 234 balanced, 2 of 265 jackpot; one nonzero
covariance eigenvalue; rank-1 affine residual ~1e-15). Those two endpoints share an
IDENTICAL tail split among {2x, 5x, 20x} to within 1e-9, and differ only in P(1x|positive):

    steady    tail (0.800000, 0.196000, 0.004000)   P(1|pos) in {0.455122, 0.753989}
    balanced  tail (0.695652, 0.260870, 0.043478)   P(1|pos) in {0.214066, 0.499853}
    jackpot   tail (0.368421, 0.473684, 0.157895)   P(1|pos) in {0.120787, 0.335146}

So the family is one-parameter per type: r = [u, (1-u) * tail_type], u in two values. The
endpoint is a MACHINE-level property, not an episode-level one: 59/90, 52/92 and 60/92
episodes contain both endpoints, and the split is close to 50/50 and near-uniform across
conditions.

CANDIDATES (all hold q byte-identical to the protected forecast; r only changes):
  protected            the deployed forecast
  tail_train           keep the forecast's own P(1|pos); replace the tail split with a
                       ratio estimated from TRAIN sampled outcomes only. Fully deployable.
  tail_train_smoothed  the same with Dirichlet smoothing, because steady has zero observed
                       20x outcomes in 55 training positives and an exact zero is unsafe.
  tail_crossfit        tail ratio estimated from validation TARGETS on each outer fold's
                       FITTING episodes, applied to its held fold. Labelled: this uses
                       held-out-style probability supervision, cross-fitted.
  snap_nearest         PRIVILEGED. Snap r to the nearer true endpoint. Measures what the
                       family buys if the endpoint were identifiable; already measured at
                       52.340264, BELOW the protected forecast, because committing to the
                       wrong endpoint costs more than hedging between them.
  true_r               PRIVILEGED upper reference, 55.713488.

These are conditional deployment replays: q and the blend form are frozen and identical
across candidates. Exposure: 96 repeatedly inspected validation episodes; exploratory.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets, write_submission

TAILS = (2, 5, 20)


def tail_from_labels(queries, labels, smoothing):
    """Per-type tail split among {2x,5x,20x} from sampled outcomes. Train-only input."""
    types = {q.query_id: q.machine_type for q in queries}
    counts = {}
    for key, value in labels.items():
        if value in TAILS:
            counts.setdefault(types[key], np.zeros(3))[TAILS.index(value)] += 1
    out = {}
    for t, c in counts.items():
        s = c + smoothing
        out[t] = (s / s.sum()).tolist()
    return out, {t: int(c.sum()) for t, c in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/endpoint_family')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    vq, _ = load_inputs(args.data / 'validation')
    vtype = {q.query_id: q.machine_type for q in vq}
    types = np.array([vtype[i] for i in ids])

    dep = read_submission(args.package / 'validation_submission.csv')
    dep = np.array([dep[q] for q in ids])
    q_fixed = 1.0 - dep[:, 0]
    r_cur = dep[:, 1:] / np.maximum(q_fixed, 1e-12)[:, None]
    r_cur /= r_cur.sum(axis=1, keepdims=True)
    r_true = truth[:, 1:] / np.maximum(truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)

    # true endpoints, used only for the privileged snap and for reporting the structure
    ends = {}
    for t in sorted(set(types.tolist())):
        m = types == t
        ends[t] = [np.array(u) for u in sorted({tuple(np.round(x, 9)) for x in r_true[m]})]

    tq, _ = load_inputs(args.data / 'train')
    labels = read_labels(args.data / 'train/labels.csv', tq)
    tail_raw, n_per_type = tail_from_labels(tq, labels, 0.0)
    tail_sm, _ = tail_from_labels(tq, labels, 1.0)

    def project(tail_by_type, mask=None):
        """Keep each forecast's own P(1|pos); replace its tail split."""
        out = r_cur.copy()
        sel = np.ones(len(ids), dtype=bool) if mask is None else mask
        for k in range(len(ids)):
            if not sel[k]:
                continue
            t = types[k]
            if t not in tail_by_type:
                continue
            u = r_cur[k, 0]
            out[k, 0] = u
            out[k, 1:] = (1.0 - u) * np.asarray(tail_by_type[t])
        return out / out.sum(axis=1, keepdims=True)

    # cross-fitted tail: estimate from validation TARGETS on each fold's fitting episodes
    r_cross = r_cur.copy()
    cross_tails = []
    for fold in range(4):
        held = np.array([folds[x] == fold for x in ep])
        fit = ~held
        tails = {}
        for t in sorted(set(types.tolist())):
            m = fit & (types == t)
            tl = r_true[m][:, 1:]
            tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1e-15)
            tails[t] = tl.mean(axis=0).tolist()
        cross_tails.append(tails)
        r_cross[held] = project(tails, held)[held]

    snap = np.empty_like(r_cur)
    for k in range(len(ids)):
        cands = ends[types[k]]
        snap[k] = cands[int(np.argmin([np.linalg.norm(r_cur[k] - c) for c in cands]))]

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(r):
        out = np.column_stack((1.0 - q_fixed, q_fixed[:, None] * r))
        return out / out.sum(axis=1, keepdims=True)

    def official(r):
        return score(mapping(compose(r)), scoring, episodes)

    variants = {
        'protected': r_cur,
        'tail_train': project(tail_raw),
        'tail_train_smoothed': project(tail_sm),
        'tail_crossfit': r_cross,
        'snap_nearest_PRIVILEGED': snap,
        'true_r_PRIVILEGED': r_true,
    }
    results = {k: official(v) for k, v in variants.items()}

    # How much of the current r error is OFF-FAMILY versus along the family direction?
    off, along = np.zeros(len(ids)), np.zeros(len(ids))
    for k in range(len(ids)):
        t = types[k]
        tail = np.asarray(ends[t][0])[1:]
        tail = tail / tail.sum()
        u_cur = r_cur[k, 0]
        on_family = np.concatenate(([u_cur], (1 - u_cur) * tail))
        off[k] = float(np.linalg.norm(r_cur[k] - on_family))
        along[k] = float(abs(u_cur - r_true[k, 0]))
    base = official(r_cur)
    best_deployable = max(['tail_train', 'tail_train_smoothed'],
                          key=lambda k: results[k]['technical_points'])
    paired = paired_bootstrap(mapping(compose(variants[best_deployable])),
                              mapping(compose(r_cur)), scoring, episodes)
    gain = results[best_deployable]['technical_points'] - base['technical_points']
    lo, hi = paired['delta_loss_ratio_ci95']
    advanced = bool(gain > 0 and hi < 0)

    report = {
        'preregistration': __doc__,
        'verified_structure': {
            'distinct_true_r_per_type': {t: len(v) for t, v in ends.items()},
            'endpoints': {t: [x.tolist() for x in v] for t, v in ends.items()},
            'tail_split_identical_across_endpoints_max_abs_diff': {
                t: float(np.max(abs(np.asarray(v[0])[1:] / np.asarray(v[0])[1:].sum()
                                    - np.asarray(v[1])[1:] / np.asarray(v[1])[1:].sum())))
                for t, v in ends.items()},
            'P1_given_positive_endpoints': {t: [float(x[0]) for x in v] for t, v in ends.items()}},
        'train_tail_estimates': {'raw': tail_raw, 'smoothed_add1': tail_sm,
                                 'training_positive_samples_per_type': n_per_type},
        'crossfit_tails_per_fold': cross_tails,
        'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                        'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                    for k, v in results.items()},
        'error_geometry': {
            'mean_off_family_L2_of_current_r': float(off.mean()),
            'p90_off_family_L2': float(np.percentile(off, 90)),
            'mean_along_family_abs_error_in_P1': float(along.mean()),
            'interpretation':
                'off-family error is what a projection can remove; along-family error '
                'requires identifying which endpoint applies, which the projection cannot do'},
        'best_deployable': best_deployable, 'gain_over_protected': gain,
        'paired_best_deployable_vs_protected': paired,
        'advancement_rule': 'held-fold points improve AND paired interval upper bound < 0',
        'advanced': advanced,
        'q_byte_identical': True,
        'exposure': 'Repeatedly inspected public validation; exploratory research round.',
        'seconds': time.perf_counter() - start}
    for k, v in variants.items():
        write_submission(args.out / 'predictions' / f'oof_{k}.csv', mapping(compose(v)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(base['by_condition'])
    print(f"{'candidate':<26}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('protected', 'tail_train', 'tail_train_smoothed', 'tail_crossfit',
              'snap_nearest_PRIVILEGED', 'true_r_PRIVILEGED'):
        v = results[k]
        print(f"{k:<26}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    eg = report['error_geometry']
    print(f"\ncurrent r: mean off-family L2 {eg['mean_off_family_L2_of_current_r']:.4f} "
          f"(p90 {eg['p90_off_family_L2']:.4f}); mean |P(1|pos) error| along the family "
          f"{eg['mean_along_family_abs_error_in_P1']:.4f}")
    print(f"best deployable: {best_deployable}  gain {gain:+.6f}  "
          f"paired 95% [{lo:+.5f}, {hi:+.5f}]")
    print(f"ADVANCEMENT: {'GRANTED' if advanced else 'WITHHELD'} under the preregistered rule")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
