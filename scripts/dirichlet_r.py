"""Conjugate Dirichlet conditional-positive distribution, the r-side analogue of the parity win.

PREREGISTRATION. The refreshed diagnosis on the 53.191542 parity winner shows the
bottleneck has moved: privileged substitution of the true conditional-positive
distribution r scores 55.7135 against the candidate's 53.1915, so r now carries about
2.52 points of headroom where it previously had almost none. The deployed r is the
two-head CatBoost conditional distribution, which is a pooled population model with no
machine-specific adaptation at all.

Hypothesis: a per-machine Dirichlet-multinomial over the four positive payout categories,
with a population prior learned on training episodes, estimates r better than the pooled
two-head head alone - the same conjugate, partially pooled mechanism that moved q from
48.4 to 53.2.

Model. For machine m, over its pre-cutoff POSITIVE outcomes with recency weights w_t,
accumulate decayed counts W_k for k in {1x, 2x, 5x, 20x}. With prior alpha_k = tau * m_k,
where m_k is the population conditional-positive distribution for that machine type:

    r_k = (tau * m_k + W_k) / (tau + sum_j W_j)

Candidates: tau in {1, 2, 4, 8, 16, 32, 64, 128}. Context: pooled over the machine
(declared) - the parity context is NOT reused here, because conditioning r on the same
cell would split already-thin positive counts four ways.
Primary metric: official points / R on concatenated held-fold predictions.
Selection: tau and the population prior m_k are learned on TRAIN episodes only, by
episode-equal conditional-positive Brier on observed training outcomes - a sampled-outcome
surrogate, labelled as such.
Matched controls: deployed two-head r; a single bounded blend between the two.
q is HELD FIXED at the parity benchmark throughout. Nothing about q is refitted.
Advancement: held-fold points must improve AND the paired interval must lie below zero.

Exposure: 96 repeatedly inspected validation episodes; exploratory, another adaptive round.
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

from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import PAYOUTS, causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from scripts.win_head_train import build

POS = (1, 2, 5, 20)          # payout values, indices 1..4 in PAYOUTS
TAUS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def positive_counts(ticks, cutoff, half_life):
    """Decay-weighted counts of the four positive payout categories, strictly pre-cutoff."""
    counts = np.zeros(4)
    for tick, value in ticks.items():
        if tick >= cutoff or value <= 0:
            continue
        w = 1.0 if not half_life else 2.0 ** (-(cutoff - 1 - tick) / half_life)
        counts[value - 1] += w
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_parity')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/dirichlet_r')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    model = args.package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')

    def half_life_of(e):
        _, d = W.predict_q_averaged({h: beta[h].get(e['type'], np.zeros(W.DIM)) for h in beta},
                                    e['phi'], e['y'], e['used'], e['cutoff'], e['star'],
                                    lam, adapt='full')
        p = d['selected_half_life']
        return None if p == 0.0 else p

    # ---- learn tau and the population conditional-positive prior on TRAIN ----
    tq, th = load_inputs(args.data / 'train')
    tindex = causal_index(tq, th)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    folds_t = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['train']
    tent = build(tq, tindex)
    epof = {k: tent[k]['episode'] for k in tent}
    byfold = {f: [k for k in tent if folds_t[epof[k]] == f] for f in range(4)}
    hl_t = {k: half_life_of(tent[k]) for k in tent
            if tent[k]['star'] is not None and len(tent[k]['y'])}
    # only queries whose observed outcome is positive carry an r target
    pos_t = {k: PAYOUTS.index(labels[k]) for k in tent if labels[k] > 0}

    records, best = [], None
    for tau in TAUS:
        oof = {}
        for fold in range(4):
            fit = [k for f in range(4) if f != fold for k in byfold[f]]
            m = np.full(4, 0.0)
            by_type = {}
            for k in fit:
                if labels[k] > 0:
                    by_type.setdefault(tent[k]['type'], np.zeros(4))[POS.index(labels[k])] += 1
            by_type = {t: v / v.sum() for t, v in by_type.items() if v.sum() > 0}
            for k in byfold[fold]:
                if k not in pos_t or k not in hl_t:
                    continue
                e = tent[k]
                ticks = tindex[(e['episode'], [q for q in tq if q.query_id == k][0].machine_id)] \
                    if False else None
                oof[k] = None
        # counts need the machine id; rebuild a direct map once
        break

    # direct map from query_id to (episode, machine, cutoff, type)
    meta = {q.query_id: (q.episode_id, q.machine_id, q.cutoff, q.machine_type) for q in tq}
    records, best = [], None
    for tau in TAUS:
        oof, target = {}, {}
        for fold in range(4):
            fit = [k for f in range(4) if f != fold for k in byfold[f]]
            by_type = {}
            for k in fit:
                if labels[k] > 0:
                    by_type.setdefault(meta[k][3], np.zeros(4))[POS.index(labels[k])] += 1
            by_type = {t: v / v.sum() for t, v in by_type.items() if v.sum() > 0}
            for k in byfold[fold]:
                if k not in pos_t or k not in hl_t:
                    continue
                ep_, mid, cut, typ = meta[k]
                counts = positive_counts(tindex[(ep_, mid)], cut, hl_t[k])
                prior = by_type.get(typ, np.full(4, 0.25))
                oof[k] = (tau * prior + counts) / (tau + counts.sum())
                target[k] = POS.index(labels[k])
        per = {}
        for k, v in oof.items():
            per.setdefault(epof[k], []).append(float(np.sum((v - np.eye(4)[target[k]]) ** 2)))
        brier = float(np.mean([np.mean(v) for v in per.values()]))
        records.append({'tau': tau, 'train_oof_conditional_positive_brier': brier,
                        'scored_queries': len(oof)})
        print(f'  tau={tau:<6} train OOF conditional-positive Brier = {brier:.6f}', flush=True)
        if best is None or brier < best['train_oof_conditional_positive_brier']:
            best = records[-1]
    tau = best['tau']
    full_prior = {}
    for k in pos_t:
        full_prior.setdefault(meta[k][3], np.zeros(4))[POS.index(labels[k])] += 1
    full_prior = {t: (v / v.sum()).tolist() for t, v in full_prior.items()}
    print(f"\nselected tau={tau} on the train surrogate; population prior {full_prior}", flush=True)

    # ---- validation ------------------------------------------------------
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds_v = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)
    vmeta = {q.query_id: (q.episode_id, q.machine_id, q.cutoff, q.machine_type) for q in vq}

    shipped = read_submission(args.package / 'validation_submission.csv')
    shipped = np.array([shipped[q] for q in ids])
    q_fixed = 1.0 - shipped[:, 0]
    r_two = shipped[:, 1:] / np.maximum(q_fixed, 1e-12)[:, None]
    r_two /= r_two.sum(axis=1, keepdims=True)

    r_dir = np.empty((len(ids), 4))
    support = np.empty(len(ids))
    for i, qid in enumerate(ids):
        e = vent[qid]
        ep_, mid, cut, typ = vmeta[qid]
        hl = half_life_of(e) if (e['star'] is not None and len(e['y'])) else None
        counts = positive_counts(vindex[(ep_, mid)], cut, hl)
        prior = np.asarray(full_prior.get(typ, [0.25] * 4))
        r_dir[i] = (tau * prior + counts) / (tau + counts.sum())
        support[i] = counts.sum()
    r_dir /= r_dir.sum(axis=1, keepdims=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(r):
        out = np.column_stack((1.0 - q_fixed, q_fixed[:, None] * r))
        return out / out.sum(axis=1, keepdims=True)

    def official(a):
        return score(mapping(a), scoring, episodes)

    def row_weights(es):
        w = np.zeros(len(ids))
        for c in sorted(set(groups.values())):
            ces = {x for i, x in enumerate(ep) if x in es and groups[ids[i]] == c}
            b = np.mean([np.mean(np.sum((ref_arr[ep == x] - truth[ep == x]) ** 2, axis=1)) for x in sorted(ces)])
            for x in ces:
                w[ep == x] = 1 / (4 * len(ces) * sum(ep == x) * max(b, 1e-9))
        return w

    # one bounded blend between the pooled two-head r and the conjugate r
    held = np.empty((len(ids), 4)); betas = []
    for fold in range(4):
        m = np.array([folds_v[x] == fold for x in ep])
        fit = {x for x in episodes if folds_v[x] != fold}
        w = row_weights(fit)
        grid = np.linspace(0, 1, 101)
        losses = []
        for g in grid:
            r = (1 - g) * r_two + g * r_dir
            r = r / r.sum(axis=1, keepdims=True)
            losses.append(float(np.sum(w * np.sum((compose(r) - truth) ** 2, axis=1))))
        g = float(grid[int(np.argmin(losses))])
        betas.append(g)
        r = (1 - g) * r_two + g * r_dir
        held[m] = (r / r.sum(axis=1, keepdims=True))[m]

    out = {'deployed_two_head_r': official(compose(r_two)),
           'conjugate_dirichlet_r': official(compose(r_dir)),
           'blend_fitted_per_fold': official(compose(held))}
    bestname = max(out, key=lambda k: out[k]['technical_points'])
    paired = paired_bootstrap(mapping(compose(held)), mapping(compose(r_two)), scoring, episodes)

    report = {'preregistration': __doc__, 'tau': tau, 'train_records': records,
              'population_prior_by_type': full_prior,
              'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                              'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                          for k, v in out.items()},
              'blend_weight_per_fold': betas,
              'positive_support': {'median': float(np.median(support)),
                                   'p10': float(np.percentile(support, 10))},
              'paired_blend_vs_deployed': paired,
              'q_held_fixed': True,
              'exposure': 'Repeatedly inspected public validation; exploratory.',
              'seconds': time.perf_counter() - start}
    for nm, r in (('dirichlet', r_dir), ('blend', held)):
        write_submission(args.out / 'predictions' / f'oof_r_{nm}.csv', mapping(compose(r)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(out['deployed_two_head_r']['by_condition'])
    print(f"\n{'r formulation':<30}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k, v in out.items():
        print(f"{k:<30}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\nblend weight per fold {[round(b,3) for b in betas]}")
    print(f"paired blend vs deployed [{lo:+.5f}, {hi:+.5f}] "
          f"{'favours conjugate r' if hi < 0 else 'overlaps zero' if lo < 0 else 'favours deployed'}")
    print(f"positive support: median {report['positive_support']['median']:.1f}, "
          f"p10 {report['positive_support']['p10']:.1f} weighted positive observations")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
