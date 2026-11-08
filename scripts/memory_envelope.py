"""Capacity envelope for the fixed memory-expert family, then causal fixed-share if justified.

PREREGISTERED BATCH (frozen before any public-validation number below was read).

Hypothesis. The parity representation is useful, but committing to one memory length may
discard information or react poorly to change. A causal mixture over the existing
half-life experts may improve the bias-variance tradeoff.

Gate first, per the brief. With r held fixed, the privileged per-query optimum is
    q_star = clip((u + r . t_pos) / (1 + r . r), 0, 1)
projected onto [min_H q_H, max_H q_H]. Scoring those projected forecasts officially gives
an OPTIMISTIC upper bound for ANY convex combination of these particular expert
predictions, including the true per-query optimal weights. It uses public targets and is
a privileged diagnostic, not a deployable selector and not a performance estimate.

Advancement rule, fixed in advance: if the envelope's improvement over the deployed
candidate is below 0.25 points, the fixed expert family provably cannot repay a causal
weighting experiment and the batch stops there. If it is above 0.25, run the causal
fixed-share comparison.

Experts: the frozen half-life grid 8, 16, 32, 64, 128, 256, infinity. The Beta prior,
kappa, feature construction, structural candidate family and r are all held fixed. For
each expert the structural choice (pair and template) is selected from the SAME strict
prefix, so the comparison isolates memory and nothing else.
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
from scipy.special import betaln

from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.win_head_train import build

HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)
TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
ENVELOPE_GATE = 0.25


def weights_at(used, anchor, half_life):
    if not half_life:
        return np.ones(len(used))
    return 2.0 ** (-(anchor - 1 - np.asarray(used, dtype=np.float64)) / half_life)


def counts_for(phi, y, w, rows, subset, template):
    ci, nc = WI._template_cells(PD.signs_from(phi[rows]), subset, template)
    wins = np.bincount(ci, weights=w[rows] * y[rows], minlength=nc)
    loss = np.bincount(ci, weights=w[rows] * (1 - y[rows]), minlength=nc)
    return wins, loss, nc


def select_structure(phi, y, w, rows, a0, b0, log_prior):
    best = (-np.inf, 0)
    for ix, (subset, template) in enumerate(TEMPLATES):
        wins, loss, _ = counts_for(phi, y, w, rows, subset, template)
        ev = float(np.sum(betaln(a0 + wins, b0 + loss) - betaln(a0, b0))) + float(log_prior[ix])
        if np.isfinite(ev) and ev > best[0]:
            best = (ev, ix)
    return best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/memory_envelope')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    model = args.package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    episodes = set(ep.tolist())

    # Deployed forecast: q from the package, r exactly as deployed (frozen).
    deployed = read_submission(args.package / 'validation_submission.csv')
    deployed = np.array([deployed[q] for q in ids])
    q_dep = 1.0 - deployed[:, 0]
    r_dep = deployed[:, 1:] / np.maximum(q_dep, 1e-12)[:, None]
    r_dep /= r_dep.sum(axis=1, keepdims=True)

    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)

    # ---- per-expert forecasts, common prefix-selected structure --------
    experts = np.full((len(ids), len(HALF_LIVES)), np.nan)
    for i, qid in enumerate(ids):
        e = vent[qid]
        if e['star'] is None or len(e['y']) == 0:
            experts[i, :] = q_dep[i]
            continue
        mu = float(prior_mean.get(e['type'], 0.44))
        a0, b0 = kappa * mu, kappa * (1 - mu)
        phi, y, used, cutoff, star = e['phi'], e['y'], e['used'], e['cutoff'], e['star']
        rows = np.arange(len(y))
        # One structural choice, selected from the full pre-cutoff prefix with the
        # DEPLOYED memory, then held common across experts so only memory varies.
        _, mix = W.predict_q_averaged(
            {h: beta[h].get(e['type'], np.zeros(W.DIM)) for h in beta},
            phi, y, used, cutoff, star, lam, adapt='full')
        picked = mix['selected_half_life']
        w_dep = weights_at(used, cutoff, None if picked == 0.0 else picked)
        ix = select_structure(phi, y, w_dep, rows, a0, b0, log_prior)
        subset, template = TEMPLATES[ix]
        si, _ = WI._template_cells(star[1:1 + W.HORIZON][None, :], subset, template)
        c = int(si[0])
        for j, h in enumerate(HALF_LIVES):
            w = weights_at(used, cutoff, h)
            wins, loss, _ = counts_for(phi, y, w, rows, subset, template)
            experts[i, j] = float((a0 + wins[c]) / (a0 + b0 + wins[c] + loss[c]))

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(q):
        out = np.column_stack((1.0 - q, q[:, None] * r_dep))
        return out / out.sum(axis=1, keepdims=True)

    def official(a):
        return score(mapping(a), scoring, episodes)

    dep = official(compose(q_dep))

    # ---- privileged capacity envelope ---------------------------------
    u = 1.0 - truth[:, 0]
    t_pos = truth[:, 1:]
    q_star = np.clip((u + np.sum(r_dep * t_pos, axis=1)) / (1.0 + np.sum(r_dep ** 2, axis=1)), 0, 1)
    lo = np.nanmin(experts, axis=1)
    hi = np.nanmax(experts, axis=1)
    projected = np.clip(q_star, lo, hi)
    env = official(compose(projected))
    # also each single expert, and the unconstrained privileged optimum for context
    singles = {('inf' if h is None else str(int(h))): official(compose(experts[:, j]))['technical_points']
               for j, h in enumerate(HALF_LIVES)}
    unconstrained = official(compose(q_star))

    gain = env['technical_points'] - dep['technical_points']
    spread = float(np.mean(hi - lo))
    report = {
        'preregistration': __doc__,
        'deployed': {'points': dep['technical_points'], 'R': dep['mean_loss_ratio']},
        'single_experts': singles,
        'expert_spread_mean_max_minus_min': spread,
        'privileged_envelope': {'points': env['technical_points'], 'R': env['mean_loss_ratio'],
                                'gain_over_deployed': gain,
                                'note': 'PRIVILEGED. Optimistic upper bound for ANY convex '
                                        'combination of these expert predictions, computed with '
                                        'public true vectors. Not deployable, not a performance '
                                        'estimate.'},
        'unconstrained_privileged_optimum': unconstrained['technical_points'],
        'gate': {'threshold_points': ENVELOPE_GATE, 'passed': bool(gain >= ENVELOPE_GATE),
                 'decision': ('run the causal fixed-share comparison' if gain >= ENVELOPE_GATE
                              else 'STOP: the fixed expert family cannot repay a causal weighting '
                                   'experiment, so no amount of weighting within it will help')},
        'r_held_fixed': True,
        'exposure': 'Repeatedly inspected public validation; exploratory research round.',
        'seconds': time.perf_counter() - start,
    }
    write_submission(args.out / 'predictions' / 'privileged_envelope.csv', mapping(compose(projected)))
    np.save(args.out / 'experts.npy', experts)
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    print(f"deployed                         {dep['technical_points']:12.6f}  R={dep['mean_loss_ratio']:.6f}")
    print('single memory experts (r fixed, structure common):')
    for k, v in singles.items():
        print(f"   H={k:<5} {v:12.6f}")
    print(f"mean expert spread (max-min q)   {spread:12.6f}")
    print(f"PRIVILEGED envelope              {env['technical_points']:12.6f}  "
          f"R={env['mean_loss_ratio']:.6f}  gain {gain:+.6f}")
    print(f"unconstrained privileged optimum {unconstrained['technical_points']:12.6f}")
    print(f"\nGATE at {ENVELOPE_GATE} points: {'PASSED' if gain >= ENVELOPE_GATE else 'NOT passed'}")
    print(f"-> {report['gate']['decision']}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
