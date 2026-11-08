"""Beta-specific causal memory selection for the shipped template dictionary.

PREREGISTRATION. The earlier memory-selector experiment that lost 2.162018 points used
the LOGISTIC pair model (scripts/compact_memory.py contains zero `betaln` references),
so its negative result does not transfer to the conjugate Beta path. The deployed
half-life still comes from the dense 37-coefficient averaged selector, which is a
different estimator from the one being deployed.

Hypothesis: selecting memory with the Beta template model's OWN causal held-out
predictive log loss fits the deployed estimator better than inheriting the dense
selector's choice.

Candidates: the frozen half-life set 8, 16, 32, 64, 128, 256, infinity, plus one single
global half-life chosen on training episodes. Primary metric: official points / R on
concatenated held-fold predictions. Selection: per machine, from that machine's own
pre-cutoff history only; template prior and kappa are the frozen trained values.
Matched control: the shipped dense-selected candidate at 53.191542 with identical
priors, eligible transitions and r. Advancement: held-fold points must improve AND the
paired whole-episode interval must lie below zero.

Procedure per machine and candidate half-life H:
  1. split complete-lag transitions by ACTUAL tick into a prefix and the most recent
     held-out block; the block is identical across H, so candidates are compared on the
     same eligible target ticks
  2. anchor decay at the historical origin (the first held-out tick), not the cutoff
  3. select the template on the prefix ALONE, so template identity never sees the block
     it is scored on
  4. predict each held-out tick and accumulate UNWEIGHTED log loss, keeping the
     comparison on a common scale across half-lives with different total weight
  5. choose H, refit on the full pre-cutoff history anchored at the cutoff, and predict
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


def decay(ticks, anchor, half_life):
    if not half_life:
        return np.ones(len(ticks))
    return 2.0 ** (-(anchor - 1 - np.asarray(ticks, dtype=np.float64)) / half_life)


def select_on(phi, y, w, rows, a0, b0, log_prior):
    """Highest-evidence template using only `rows`. Returns (index, subset, template)."""
    signs = PD.signs_from(phi[rows])
    best = (-np.inf, 0)
    for index, (subset, template) in enumerate(TEMPLATES):
        ci, nc = WI._template_cells(signs, subset, template)
        wins = np.bincount(ci, weights=w[rows] * y[rows], minlength=nc)
        loss = np.bincount(ci, weights=w[rows] * (1 - y[rows]), minlength=nc)
        ev = float(np.sum(betaln(a0 + wins, b0 + loss) - betaln(a0, b0))) + float(log_prior[index])
        if np.isfinite(ev) and ev > best[0]:
            best = (ev, index)
    return best[1]


def predict_rows(phi, y, w, fit_rows, target_rows, index_t, a0, b0):
    """Predictive means for `target_rows` from counts accumulated over `fit_rows`."""
    subset, template = TEMPLATES[index_t]
    ci, nc = WI._template_cells(PD.signs_from(phi[fit_rows]), subset, template)
    wins = np.bincount(ci, weights=w[fit_rows] * y[fit_rows], minlength=nc)
    loss = np.bincount(ci, weights=w[fit_rows] * (1 - y[fit_rows]), minlength=nc)
    ti, _ = WI._template_cells(PD.signs_from(phi[target_rows]), subset, template)
    return (a0 + wins[ti]) / (a0 + b0 + wins[ti] + loss[ti])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/beta_memory')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    model = ROOT / 'artifacts/win_head_release_parity/code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    episodes = set(ep.tolist())
    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)

    shipped = read_submission(ROOT / 'artifacts/win_head_release_parity/validation_submission.csv')
    shipped = np.array([shipped[q] for q in ids])
    q_ship = 1.0 - shipped[:, 0]
    thp = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    thp = np.array([thp[q] for q in ids])
    r_two = thp[:, 1:] / np.maximum(1.0 - thp[:, 0], 1e-12)[:, None]
    r_two /= r_two.sum(axis=1, keepdims=True)

    q_beta = np.empty(len(ids))

    picks, short = [], 0
    for i, qid in enumerate(ids):
        e = vent[qid]
        if e['star'] is None or len(e['y']) == 0:
            q_beta[i] = q_ship[i]
            picks.append(None)
            continue
        mu = float(prior_mean.get(e['type'], 0.44))
        a0, b0 = kappa * mu, kappa * (1 - mu)
        phi, y, used, cutoff, star = e['phi'], e['y'], e['used'], e['cutoff'], e['star']
        n = len(y)
        order = np.argsort(used)
        allrows = np.arange(n)
        hold = min(40, max(8, n // 3))
        if n - hold < 12:
            q_beta[i] = q_ship[i]; picks.append(None); short += 1; continue
        fit_rows, val_rows = order[:-hold], order[-hold:]
        origin = float(used[val_rows].min())
        best, h_pick = -np.inf, None
        for h in HALF_LIVES:
            w = decay(used, origin, h)
            idx = select_on(phi, y, w, fit_rows, a0, b0, log_prior)
            p = predict_rows(phi, y, w, fit_rows, val_rows, idx, a0, b0)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            ll = float(np.sum(y[val_rows] * np.log(p) + (1 - y[val_rows]) * np.log(1 - p)))
            if ll > best:
                best, h_pick = ll, h
        w = decay(used, cutoff, h_pick)
        idx = select_on(phi, y, w, allrows, a0, b0, log_prior)
        subset, template = TEMPLATES[idx]
        ci, nc = WI._template_cells(PD.signs_from(phi), subset, template)
        si, _ = WI._template_cells(star[1:1 + W.HORIZON][None, :], subset, template)
        wins = np.bincount(ci, weights=w * y, minlength=nc)
        loss = np.bincount(ci, weights=w * (1 - y), minlength=nc)
        c = int(si[0])
        q_beta[i] = float((a0 + wins[c]) / (a0 + b0 + wins[c] + loss[c]))
        picks.append(h_pick if h_pick else 0.0)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(q):
        out = np.column_stack((1.0 - q, q[:, None] * r_two))
        return out / out.sum(axis=1, keepdims=True)

    def official(a):
        return score(mapping(a), scoring, episodes)

    ship = official(compose(q_ship))
    new = official(compose(q_beta))
    paired = paired_bootstrap(mapping(compose(q_beta)), mapping(compose(q_ship)), scoring, episodes)
    hist = {}
    for p in picks:
        if p is not None:
            hist[str(int(p))] = hist.get(str(int(p)), 0) + 1
    report = {'preregistration': __doc__,
              'estimator_identity_check':
                  'scripts/compact_memory.py contains 0 betaln references, so the earlier '
                  '-2.162018 memory result tested the logistic pair model, not Beta. This '
                  'experiment is the Beta-specific test that negative did not cover.',
              'shipped_dense_selected': {'points': ship['technical_points'], 'R': ship['mean_loss_ratio'],
                                         'by_condition': {c: v['loss_ratio'] for c, v in ship['by_condition'].items()}},
              'beta_selected': {'points': new['technical_points'], 'R': new['mean_loss_ratio'],
                                'by_condition': {c: v['loss_ratio'] for c, v in new['by_condition'].items()}},
              'gain': new['technical_points'] - ship['technical_points'],
              'half_life_histogram': hist, 'too_short_to_select': short,
              'q_max_abs_change': float(np.max(abs(q_beta - q_ship))),
              'paired_vs_shipped': paired,
              'exposure': 'Repeatedly inspected public validation; exploratory.',
              'seconds': time.perf_counter() - start}
    write_submission(args.out / 'predictions' / 'oof_beta_memory.csv', mapping(compose(q_beta)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(ship['by_condition'])
    print(f"{'policy':<34}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for nm, v in (('shipped: dense-selected H', ship), ('Beta-selected H', new)):
        print(f"{nm:<34}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\ngain {report['gain']:+.6f}; paired 95% [{lo:+.5f}, {hi:+.5f}] "
          f"{'favours Beta selection' if hi < 0 else 'overlaps zero' if lo < 0 else 'favours shipped'}")
    print(f"half-life picks {hist}; too short to select {short}; max |dq| {report['q_max_abs_change']:.4f}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
