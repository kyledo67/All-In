"""Select the per-machine half-life averaging settings on TRAIN only.

Why this exists. The single frozen half-life chosen by scripts/win_head_train.py landed
on H=16 and lam_inter=2, both at the EDGE of the predeclared grid, and the validation
evidence showed the compromise directly: H=128 reached a 0.2177 stable ratio while
pushing drift to 1.0972, and H=16 reached a 0.5852 drift ratio while giving up stable
to 0.4676. Which regime a machine is in is visible in its own pre-cutoff history, so
the half-life can be chosen per machine without a condition label and without a change
detector, by scoring each candidate on the machine's most recent held-out transitions.

DISCLOSURE. This grid is an extension of the original predeclared one, made after
seeing that the original selection sat at a grid boundary. The extension and the
selection use the TRAIN sampled-outcome surrogate exclusively; no public-validation
target, condition label or reference forecast is read in this script. The extension is
recorded here rather than silently folded into the first grid.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import win_head as W
from slot_model.features import causal_index
from slot_forecasting.io import load_inputs, read_labels
from scripts.win_head_train import build, weights, pooled

# Extended candidate half-lives, including a uniform (None) schedule.
HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)
LAM_MAIN = (4.0,)
LAM_INTER = (0.5, 1.0, 2.0)
LAM_MAGNITUDE = (1.0,)
LAM_NEIGHBOUR = (1.0,)
LAM_INTERCEPT = 0.01
PAYOUTS = (0, 1, 2, 5, 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/win_head')
    parser.add_argument('--magnitude', action='store_true',
                        help='Extend the basis with the 8 large-payout indicators.')
    parser.add_argument('--neighbour-lags', type=int, default=0,
                        help='Append the mean neighbour win sign at lags 1..N (0 disables).')
    parser.add_argument('--name', default='population_averaged.json')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    tq, th = load_inputs(args.data / 'train')
    index = causal_index(tq, th)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['train']
    entries = build(tq, index, magnitude=args.magnitude, neighbour_lags=args.neighbour_lags)
    wins = {k: 1.0 if PAYOUTS.index(labels[k]) > 0 else 0.0 for k in entries}
    episode_of = {k: entries[k]['episode'] for k in entries}
    by_fold = {f: [k for k in entries if folds[episode_of[k]] == f] for f in range(4)}

    def episode_brier(pred):
        per = {}
        for k, v in pred.items():
            per.setdefault(episode_of[k], []).append((v - wins[k]) ** 2)
        return float(np.mean([np.mean(v) for v in per.values()]))

    cache = {}

    def population(fold, half_life, lam_key, lam):
        key = (fold, half_life, lam_key)
        if key not in cache:
            fit_keys = [k for f in range(4) if f != fold for k in by_fold[f]]
            cache[key] = pooled(entries, fit_keys, half_life, lam)[0]
        return cache[key]

    records = []
    for lam_main in LAM_MAIN:
        for lam_inter in LAM_INTER:
            lam_key = (lam_main, lam_inter)
            lam = W.penalty(LAM_INTERCEPT, lam_main, lam_inter,
                            LAM_MAGNITUDE[0] if args.magnitude else None,
                            LAM_NEIGHBOUR[0] if args.neighbour_lags else None,
                            neighbour_lags=args.neighbour_lags)
            began = time.perf_counter()
            oof, picks, entropies = {}, [], []
            for fold in range(4):
                betas = {h: population(fold, h, lam_key, lam) for h in HALF_LIVES}
                for k in by_fold[fold]:
                    e = entries[k]
                    if e['star'] is None or len(e['y']) == 0:
                        continue
                    by_h = {h: betas[h].get(e['type'], np.zeros(len(lam))) for h in HALF_LIVES}
                    q, detail = W.predict_q_averaged(by_h, e['phi'], e['y'], e['used'], e['cutoff'],
                                                     e['star'], lam, adapt='full')
                    oof[k] = q
                    picks.append(detail['selected_half_life'])
                    entropies.append(detail['mix_entropy'])
            metric = episode_brier(oof)
            records.append({'mode': 'per_machine_half_life_average', 'lam_main': lam_main,
                            'lam_inter': lam_inter, 'lam_intercept': LAM_INTERCEPT,
                            'lam_magnitude': LAM_MAGNITUDE[0] if args.magnitude else None,
                            'magnitude': bool(args.magnitude), 'neighbour_lags': args.neighbour_lags,
                            'half_lives': [h if h else 0.0 for h in HALF_LIVES],
                            'train_oof_episode_win_brier': metric, 'scored_machines': len(oof),
                            'selected_half_life_histogram': {str(int(h)): int(np.sum(np.asarray(picks) == h))
                                                             for h in sorted(set(picks))},
                            'mix_entropy_median': float(np.median(entropies)),
                            'seconds': time.perf_counter() - began})
            print(f'  averaged  lam_main={lam_main:<5} lam_inter={lam_inter:<5} '
                  f'train OOF episode win Brier = {metric:.6f}  '
                  f'({time.perf_counter() - began:.0f}s)', flush=True)
            print(f'      per-machine half-life picks: {records[-1]["selected_half_life_histogram"]}', flush=True)

    records.sort(key=lambda r: (r['train_oof_episode_win_brier'], r['lam_inter']))
    best = records[0]
    previous = json.loads((args.out / 'population.json').read_text())
    single = previous['selected']
    print(f"\nsingle frozen H baseline : {single['train_oof_episode_win_brier']:.6f} "
          f"(H={single['half_life']}, lam_inter={single['lam_inter']})")
    print(f"per-machine averaged     : {best['train_oof_episode_win_brier']:.6f} "
          f"(lam_inter={best['lam_inter']})")
    improved = best['train_oof_episode_win_brier'] < single['train_oof_episode_win_brier']
    print(f"train surrogate {'prefers averaging' if improved else 'prefers the single half-life'}")

    lam = W.penalty(LAM_INTERCEPT, best['lam_main'], best['lam_inter'],
                    LAM_MAGNITUDE[0] if args.magnitude else None,
                    LAM_NEIGHBOUR[0] if args.neighbour_lags else None,
                    neighbour_lags=args.neighbour_lags)
    beta_by_h = {}
    for half_life in HALF_LIVES:
        beta, _ = pooled(entries, list(entries), half_life, lam)
        beta_by_h[str(half_life if half_life else 0.0)] = {t: v.tolist() for t, v in beta.items()}
    payload = {'selected': best, 'single_half_life_baseline': single,
               'averaging_preferred_on_train_surrogate': bool(improved),
               'grid_extension_disclosure':
                   'Half-life grid extended to include 8 and a uniform schedule, and lam_inter '
                   'extended down to 0.5 and 1, after the original predeclared selection landed on '
                   'the grid boundary (H=16, lam_inter=2). Extension and selection use the TRAIN '
                   'sampled-outcome surrogate only.',
               'all_records': records, 'half_lives': [h if h else 0.0 for h in HALF_LIVES],
               'beta_by_half_life': beta_by_h, 'basis': W.basis_names(args.magnitude, args.neighbour_lags),
               'dim': len(lam), 'magnitude': bool(args.magnitude),
               'fit': 'MAP (penalized maximum likelihood) with a Laplace predictive integral over '
                      'the one-dimensional logit variance; not posterior inference',
               'folds_sha256': hashlib.sha256((ROOT / 'artifacts/phase2/folds.json').read_bytes()).hexdigest(),
               'validation_untouched': True, 'seconds': time.perf_counter() - start}
    (args.out / args.name).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f"wrote {args.out / args.name} in {payload['seconds']:.1f}s")


if __name__ == '__main__':
    main()
