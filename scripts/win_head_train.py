"""Fit the hierarchical logistic win head's population effects and shrinkage on TRAIN only.

Protocol. Population effects beta_type and the shrinkage/recency hyperparameters are
learned from public/train observations exclusively. No public-validation target,
condition label or reference forecast is read anywhere in this script.

Selection metric is the episode-equal win-indicator Brier on TRAIN out-of-fold
predictions. This is a SAMPLED-OUTCOME SURROGATE, used only because true probability
vectors do not exist for train. The official true-probability score is never used for
tuning; it appears only in scripts/win_head_evaluate.py on validation.

Out-of-fold means: for a machine in train fold f, the pooled beta it is adapted
against is fitted on train episodes outside fold f, using artifacts/phase2/folds.json.
Whole episodes stay together.

The grid is predeclared here and is deliberately small. One recency half-life is
chosen and then frozen across every matched control in the evaluation.
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

# Predeclared tuning grid. Frozen before any validation score is computed.
HALF_LIVES = (16.0, 32.0, 64.0, 128.0, 256.0)
LAM_MAIN = (0.5, 1.0, 2.0, 4.0)
LAM_INTER = (2.0, 4.0, 8.0, 16.0, 32.0)
LAM_INTERCEPT = 0.01
PAYOUTS = (0, 1, 2, 5, 20)


def build(queries, index, magnitude=False, neighbour_lags=0):
    """Per-machine complete-lag design, recency weights deferred so H is cheap to vary."""
    out = {}
    by_episode = {}
    for (episode, machine), history in index.items():
        by_episode.setdefault(episode, {})[machine] = history
    for q in queries:
        ticks = index[(q.episode_id, q.machine_id)]
        # Episodes share one cutoff, so every neighbour tick below it is a legal input.
        nb = (W.neighbour_signs(by_episode[q.episode_id], q.machine_id)
              if neighbour_lags else None)
        phi, y, _, used = W.design(ticks, q.cutoff, half_life=None, magnitude=magnitude,
                                   neighbours=nb, neighbour_lags=neighbour_lags)
        star = W.query_row(ticks, q.cutoff, magnitude=magnitude,
                           neighbours=nb, neighbour_lags=neighbour_lags)
        out[q.query_id] = {'phi': phi, 'y': y, 'used': np.asarray(used, dtype=np.float64),
                           'star': star, 'cutoff': q.cutoff, 'type': q.machine_type,
                           'episode': q.episode_id}
    return out


def weights(entry, half_life):
    if not half_life:
        return np.ones(len(entry['y']))
    return 2.0 ** (-(entry['cutoff'] - 1 - entry['used']) / half_life)


def pooled(entries, keys, half_life, lam):
    groups = {}
    for k in keys:
        e = entries[k]
        if len(e['y']) == 0:
            continue
        g = groups.setdefault(e['type'], [[], [], []])
        g[0].append(e['phi']); g[1].append(e['y']); g[2].append(weights(e, half_life))
    groups = {t: (np.vstack(v[0]), np.concatenate(v[1]), np.concatenate(v[2])) for t, v in groups.items()}
    return W.fit_population(groups, lam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/win_head')
    parser.add_argument('--scope', default='full', choices=['full', 'main'],
                        help='Adaptation scope used for hyperparameter selection.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    tq, th = load_inputs(args.data / 'train')
    index = causal_index(tq, th)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['train']
    entries = build(tq, index)
    wins = {k: 1.0 if PAYOUTS.index(labels[k]) > 0 else 0.0 for k in entries}
    episodes = {k: entries[k]['episode'] for k in entries}
    by_fold = {f: [k for k in entries if folds[episodes[k]] == f] for f in range(4)}
    usable = {k: int(len(entries[k]['y'])) for k in entries}
    print(f'{len(entries)} train machines; median usable transitions '
          f'{int(np.median(list(usable.values())))}; '
          f'{sum(1 for k in entries if entries[k]["star"] is None)} without a query-tick basis',
          flush=True)

    def episode_brier(pred):
        per = {}
        for k, v in pred.items():
            per.setdefault(episodes[k], []).append((v - wins[k]) ** 2)
        return float(np.mean([np.mean(v) for v in per.values()]))

    records = []
    for half_life in HALF_LIVES:
        for lam_main in LAM_MAIN:
            for lam_inter in LAM_INTER:
                lam = W.penalty(LAM_INTERCEPT, lam_main, lam_inter)
                oof, adapted = {}, 0
                for fold in range(4):
                    fit_keys = [k for f in range(4) if f != fold for k in by_fold[f]]
                    beta, _ = pooled(entries, fit_keys, half_life, lam)
                    for k in by_fold[fold]:
                        e = entries[k]
                        if e['star'] is None:
                            oof[k] = None
                            continue
                        b = beta.get(e['type'], np.zeros(W.DIM))
                        q, detail = W.predict_q(b, e['phi'], e['y'], weights(e, half_life),
                                                e['star'], lam, adapt=args.scope)
                        oof[k] = q
                        adapted += int(detail['adapted'])
                scored = {k: v for k, v in oof.items() if v is not None}
                metric = episode_brier(scored)
                records.append({'half_life': half_life, 'lam_main': lam_main, 'lam_inter': lam_inter,
                                'lam_intercept': LAM_INTERCEPT, 'scope': args.scope,
                                'train_oof_episode_win_brier': metric,
                                'scored_machines': len(scored), 'adapted': adapted})
                print(f'  H={half_life:<6} lam_main={lam_main:<5} lam_inter={lam_inter:<5} '
                      f'train OOF episode win Brier = {metric:.6f}', flush=True)

    records.sort(key=lambda r: (r['train_oof_episode_win_brier'], r['half_life'],
                                r['lam_inter'], r['lam_main']))
    best = records[0]
    print(f"\nselected (train surrogate only): H={best['half_life']} "
          f"lam_main={best['lam_main']} lam_inter={best['lam_inter']} "
          f"Brier={best['train_oof_episode_win_brier']:.6f}", flush=True)

    # Population fitted on ALL train episodes with the selected settings, for validation use.
    lam = W.penalty(LAM_INTERCEPT, best['lam_main'], best['lam_inter'])
    beta, report = pooled(entries, list(entries), best['half_life'], lam)
    payload = {'selected': best, 'grid': {'half_lives': list(HALF_LIVES), 'lam_main': list(LAM_MAIN),
                                          'lam_inter': list(LAM_INTER), 'lam_intercept': LAM_INTERCEPT},
               'grid_size': len(records), 'all_records': records,
               'beta': {t: v.tolist() for t, v in beta.items()},
               'basis': W.basis_names(), 'dim': W.DIM,
               'population_report': report,
               'fit': 'MAP (penalized maximum likelihood); not posterior inference',
               'selection_metric': 'episode-equal win-indicator Brier on TRAIN out-of-fold predictions; '
                                   'a sampled-outcome surrogate, not the official probability score',
               'validation_untouched': True,
               'usable_transitions': {'median': float(np.median(list(usable.values()))),
                                      'min': int(min(usable.values())), 'max': int(max(usable.values())),
                                      'p10': float(np.percentile(list(usable.values()), 10))},
               'folds_sha256': hashlib.sha256((ROOT / 'artifacts/phase2/folds.json').read_bytes()).hexdigest(),
               'seconds': time.perf_counter() - start}
    (args.out / 'population.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f"wrote {args.out / 'population.json'} in {payload['seconds']:.1f}s")


if __name__ == '__main__':
    main()
