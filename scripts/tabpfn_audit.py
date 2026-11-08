"""Leakage audit for the TabPFN feature path.

Three independent checks, each on a deterministic sample of queries:

1. Post-cutoff invariance. Adversarial plays are appended at ticks cutoff..cutoff+8 for
   the target machine *and* its episode neighbours, then the whole feature table is
   rebuilt. Every value must be bitwise identical. This is the strong form of the rule:
   not "we filtered future rows" but "future rows provably cannot move a feature".
   The organizer loader rejects future rows in released files, so the mutated stream is
   fed straight to `causal_index`, which is the actual feature entry point.

2. Lag arithmetic. Each `lag{k}_c{c}` and `lag{k}_missing` value is recomputed from an
   independently built tick->payout dictionary using `cutoff - k` arithmetic, never a
   row offset, and must agree. A missing tick must present as missing, never as 0x.

3. Window containment. Every tick in every example's history must be strictly below
   that example's cutoff.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model.features import PAYOUTS, causal_index
from slot_model.phase2_features import examples, fit_prior, matrix
from slot_model.phase2_predict import hmm_summary
from slot_model.phase2_tables import PHASE2, load_population
from slot_forecasting.io import load_inputs
from slot_forecasting.schema import Play

HORIZON = 8


def sample_queries(queries, count, seed='tabpfn-audit'):
    ordered = sorted(queries, key=lambda q: hashlib.sha256(f'{seed}|{q.query_id}'.encode()).hexdigest())
    return ordered[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--sample', type=int, default=64)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/tabpfn/leakage_audit.json')
    args = parser.parse_args()

    folds = json.loads((PHASE2 / 'folds.json').read_text())
    train_queries, train_history = load_inputs(args.data / 'train')
    train_index = causal_index(train_queries, train_history)
    prior = fit_prior(train_queries, train_index, set(folds['train']))
    population, _ = load_population(set(folds['train']))

    queries, history = load_inputs(args.data / args.split)
    index = causal_index(queries, history)
    es = examples(queries, index)
    priors = {q.episode_id: prior for q in queries}
    baseline, names = matrix(es, index, priors)
    _, baseline_hmm = hmm_summary(population, es)

    chosen = sample_queries(queries, args.sample)
    chosen_ids = {q.query_id for q in chosen}
    cutoffs = {(q.episode_id, q.machine_id): q.cutoff for q in queries}
    episodes = {q.episode_id for q in chosen}

    # ---- 1. post-cutoff invariance -----------------------------------------
    # Append adversarial future plays for every machine in a sampled episode, so the
    # target machine's own future AND its neighbours' futures are both present.
    extra = []
    for (episode, machine), cutoff in sorted(cutoffs.items()):
        if episode not in episodes:
            continue
        observed = index[(episode, machine)]
        for offset in range(HORIZON):
            tick = cutoff + offset
            if tick in observed:
                continue
            # Pick a payout that differs from the most recent observed one.
            recent = observed.get(cutoff - 1, 0)
            extra.append(Play(episode, machine, tick, PAYOUTS[(recent + 3) % len(PAYOUTS)]))
    if not extra:
        raise RuntimeError('Audit built no adversarial future plays; cannot demonstrate invariance')
    mutated_index = causal_index(queries, tuple(history) + tuple(extra))
    mutated_es = examples(queries, mutated_index)
    mutated, mutated_names = matrix(mutated_es, mutated_index, priors)
    _, mutated_hmm = hmm_summary(population, mutated_es)
    if mutated_names != names:
        raise AssertionError('Feature names changed under mutation')
    observable_identical = bool(np.array_equal(baseline, mutated))
    hmm_identical = bool(np.array_equal(baseline_hmm, mutated_hmm))
    observable_max_delta = float(np.max(abs(baseline - mutated))) if baseline.size else 0.0
    hmm_max_delta = float(np.max(abs(baseline_hmm - mutated_hmm))) if baseline_hmm.size else 0.0

    # ---- 2. lag arithmetic on the sampled rows ------------------------------
    column = {name: i for i, name in enumerate(names)}
    lag_rows = 0
    lag_failures = []
    for row, example in enumerate(es):
        q = example['q']
        if q.query_id not in chosen_ids:
            continue
        lag_rows += 1
        ticks = {t: y for t, y in index[(q.episode_id, q.machine_id)].items() if t < q.cutoff}
        for lag in range(1, 9):
            observed = ticks.get(q.cutoff - lag)
            if baseline[row, column[f'lag{lag}_missing']] != float(observed is None):
                lag_failures.append(f'{q.query_id} lag{lag}_missing')
            for c in range(5):
                want = float(observed == c) if observed is not None else 0.0
                if baseline[row, column[f'lag{lag}_c{c}']] != want:
                    lag_failures.append(f'{q.query_id} lag{lag}_c{c}')

    # ---- 3. window containment ----------------------------------------------
    containment_failures = [e['q'].query_id for e in es if e['h'] and max(e['h']) >= e['q'].cutoff]

    report = {
        'split': args.split, 'queries': len(queries), 'examples': len(es),
        'sampled_queries': sorted(chosen_ids), 'sample_size': len(chosen_ids),
        'post_cutoff_invariance': {
            'adversarial_plays_added': len(extra),
            'episodes_mutated': sorted(episodes),
            'machines_mutated': len({(p.episode_id, p.machine_id) for p in extra}),
            'horizon_ticks': HORIZON,
            'observable_bitwise_identical': observable_identical,
            'hmm_summary_bitwise_identical': hmm_identical,
            'observable_max_abs_delta': observable_max_delta,
            'hmm_summary_max_abs_delta': hmm_max_delta,
        },
        'lag_arithmetic': {'rows_checked': lag_rows, 'checks_per_row': 8 * 6,
                           'failures': lag_failures[:20], 'failure_count': len(lag_failures)},
        'window_containment': {'failures': containment_failures[:20],
                               'failure_count': len(containment_failures)},
    }
    passed = (observable_identical and hmm_identical and not lag_failures and not containment_failures)
    report['passed'] = passed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    print(f"post-cutoff invariance: {len(extra)} adversarial plays across "
          f"{report['post_cutoff_invariance']['machines_mutated']} machines in {len(episodes)} episodes")
    print(f"  observable table bitwise identical: {observable_identical} (max delta {observable_max_delta:.1e})")
    print(f"  HMM summary bitwise identical:      {hmm_identical} (max delta {hmm_max_delta:.1e})")
    print(f"lag arithmetic: {lag_rows} rows x 48 checks, {len(lag_failures)} failures")
    print(f"window containment: {len(containment_failures)} failures over {len(es)} examples")
    print('AUDIT', 'PASSED' if passed else 'FAILED')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
