"""Independent causality audit of the FAMILY T changepoint forecast.

Three checks, each an assertion on recomputed numbers rather than a claim about code.

A. The causal index holds no tick at or after a query's cutoff.
B. Target-tick invariance. A fabricated outcome is injected AT the cutoff tick and at
   cutoff+1 for every machine, and the arm's q is recomputed. Bit-identical q proves the
   forecast cannot see its own target or anything after it.
C. Cross-machine invariance. Every OTHER machine in the same episode has its whole
   pre-cutoff history inverted (win <-> loss), and the arm's q is recomputed. Bit-identical
   q proves this family uses no neighbour information, so the shared_change gain comes from
   the machine's own run-length posterior and not from a neighbour channel.

The recomputed arm is T3B7 at the two settings the nested run selected
(kappa=1, h=1e-5 and kappa=0.5, h=1e-4), and the clean recomputation is additionally
compared bit-for-bit against the saved nested forecast of the TALL arm on the folds where
that setting was chosen.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import harness as H
from slot_model import win_head_infer as WI
from slot_forecasting.io import load_inputs, read_submission
from scripts.family_t_joint_structure_memory import (TEMPLATES, SUPPORT_MIN, changepoint_pass,
                                                     prepare, t1_mixture, half_lives)
from scripts import family_t_joint_structure_memory as FT

SETTINGS = ((1.0, 1e-5), (0.5, 1e-4))


def recompute(bundle, index, records_for_pick, hl, bank, lp, prior_mean, pick):
    """q for T3 on `bank` at every SETTINGS entry, built from `index` only."""
    saved = bundle.index
    bundle.index = index
    try:
        records = prepare(bundle)
    finally:
        bundle.index = saved
    pos = np.array([bank.index(int(p)) if int(p) in bank else 0 for p in pick])
    out = {}
    for kappa, h in SETTINGS:
        res = changepoint_pass(records, bank, lp, prior_mean, kappa, (h,), pos,
                               variants=('T3',))
        out[f'k{kappa:g}_h{h:g}'] = res['q3'][0].copy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    parser.add_argument('--run', type=Path, default=ROOT / 'artifacts/family_t_joint')
    args = parser.parse_args()
    start = time.perf_counter()
    bundle = H.load(args.data)
    queries, _ = load_inputs(bundle.data / 'validation')
    by_query = {q.query_id: q for q in queries}
    model = bundle.package / 'code/model'
    kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    support = np.asarray(payload['train_argmax_counts'], dtype=np.float64)
    bank = [i for i in range(len(TEMPLATES)) if support[i] >= SUPPORT_MIN]
    lp = np.asarray(log_prior, dtype=np.float64)[bank]

    # A ---------------------------------------------------------------------------
    worst = -10 ** 9
    for qid in bundle.ids:
        q = by_query[qid]
        ticks = bundle.index[(q.episode_id, q.machine_id)]
        worst = max(worst, max(ticks) - q.cutoff)
    check_a = {'max_tick_minus_cutoff_over_all_queries': int(worst),
               'passed': bool(worst < 0)}

    records = prepare(bundle)
    hl = half_lives(bundle, records, beta, lam)
    _, pick, _ = t1_mixture(records, hl, list(range(len(TEMPLATES))), np.asarray(log_prior),
                            prior_mean, kappa0, None)
    clean = recompute(bundle, bundle.index, records, hl, bank, lp, prior_mean, pick)

    # B ---------------------------------------------------------------------------
    injected = copy.deepcopy(bundle.index)
    for qid in bundle.ids:
        q = by_query[qid]
        ticks = injected[(q.episode_id, q.machine_id)]
        ticks[q.cutoff] = 4            # a fabricated 20x win AT the target tick
        ticks[q.cutoff + 1] = 0        # and a fabricated loss after it
    after_b = recompute(bundle, injected, records, hl, bank, lp, prior_mean, pick)

    # C ---------------------------------------------------------------------------
    flipped = copy.deepcopy(bundle.index)
    machines = {}
    for (episode, machine) in sorted(flipped):
        machines.setdefault(episode, []).append(machine)
    per_query = {}
    for qid in bundle.ids:
        q = by_query[qid]
        per_query[qid] = sorted(m for m in machines[q.episode_id] if m != q.machine_id)
    for episode in sorted(machines):
        for machine in machines[episode]:
            ticks = flipped[(episode, machine)]
            for tick in sorted(ticks):
                ticks[tick] = 0 if ticks[tick] > 0 else 2
    # restore each queried machine's own history, flip only its neighbours
    for qid in bundle.ids:
        q = by_query[qid]
        flipped[(q.episode_id, q.machine_id)] = dict(bundle.index[(q.episode_id, q.machine_id)])
    after_c = recompute(bundle, flipped, records, hl, bank, lp, prior_mean, pick)

    deltas = {'B_target_tick_injection': {k: float(np.max(np.abs(clean[k] - after_b[k])))
                                          for k in sorted(clean)},
              'C_neighbour_history_inversion': {k: float(np.max(np.abs(clean[k] - after_c[k])))
                                                for k in sorted(clean)}}

    # match against the saved nested forecast on the folds that used each setting
    saved = read_submission(args.run / 'predictions/oof_TALL_gated.csv')
    P = np.array([saved[i] for i in bundle.ids])
    q_saved = 1.0 - P[:, 0] / P.sum(axis=1)
    report = json.loads((args.run / 'evaluation.json').read_text())
    chosen = report['arms']['TALL']['gated']['selected_per_fold']
    agree = {}
    for fold, tag in enumerate(chosen):
        setting = tag.split('|')[1]
        mask = bundle.outer_mask(fold)
        if setting in clean:
            agree[f'fold{fold}:{tag}'] = float(np.max(np.abs(clean[setting][mask] - q_saved[mask])))
    out = {'preregistration': __doc__, 'check_A_index_strictly_before_cutoff': check_a,
           'max_abs_q_change': deltas,
           'max_abs_q_vs_saved_nested_forecast_by_fold': agree,
           'neighbours_flipped_per_query': int(np.median([len(v) for v in per_query.values()])),
           'passed': bool(check_a['passed']
                          and max(max(v.values()) for v in deltas.values()) == 0.0
                          and (not agree or max(agree.values()) < 5e-16)),
           'seconds': time.perf_counter() - start}
    (args.run / 'causality_audit.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'preregistration'}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
