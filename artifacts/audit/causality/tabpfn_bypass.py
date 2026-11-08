"""Defence-in-depth extension of scripts/tabpfn_audit.py for the TabPFN feature path.

The existing audit feeds adversarial post-cutoff plays through `causal_index`, which
itself drops `tick >= cutoff`. That proves the end-to-end property but cannot distinguish
"the feature code ignores the future" from "causal_index filtered the future first".

This test removes that single point of defence: the same post-cutoff ticks are injected
DIRECTLY into the index dicts that causal_index would have returned. Invariance can then
only come from the `t < q.cutoff` guards inside sample_history() / observable() and from
the tick arithmetic in hmm_arrays(). The TabPFN model itself is never run; only the
feature table and the 7 portable HMM summary columns are rebuilt, so no forward pass
is needed.

NOTE ON HOLDOUT STATUS: public validation, inspected many times. Not an independent holdout.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np

from slot_model.features import PAYOUTS, causal_index
from slot_model.phase2_features import examples, fit_prior, matrix
from slot_model.phase2_predict import hmm_summary
from slot_model.phase2_tables import PHASE2, load_population
from slot_forecasting.io import load_inputs

DATA = ROOT / 'main-participants-60-40/public'
HORIZON = 8
OUT = Path(__file__).resolve().parent / 'tabpfn_bypass.json'


def main():
    t0 = time.perf_counter()
    folds = json.loads((PHASE2 / 'folds.json').read_text())
    tq, th = load_inputs(DATA / 'train')
    tidx = causal_index(tq, th)
    prior = fit_prior(tq, tidx, set(folds['train']))
    population, pkey = load_population(set(folds['train']))

    vq, vh = load_inputs(DATA / 'validation')
    idx0 = causal_index(vq, vh)
    priors = {q.episode_id: prior for q in vq}

    def build(index):
        es = examples(vq, index)
        X, names = matrix(es, index, priors)
        _, H = hmm_summary(population, es)
        return X, names, H, es

    X0, names, H0, es0 = build(idx0)
    report = {'holdout_status': 'public validation, inspected many times; NOT an independent '
                                'holdout. Causality checks only.',
              'queries': len(vq), 'examples': len(es0),
              'observable_columns': len(names), 'hmm_summary_columns': int(H0.shape[1]),
              'population_key': pkey, 'horizon_ticks': HORIZON,
              'method': 'post-cutoff ticks injected DIRECTLY into the index dicts, bypassing '
                        "causal_index's own tick >= cutoff drop"}

    # ---- adversarial injection, bypassing causal_index ---------------------
    idx1 = {k: dict(v) for k, v in idx0.items()}
    cutoffs = {(q.episode_id, q.machine_id): q.cutoff for q in vq}
    injected = 0
    for (e, m), cut in sorted(cutoffs.items()):
        obs = idx1[(e, m)]
        for off in range(HORIZON):
            tick = cut + off
            if tick in obs:
                continue
            recent = idx0[(e, m)].get(cut - 1, 0)
            obs[tick] = PAYOUTS.index(PAYOUTS[(recent + 3) % len(PAYOUTS)])
            injected += 1
    X1, names1, H1, _ = build(idx1)
    assert names1 == names, 'feature names changed under mutation'
    obs_ok = bool(np.array_equal(X0, X1))
    hmm_ok = bool(np.array_equal(H0, H1))
    report['post_cutoff_invariance_bypassing_causal_index'] = {
        'ticks_injected': injected,
        'machines_mutated': len(cutoffs), 'episodes_mutated': len({e for e, _ in cutoffs}),
        'queries_attacked': len(vq),
        'observable_bitwise_identical': obs_ok,
        'observable_max_abs_delta': float(np.max(np.abs(X0 - X1))),
        'hmm_summary_bitwise_identical': hmm_ok,
        'hmm_summary_max_abs_delta': float(np.max(np.abs(H0 - H1)))}
    print(f'bypass injection: {injected} post-cutoff ticks forced into the index -> '
          f'observable identical={obs_ok}, hmm summary identical={hmm_ok}', flush=True)

    # ---- power control ------------------------------------------------------
    detail = []
    for q in vq:
        obs = idx0[(q.episode_id, q.machine_id)]
        gap = next((t for t in range(q.cutoff - 128, q.cutoff - 40) if t >= 0 and t not in obs), None)
        if gap is None:
            continue
        idx2 = {k: dict(v) for k, v in idx0.items()}
        idx2[(q.episode_id, q.machine_id)][gap] = PAYOUTS.index(2)
        X2, _, H2, _ = build(idx2)
        detail.append({'query_id': q.query_id, 'injected_tick': gap, 'cutoff': q.cutoff,
                       'observable_max_abs_delta': float(np.max(np.abs(X0 - X2))),
                       'hmm_summary_max_abs_delta': float(np.max(np.abs(H0 - H2)))})
        if len(detail) >= 3:
            break
    report['power_control_pre_cutoff'] = {
        'probes': len(detail),
        'probes_that_changed_observable': sum(1 for d in detail if d['observable_max_abs_delta'] > 0),
        'test_has_power': bool(detail) and all(d['observable_max_abs_delta'] > 0 for d in detail),
        'detail': detail}
    print(f"power control: {report['power_control_pre_cutoff']['probes_that_changed_observable']}"
          f"/{len(detail)} pre-cutoff injections changed the observable table "
          f"(test has power = {report['power_control_pre_cutoff']['test_has_power']})", flush=True)

    # ---- window containment on the injected index ---------------------------
    bad = [e['q'].query_id for e in examples(vq, idx1) if e['h'] and max(e['h']) >= e['q'].cutoff]
    report['window_containment_under_injection'] = {
        'failure_count': len(bad), 'failures': bad[:10],
        'note': 'sample_history() must re-filter even though the index now holds future ticks'}
    print(f'window containment under injection: {len(bad)} failures', flush=True)

    report['seconds'] = time.perf_counter() - t0
    report['passed'] = bool(obs_ok and hmm_ok and not bad and
                            report['power_control_pre_cutoff']['test_has_power'])
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('TABPFN BYPASS AUDIT', 'PASSED' if report['passed'] else 'FAILED',
          f"({report['seconds']:.1f}s)")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
