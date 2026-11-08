"""Adversarial causality mutation test for the MTD component of the audited candidate.

Target under test: PairMTDModel(LAG7_PAIRS) over MTDModel.load(artifacts/mtd_01/change.json),
which is exactly what produced artifacts/mtd_pairs_01/validation_lag7.csv.

Why this is a strong test: slot_model.mtd.index() applies NO cutoff filter at all
(unlike slot_model.features.causal_index, which drops tick >= cutoff). So post-cutoff
rows reach the model's own index untouched; invariance can only come from the model's
own `t < cutoff` guards and `t - k` tick arithmetic.

NOTE ON HOLDOUT STATUS: this runs on public validation, which has been inspected many
times across this project. Nothing here is an independent holdout. These are causality
(information-flow) checks, not generalisation evidence.

Checks
  R  reproducibility: re-predict vs saved validation_lag7.csv, plus in-process determinism
  A  post-cutoff invariance: append plays at ticks cutoff..cutoff+8 for EVERY machine
     (target machine AND all 7 episode neighbours) of all 96 episodes -> bitwise identical
  P  power/sensitivity control: insert ONE play at a MISSING tick strictly below cutoff
     -> prediction MUST change (otherwise check A is vacuous)
  U1 unknown symbol census: where symbol 5 (unknown) actually occurs, per condition,
     at the query cutoff and across the historical rows predict() fits on
  U2 unknown != 0x: materialise a missing tick as an actual 0x play; the symbol must move
     5 -> 0 and the prediction must change
  U3 lag arithmetic: inputs(q, t, idx) recomputed from an independent tick->payout dict
     using t-k (own lags 1..8, neighbour lags 1..3) must agree, at the cutoff AND on
     every historical row
  U4 structural: the learned Q tensor has 6 input symbols (5 payouts + unknown) and the
     unknown row is not a copy of the 0x row
"""
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np

from slot_model.mtd import MTDModel, PAYOUTS, index as mtd_index, inputs as mtd_inputs
from slot_model.mtd_pairs import PairMTDModel, LAG7_PAIRS
from slot_forecasting.io import load_inputs, read_submission
from slot_forecasting.schema import Play

DATA = ROOT / 'main-participants-60-40/public'
HORIZON = 8
UNKNOWN = 5
OUT = Path(__file__).resolve().parent / 'mtd_mutation.json'


def model():
    return PairMTDModel(pairs=LAG7_PAIRS, base=MTDModel.load(ROOT / 'artifacts/mtd_01/change.json'))


def compare(a, b):
    assert set(a) == set(b), 'key sets differ'
    worst, changed = 0.0, []
    for k in sorted(a):
        va, vb = np.asarray(a[k], float), np.asarray(b[k], float)
        worst = max(worst, float(np.max(np.abs(va - vb))))
        if not np.array_equal(va, vb):
            changed.append(k)
    return (not changed), changed, worst


def main():
    t0 = time.perf_counter()
    vq, vh = load_inputs(DATA / 'validation')
    idx = mtd_index(vh)
    cond = {r['query_id']: r['condition'] for r in
            csv.DictReader(open(DATA / 'validation/conditions.csv'))}
    cutoffs = {(q.episode_id, q.machine_id): q.cutoff for q in vq}
    report = {'holdout_status': 'public validation, inspected many times; NOT an independent '
                                'holdout. Causality checks only, no generalisation claim.',
              'queries': len(vq), 'machines': len(idx), 'horizon_ticks': HORIZON,
              'mtd_index_applies_no_cutoff_filter': True}

    # ---- R reproducibility + determinism ----------------------------------
    base = model().predict(vq, vh)
    base2 = model().predict(vq, vh)
    det_ok, _, det_delta = compare(base, base2)
    saved = read_submission(ROOT / 'artifacts/mtd_pairs_01/validation_lag7.csv')
    rep_ok, rep_changed, rep_delta = compare(base, saved)
    report['reproducibility'] = {
        'in_process_bitwise_deterministic': det_ok, 'in_process_max_abs_delta': det_delta,
        'saved_bitwise_identical': rep_ok, 'saved_max_abs_delta': rep_delta,
        'saved_queries_differing': len(rep_changed),
        'saved_within_1e_15': rep_delta < 1e-15,
        'note': 'submissions are written with .17g (exact float64 round-trip), so a nonzero '
                'delta is cross-run floating-point reduction order, not CSV truncation'}
    print(f'R determinism={det_ok}; vs saved: bitwise={rep_ok} max_delta={rep_delta:.3e} '
          f'({len(rep_changed)} queries)', flush=True)

    # ---- A post-cutoff invariance -----------------------------------------
    extra = []
    for (e, m), cut in sorted(cutoffs.items()):
        observed = idx[(e, m)]
        for off in range(HORIZON):
            tick = cut + off
            if tick in observed:
                continue
            recent = observed.get(cut - 1, 0)
            extra.append(Play(e, m, tick, PAYOUTS[(recent + 3) % len(PAYOUTS)]))
    assert extra, 'built no adversarial future plays'
    mutated = model().predict(vq, tuple(vh) + tuple(extra))
    ok_a, bad_a, worst_a = compare(base, mutated)
    report['post_cutoff_invariance'] = {
        'adversarial_plays_added': len(extra),
        'machines_mutated': len({(p.episode_id, p.machine_id) for p in extra}),
        'episodes_mutated': len({p.episode_id for p in extra}),
        'queries_attacked': len(vq), 'bitwise_identical': ok_a,
        'max_abs_delta': worst_a, 'changed_queries': bad_a[:10],
        'covers': 'own machine future AND all 7 episode-neighbour futures, every episode'}
    print(f'A post-cutoff invariance: {len(extra)} plays / '
          f"{report['post_cutoff_invariance']['machines_mutated']} machines -> "
          f'identical={ok_a} (max delta {worst_a:.3e})', flush=True)

    # ---- P power control ----------------------------------------------------
    probe, per_probe = [], []
    for q in vq:
        obs = idx[(q.episode_id, q.machine_id)]
        gap = next((t for t in range(q.cutoff - 128, q.cutoff - 40) if t >= 0 and t not in obs), None)
        if gap is not None:
            probe.append((q, Play(q.episode_id, q.machine_id, gap, PAYOUTS[2])))
        if len(probe) >= 12:
            break
    for q, play in probe:
        got = model().predict([q], tuple(vh) + (play,))
        d = float(np.max(np.abs(np.asarray(got[q.query_id]) - np.asarray(base[q.query_id]))))
        per_probe.append({'query_id': q.query_id, 'inserted_tick': play.tick,
                          'cutoff': q.cutoff, 'max_abs_delta': d})
    changed = sum(1 for p in per_probe if p['max_abs_delta'] > 0)
    report['power_control_pre_cutoff'] = {
        'probes': len(probe), 'probes_that_changed': changed,
        'test_has_power': bool(probe) and changed == len(probe), 'detail': per_probe}
    print(f'P power control: {changed}/{len(probe)} pre-cutoff insertions changed the '
          f'prediction (test has power = {bool(probe) and changed == len(probe)})', flush=True)

    # ---- U1 census + U3 lag arithmetic over cutoff AND historical rows -----
    raw = defaultdict(dict)
    for p in vh:
        raw[(p.episode_id, p.machine_id)][p.tick] = PAYOUTS.index(p.payout)
    census = defaultdict(lambda: dict(queries=0, hist_rows=0, cutoff_own_slots=0,
                                      cutoff_own_unknown=0, cutoff_nbr_slots=0,
                                      cutoff_nbr_unknown=0, hist_own_slots=0,
                                      hist_own_unknown=0, hist_nbr_slots=0, hist_nbr_unknown=0))
    own_fail, nbr_fail, rows_checked = [], [], 0

    def verify(q, t):
        got = mtd_inputs(q, t, idx)
        own = raw[(q.episode_id, q.machine_id)]
        want = [own.get(t - k, UNKNOWN) for k in range(1, 9)]
        if got[:8] != want:
            own_fail.append(f'{q.query_id}@t{t}')
        for k in range(1, 4):
            vals = [v[t - k] > 0 for (e, m), v in raw.items()
                    if e == q.episode_id and m != q.machine_id and t - k in v]
            w = UNKNOWN if not vals else min(2, int(3 * sum(vals) / len(vals)))
            if got[7 + k] != w:
                nbr_fail.append(f'{q.query_id}@t{t}|nbr_lag{k}')
        return got

    for q in vq:
        c = census[cond[q.query_id]]
        c['queries'] += 1
        xx = verify(q, q.cutoff)
        rows_checked += 1
        c['cutoff_own_slots'] += 8
        c['cutoff_own_unknown'] += sum(1 for v in xx[:8] if v == UNKNOWN)
        c['cutoff_nbr_slots'] += 3
        c['cutoff_nbr_unknown'] += sum(1 for v in xx[8:] if v == UNKNOWN)
        h = {t: v for t, v in idx[q.episode_id, q.machine_id].items() if t < q.cutoff}
        for t in sorted(h):
            x = verify(q, t)
            rows_checked += 1
            c['hist_rows'] += 1
            c['hist_own_slots'] += 8
            c['hist_own_unknown'] += sum(1 for v in x[:8] if v == UNKNOWN)
            c['hist_nbr_slots'] += 3
            c['hist_nbr_unknown'] += sum(1 for v in x[8:] if v == UNKNOWN)
    report['lag_arithmetic'] = {
        'rows_checked': rows_checked, 'queries': len(vq),
        'own_lag_failure_count': len(own_fail), 'own_lag_failures': own_fail[:10],
        'neighbour_lag_failure_count': len(nbr_fail), 'neighbour_lag_failures': nbr_fail[:10],
        'note': 'every own lag recomputed as own.get(t-k); every neighbour lag as t-k over '
                'other machines only. Neighbour lags are 1..3, so at t=cutoff they read '
                'cutoff-1..cutoff-3 and can never read tick == cutoff.'}
    report['unknown_symbol_census'] = {
        'unknown_code': UNKNOWN, 'payout_0x_code': PAYOUTS.index(0),
        'unknown_is_distinct_code_from_0x': UNKNOWN != PAYOUTS.index(0),
        'by_condition': {k: dict(v) for k, v in sorted(census.items())}}
    print(f'U3 lag arithmetic: {rows_checked} rows, {len(own_fail)} own-lag and '
          f'{len(nbr_fail)} neighbour-lag failures', flush=True)
    for k, v in sorted(census.items()):
        print(f"   U1 {k:14s} cutoff_own_unknown={v['cutoff_own_unknown']:5d}/"
              f"{v['cutoff_own_slots']:6d}  hist_own_unknown={v['hist_own_unknown']:6d}/"
              f"{v['hist_own_slots']:7d} ({100*v['hist_own_unknown']/max(1,v['hist_own_slots']):5.2f}%)"
              f"  hist_nbr_unknown={v['hist_nbr_unknown']:5d}/{v['hist_nbr_slots']:7d}", flush=True)

    # ---- U2 unknown != 0x, probed on the rows where unknown actually fires --
    u2 = []
    for q in vq:
        h = {t: v for t, v in idx[q.episode_id, q.machine_id].items() if t < q.cutoff}
        hit = None
        for t in sorted(h):
            x = mtd_inputs(q, t, idx)
            k = next((j for j in range(1, 9) if x[j - 1] == UNKNOWN and t - j >= 0), None)
            if k is not None:
                hit = (t, k, t - k)
                break
        if hit is None:
            continue
        t, k, gap = hit
        play = Play(q.episode_id, q.machine_id, gap, 0)
        hist2 = tuple(vh) + (play,)
        x2 = mtd_inputs(q, t, mtd_index(hist2))
        pred2 = model().predict([q], hist2)
        d = float(np.max(np.abs(np.asarray(pred2[q.query_id]) - np.asarray(base[q.query_id]))))
        u2.append({'query_id': q.query_id, 'condition': cond[q.query_id],
                   'historical_row_tick': t, 'lag': k, 'materialised_tick': gap,
                   'symbol_before': x2 and mtd_inputs(q, t, idx)[k - 1],
                   'symbol_after': x2[k - 1], 'max_abs_delta': d})
        if len(u2) >= 8:
            break
    report['unknown_distinct_from_zero'] = {
        'probes': len(u2), 'probes_found': bool(u2),
        'all_symbol_5_to_0': bool(u2) and all(p['symbol_before'] == UNKNOWN and
                                              p['symbol_after'] == 0 for p in u2),
        'all_predictions_changed': bool(u2) and all(p['max_abs_delta'] > 0 for p in u2),
        'detail': u2,
        'note': 'a missing tick is the unknown symbol 5; materialising it as a real 0x play '
                'moves the symbol to 0 and moves the forecast, so UNKNOWN is not folded into 0x'}
    print(f"U2 unknown vs 0x: {len(u2)} probes, symbol 5->0 "
          f"{report['unknown_distinct_from_zero']['all_symbol_5_to_0']}, predictions changed "
          f"{report['unknown_distinct_from_zero']['all_predictions_changed']}", flush=True)

    # ---- U4 structural ------------------------------------------------------
    ch = json.loads((ROOT / 'artifacts/mtd_01/change.json').read_text())
    shapes, distinct = set(), []
    for name, ctx in ch['contexts'].items():
        Q = np.asarray(ctx['Q'], float)
        shapes.add(Q.shape)
        distinct.append(bool(np.any(Q[:, UNKNOWN, :] != Q[:, 0, :])))
    report['q_tensor'] = {'shapes': sorted(str(s) for s in shapes),
                          'input_symbols': sorted({s[1] for s in shapes}),
                          'unknown_row_differs_from_0x_row_in_every_context': all(distinct),
                          'contexts': len(distinct), 'changepoint': ch['changepoint'],
                          'strength': ch['strength']}
    print(f"U4 Q tensor shapes {report['q_tensor']['shapes']}, unknown row distinct from 0x: "
          f"{report['q_tensor']['unknown_row_differs_from_0x_row_in_every_context']}", flush=True)

    report['seconds'] = time.perf_counter() - t0
    report['passed'] = bool(
        det_ok and report['reproducibility']['saved_within_1e_15'] and ok_a and
        report['power_control_pre_cutoff']['test_has_power'] and
        not own_fail and not nbr_fail and
        report['unknown_distinct_from_zero']['all_symbol_5_to_0'] and
        report['unknown_distinct_from_zero']['all_predictions_changed'] and
        report['q_tensor']['unknown_row_differs_from_0x_row_in_every_context'])
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('MTD MUTATION AUDIT', 'PASSED' if report['passed'] else 'FAILED',
          f"({report['seconds']:.1f}s)")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
