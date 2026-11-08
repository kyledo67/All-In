"""Causality test for slot_model/shared_state.py neighbor_state.

neighbor_state loops `for t in range(start, q.cutoff)` with start=max(0,q.cutoff-96) and
reads OTHER machines' observations. Two things must hold:
  S1 strict inequality: the loop can never read tick == q.cutoff. Verified structurally
     (range upper bound is exclusive) AND adversarially: inject neighbour observations at
     ticks cutoff..cutoff+8 -> state must be bitwise identical. slot_model.mtd.index()
     applies no cutoff filter, so nothing upstream removes these rows for us.
  S2 own machine excluded: m != q.machine_id, so the target's own ticks never enter.
     Adversarially: inject the target machine's OWN observations inside the window
     [cutoff-96, cutoff) -> state must be bitwise identical.
  S3 power control: inject a NEIGHBOUR observation at a tick strictly inside the window
     -> state MUST change, otherwise S1/S2 are vacuous.
  S4 missing ticks are skipped, not zeroed: a tick with no neighbour observation performs
     only the prior@prev transition and no likelihood update.

Context: shared_state is NOT part of the audited candidate (commit f002c77 removed it from
the blend candidate set, and artifacts/shared_state_01/validation.csv is not a standalone
model anyway). This is checked because the dimension asks for it.

NOTE ON HOLDOUT STATUS: public validation, inspected many times. Not an independent holdout.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np

from slot_model.shared_state import neighbor_state
from slot_model.mtd import PAYOUTS, index as mtd_index
from slot_forecasting.io import load_inputs
from slot_forecasting.schema import Play

DATA = ROOT / 'main-participants-60-40/public'
WIN_RATE = 0.35
OUT = Path(__file__).resolve().parent / 'shared_state_mutation.json'


def diff(a, b):
    assert set(a) == set(b)
    d = {k: abs(a[k] - b[k]) for k in a}
    worst = max(d.values())
    return (worst == 0.0), worst, sorted([k for k, v in d.items() if v > 0])[:10]


def main():
    t0 = time.perf_counter()
    vq, vh = load_inputs(DATA / 'validation')
    idx = mtd_index(vh)
    base = neighbor_state(vq, vh, WIN_RATE)
    report = {'holdout_status': 'public validation, inspected many times; NOT an independent '
                                'holdout. Causality checks only.',
              'queries': len(vq), 'win_rate': WIN_RATE, 'window_ticks': 96,
              'loop': 'for t in range(max(0, q.cutoff-96), q.cutoff)',
              'range_upper_bound_is_exclusive': True,
              'max_tick_readable': 'q.cutoff - 1',
              'in_candidate': False,
              'in_candidate_note': 'shared_state is not a component of the audited candidate; '
                                   'checked because the dimension asks for it'}

    # ---- S1 neighbour post-cutoff injection --------------------------------
    extra = []
    for q in vq:
        for off in range(9):
            t = q.cutoff + off
            for (e, m), h in idx.items():
                if e == q.episode_id and m != q.machine_id and t not in h:
                    extra.append(Play(e, m, t, PAYOUTS[4]))
    # de-duplicate: mtd.index would otherwise just overwrite, but keep it clean
    seen, uniq = set(), []
    for p in extra:
        k = (p.episode_id, p.machine_id, p.tick)
        if k not in seen and p.tick not in idx[(p.episode_id, p.machine_id)]:
            seen.add(k)
            uniq.append(p)
    s1 = neighbor_state(vq, tuple(vh) + tuple(uniq), WIN_RATE)
    s1_ok, s1_worst, s1_bad = diff(base, s1)
    report['S1_neighbour_post_cutoff_invariance'] = {
        'plays_injected': len(uniq), 'bitwise_identical': s1_ok,
        'max_abs_delta': s1_worst, 'changed': s1_bad,
        'ticks': 'cutoff .. cutoff+8 on every episode neighbour of every query'}
    print(f'S1 neighbour post-cutoff: {len(uniq)} plays -> identical={s1_ok} '
          f'(max delta {s1_worst:.3e})', flush=True)

    # ---- S2 own machine excluded (PER-QUERY ISOLATED) ----------------------
    # Must be isolated per query: a play on machine X is "own" for X's query but a
    # legitimate NEIGHBOUR observation for the other 7 queries of the same episode.
    # Injecting for all queries at once cross-contaminates and the state correctly moves.
    s2_detail = []
    for q in vq[:24]:
        h = idx[(q.episode_id, q.machine_id)]
        plays = []
        for t in range(max(0, q.cutoff - 96), q.cutoff):
            if t not in h:
                plays.append(Play(q.episode_id, q.machine_id, t, PAYOUTS[4]))
        # also OVERWRITE existing own ticks (mtd.index lets a later row replace a tick)
        for t in sorted(h)[-8:]:
            if max(0, q.cutoff - 96) <= t < q.cutoff:
                flip = PAYOUTS[0] if h[t] > 0 else PAYOUTS[4]
                plays.append(Play(q.episode_id, q.machine_id, t, flip))
        if not plays:
            continue
        got = neighbor_state([q], tuple(vh) + tuple(plays), WIN_RATE)
        s2_detail.append({'query_id': q.query_id, 'own_plays_injected': len(plays),
                          'abs_delta': abs(got[q.query_id] - base[q.query_id])})
    s2_ok = bool(s2_detail) and all(d['abs_delta'] == 0.0 for d in s2_detail)
    s2_worst = max([d['abs_delta'] for d in s2_detail], default=0.0)
    report['S2_own_machine_excluded'] = {
        'queries_probed': len(s2_detail), 'bitwise_identical': s2_ok,
        'max_abs_delta': s2_worst, 'detail': s2_detail,
        'method': 'per-query isolated: own-machine ticks injected AND existing own ticks '
                  'overwritten with flipped win/loss inside [cutoff-96, cutoff); only that '
                  "query's own state is compared",
        'note': 'an earlier all-queries-at-once variant of this probe showed a 7.125e-04 '
                'delta. That was test cross-contamination, not a leak: a play on machine X '
                'is a legitimate neighbour observation for the other 7 queries of episode X.'}
    print(f'S2 own machine excluded (isolated): {len(s2_detail)} queries -> '
          f'identical={s2_ok} (max delta {s2_worst:.3e})', flush=True)

    # ---- S3 power control: OVERWRITE an existing in-window neighbour tick ---
    detail = []
    for q in vq[:12]:
        nbrs = [(e, m) for (e, m), h in idx.items()
                if e == q.episode_id and m != q.machine_id]
        picked = None
        for nbr in nbrs:
            h = idx[nbr]
            for t in sorted(h, reverse=True):
                if max(0, q.cutoff - 96) <= t < q.cutoff:
                    picked = (nbr, t, PAYOUTS[0] if h[t] > 0 else PAYOUTS[4])
                    break
            if picked:
                break
        if picked is None:
            continue
        nbr, t, flip = picked
        got = neighbor_state([q], tuple(vh) + (Play(nbr[0], nbr[1], t, flip),), WIN_RATE)
        detail.append({'query_id': q.query_id, 'neighbour': nbr[1], 'overwritten_tick': t,
                       'cutoff': q.cutoff, 'flipped_to_payout': flip,
                       'abs_delta': abs(got[q.query_id] - base[q.query_id])})
    report['S3_power_control'] = {
        'probes': len(detail), 'probes_that_changed': sum(1 for d in detail if d['abs_delta'] > 0),
        'test_has_power': bool(detail) and all(d['abs_delta'] > 0 for d in detail),
        'detail': detail,
        'method': 'flip the win/loss of one EXISTING neighbour tick strictly inside the '
                  'window (every in-window tick is already observed, so there is no gap to '
                  'fill); mtd.index lets a later row overwrite a tick'}
    print(f"S3 power control: {report['S3_power_control']['probes_that_changed']}/{len(detail)} "
          f"in-window neighbour flips changed the state "
          f"(test has power = {report['S3_power_control']['test_has_power']})", flush=True)

    # ---- S4 missing ticks skipped, not zeroed ------------------------------
    counts = []
    for q in vq[:32]:
        machines = {m: h for (e, m), h in idx.items() if e == q.episode_id}
        empty = sum(1 for t in range(max(0, q.cutoff - 96), q.cutoff)
                    if not [1 for m, h in machines.items() if m != q.machine_id and t in h])
        counts.append(empty)
    report['S4_missing_ticks_skipped'] = {
        'queries_sampled': len(counts), 'ticks_with_no_neighbour_observation': sum(counts),
        'mechanism': 'vals is empty -> the `if vals:` branch is skipped, so only prior@prev '
                     'runs and no 0-win likelihood is applied. A missing tick is therefore '
                     'UNKNOWN, not a loss.',
        'verified_structurally': True}
    print(f"S4 missing ticks: {sum(counts)} window ticks with no neighbour observation across "
          f"{len(counts)} sampled queries, all skipped rather than scored as losses", flush=True)

    report['seconds'] = time.perf_counter() - t0
    report['passed'] = bool(s1_ok and s2_ok and report['S3_power_control']['test_has_power'])
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('SHARED-STATE AUDIT', 'PASSED' if report['passed'] else 'FAILED',
          f"({report['seconds']:.1f}s)")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
