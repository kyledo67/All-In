"""Three supporting probes, saved so every number in report.json is re-runnable.

P1 cutoff homogeneity and the real neighbour-leak surface in the RELEASED files.
P2 empirical benignity of the one positional slice in any component, slot_model/mtd.py:68.
P3 condition/target label reachability from model code.
"""
import collections, csv, json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np

from slot_model.features import causal_index
from slot_model.mtd import index as mtd_index
from slot_forecasting.io import load_inputs, read_queries

DATA = ROOT / 'main-participants-60-40/public'
OUT = Path(__file__).resolve().parent / 'surface_probes.json'


def main():
    t0 = time.perf_counter()
    report = {}

    # ---- P1 cutoff homogeneity + neighbour-leak surface --------------------
    p1 = {}
    for split in ('train', 'validation', 'test'):
        qs = read_queries(DATA / split / 'queries.csv')
        byep = collections.defaultdict(list)
        for q in qs:
            byep[q.episode_id].append(q.cutoff)
        spread = [max(v) - min(v) for v in byep.values()]
        p1[split] = {'queries': len(qs), 'episodes': len(byep),
                     'machines_per_episode': sorted({len(v) for v in byep.values()}),
                     'max_cutoff_spread_within_episode': max(spread),
                     'episodes_with_one_shared_cutoff': f'{sum(1 for s in spread if s == 0)}/{len(spread)}'}
    vq, vh = load_inputs(DATA / 'validation')
    idx = causal_index(vq, vh)
    pairs = exposed = exposed_ticks = 0
    for q in vq:
        for (e, m), h in idx.items():
            if e != q.episode_id or m == q.machine_id:
                continue
            pairs += 1
            n = sum(1 for t in h if t >= q.cutoff)
            if n:
                exposed += 1
                exposed_ticks += n
    report['P1_cutoff_homogeneity_and_surface'] = {
        'by_split': p1, 'validation_target_neighbour_pairs': pairs,
        'pairs_with_neighbour_ticks_at_or_after_target_cutoff': exposed,
        'neighbour_ticks_at_or_after_a_target_cutoff': exposed_ticks,
        'conclusion':
            'Every episode in all three splits gives its 8 machines one identical cutoff, so '
            'the released files contain ZERO neighbour ticks at or after a target cutoff. The '
            'neighbour-future leak condition never arises naturally on this data, and the '
            "organizer's index_history() would reject a file where it did. The mutation tests "
            'therefore probe a counterfactual: they prove the code stays causal under a '
            'broader event stream, not that a live risk was caught.'}
    print('P1 cutoff homogeneity:', flush=True)
    for s, v in p1.items():
        print(f"   {s:11s} {v['queries']:5d} queries / {v['episodes']:4d} episodes, "
              f"max cutoff spread within episode = {v['max_cutoff_spread_within_episode']}, "
              f"single-cutoff episodes {v['episodes_with_one_shared_cutoff']}", flush=True)
    print(f'   validation: {exposed}/{pairs} target-neighbour pairs expose a tick at/after the '
          f'target cutoff ({exposed_ticks} ticks)', flush=True)

    # ---- P2 benignity of slot_model/mtd.py:68 rows[20:-19] -----------------
    cond = {r['query_id']: r['condition'] for r in
            csv.DictReader(open(DATA / 'validation/conditions.csv'))}
    midx = mtd_index(vh)
    agg = collections.defaultdict(lambda: dict(candidates=0, passed=0, in_dense=0, outside=0))
    for q in vq:
        h = {t: v for t, v in midx[q.episode_id, q.machine_id].items() if t < q.cutoff}
        rows = sorted(h)
        a = agg[cond[q.query_id]]
        if len(rows) < 40:
            continue
        for t in rows[20:-19]:
            a['candidates'] += 1
            A = [h[s] > 0 for s in range(t - 20, t) if s in h]
            B = [h[s] > 0 for s in range(t, t + 20) if s in h]
            if len(A) >= 15 and len(B) >= 15:
                a['passed'] += 1
                if t >= q.cutoff - 40:
                    a['in_dense'] += 1
                else:
                    a['outside'] += 1
    report['P2_changepoint_positional_slice'] = {
        'location': 'slot_model/mtd.py:68  for t in rows[20:-19]:',
        'by_condition': {k: dict(v) for k, v in sorted(agg.items())},
        'conclusion':
            'The slice only chooses WHICH ticks are candidate changepoints. The statistic uses '
            'tick arithmetic range(t-20,t) / range(t,t+20) guarded by `if s in h` and >=15 '
            'present observations per side. For short_history only 1329 of 4800 candidates '
            'survive, and all 1329 sit inside the dense recent-40 region, so sparse older '
            'rows are never averaged as 20 adjacent observations.'}
    print('P2 changepoint slice:', flush=True)
    for k, v in sorted(agg.items()):
        print(f"   {k:14s} candidates={v['candidates']:6d} passed>=15 guard={v['passed']:6d} "
              f"in recent-40={v['in_dense']:6d} outside={v['outside']:6d}", flush=True)

    # ---- P3 condition/target reachability ----------------------------------
    pat = re.compile(r'conditions|targets|reference\.csv')
    hits = []
    for p in sorted((ROOT / 'slot_model').glob('*.py')):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append({'file': f'slot_model/{p.name}', 'line': i, 'text': line.strip()[:120]})
    report['P3_label_reachability'] = {
        'slot_model_hits': hits,
        'conclusion':
            'Inside slot_model/ the only module that touches conditions.csv, targets.csv or '
            'reference.csv is official_score.py, a scoring module. No feature builder '
            '(features.py, phase2_features.py, phase2_tables.py, mtd.py, mtd_pairs.py, '
            'shared_state.py) can reach a condition or target label. In the component scripts '
            'they are read only after write_submission, feeding contest_score.'}
    print(f'P3 label reachability: {len(hits)} hits in slot_model/, all in '
          f"{sorted({h['file'] for h in hits})}", flush=True)

    report['seconds'] = time.perf_counter() - t0
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('SURFACE PROBES written', f"({report['seconds']:.1f}s)")


if __name__ == '__main__':
    main()
