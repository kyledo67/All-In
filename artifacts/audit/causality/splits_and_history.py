"""Split integrity (whole episodes, no fold overlap, train/validation disjoint) and
short-history semantics (40 consecutive recent ticks + sparse older ticks, never treated
as adjacent) for every component of the audited candidate.

NOTE ON HOLDOUT STATUS: the 4 validation folds are splits of the SAME 96 public-validation
episodes that have been inspected many times across this project. A clean fold partition
makes the OOF construction internally honest; it does NOT make the resulting number an
independent holdout, and it is not confirmation of anything. The candidate's simplex
weights are fitted against public-validation TARGETS on 3 folds and applied to the 4th,
so the 32.260531 figure is an in-sample-at-the-corpus-level estimate.
"""
import csv, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np

from slot_model.features import causal_index
from slot_forecasting.io import load_inputs, read_submission

DATA = ROOT / 'main-participants-60-40/public'
OUT = Path(__file__).resolve().parent / 'splits_and_history.json'


def main():
    t0 = time.perf_counter()
    report = {'holdout_status': __doc__.split('NOTE ON HOLDOUT STATUS:')[1].strip()}
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())
    tq, th = load_inputs(DATA / 'train')
    vq, vh = load_inputs(DATA / 'validation')

    # ---- 5. split integrity -------------------------------------------------
    split = {}
    for name, qs in (('train', tq), ('validation', vq)):
        fmap = folds[name]
        eps = {q.episode_id for q in qs}
        per_ep = Counter(q.episode_id for q in qs)
        # whole episodes: the fold map is keyed by episode, so every query of an episode
        # necessarily lands in the same fold. Verify explicitly anyway.
        by_q = defaultdict(set)
        for q in qs:
            by_q[q.episode_id].add(fmap.get(q.episode_id, 'MISSING'))
        split[name] = {
            'queries': len(qs), 'episodes': len(eps),
            'fold_map_entries': len(fmap),
            'episodes_missing_from_fold_map': sorted(eps - set(fmap)),
            'fold_map_entries_with_no_query': sorted(set(fmap) - eps),
            'queries_per_episode': sorted(set(per_ep.values())),
            'episodes_with_more_than_one_fold': sorted(e for e, f in by_q.items() if len(f) != 1),
            'fold_sizes_episodes': {str(f): sum(1 for v in fmap.values() if v == f)
                                    for f in sorted(set(fmap.values()))},
            'fold_sizes_queries': {str(f): sum(1 for q in qs if fmap[q.episode_id] == f)
                                   for f in sorted(set(fmap.values()))},
            'folds_partition_all_queries':
                sum(sum(1 for q in qs if fmap[q.episode_id] == f)
                    for f in sorted(set(fmap.values()))) == len(qs)}
    # held vs fit disjointness, per fold, as the candidate's simplex uses it
    vmap = folds['validation']
    per_fold = []
    for f in sorted(set(vmap.values())):
        held = {e for e, g in vmap.items() if g == f}
        fit = {e for e, g in vmap.items() if g != f}
        per_fold.append({'fold': f, 'held_episodes': len(held), 'fit_episodes': len(fit),
                         'overlap': len(held & fit),
                         'held_queries': sum(1 for q in vq if vmap[q.episode_id] == f)})
    union = set()
    double = []
    for f in sorted(set(vmap.values())):
        held = {e for e, g in vmap.items() if g == f}
        double += sorted(union & held)
        union |= held
    report['split_integrity'] = {
        'per_split': split,
        'validation_per_fold': per_fold,
        'episodes_in_two_folds': double,
        'validation_fold_union_covers_all_episodes':
            union == {q.episode_id for q in vq},
        'train_validation_episode_overlap':
            sorted({q.episode_id for q in tq} & {q.episode_id for q in vq}),
        'train_validation_machine_overlap':
            len({(q.episode_id, q.machine_id) for q in tq} &
                {(q.episode_id, q.machine_id) for q in vq}),
        'candidate_oof_rows': None}
    cand = read_submission(ROOT / 'artifacts/audit/oof_candidate_raw.csv')
    report['split_integrity']['candidate_oof_rows'] = {
        'rows': len(cand), 'unique_query_ids': len(set(cand)),
        'matches_validation_query_set': set(cand) == {q.query_id for q in vq},
        'episodes_covered': len({k.rsplit('-m', 1)[0] for k in cand})}
    print('5 split integrity:', flush=True)
    for name, v in split.items():
        print(f"   {name:11s} {v['queries']} queries / {v['episodes']} episodes, "
              f"queries_per_episode={v['queries_per_episode']}, "
              f"fold sizes {v['fold_sizes_episodes']}, "
              f"episodes with >1 fold: {len(v['episodes_with_more_than_one_fold'])}, "
              f"missing from map: {len(v['episodes_missing_from_fold_map'])}", flush=True)
    print(f"   episodes in two folds: {double}; train/validation episode overlap: "
          f"{report['split_integrity']['train_validation_episode_overlap']}; "
          f"machine overlap: {report['split_integrity']['train_validation_machine_overlap']}",
          flush=True)
    print(f"   candidate OOF: {report['split_integrity']['candidate_oof_rows']}", flush=True)

    # ---- 6. short-history semantics ----------------------------------------
    cond = {r['query_id']: r['condition'] for r in
            csv.DictReader(open(DATA / 'validation/conditions.csv'))}
    idx = causal_index(vq, vh)
    agg = defaultdict(lambda: dict(queries=0, hist=[], recent40_complete=0, recent40_gaps=0,
                                   older=[], span=[], max_older_gap=[], older_adjacent_pairs=0,
                                   older_pairs=0))
    for q in vq:
        h = idx[(q.episode_id, q.machine_id)]
        pre = sorted(h)
        a = agg[cond[q.query_id]]
        a['queries'] += 1
        a['hist'].append(len(pre))
        rec = [t for t in range(q.cutoff - 40, q.cutoff) if t in h]
        a['recent40_complete'] += (len(rec) == 40)
        a['recent40_gaps'] += 40 - len(rec)
        old = [t for t in pre if t < q.cutoff - 40]
        a['older'].append(len(old))
        a['span'].append(q.cutoff - pre[0])
        if len(old) > 1:
            gaps = np.diff(old)
            a['max_older_gap'].append(int(gaps.max()))
            a['older_pairs'] += len(gaps)
            a['older_adjacent_pairs'] += int((gaps == 1).sum())
    hist_report = {}
    for k, a in sorted(agg.items()):
        hist_report[k] = {
            'queries': a['queries'],
            'history_ticks_min': min(a['hist']), 'history_ticks_max': max(a['hist']),
            'recent40_complete': f"{a['recent40_complete']}/{a['queries']}",
            'recent40_missing_ticks_total': a['recent40_gaps'],
            'older_ticks_min': min(a['older']), 'older_ticks_max': max(a['older']),
            'span_min': min(a['span']), 'span_max': max(a['span']),
            'older_gap_max': max(a['max_older_gap']) if a['max_older_gap'] else None,
            'older_adjacent_fraction': (a['older_adjacent_pairs'] / a['older_pairs']
                                        if a['older_pairs'] else None)}
    report['short_history_semantics'] = {
        'by_condition': hist_report,
        'short_history_episodes': sorted({k.rsplit('-m', 1)[0] for k, c in cond.items()
                                          if c == 'short_history'}),
        'interpretation':
            'short_history machines hold exactly 64 pre-cutoff ticks: the 40 ticks '
            '[cutoff-40, cutoff) are all present and consecutive, plus exactly 24 older '
            'ticks scattered over a much longer span, only a small fraction of which are '
            'tick-adjacent. The dense conditions have history count == span, i.e. fully '
            'contiguous with no gaps at all.'}
    print('6 short-history semantics:', flush=True)
    for k, v in hist_report.items():
        print(f"   {k:14s} hist={v['history_ticks_min']}-{v['history_ticks_max']} "
              f"recent40_complete={v['recent40_complete']} "
              f"older={v['older_ticks_min']}-{v['older_ticks_max']} "
              f"span={v['span_min']}-{v['span_max']} "
              f"older_gap_max={v['older_gap_max']} "
              f"older_adjacent_frac={v['older_adjacent_fraction']}", flush=True)

    # ---- 6b. row-offset vs tick-arithmetic scan ----------------------------
    files = ['slot_model/features.py', 'slot_model/phase2_features.py',
             'slot_model/phase2_predict.py', 'slot_model/mtd.py', 'slot_model/mtd_pairs.py',
             'slot_model/shared_state.py', 'scripts/dense_catboost.py',
             'scripts/two_head_catboost.py']
    # positional slicing / indexing of a tick-ordered row list
    suspicious = re.compile(r'rows\s*\[|sorted\([^)]*\)\s*\[|\.values\(\)\s*\[|iloc|shift\(')
    hits = []
    for rel in files:
        for i, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
            if suspicious.search(line):
                hits.append({'file': rel, 'line': i, 'text': line.strip()})
    report['row_offset_scan'] = {
        'files_scanned': files, 'pattern': suspicious.pattern, 'hits': hits,
        'assessment':
            'The only positional slicing of a tick-ordered list in any candidate component '
            'is slot_model/mtd.py `for t in rows[20:-19]` inside the changepoint search. It '
            'selects WHICH ticks are candidate changepoints; the statistic itself is built '
            'with tick arithmetic (range(t-20,t) and range(t,t+20)) guarded by `if s in h` '
            'and a len>=15 minimum per side, so sparse older rows cannot masquerade as 20 '
            'adjacent observations. Every lag, decay and window in every component is tick '
            'arithmetic: h.get(t-k), q.cutoff-lag, q.cutoff-1-t, q.cutoff-w <= t < q.cutoff.'}
    print(f"6b row-offset scan: {len(hits)} positional-indexing hits across "
          f"{len(files)} component files", flush=True)
    for h in hits:
        print(f"   {h['file']}:{h['line']}  {h['text'][:100]}", flush=True)

    ok = (not double
          and not report['split_integrity']['train_validation_episode_overlap']
          and report['split_integrity']['train_validation_machine_overlap'] == 0
          and report['split_integrity']['validation_fold_union_covers_all_episodes']
          and all(not v['episodes_with_more_than_one_fold'] and
                  not v['episodes_missing_from_fold_map'] and
                  v['queries_per_episode'] == [8] and v['folds_partition_all_queries']
                  for v in split.values())
          and all(p['overlap'] == 0 for p in per_fold)
          and report['split_integrity']['candidate_oof_rows']['matches_validation_query_set']
          and all(v['recent40_missing_ticks_total'] == 0 for v in hist_report.values()))
    report['seconds'] = time.perf_counter() - t0
    report['passed'] = bool(ok)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')
    print('SPLITS + HISTORY AUDIT', 'PASSED' if ok else 'FAILED', f"({report['seconds']:.1f}s)")
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
