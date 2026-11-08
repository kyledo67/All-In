"""Wider versions of the prior agent's two greps.

(a) row-offset / positional-indexing scan over the COMPLETE component file set.
    The prior agent scanned 8 files and omitted phase2_tables.py, tabpfn_model.py,
    phase2_predict.py's callers, portable.py, frozen_blend.py and the scripts that
    actually produced the three active components.
(b) condition/target/reference/label reachability over slot_model/ AND scripts/,
    with the import graph, instead of slot_model/ only.
"""
import ast, json, re, sys
from pathlib import Path
ROOT = Path('/Users/ayan/Downloads/All-In')

# every module that is on the inference path of at least one candidate component,
# plus every script that produced one of the four component CSVs
FILES = sorted(set([
 'slot_model/features.py','slot_model/phase2_features.py','slot_model/phase2_predict.py',
 'slot_model/phase2_tables.py','slot_model/portable.py','slot_model/tabpfn_model.py',
 'slot_model/mtd.py','slot_model/mtd_pairs.py','slot_model/shared_state.py',
 'slot_model/frozen_blend.py','slot_model/model.py','slot_model/runtime.py',
 'scripts/dense_catboost.py','scripts/two_head_catboost.py','scripts/evaluate_mtd.py',
 'scripts/evaluate_mtd_pairs.py','scripts/tabpfn_train.py','scripts/tabpfn_evaluate.py',
 'scripts/phase2_evaluate.py','scripts/phase2_train.py','scripts/combined_evaluate.py',
 'organizer/slot_forecasting/transition.py',
]))
FILES = [f for f in FILES if (ROOT/f).exists()]

# broader than the prior agent's pattern: any subscript/slice of a name that looks like a
# tick-ordered sequence, plus pandas positional ops
pat = re.compile(r'(rows|pre|ticks|hist(?:ory)?|seq|order(?:ed)?|sorted_\w*|times?)\s*\[|'
                 r'sorted\([^)]*\)\s*\[|\.values\(\)\s*\[|\.keys\(\)\s*\[|iloc|'
                 r'shift\(|np\.roll|\[\s*-?\d+\s*:|\[\s*:\s*-?\d+\s*\]|argsort|'
                 r'enumerate\(sorted|zip\([^)]*\[1:\]')
hits = []
for rel in FILES:
    for i, line in enumerate((ROOT/rel).read_text().splitlines(), 1):
        s = line.strip()
        if s.startswith('#') or s.startswith('"""'):
            continue
        if pat.search(line):
            hits.append({'file': rel, 'line': i, 'text': s[:130]})

labelpat = re.compile(r'conditions\.csv|targets\.csv|reference\.csv|labels\.csv|'
                      r'read_targets|read_labels|contest_score|subset_score|scoring_inputs')
lab = []
for rel in sorted(set(FILES) | {f'slot_model/{p.name}' for p in (ROOT/'slot_model').glob('*.py')}):
    p = ROOT/rel
    if not p.exists():
        continue
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if labelpat.search(line):
            lab.append({'file': rel, 'line': i, 'text': line.strip()[:150]})

out = {'files_scanned': FILES, 'files_scanned_count': len(FILES),
       'prior_agent_files_scanned': 8,
       'positional_pattern': pat.pattern,
       'positional_hits': hits, 'positional_hit_count': len(hits),
       'label_reachability_hits': lab, 'label_hit_count': len(lab)}
print(json.dumps({k: v for k, v in out.items() if k != 'label_reachability_hits'}, indent=1)[:7000])
print('--- label hits ---')
for h in lab:
    print(f"{h['file']}:{h['line']}  {h['text']}")
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2)+'\n')
