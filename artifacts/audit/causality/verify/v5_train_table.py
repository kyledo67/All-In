"""Containment on the TRAINING table, which no prior audit checked.

Every prior containment check ran on the inference examples (768 validation / 1536 test
rows, all real queries). The TabPFN context and the two_head/CatBoost fitting rows are
3134 TRAIN rows built with examples(..., historical=True): up to 8 historical pseudo-
queries per machine whose LABEL is the machine's own payout at the pseudo-cutoff. Three
things must hold and none was measured:
  C1 every pseudo-row's own history is strictly below its own pseudo-cutoff
  C2 a pseudo-row's own label tick is NOT in its own feature history (self-label leak)
  C3 a pseudo-row's neighbour window excludes the simultaneous labels of the other 7
     machines at the same pseudo-cutoff (cross-label leak among simultaneous rows)
"""
import json, sys
from pathlib import Path
ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from slot_model.features import causal_index
from slot_model.phase2_features import examples, fit_prior, matrix, observable
from slot_forecasting.io import load_inputs, read_labels

DATA = ROOT / 'main-participants-60-40/public'
tq, th = load_inputs(DATA / 'train')
tidx = causal_index(tq, th)
labels = read_labels(DATA / 'train/labels.csv', tq)
es = examples(tq, tidx, labels, historical=True)

c1 = [e['q'].query_id for e in es if e['h'] and max(e['h']) >= e['q'].cutoff]
c2 = [e['q'].query_id for e in es if e['q'].cutoff in e['h']]
import re as _re
pseudo = [e for e in es if _re.search(r'-t\d+$', e['q'].query_id)]
real = [e for e in es if not _re.search(r'-t\d+$', e['q'].query_id)]
# C2b: for a pseudo-row the stored label must equal the machine's observed payout at the
# pseudo-cutoff, and that tick must be absent from the row's own history
lab_ok, lab_bad = 0, []
for e in pseudo:
    q = e['q']
    want = tidx[(q.episode_id, q.machine_id)][q.cutoff]
    if e['y'] != want:
        lab_bad.append(q.query_id)
    else:
        lab_ok += 1
# C3: neighbour window used by observable() for a pseudo-row must hold no tick >= cutoff
c3 = []
for e in pseudo[:400]:
    q = e['q']
    for (ep, m), seq in tidx.items():
        if ep != q.episode_id or m == q.machine_id:
            continue
        if any(t >= q.cutoff for t in seq if q.cutoff - 16 <= t):
            c3.append(q.query_id)
            break
# C4: do pseudo-rows carry signal? must be 0 and signal_known False
sig = {'nonzero_signal': sum(1 for e in pseudo if e['q'].signal != 0),
       'signal_known_true': sum(1 for e in pseudo if e['signal_known'])}
out = {'train_examples': len(es), 'pseudo_rows': len(pseudo), 'real_rows': len(real),
       'C1_history_at_or_after_own_cutoff': {'failures': len(c1), 'sample': c1[:5]},
       'C2_own_label_tick_inside_own_history': {'failures': len(c2), 'sample': c2[:5]},
       'C2b_pseudo_label_matches_observed_payout_at_pseudo_cutoff':
           {'ok': lab_ok, 'failures': len(lab_bad), 'sample': lab_bad[:5]},
       'C3_neighbour_window_holds_a_tick_at_or_after_pseudo_cutoff':
           {'pseudo_rows_checked': min(400, len(pseudo)), 'failures': len(c3), 'sample': c3[:5]},
       'C4_pseudo_rows_signal': sig,
       'verdict': 'clean' if not (c1 or c2 or lab_bad or c3 or sig['nonzero_signal']
                                  or sig['signal_known_true']) else 'FAILURES PRESENT'}
print(json.dumps(out, indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
