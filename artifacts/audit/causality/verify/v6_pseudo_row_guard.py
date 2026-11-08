"""The one place where post-cutoff data REALLY exists in the index at build time.

For an inference example, causal_index has already deleted every tick >= cutoff, so the
prior agent had to invent a counterfactual stream to test the guards. For the TRAINING
table it is not counterfactual: examples(..., historical=True) builds up to 8 pseudo-rows
per machine at cutoffs 192, 200, ... < q.cutoff, and the index legitimately holds real
observed ticks AT and AFTER each pseudo-cutoff (up to q.cutoff-1). The ONLY thing stopping
a pseudo-row from reading its own future is the `t < q.cutoff` guard inside
sample_history() / observable(). No causal_index layer protects it.

Test: overwrite the payout at ONE tick T on ALL 8 machines of an episode.
  - every pseudo-row with cutoff <= T must have a BITWISE IDENTICAL feature row
    (its own history and its neighbour window are entirely below its own cutoff <= T)
  - rows with cutoff > T must change -> built-in power control
This is run on the released train split, so it is a live property, not a counterfactual.
"""
import json, re, sys, time
from pathlib import Path
ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from slot_model.features import PAYOUTS, causal_index
from slot_model.phase2_features import examples, fit_prior, matrix
from slot_model.phase2_tables import PHASE2
from slot_forecasting.io import load_inputs, read_labels

DATA = ROOT / 'main-participants-60-40/public'
t0 = time.perf_counter()
folds = json.loads((PHASE2 / 'folds.json').read_text())
tq, th = load_inputs(DATA / 'train')
tidx0 = causal_index(tq, th)
labels = read_labels(DATA / 'train/labels.csv', tq)
prior = fit_prior(tq, tidx0, set(folds['train']))
priors = {q.episode_id: prior for q in tq}

es0 = examples(tq, tidx0, labels, historical=True)
X0, names = matrix(es0, tidx0, priors)
key = [(e['q'].query_id, e['q'].cutoff) for e in es0]

eps = sorted({q.episode_id for q in tq})[:3]
results = []
for epi in eps:
    machines = [q for q in tq if q.episode_id == epi]
    # a tick that every machine of the episode observes and that sits above the lowest
    # pseudo-cutoff but below the real cutoff, so some rows are below it and some above
    cutoffs = sorted({q.cutoff for q in machines})
    T = 208
    common = all(T in tidx0[(epi, q.machine_id)] for q in machines)
    if not common:
        continue
    idx1 = {k: dict(v) for k, v in tidx0.items()}
    changed_ticks = 0
    for q in machines:
        old = idx1[(epi, q.machine_id)][T]
        idx1[(epi, q.machine_id)][T] = 4 if old != 4 else 0   # force a different payout
        changed_ticks += 1
    es1 = examples(tq, idx1, labels, historical=True)
    assert [(e['q'].query_id, e['q'].cutoff) for e in es1] == key, 'row set changed'
    X1, _ = matrix(es1, idx1, priors)
    below, below_bad, above, above_changed = 0, [], 0, 0
    for i, (qid, cut) in enumerate(key):
        if not qid.startswith(epi):
            continue
        same = np.array_equal(X0[i], X1[i])
        if cut <= T:
            below += 1
            if not same:
                below_bad.append({'query_id': qid, 'cutoff': cut,
                                  'max_abs_delta': float(np.max(np.abs(X0[i] - X1[i]))),
                                  'columns': [names[j] for j in np.where(X0[i] != X1[i])[0]][:8]})
        else:
            above += 1
            above_changed += (not same)
    results.append({'episode': epi, 'overwritten_tick': T, 'machines_touched': changed_ticks,
                    'rows_with_cutoff_le_T': below,
                    'rows_with_cutoff_le_T_that_changed': len(below_bad),
                    'violations': below_bad[:5],
                    'rows_with_cutoff_gt_T': above,
                    'rows_with_cutoff_gt_T_that_changed': above_changed,
                    'power_control_ok': above > 0 and above_changed == above})
    print(f"{epi}: tick {T} overwritten on {changed_ticks} machines -> "
          f"{below - len(below_bad)}/{below} rows at cutoff<=T bitwise identical; "
          f"{above_changed}/{above} rows at cutoff>T changed", flush=True)

out = {'train_examples': len(es0), 'observable_columns': len(names),
       'episodes_probed': results,
       'guard_holds': all(not r['violations'] for r in results),
       'power_controls_fired': all(r['power_control_ok'] for r in results),
       'why_this_matters': ('for a pseudo-row the post-cutoff ticks are REALLY in the index '
                            '(they are released observations below the real query cutoff), so '
                            'this exercises the t < q.cutoff guards on live data rather than on '
                            'an injected counterfactual stream'),
       'seconds': time.perf_counter() - t0}
print(json.dumps({k: v for k, v in out.items() if k != 'episodes_probed'}, indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
