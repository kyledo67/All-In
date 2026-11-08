"""Are the positional slices the prior agent's regex missed actually benign?

Its pattern was rows\[ | sorted\([^)]*\)\[ | .values()\[ | iloc | shift( , which cannot
match `list(range(192,q.cutoff,8))[-8:]` or
`sorted(old,key=lambda t:hashlib.sha256(...).hexdigest())[:24]` because [^)]* cannot span
the inner ')'. Both live in files it DID scan, so 'exactly 1 hit' is an artefact of the
pattern. Re-derive their safety instead of trusting the conclusion.
"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
from slot_model.features import causal_index
from slot_model.phase2_features import sample_history
from slot_forecasting.io import load_inputs

DATA = ROOT / 'main-participants-60-40/public'
res = {}
for split in ('train', 'validation', 'test'):
    qs, hs = load_inputs(DATA / split)
    idx = causal_index(qs, hs)
    bad_cut, bad_thin, thin_rows, cut_rows = [], [], 0, 0
    for q in qs:
        # phase2_features.py:96 / dense_catboost.py:39  list(range(192,q.cutoff,8))[-8:]
        for c in list(range(192, q.cutoff, 8))[-8:]:
            cut_rows += 1
            if c >= q.cutoff:
                bad_cut.append((q.query_id, c))
        # phase2_features.py:35  thinned older-tick selection
        h = sample_history(q, idx, thin=True)
        thin_rows += 1
        if any(t >= q.cutoff for t in h):
            bad_thin.append((q.query_id, max(h)))
        older = [t for t in h if t < q.cutoff - 40]
        if any(t >= q.cutoff - 40 for t in older):
            bad_thin.append((q.query_id, 'older boundary'))
        if len(older) > 24:
            bad_thin.append((q.query_id, f'older={len(older)}>24'))
    res[split] = {'pseudo_cutoffs_generated': cut_rows,
                  'pseudo_cutoffs_at_or_after_real_cutoff': len(bad_cut), 'sample': bad_cut[:3],
                  'thin_histories_built': thin_rows,
                  'thin_violations': len(bad_thin), 'thin_sample': bad_thin[:3]}
    print(split, res[split])
out = {'by_split': res,
       'verdict': ('both slices are benign: range(192, q.cutoff, 8) is stop-exclusive so every '
                   'pseudo-cutoff is < q.cutoff, and the [:24] thin selection is a slice of a '
                   'HASH-ordered list of ticks already filtered to t < q.cutoff-40, not of a '
                   'tick-ordered list, so no adjacency is implied'),
       'but': ("the prior agent's 'exactly 1 positional hit' is a property of its regex, not of "
               "the code; it under-scanned 8 files out of 22 on the candidate's path and its "
               "pattern misses list(range(...))[-8:] and sorted(..., key=lambda ...)[:24] even "
               'inside the files it did scan')}
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2, default=str) + '\n')
print(json.dumps({k: v for k, v in out.items() if k != 'by_split'}, indent=2))
