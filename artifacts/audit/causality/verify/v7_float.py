"""Is the 2.22e-16 MTD delta really 'cross-run reduction order, not CSV truncation'?

The agent ruled out truncation (.17g round-trips float64 exactly) but not the OTHER
transform on the path: slot_forecasting.schema.probabilities() renormalises on READ,
returning v / math.fsum(values). If fsum != 1.0 exactly, read_submission never returns the
bytes that were written, so a nonzero delta is partly a read-back artefact.
"""
import csv, json, math, sys
from pathlib import Path
ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from slot_forecasting.io import read_submission

P = ROOT / 'artifacts/mtd_pairs_01/validation_lag7.csv'
cols = ['p_0', 'p_1', 'p_2', 'p_5', 'p_20']
raw = {}
for r in csv.DictReader(open(P)):
    raw[r['query_id']] = [float(r[c]) for c in cols]
back = read_submission(P)

n_sum_ne_1 = 0
worst_renorm = 0.0
exact_roundtrip = 0
for k, v in raw.items():
    s = math.fsum(v)
    if s != 1.0:
        n_sum_ne_1 += 1
    d = max(abs(a - b) for a, b in zip(v, back[k]))
    worst_renorm = max(worst_renorm, d)
    if d == 0.0:
        exact_roundtrip += 1
# does .17g round-trip the raw float exactly?
fmt_ok = all(float(format(x, '.17g')) == x for v in raw.values() for x in v)
out = {'rows': len(raw),
       'rows_whose_written_values_do_not_fsum_to_exactly_1': n_sum_ne_1,
       'rows_where_read_submission_returns_the_written_values_unchanged': exact_roundtrip,
       'max_abs_change_introduced_by_read_side_renormalisation': worst_renorm,
       'format_17g_is_an_exact_float64_roundtrip': fmt_ok,
       'finding': ('read_submission() divides by math.fsum(values) (organizer '
                   'slot_forecasting/schema.py:73), so for the rows whose written values do '
                   'not fsum to exactly 1.0 the value read back differs from the value '
                   'written. The agent ruled out .17g truncation but not this read-side '
                   'renormalisation, which is of the same 1-2 ULP magnitude as the delta it '
                   'attributed solely to cross-run reduction order.')}
print(json.dumps(out, indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
