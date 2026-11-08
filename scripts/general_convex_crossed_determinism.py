"""Byte-identity check for scripts/general_convex_crossed.py across two processes.

The campaign's determinism failure mode is specific: episode_id is a string, so iterating a
Python set of episode ids has an order that depends on CPython's per-process string hash
randomisation, and summing floats in that order previously flipped near-tied discrete grid
choices between runs. This compares two complete runs -- the second with PYTHONHASHSEED=777
-- on the predictions CSVs and on the selection digest, which records every discrete choice
(the X0 pair, every shortlist size, every shrinkage value, the per-type/pooled/no-change
branch) as well as the final points and R. evaluation.json is excluded because it stores a
wall-clock duration.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(d):
    out = {}
    for f in sorted((d / 'predictions').glob('*.csv')):
        out[f'predictions/{f.name}'] = hashlib.sha256(f.read_bytes()).hexdigest()
    sel = d / 'selection_digest.txt'
    if sel.exists():
        out['selection_digest.txt'] = hashlib.sha256(sel.read_bytes()).hexdigest()
    return out


def main():
    a, b = ROOT / sys.argv[1], ROOT / sys.argv[2]
    da, db = digest(a), digest(b)
    keys = sorted(set(da) | set(db))
    bad = 0
    for k in keys:
        ok = da.get(k) == db.get(k) and k in da
        bad += 0 if ok else 1
        print(f"{'OK  ' if ok else 'DIFF'} {k} {str(da.get(k))[:16]} {str(db.get(k))[:16]}")
    print(f"\n{len(keys) - bad}/{len(keys)} byte-identical")
    print('DETERMINISM VERIFIED' if bad == 0 and keys else 'DETERMINISM FAILED')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
