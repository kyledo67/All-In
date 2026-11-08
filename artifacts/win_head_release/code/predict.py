"""Offline prediction entry point. One command, no network, no training.

    python predict.py --data /data --out /output/submission.csv

Environment variables are pinned before numpy/torch are imported so the organizer's
two-core budget is respected and no library tries to write a cache into the read-only
/submission mount.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get('TMPDIR') or '/tmp')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
# /submission is read-only; every cache a dependency might create goes to scratch.
os.environ.setdefault('MPLCONFIGDIR', str(SCRATCH / 'winhead-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(SCRATCH / 'winhead-cache'))
os.environ.setdefault('HF_HOME', str(SCRATCH / 'winhead-hf'))
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
for name in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'HF_HOME'):
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from slot_model.win_head_infer import predict


def main():
    parser = argparse.ArgumentParser(description='Win-head candidate forecaster')
    parser.add_argument('--data', type=Path, required=True,
                        help="Round data folder holding the split subdirectories")
    parser.add_argument('--out', type=Path, required=True, help='Submission CSV to write')
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    parser.add_argument('--model', type=Path, default=ROOT / 'model')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--diagnostics', type=Path,
                        help='Optional JSON dump of q and the reference block')
    args = parser.parse_args()
    report = predict(args.data, args.out, args.model, split=args.split,
                     threads=args.threads, diagnostics=args.diagnostics)
    report['out'] = str(args.out)
    print(json.dumps({k: v for k, v in report.items()
                      if k != 'win_head_fallback_queries'}, indent=2, sort_keys=True))
    if report['win_head_fallbacks']:
        print(f"win-head query-basis fallbacks: {report['win_head_fallbacks']}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
