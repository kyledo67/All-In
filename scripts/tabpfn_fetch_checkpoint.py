"""One-time fetch of the TabPFN v2 classifier checkpoint into the repository.

The 28 MB weight file is not committed, so a fresh clone runs this once, with network
access, before any offline prediction. Everything afterwards sets HF_HUB_OFFLINE=1 and
reads only the cached file, which is what the competition's offline rule requires.

Built with PriorLabs-TabPFN. These are upstream weights under the Prior Labs License
v1.1; the license copy at research/tabpfn/LICENSE must travel with them wherever they
are redistributed, including inside a submission archive. See THIRD_PARTY_NOTICES.md.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/tabpfn/checkpoints'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--check-only', action='store_true',
                        help='Report whether a usable checkpoint is already present.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob('*.ckpt'))

    if args.check_only or existing:
        report = {'directory': str(args.out), 'checkpoints': [p.name for p in existing],
                  'bytes': {p.name: p.stat().st_size for p in existing},
                  'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in existing},
                  'offline_ready': bool(existing),
                  'license': 'Prior Labs License v1.1; copy at research/tabpfn/LICENSE',
                  'attribution': 'Built with PriorLabs-TabPFN'}
        print(json.dumps(report, indent=2))
        if args.check_only:
            return 0 if existing else 1
        print('Checkpoint already present; nothing downloaded.')
        return 0

    os.environ['TABPFN_MODEL_CACHE_DIR'] = str(args.out)
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
    import numpy as np
    from tabpfn import TabPFNClassifier
    start = time.perf_counter()
    # The smallest possible fit; the point is the checkpoint download, not the model.
    TabPFNClassifier(device='cpu', random_state=0, n_estimators=1).fit(
        np.arange(20, dtype=np.float64).reshape(10, 2), np.arange(10) % 5)
    found = sorted(args.out.glob('*.ckpt'))
    if not found:
        raise SystemExit('No checkpoint appeared; check network access and the tabpfn version.')
    print(json.dumps({'downloaded': [p.name for p in found],
                      'bytes': {p.name: p.stat().st_size for p in found},
                      'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in found},
                      'seconds': time.perf_counter() - start,
                      'attribution': 'Built with PriorLabs-TabPFN'}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
