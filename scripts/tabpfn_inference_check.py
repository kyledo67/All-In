"""Confirm the deployed TabPFN artifact reproduces the predictions that were scored.

`scripts/tabpfn_evaluate.py` scored the seed average of the two `hmm`-variant runs.
This loads the saved deployment directory through `TabPFNEnsemble` — the same path
`predict_blend.py` uses at inference — and requires it to reproduce that average
exactly. Without this, the measured component and the shipped component could drift
apart silently.

Runs offline against the cached checkpoint.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model.phase2_tables import cached_design, features_for
from slot_model.tabpfn_model import TabPFNEnsemble
from slot_forecasting.io import read_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/blend_release/model/tabpfn')
    parser.add_argument('--predictions', type=Path, default=ROOT / 'artifacts/tabpfn/predictions')
    parser.add_argument('--variant', default='hmm')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/tabpfn/inference_checks.json')
    args = parser.parse_args()
    start = time.perf_counter()

    tables = cached_design(args.data, ROOT / 'artifacts/tabpfn/cache')
    validation = tables['validation']
    ids = validation['ids'].tolist()
    values = features_for(validation, args.variant)

    ensemble = TabPFNEnsemble.load(args.model)
    got = ensemble.predict_proba(values)
    want = np.mean([np.array([read_submission(args.predictions / f'tabpfn_{args.variant}_{seed}.csv')[q]
                              for q in ids]) for seed in ensemble.seeds], axis=0)
    # The evaluated average is renormalized by `simplex`, so compare after the same step.
    want = np.clip(want, 1e-6, 1 - 1e-6)
    want = want / want.sum(axis=1, keepdims=True)
    deviation = float(np.max(abs(got - want)))

    report = {'model': str(args.model), 'variant': args.variant, 'seeds': ensemble.seeds,
              'components': len(ensemble.components), 'rows': int(len(got)),
              'max_abs_deviation': deviation,
              'reproduces_evaluated_predictions': deviation < 1e-9,
              'all_finite': bool(np.isfinite(got).all()),
              'min_probability': float(got.min()),
              'max_sum_error': float(np.max(abs(got.sum(1) - 1))),
              'payout_order': [0, 1, 2, 5, 20],
              'offline': os.environ.get('HF_HUB_OFFLINE') == '1',
              'seconds': time.perf_counter() - start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report['reproduces_evaluated_predictions']:
        raise SystemExit(f'Deployed artifact deviates from the evaluated predictions by {deviation:.3e}')
    print('DEPLOYED ARTIFACT MATCHES THE EVALUATED PREDICTIONS')


if __name__ == '__main__':
    main()
