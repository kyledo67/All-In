"""Fit the TabPFN component on the validated phase-two feature table.

Mirrors scripts/phase2_train.py: same examples, same 182 observable features, same
optional 7 portable HMM summary columns, same seeds from artifacts/phase2/protocol.json.
Writes validation predictions in the same place and format as the CatBoost runs so
scripts/tabpfn_evaluate.py can treat TabPFN as one more blend component.

Needs no JAX: the feature tables are rebuilt from the committed cross-fit HMM
checkpoints through slot_model/phase2_tables.py, and that rebuild is gated on exactly
reproducing the committed CatBoost predictions before anything is fitted.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model.phase2_tables import PHASE2, cached_design, features_for
from slot_model.tabpfn_model import TabPFNComponent, checkpoint_sha256
from slot_forecasting.io import read_submission, write_submission

VARIANTS = ('observable', 'hmm')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify_rebuild(tables, tolerance=1e-9):
    """Require the JAX-free tables to reproduce every committed CatBoost prediction.

    This is the gate that lets us trust a table rebuilt outside the original JAX
    environment. A float64 rebuild of the same features through the same frozen trees
    must land within rounding of the committed CSVs.
    """
    from catboost import CatBoostClassifier
    protocol = json.loads((PHASE2 / 'protocol.json').read_text())
    validation = tables['validation']
    ids = validation['ids'].tolist()
    results = {}
    worst = 0.0
    for variant in VARIANTS:
        values = features_for(validation, variant)
        for index in range(len(protocol['catboost'])):
            for seed in protocol['seeds']:
                name = f'cb_{variant}_{index}_{seed}'
                model = CatBoostClassifier()
                model.load_model(str(PHASE2 / 'models' / f'{name}.cbm'))
                if list(model.classes_) != list(range(5)):
                    raise ValueError('Payout class order changed')
                got = model.predict_proba(values, thread_count=2)
                want = np.array([read_submission(PHASE2 / 'predictions' / f'{name}.csv')[q] for q in ids])
                deviation = float(np.max(abs(got - want)))
                results[name] = deviation
                worst = max(worst, deviation)
    if worst > tolerance:
        raise ValueError(f'Rebuilt feature table does not reproduce committed CatBoost predictions '
                         f'(worst deviation {worst:.3e} > {tolerance:.0e})')
    return {'max_abs_deviation': worst, 'tolerance': tolerance, 'per_model': results,
             'conclusion': 'JAX-free feature rebuild reproduces every committed CatBoost prediction'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/tabpfn')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--n-estimators', type=int, default=1,
                        help='1 is what every recorded result used (artifacts/tabpfn/training_runs.json, '
                             'docs/tabpfn-component.md); a rerun at any other value is a different model '
                             'from the one that was scored. Cost per 768-row predict against the 3,134-row '
                             'context: ~118 s at 1, ~305 s at 2, ~950 s at 4 (default: 1).')
    parser.add_argument('--max-train-rows', type=int, default=10000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--skip-verify', action='store_true',
                        help='Only for reruns that already recorded a passing rebuild gate.')
    args = parser.parse_args()

    out = Path(args.out)
    protocol = json.loads((PHASE2 / 'protocol.json').read_text())
    seeds = args.seeds or protocol['seeds']
    # Resolved first, so a missing or swapped checkpoint fails now rather than after
    # the fits, and so the manifest below can name the exact weights that were used.
    checkpoints = checkpoint_sha256()
    print('checkpoints: ' + ', '.join(f'{n}={h[:16]}' for n, h in checkpoints.items()), flush=True)
    start = time.perf_counter()

    tables = cached_design(args.data, out / 'cache')
    print(f"features: {tables['provenance']['train_rows']} train rows, "
          f"{tables['provenance']['validation_rows']} validation rows, "
          f"cache_key={tables['cache_key']} cached={tables['cached']}", flush=True)

    if args.skip_verify:
        verification = {'skipped': True}
    else:
        verification = verify_rebuild(tables)
        print(f"rebuild gate passed: max deviation {verification['max_abs_deviation']:.3e}", flush=True)
    save(out / 'rebuild_verification.json', verification)

    train, validation = tables['train'], tables['validation']
    records = []
    for variant in args.variants:
        x, xv = features_for(train, variant), features_for(validation, variant)
        for seed in seeds:
            name = f'tabpfn_{variant}_{seed}'
            begin = time.perf_counter()
            component = TabPFNComponent(seed=seed, n_estimators=args.n_estimators, variant=variant,
                                        max_train_rows=args.max_train_rows, threads=args.threads)
            component.fit(x, train['y'], episodes=train['ep'], weight=train['weight'],
                          names=(train['names'].tolist() if variant == 'observable' else None))
            fit_seconds = time.perf_counter() - begin
            mark = time.perf_counter()
            probabilities = component.predict_proba(xv)
            inference_seconds = time.perf_counter() - mark
            write_submission(out / 'predictions' / f'{name}.csv',
                             dict(zip(validation['ids'].tolist(), probabilities.tolist())))
            config = component.save(out / 'models' / name)
            record = {'name': name, 'variant': variant, 'seed': seed,
                      'n_estimators': args.n_estimators, 'features': int(x.shape[1]),
                      'context_rows': config['context_rows'], 'subsample': config['subsample'],
                      'weight_handling': config['weight_handling'],
                      'fit_seconds': fit_seconds, 'validation_inference_seconds': inference_seconds,
                      'context_sha256': config['context_sha256'],
                      'probability_min': float(probabilities.min()),
                      'max_sum_error': float(np.max(abs(probabilities.sum(1) - 1)))}
            save(out / 'models' / name / 'diagnostics.json', record)
            records.append(record)
            print(f"{name}: fit {fit_seconds:.1f}s predict {inference_seconds:.1f}s "
                  f"features {x.shape[1]} rows {config['context_rows']}", flush=True)

    save(out / 'training_runs.json',
         {'runs': records, 'seeds': seeds, 'variants': args.variants,
          'n_estimators': args.n_estimators, 'provenance': tables['provenance'],
          'cache_key': tables['cache_key'], 'rebuild_verification': verification,
          'package': 'tabpfn (local); tabpfn_client never imported',
          'checkpoint_dir': str(ROOT / 'artifacts/tabpfn/checkpoints'),
          'checkpoint_sha256': checkpoints,
          'total_seconds': time.perf_counter() - start})
    print(f'done in {time.perf_counter() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
