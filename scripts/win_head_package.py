"""Stage, build and checksum the two win-head candidate submission packages.

PRIMARY  -> artifacts/win_head_release/        q from the win head, r from the frozen
                                               three-component blend (TabPFN + MTD +
                                               two-head CatBoost). Ships TabPFN weights.
FALLBACK -> artifacts/win_head_release_nofx/   same q, r from the two-head CatBoost
                                               alone. No torch, no TabPFN, no checkpoint.

Nothing under organizer/, artifacts/blend_release/, artifacts/final/ or
artifacts/active_champion.json is written. This script only reads them.

Steps performed, in order:
  1. verify the extracted inference feature builder against artifacts/two_head_01/validation.csv
  2. stage each package's code and frozen model files
  3. run the STAGED package on validation and on test, in-repo
  4. write checksums.json, requirements.txt, reproduce.json, README.md
  5. build entry.zip through the organizer's own slot_forecasting.submission_contract
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))

TABPFN_CKPT = 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'
SEEDS = (20261108, 20261109)

# Mean of the four outer-fold simplex fits recorded in artifacts/audit/recompute.json.
# NOT tuned on anything further. frozen_blend is dropped: its mean weight is
# 5.049528681541388e-17 and the remaining three already sum to exactly 1.0 in float64,
# so dropping it changes no weight at all (measured delta 0.0 for each of the three).
DEPLOYMENT_WEIGHTS = {
    'tabpfn_raw': 0.4829584454464017,
    'mtd_lag7': 0.19998414057692182,
    'two_head': 0.31705741397667647,
}

COMMON_SLOT_MODEL = ('__init__.py', 'features.py', 'phase2_features.py',
                     'two_head_infer.py', 'win_head.py', 'win_head_infer.py',
                     'pair_dict.py')
PRIMARY_SLOT_MODEL = ('portable.py', 'phase2_predict.py', 'mtd.py', 'mtd_pairs.py',
                      'tabpfn_model.py')
KIT_MODULES = ('__init__.py', 'schema.py', 'io.py', 'baseline.py', 'transition.py')

PREDICT = '''"""Offline prediction entry point. One command, no network, no training.

    python predict.py --data /data --out /output/submission.csv

Environment variables are pinned before numpy or any native library is imported, so
the organizer's two-core budget is respected and no library tries to write a cache into the read-only
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
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).is_symlink():
        raise ValueError(f'{source} is a symlink; code ZIPs cannot contain symlinks')
    shutil.copyfile(source, target)
    return target


def stage(name, code, win_model='dense'):
    """Create one package's source tree and return its description.

    `win_model` selects the LOCAL win model: 'dense' is the measured 37-coefficient
    hierarchical model (`local_interactions_averaged`, the 40.705473 / 40.589853
    candidate), 'pair_dict' is the compact pair-dictionary alternative. The pooled
    population coefficients and the half-life averaging are identical either way.
    """
    if win_model not in ('dense', 'pair_dict'):
        raise ValueError(f'Unknown win model {win_model!r}')
    primary = name == 'primary'
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True)

    modules = COMMON_SLOT_MODEL + (PRIMARY_SLOT_MODEL if primary else ())
    for module in modules:
        copy(ROOT / 'slot_model' / module, code / 'slot_model' / module)
    for module in KIT_MODULES:
        copy(ROOT / 'organizer/slot_forecasting' / module, code / 'slot_forecasting' / module)

    model = code / 'model'
    copy(ROOT / 'artifacts/blend_release/model/population_prior.json', model / 'population_prior.json')
    copy(ROOT / 'artifacts/blend_release/model/feature_schema.json', model / 'feature_schema.json')
    copy(ROOT / 'artifacts/win_head/population_averaged.json',
         model / 'win_head/population_averaged.json')
    copy(ROOT / 'artifacts/win_head/pair_dict_deploy.json',
         model / 'win_head/pair_dict_deploy.json')
    for seed in SEEDS:
        for head in ('win', 'size'):
            copy(ROOT / f'artifacts/two_head_01/{head}_{seed}.cbm',
                 model / f'two_head/{head}_{seed}.cbm')

    deployment = {
        'candidate': 'win_head_q_times_three_component_r' if primary
                     else 'win_head_q_times_two_head_r',
        'reference_variant': 'three_component' if primary else 'two_head',
        'alpha': 1.0,
        'win_head': 'population_averaged.json',
        'payout_order': [0, 1, 2, 5, 20],
        'local_win_model': win_model,
    }
    if win_model == 'pair_dict':
        deployment['pair_dict'] = 'pair_dict_deploy.json'
        deployment['pair_select'] = 'hard'
    if primary:
        copy(ROOT / 'artifacts/blend_release/model/population.json', model / 'population.json')
        copy(ROOT / 'artifacts/mtd_01/change.json', model / 'mtd/change.json')
        for seed in SEEDS:
            for part in ('config.json', 'context.npz'):
                copy(ROOT / f'artifacts/blend_release/model/tabpfn/seed_{seed}/{part}',
                     model / f'tabpfn/seed_{seed}/{part}')
        copy(ROOT / 'artifacts/tabpfn/checkpoints' / TABPFN_CKPT,
             model / 'tabpfn_checkpoints' / TABPFN_CKPT)
        deployment['weights'] = dict(DEPLOYMENT_WEIGHTS)
        deployment['weights_provenance'] = (
            'Arithmetic mean of the four outer-fold simplex fits in '
            'artifacts/audit/recompute.json. Not tuned on anything further. The '
            'frozen_blend component was dropped because its mean weight is 5.05e-17 and '
            'it has no test-time analogue (its measured column was a held-fold '
            'prediction); the remaining three already sum to exactly 1.0 in float64.')
        copy(ROOT / 'research/tabpfn/LICENSE', code / 'licenses/PRIORLABS-TABPFN-LICENSE')
    copy(ROOT / 'artifacts/blend_release/CATBOOST_LICENSE', code / 'licenses/CATBOOST-LICENSE')

    (model / 'deployment.json').write_text(json.dumps(deployment, indent=2, sort_keys=True) + '\n')
    (code / 'predict.py').write_text(PREDICT)
    (code / 'reproduce.json').write_text(json.dumps(
        {'image': 'python',
         'command': ['python', 'predict.py', '--data', '/data', '--out', '/output/submission.csv']},
        indent=2) + '\n')
    return deployment


def requirements(primary):
    """Pinned inference closure. Union of the two frozen locks already in the repo."""
    blend = [l.strip() for l in (ROOT / 'artifacts/blend_release/requirements.txt')
             .read_text().splitlines() if l.strip() and not l.startswith('#')]
    if not primary:
        return blend, {}
    tabpfn = [l.split('#')[0].strip() for l in (ROOT / 'requirements-tabpfn.lock')
              .read_text().splitlines()]
    tabpfn = [l for l in tabpfn if l and '==' in l]
    merged, notes = {}, {}
    for source, lines in (('tabpfn', tabpfn), ('blend', blend)):
        for line in lines:
            package, version = line.split('==')
            key = package.lower().replace('_', '-')
            if key in merged and merged[key][1] != version:
                # requirements-tabpfn.lock documents these two downgrades explicitly.
                notes[package] = f'{merged[key][1]} kept over {version} (from {source})'
                continue
            merged.setdefault(key, (package, version))
    return [f'{p}=={v}' for p, v in sorted(merged.values(), key=lambda t: t[0].lower())], notes


def run_staged(code, data, split, out, diagnostics):
    """Run the staged package exactly as the organizer would, timed, 2 threads."""
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       VECLIB_MAXIMUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                       PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, 'predict.py', '--data', str(data), '--out', str(out),
               '--split', split, '--diagnostics', str(diagnostics)]
    start = time.perf_counter()
    done = subprocess.run(command, cwd=code, env=environment, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if done.returncode != 0:
        raise RuntimeError(f'staged run failed ({done.returncode}):\n{done.stdout}\n{done.stderr}')
    return {'seconds': seconds, 'command': ' '.join(command), 'cwd': str(code),
            'report': json.loads(done.stdout[done.stdout.index('{'):]),
            'stderr_tail': done.stderr.strip().splitlines()[-3:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    parser.add_argument('--only', choices=['primary', 'fallback'])
    parser.add_argument('--win-model', choices=['dense', 'pair_dict'], default='dense',
                        help="Local win model to deploy. 'dense' (default) is the measured "
                             "37-coefficient local_interactions_averaged head whose "
                             "development scores the READMEs quote; 'pair_dict' deploys the "
                             "compact pair-dictionary alternative instead.")
    parser.add_argument('--reuse-runs', action='store_true',
                        help='Re-stage, re-render README and rebuild the ZIP, reusing the '
                             'already-measured runs recorded in packaging.json instead of '
                             'running the expensive prediction again. The staged code is '
                             'byte-identical, so the measurements still describe it.')
    args = parser.parse_args()
    import numpy as np
    from slot_model.features import causal_index
    from slot_model.two_head_infer import feature_table, predict_proba
    from slot_forecasting.io import load_inputs, read_submission
    from slot_forecasting.submission_contract import package_code

    # ---- 1. the extracted feature builder must reproduce the measured component ----
    prior = json.loads((ROOT / 'artifacts/blend_release/model/population_prior.json').read_text())
    schema = json.loads((ROOT / 'artifacts/blend_release/model/feature_schema.json').read_text())['observable']
    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    rows, x, names = feature_table(vq, index, prior, schema=schema)
    rebuilt = predict_proba(x, ROOT / 'artifacts/two_head_01')
    saved = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    ids = [e['q'].query_id for e in rows]
    two_head_deviation = float(np.max(np.abs(rebuilt - np.array([saved[q] for q in ids]))))
    print(f'two-head rebuild max abs deviation vs artifacts/two_head_01/validation.csv: '
          f'{two_head_deviation:.6e}  (rows {x.shape[0]}, columns {x.shape[1]})', flush=True)

    measured_q = read_submission(ROOT / 'artifacts/win_head/predictions/oof_win_head.csv')

    packages = [('primary', ROOT / 'artifacts/win_head_release'),
                ('fallback', ROOT / 'artifacts/win_head_release_nofx')]
    if args.only:
        packages = [p for p in packages if p[0] == args.only]

    for name, out in packages:
        print(f'\n===== {name}: {out} =====', flush=True)
        code = out / 'code'
        out.mkdir(parents=True, exist_ok=True)
        deployment = stage(name, code, win_model=args.win_model)
        pins, conflicts = requirements(name == 'primary')
        header = ('# Pinned inference closure. Base environment is Python 3.13 + stdlib;\n'
                  '# organizers prepare these before the offline prediction run.\n')
        if conflicts:
            header += ''.join(f'# resolved: {k} {v}\n' for k, v in sorted(conflicts.items()))
        (code / 'requirements.txt').write_text(header + '\n'.join(pins) + '\n')

        runs = {}
        previous = out / 'packaging.json'
        if args.reuse_runs and previous.exists():
            stored = json.loads(previous.read_text())['csv']
            for split, entry in stored.items():
                runs[split] = {'seconds': entry['seconds'], 'report': entry['report'],
                               'reused_from': str(previous.relative_to(ROOT))}
                print(f'  {split}: reusing measured run '
                      f'({entry["seconds"]:.1f}s, {entry["report"]["queries"]} queries)', flush=True)
        else:
            for split in ('validation', 'test'):
                runs[split] = run_staged(code, args.data.resolve(), split,
                                         out / f'{split}_submission.csv',
                                         out / f'{split}_diagnostics.json')
                print(f'  {split}: {runs[split]["seconds"]:.1f}s  '
                      f'{runs[split]["report"]["queries"]} queries  '
                      f'fallbacks {runs[split]["report"]["win_head_fallbacks"]}', flush=True)

        mine = read_submission(out / 'validation_submission.csv')
        order = sorted(measured_q)
        q_deviation = float(np.max(np.abs(
            np.array([1 - measured_q[i][0] for i in order])
            - np.array([1 - mine[i][0] for i in order]))))
        print(f'  win-head q max abs deviation vs artifacts/win_head/predictions/'
              f'oof_win_head.csv: {q_deviation:.6e}', flush=True)

        for cache in sorted(code.rglob('__pycache__'), reverse=True):
            shutil.rmtree(cache, ignore_errors=True)
        files = sorted(p for p in code.rglob('*') if p.is_file())
        checksums = {str(p.relative_to(code)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                     for p in files}
        (code / 'checksums.json').write_text(json.dumps(
            {'note': 'sha256 of every shipped file except README.md and checksums.json itself',
             'files': checksums}, indent=2, sort_keys=True) + '\n')

        readme = render_readme(name, deployment, runs, two_head_deviation, q_deviation,
                               out, code, pins, conflicts)
        (code / 'README.md').write_text(readme)

        archive = out / 'entry.zip'
        if archive.exists():
            archive.unlink()
        package_code(code, archive)
        staged = sorted(p for p in code.rglob('*') if p.is_file())
        manifest = {
            'package': name,
            'local_win_model': deployment['local_win_model'],
            'deployment': deployment,
            'entry_zip_bytes': archive.stat().st_size,
            'entry_zip_sha256': digest(archive),
            'staged_file_count': len(staged),
            'staged_bytes': sum(p.stat().st_size for p in staged),
            'event_zip_limit_bytes': 100000000,
            'kit_enforced_zip_limit_bytes': 134217728,
            'within_event_zip_limit': archive.stat().st_size <= 100000000,
            'csv': {split: {'bytes': (out / f'{split}_submission.csv').stat().st_size,
                            'sha256': digest(out / f'{split}_submission.csv'),
                            'seconds': runs[split]['seconds'],
                            'report': runs[split]['report']} for split in runs},
            'event_csv_limit_bytes': 10000000,
            'two_head_rebuild_max_abs_deviation': two_head_deviation,
            'win_head_q_max_abs_deviation_vs_measured': q_deviation,
            'requirements': pins,
            'requirements_conflicts_resolved': conflicts,
            'readme_bytes': len(readme.encode()),
            'reproduce_json_bytes': (code / 'reproduce.json').stat().st_size,
        }
        if name == 'primary':
            from slot_model.tabpfn_model import checkpoint_sha256
            manifest['tabpfn_checkpoint_sha256'] = checkpoint_sha256(
                code / 'model/tabpfn_checkpoints')
        (out / 'packaging.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        print(f'  entry.zip {archive.stat().st_size:,} bytes '
              f'({100 * archive.stat().st_size / 100000000:.1f}% of the 100,000,000-byte cap), '
              f'{len(staged)} files', flush=True)


def render_readme(name, deployment, runs, two_head_deviation, q_deviation, out, code,
                  pins, conflicts):
    primary = name == 'primary'
    title = 'TabPFN-WinHead-ThreeComponent' if primary else 'WinHead-TwoHead'
    metadata = json.loads((ROOT / 'main-participants-60-40/public/metadata.json').read_text())
    csv_sha = digest(out / 'test_submission.csv')
    weights = deployment.get('weights', {})
    weight_block = '\n'.join(f'  - `{k}` = `{v!r}`' for k, v in sorted(weights.items()))
    dense = deployment.get('local_win_model', 'dense') == 'dense'
    development = (('40.705473 / 60 (R = 0.474003)' if primary
                    else '40.589853 / 60 (R = 0.478200)') if dense else
                   'not quoted here - this archive deploys the compact pair-dictionary '
                   'local win model, whose own development score is recorded in '
                   '`artifacts/win_head/pair_dict_deploy.json` of the development '
                   'repository, not the dense-model score')
    reference = ('a frozen convex blend of three components (raw two-seed TabPFN(hmm) '
                 'average, PairMTD lag-7 over the saved changepoint MTD, and the two-head '
                 'CatBoost)' if primary else 'the two-head CatBoost alone')

    extra = []
    if primary:
        extra = [
            '- `model/population.json` - frozen HMM whose 7 summary columns extend the '
            'observable table for TabPFN',
            '- `model/mtd/change.json` - saved changepoint MTD, the base of the lag-7 '
            'pair-correction model',
            '- `model/tabpfn/seed_*/{context.npz,config.json}` - saved TabPFN context '
            'tables, seeds and context hashes',
            f'- `model/tabpfn_checkpoints/{TABPFN_CKPT}` - TabPFN v2 classifier weights',
        ]
    extra_bullets = '\n'.join(extra)

    if primary:
        weights_section = (
            'The three-component `r` is a FIXED convex combination, recorded here '
            'explicitly because test episodes carry no fold assignment:\n\n'
            + weight_block + '\n\n'
            'These are the **arithmetic mean of the four outer-fold simplex fits** '
            'recorded in `artifacts/audit/recompute.json` of the development repository, '
            'and they were **NOT tuned on anything further** - no search, no validation '
            're-fit, no per-condition adjustment. The same vector is stored in '
            '`model/deployment.json`, which is where the running code reads it from.\n\n'
            'The fourth component of that fit, the earlier frozen four-recipe blend, had '
            'mean weight `5.049528681541388e-17` and has no test-time analogue (its '
            'measured column was a held-fold prediction), so it is dropped. The remaining '
            'three already sum to exactly `1.0` in float64, so dropping it changes each '
            'surviving weight by exactly `0.0`.\n\n'
            '`alpha = 1.0` is frozen too: the development fit put alpha at 1.0000 in all '
            'four outer folds, i.e. the win head replaces rather than blends with the '
            'reference win mass.')
    else:
        weights_section = (
            'There are no blend weights in this package. `r` comes from the two-head '
            'CatBoost alone; the two seeds are averaged after each seed\'s five-vector is '
            'composed, which is the order the measured component used. `alpha = 1.0` is '
            'frozen: the development fit put alpha at 1.0000 in all four outer folds.')

    hardware = ('CPU only. Two cores and 16 GiB RAM are sufficient; no GPU is used or '
                'required (`"device": "cpu"` in every saved TabPFN config).' if primary else
                'CPU only. Two cores and 16 GiB RAM are sufficient; no GPU is used or '
                'required. `torch` is not a dependency of this package.')
    seeds_note = ('; each saved TabPFN component carries its own `random_state` (20261108, '
                  '20261109), `n_estimators = 1`, and runs on CPU with a fixed thread count'
                  if primary else '')
    deployment_delta = (
        'The deployment configuration in this package differs from the measured '
        'configuration in exactly one way: the three-component `r` uses the '
        'mean-of-folds weight vector above instead of each fold\'s own fitted weights, '
        'because test episodes have no fold assignment.' if primary else
        'The deployment configuration in this package is the measured configuration: '
        '`r` comes from the two-head CatBoost, which has no fold-dependent quantity, and '
        '`alpha` is the frozen 1.0 that the development fit chose in every fold.')

    if primary:
        license_section = f'''### Built with PriorLabs-TabPFN

This package distributes the TabPFN v2 classifier model weights
(`model/tabpfn_checkpoints/{TABPFN_CKPT}`). TabPFN is **not** Apache-2.0: it is
licensed under the **Prior Labs License Version 1.1 (May 2025)**, whose Paragraph 10
requires that a distributor of the source or model weights (A) provide a copy of the
license and (B) prominently display "Built with PriorLabs-TabPFN". A verbatim copy of
that license is shipped at `licenses/PRIORLABS-TABPFN-LICENSE`, and the heading above
satisfies (B). Paragraph 10 also asks that a distributed AI model built from TabPFN
source, weights or outputs carry "TabPFN" at the beginning of its name, so this
candidate is named **TabPFN-WinHead-ThreeComponent**.

TabPFN upstream: https://github.com/priorlabs/tabpfn, pinned to `tabpfn==2.2.1`.
Only the local `tabpfn` package is used; `tabpfn-client` would post features to a
remote API and is never imported (the inference code raises if it is present).

### CatBoost'''
    else:
        license_section = '''### No TabPFN in this package

This package ships **no** TabPFN code, weights or outputs, and therefore makes **no**
"Built with PriorLabs-TabPFN" claim and no Prior Labs License obligations apply to it.
`torch`, `tabpfn` and the 29 MB v2 checkpoint are all absent - `requirements.txt`
contains neither, and nothing under `model/` is a TabPFN artifact.

`slot_model/win_head_infer.py` is the single inference module shared with the
three-component package, so it still *contains* a `three_component` branch. That branch
is unreachable here: `model/deployment.json` selects `two_head`, and the modules the
branch would import (`tabpfn_model.py`, `mtd.py`, `mtd_pairs.py`, `phase2_predict.py`)
are deliberately not shipped in this archive, so editing the deployment file would raise
`ModuleNotFoundError` rather than quietly reach for a network.

### CatBoost'''

    return f'''# House Edge submission - {title}

## Entry

- Team ID: [fill in before upload]
- Round: main
- Season ID: {metadata['season_id']}
- Forecast file: `submission.csv`, written by the command below to `/output/submission.csv`
- Forecast SHA-256: `{csv_sha}`
  (the development copy is `{(out / 'test_submission.csv').relative_to(ROOT)}`;
  `check_submission.py` reprints this hash)
- Code version: repository commit `{commit()}` plus the uncommitted win-head work
- Environment: `"image": "python"` in `reproduce.json`; extra packages pinned in
  `requirements.txt` at this archive's root

## What this entry predicts

Five payout probabilities per query, in the order `p_0,p_1,p_2,p_5,p_20`, assembled as

    [1 - q,  q*r_1,  q*r_2,  q*r_5,  q*r_20]

- `q` is the total win probability from a hierarchical logistic autoregressive win head.
  For machine m at tick t it uses `s_l = 2*1(Y_(t-l) > 0) - 1` over ACTUAL ticks
  `l = 1..8` and the predeclared 37-dimensional basis `[1, s_1..s_8, {{s_i s_j : i<j}}]`,
  with `logit q = beta_type . phi + u_m . phi`. `beta` is pooled within machine type and
  was fitted on training observations only. `u_m` are regularized machine-specific
  deviations, free in sign, fitted at prediction time from that machine's own pre-cutoff
  outcomes. Recency half-lives `[8, 16, 32, 64, 128, 256, uniform]` are averaged per
  machine by a softmax over that machine's own held-out most recent transition block, and
  the predictive probability applies a Laplace scaling of the one-dimensional logit
  variance. Fitting is MAP (penalized maximum likelihood) with a one-dimensional Laplace
  predictive integral. It is **not** posterior inference and is not Bayesian.
- `r` are the positive-payout proportions of {reference}.
- A missing lag is never imputed as an observed loss: a transition with any unobserved
  lag in 1..8 is dropped, and if the prediction tick's own basis row cannot be built the
  reference predictor's win mass is used for that query instead.

Why this split: on the earlier three-component forecaster the `p_0` coordinate carried
67.6% of all squared probability error, so the total win mass `q` is where the remaining
headroom was. `r` matters very little - swapping the three-component `r` for the two-head
`r` moved the development score by 0.1156 points.

### Model and config files (all required)

- `model/deployment.json` - which reference predictor, the frozen `alpha`, and the
  deployment weights
- `model/win_head/population_averaged.json` - pooled win-head coefficients per machine
  type per half-life, and the frozen penalties
  (`lam_intercept = 0.01`, `lam_main = 4`, `lam_inter = 1`)
- `model/two_head/{{win,size}}_{{20261108,20261109}}.cbm` - the two CatBoost heads, two seeds
- `model/population_prior.json` - 3x5 payout prior per machine type, frozen from training
- `model/feature_schema.json` - the 182 observable column names, asserted at inference
{extra_bullets}

### Deployment weights

{weights_section}

## Setup

- Python/runtime version: Python 3.13 base environment (`"image": "python"`).
  Developed and measured on CPython 3.12.16.
- OS tested: macOS 15.x (Darwin 25.3.0), arm64.
- Dependencies: `requirements.txt`, exact versions ({len(pins)} pins). Every pin has a
  wheel installable on CPython 3.13 / manylinux x86_64 (or is a pure-Python wheel).
{"".join(f"  - resolved pin conflict: {k} {v}" + chr(10) for k, v in sorted(conflicts.items()))}- Required hardware: {hardware}
- Put the released data at: anywhere; pass it with `--data`. The organizer rerun mounts
  it at `/data` and this entry reads `/data/test/`.

## Recreate the forecast

Run from: this archive's root (the organizer mounts it at `/submission` and runs there).

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Required model/config files: everything under `model/`, listed above. The command
  loads saved models; it trains nothing.
- Random seeds and settings: no RNG is drawn anywhere in this path. The seeds 20261108
  and 20261109 name saved CatBoost checkpoints rather than seeding anything at prediction
  time{seeds_note}. The machine-specific deviations `u_m` are the unique mode of a
  strictly convex penalized objective, found by damped Newton/IRLS.
- What the command does: loads saved models and predicts. No population effect, blend
  weight or hyperparameter is fitted at prediction time. The only quantities computed
  from the query's own machine are `u_m` and the half-life mixture, both read strictly
  from ticks below the cutoff.
- Measured runtime and hardware: **{runs['test']['seconds']:.1f} s** for the
  {runs['test']['report']['queries']} test queries and {runs['validation']['seconds']:.1f} s
  for the {runs['validation']['report']['queries']} validation queries, with
  `OMP_NUM_THREADS=2` on an arm64 macOS laptop. The event budget is 30 minutes (1800 s)
  on two cores.
- Training command, if separate: training is not part of this package. The pooled
  win-head coefficients come from `scripts/win_head_tune_average.py` and the two CatBoost
  heads from `scripts/two_head_catboost.py` in the development repository; both read only
  the released training split.
- Main-round data/model reused in adaptation, if any: not applicable (main round).

Reruns are deterministic: the same inputs give the same CSV. See "Verification" below for
the measured reproduction gap, which is floating-point only.

## Local check

```sh
python check_submission.py /output/submission.csv --data /path/to/kit/public
```

- Checker result: `VALID FORMAT`. The recorded run is in `verification.json` beside this
  package in the development repository
  (`{(out / 'verification.json').relative_to(ROOT)}`).
- Fresh-folder reproduction checked: yes. This ZIP was extracted into an empty temporary
  directory through the kit's own `submission_contract.extract_archive` and run there in a
  stripped environment with `HF_HUB_OFFLINE=0` and every outbound HTTP route pointed at a
  closed port, so any download attempt would fail rather than silently succeed. The
  regenerated CSV was then compared with the development copy.

## Verification (measured)

- The extracted inference feature builder reproduces
  `artifacts/two_head_01/validation.csv` from the saved heads: max absolute deviation
  **{two_head_deviation:.3e}**.
- The win head `q` reproduces the measured held-fold prediction in
  `artifacts/win_head/predictions/oof_win_head.csv`: max absolute deviation
  **{q_deviation:.3e}**.
- Composed five-vectors on test: all finite and non-negative, minimum probability
  {runs['test']['report']['min_probability']:.3e}, max absolute sum error
  {runs['test']['report']['max_abs_sum_error']:.3e} (the contract allows 1e-6).
- Win-head query-basis fallbacks: {runs['test']['report']['win_head_fallbacks']} on test,
  {runs['validation']['report']['win_head_fallbacks']} on validation.

### Known cosmetic warning

On numpy built against Apple Accelerate (this development machine) every `np.matmul`
call reports `divide by zero`, `overflow` and `invalid value` RuntimeWarnings even when
the result is exact; `np.ones((213, 37)) @ np.ones(37)` reproduces all three while
returning exactly 37.0. These warnings are spurious BLAS floating-point flags, not model
values. Every value this code produces is explicitly checked for finiteness, and the
composed five-vector is validated before it is written.

## Development score, and what it is not

Development score of this candidate: **{development}** on the released public validation
split, with the organizer's own scorer and held-fold discipline.

This number is **exploratory and not confirmed**. All 96 public-validation episodes were
inspected repeatedly during development, so a new split of them is not an untouched
holdout, and no independent organizer evaluation set exists outside the private test
answers. {deployment_delta} Re-scoring this package's own validation output is therefore
an in-sample diagnostic, not an achieved forecasting score, and is not presented as one.

## License and attribution

{license_section}

- CatBoost (`catboost==1.2.8`, Yandex LLC) powers the two-head component under the Apache
  License 2.0; a verbatim copy is at `licenses/CATBOOST-LICENSE`.

### Organizer kit code

`slot_forecasting/{{__init__,schema,io,baseline,transition}}.py` are copied without
modification from the released `main-participants-60-40` kit, the same way the kit's own
`build_entry.py` packages them for the starter entry. The kit ships no separate LICENSE
file and no additional license is asserted over them.

### This team's code

`predict.py` and everything under `slot_model/` is this team's own work.

## File integrity

`checksums.json` lists the SHA-256 and byte size of every shipped file except itself and
this README.
'''


def commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    main()
