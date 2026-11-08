"""Write artifacts/audit/reliability/environment.json: the provenance record that was missing.

Commit, interpreter, resolved package versions, the TabPFN checkpoint hash, the
phase-2 protocol/folds/schema hashes, the feature cache key, and the sha256 of every
CSV that feeds the audited candidate. Re-runnable: it only reads.
"""
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
from slot_model.phase2_tables import cache_inputs  # noqa: E402
from slot_model.tabpfn_model import checkpoint_sha256  # noqa: E402

DATA = ROOT / 'main-participants-60-40/public'
# The only shared source files this reliability pass edited. Everything else listed
# under git.uncommitted_paths was already there or belongs to another agent's work.
RELIABILITY_EDITS = ('scripts/report_mtd_pairs.py', 'scripts/tabpfn_train.py',
                     'slot_model/official_score.py', 'slot_model/phase2_tables.py',
                     'slot_model/tabpfn_model.py')
PACKAGES = ('numpy', 'scipy', 'catboost', 'torch', 'tabpfn')

# Every CSV the candidate is built from, by the role it plays in the per-fold simplex.
COMPONENT_CSVS = {
    'frozen_blend': 'artifacts/phase2/predictions/oof_ensemble.csv',
    'tabpfn_component_as_blended': 'artifacts/tabpfn/predictions/oof_tabpfn.csv',
    'tabpfn_raw_seed_20261108': 'artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv',
    'tabpfn_raw_seed_20261109': 'artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv',
    'mtd_lag7': 'artifacts/mtd_pairs_01/validation_lag7.csv',
    'two_head': 'artifacts/two_head_01/validation.csv',
    'candidate_oof': 'artifacts/audit/oof_candidate_raw.csv',
}
PROTOCOL_FILES = {
    'folds': 'artifacts/phase2/folds.json',
    'protocol': 'artifacts/phase2/protocol.json',
    'feature_schema': 'artifacts/phase2/feature_schema.json',
    'recompute': 'artifacts/audit/recompute.json',
    'kit_metadata': 'main-participants-60-40/public/metadata.json',
}


def git(*args):
    return subprocess.run(('git', *args), cwd=ROOT, capture_output=True, text=True, check=True).stdout


def status(*paths):
    """Porcelain rows as (code, path). Never .strip() the whole blob: that eats the
    leading space of a ' M path' row and so truncates the first filename."""
    return [(line[:2], line[3:]) for line in git('status', '--porcelain', *paths).splitlines() if line]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    key, inputs = cache_inputs(DATA)
    dirty = status()
    record = {
        'git': {'commit': git('rev-parse', 'HEAD').strip(),
                'commit_short': git('rev-parse', '--short', 'HEAD').strip(),
                'branch': git('rev-parse', '--abbrev-ref', 'HEAD').strip(),
                'committed_tree_clean': not dirty,
                'uncommitted_paths': {path: code for code, path in sorted(dirty, key=lambda r: r[1])},
                'reliability_pass_edited': RELIABILITY_EDITS,
                'organizer_rows': len(status('organizer/'))},
        'python': {'version': platform.python_version(),
                   'implementation': platform.python_implementation(),
                   'executable': str(ROOT / '.venv/bin/python'),
                   'sys_version': sys.version.replace('\n', ' ')},
        'platform': {'system': platform.system(), 'release': platform.release(),
                     'machine': platform.machine(), 'processor': platform.processor()},
        'packages': {name: metadata.version(name) for name in PACKAGES},
        'tabpfn_checkpoint_sha256': checkpoint_sha256(),
        'phase2_feature_cache': {'cache_key': key,
                                 'population_keys': inputs['configuration']['population_keys'],
                                 'hmm_checkpoint_sha256': {n: h for n, h in inputs['sha256'].items()
                                                           if n.startswith('hmm_models/')}},
        'protocol_sha256': {name: sha256(ROOT / path) for name, path in sorted(PROTOCOL_FILES.items())},
        'component_csv_sha256': {name: sha256(ROOT / path) for name, path in sorted(COMPONENT_CSVS.items())},
        'component_csv_paths': dict(sorted(COMPONENT_CSVS.items())),
        'scope': ('Environment and input provenance for the 32.260531/60 candidate. Every score it '
                  'refers to is public-validation over the same 96 episodes this project has inspected '
                  'many times; none of it is an independent holdout.'),
    }
    out = Path(__file__).parent / 'environment.json'
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
