"""Prove scripts/tabpfn_train.py defaults to n_estimators=1 and records the checkpoint hash.

TabPFN itself must not be re-run (~4 min per 768-row forward pass), so the forward pass
is replaced by a stub that returns the committed predictions. Everything else in main()
executes for real: argparse defaults, checkpoint_sha256(), cached_design(), the manifest
write. Outputs go to a scratch --out, never to artifacts/tabpfn/.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer'), str(ROOT / 'scripts')]
import numpy as np  # noqa: E402

from slot_forecasting.io import read_submission  # noqa: E402
from slot_model.tabpfn_model import CHECKPOINTS, checkpoint_sha256  # noqa: E402

CKPT = CHECKPOINTS / 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'


class StubComponent:
    """Stands in for TabPFNComponent so no forward pass runs."""

    def __init__(self, *, seed, n_estimators, variant, max_train_rows, threads):
        self.seed, self.n_estimators, self.variant = seed, n_estimators, variant
        self.max_train_rows, self.threads = max_train_rows, threads

    def fit(self, x, y, *, episodes=None, weight=None, names=None):
        self.rows = int(len(x))
        return self

    def predict_proba(self, x):
        committed = read_submission(ROOT / f'artifacts/tabpfn/predictions/tabpfn_{self.variant}_{self.seed}.csv')
        return np.array([committed[q] for q in sorted(committed)])

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        return {'context_rows': self.rows, 'subsample': {'stub': True},
                'weight_handling': 'stub', 'context_sha256': '0' * 64}


def main():
    out = {}
    help_text = subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/tabpfn_train.py'), '--help'],
                               capture_output=True, text=True, check=True, cwd=ROOT).stdout
    body = help_text[help_text.index('options:'):]
    estimators = body[body.index('--n-estimators'):]
    estimators = estimators[:estimators.index('--max-train-rows')]
    estimators = ' '.join(estimators.split())  # argparse hard-wraps the help column
    out['help_mentions_default_1'] = '(default: 1)' in estimators
    out['help_mentions_cost_ladder'] = '~118 s at 1' in estimators and '~950 s at 4' in estimators
    out['help_text'] = estimators

    out['checkpoint_sha256'] = checkpoint_sha256()
    out['checkpoint_sha256_matches_shasum'] = (
        out['checkpoint_sha256'][CKPT.name]
        == hashlib.sha256(CKPT.read_bytes()).hexdigest()
        == subprocess.run(['shasum', '-a', '256', str(CKPT)], capture_output=True, text=True,
                          check=True).stdout.split()[0])

    import tabpfn_train  # noqa: E402  (the script under test)
    tabpfn_train.TabPFNComponent = StubComponent
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        argv = sys.argv
        sys.argv = ['tabpfn_train.py', '--data', str(ROOT / 'main-participants-60-40/public'),
                    '--out', str(tmp), '--variants', 'hmm', '--seeds', '20261108', '--skip-verify']
        try:
            tabpfn_train.main()
        finally:
            sys.argv = argv
        manifest = json.loads((tmp / 'training_runs.json').read_text())
    out['manifest_keys'] = sorted(manifest)
    out['manifest_n_estimators_default'] = manifest['n_estimators']
    out['manifest_checkpoint_sha256'] = manifest.get('checkpoint_sha256')
    out['manifest_records_checkpoint'] = manifest.get('checkpoint_sha256') == out['checkpoint_sha256']
    out['manifest_cache_key'] = manifest['cache_key']
    out['manifest_cache_inputs_population_keys'] = \
        manifest['provenance']['cache_inputs']['configuration']['population_keys']
    out['manifest_run_n_estimators'] = [r['n_estimators'] for r in manifest['runs']]

    (Path(__file__).parent / 'tabpfn_manifest_check.json').write_text(
        json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2, sort_keys=True))
    assert out['help_mentions_default_1'] and out['help_mentions_cost_ladder']
    assert out['checkpoint_sha256_matches_shasum']
    assert out['manifest_records_checkpoint']
    assert out['manifest_n_estimators_default'] == 1 and out['manifest_run_n_estimators'] == [1]
    print('TABPFN MANIFEST + DEFAULT: PASSED')


if __name__ == '__main__':
    main()
