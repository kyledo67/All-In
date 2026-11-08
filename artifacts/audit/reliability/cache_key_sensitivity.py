"""Prove the phase-2 feature cache key now covers the HMM checkpoints and the schema.

Nothing real is modified: every mutation happens in a scratch copy handed to
cached_design/cache_inputs through the existing `models=` / new `schema=` parameters.
Also recomputes the OLD digest to show it was blind to both inputs.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
from slot_model.phase2_tables import PHASE2, cache_inputs, population_keys  # noqa: E402

DATA = ROOT / 'main-participants-60-40/public'


def old_key(data, models=None, schema=None):
    """The digest as committed at f002c77: module sources + protocol + folds + CSVs."""
    digest = hashlib.sha256()
    for path in (ROOT / 'slot_model/phase2_tables.py', ROOT / 'slot_model/phase2_features.py',
                 ROOT / 'slot_model/features.py', ROOT / 'slot_model/phase2_predict.py',
                 ROOT / 'slot_model/portable.py', ROOT / 'organizer/slot_forecasting/transition.py',
                 PHASE2 / 'protocol.json', PHASE2 / 'folds.json'):
        digest.update(path.read_bytes())
    for name in ('train/queries.csv', 'train/history.csv', 'train/labels.csv',
                 'validation/queries.csv', 'validation/history.csv'):
        digest.update((Path(data) / name).read_bytes())
    return digest.hexdigest()[:16]


def main():
    loaded = population_keys(json.loads((PHASE2 / 'folds.json').read_text()))
    unloaded = sorted(p.stem for p in (PHASE2 / 'hmm_models').glob('*.json')
                      if not p.stem.endswith('_diagnostics') and p.stem not in loaded)
    base_new, _ = cache_inputs(DATA)
    base_old = old_key(DATA)
    out = {'loaded_population_keys': loaded, 'committed_but_unloaded_keys': unloaded,
           'baseline_new_key': base_new, 'baseline_old_key': base_old, 'cases': {}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. a loaded checkpoint's CONTENTS change
        models = tmp / 'models_touched'
        shutil.copytree(PHASE2 / 'hmm_models', models)
        victim = models / f'{loaded[0]}.json'
        payload = json.loads(victim.read_text())
        field = next(k for k, v in payload.items() if isinstance(v, list))
        flat = json.dumps(payload[field])
        payload[field] = json.loads(flat.replace('0.', '0.1', 1)) if '0.' in flat else payload[field]
        victim.write_text(json.dumps(payload))
        key, _ = cache_inputs(DATA, models=models)
        out['cases']['loaded_checkpoint_contents_changed'] = {
            'file': f'{loaded[0]}.json', 'mutated_field': field,
            'new_key': key, 'new_key_changed': key != base_new,
            'old_key': old_key(DATA, models=models), 'old_key_changed': old_key(DATA, models=models) != base_old}

        # 2. an UNLOADED checkpoint's contents change -> must NOT move the key
        models2 = tmp / 'models_unloaded'
        shutil.copytree(PHASE2 / 'hmm_models', models2)
        spare = models2 / f'{unloaded[0]}.json'
        spare.write_text(spare.read_text() + '\n')
        key2, _ = cache_inputs(DATA, models=models2)
        out['cases']['unloaded_checkpoint_contents_changed'] = {
            'file': f'{unloaded[0]}.json', 'new_key': key2, 'new_key_changed': key2 != base_new}

        # 3. the feature schema order changes
        schema = tmp / 'feature_schema.json'
        payload = json.loads((PHASE2 / 'feature_schema.json').read_text())
        payload['observable'][0], payload['observable'][1] = payload['observable'][1], payload['observable'][0]
        schema.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        key3, _ = cache_inputs(DATA, schema=schema)
        out['cases']['feature_schema_order_changed'] = {
            'swapped': payload['observable'][:2], 'new_key': key3, 'new_key_changed': key3 != base_new,
            'old_key': old_key(DATA, schema=schema), 'old_key_changed': old_key(DATA, schema=schema) != base_old}

        # 4. an unchanged scratch copy of the checkpoints reproduces the key
        models3 = tmp / 'models_identical'
        shutil.copytree(PHASE2 / 'hmm_models', models3)
        key4, _ = cache_inputs(DATA, models=models3)
        out['cases']['identical_copy_of_checkpoints'] = {'new_key': key4, 'new_key_changed': key4 != base_new}

    out['verdict'] = {
        'new_key_detects_checkpoint_contents': out['cases']['loaded_checkpoint_contents_changed']['new_key_changed'],
        'new_key_detects_schema_order': out['cases']['feature_schema_order_changed']['new_key_changed'],
        'old_key_was_blind_to_checkpoint': not out['cases']['loaded_checkpoint_contents_changed']['old_key_changed'],
        'old_key_was_blind_to_schema': not out['cases']['feature_schema_order_changed']['old_key_changed'],
        'new_key_ignores_unloaded_checkpoint': not out['cases']['unloaded_checkpoint_contents_changed']['new_key_changed'],
        'new_key_is_path_independent': not out['cases']['identical_copy_of_checkpoints']['new_key_changed']}
    (Path(__file__).parent / 'cache_key_sensitivity.json').write_text(
        json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out['verdict'], indent=2, sort_keys=True))
    print('baseline new key', base_new, '| old key', base_old)
    for name, case in out['cases'].items():
        print(f"  {name}: new_key={case['new_key']} changed={case['new_key_changed']}")
    assert all(out['verdict'].values()), out['verdict']
    print('CACHE KEY SENSITIVITY: PASSED')


if __name__ == '__main__':
    main()
