"""Rebuild the exact phase-two feature tables without a JAX runtime.

`scripts/phase2_train.py` built these tables with Dynamax/JAX and cached them under
`artifacts/phase2/cache/`, which is not versioned. The eight cross-fit HMM population
checkpoints *are* versioned, and `phase2_predict.hmm_summary` is the portable float64
forward recurrence already checked against the JAX path, so the tables can be
reconstructed exactly with numpy and the standard library alone.

Reconstruction is verified, not assumed: `scripts/tabpfn_train.py` feeds the rebuilt
validation table through the committed CatBoost checkpoints and requires the result to
match the committed `artifacts/phase2/predictions/cb_*.csv` rows.

Every feature still comes from `phase2_features.observable`, which reads only ticks
strictly below each cutoff.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .features import causal_index
from .phase2_features import examples, fit_prior, local_probability, matrix
from .phase2_predict import hmm_summary
from slot_forecasting.io import load_inputs, read_labels

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / 'artifacts/phase2'
HMM_COLUMNS = 7


def population_key(allowed):
    """Same key derivation as scripts/phase2_train.py, so checkpoints line up."""
    return hashlib.sha256('|'.join(sorted(allowed)).encode()).hexdigest()[:16]


def load_population(allowed, models=None):
    models = Path(models or PHASE2 / 'hmm_models')
    key = population_key(allowed)
    path = models / f'{key}.json'
    if not path.exists():
        raise FileNotFoundError(
            f'No committed HMM checkpoint for this episode set (key {key}). '
            'Refitting it needs the JAX training environment.')
    return json.loads(path.read_text()), key


def produce(es, index, upstream_sets, train_queries, train_index, local_configs, models=None):
    """Port of scripts/phase2_train.py `produce`, with the portable HMM recurrence.

    `upstream_sets` maps each example episode to the training episodes its upstream
    prior and HMM may be fitted on, so an episode never informs its own features.
    """
    priors = {}
    hmm = np.zeros((len(es), HMM_COLUMNS))
    sets = {tuple(sorted(a)) for a in upstream_sets.values()}
    model_keys = []
    for allowed_tuple in sorted(sets):
        allowed = set(allowed_tuple)
        population, key = load_population(allowed, models)
        prior = fit_prior(train_queries, train_index, allowed)
        selected = [i for i, e in enumerate(es) if upstream_sets[e['q'].episode_id] == allowed]
        if not selected:
            continue
        batch = [es[i] for i in selected]
        if any(e['q'].episode_id in allowed for e in batch):
            raise ValueError('An example episode leaked into its own upstream fitting set')
        for e in batch:
            priors[e['q'].episode_id] = prior
        _, summary = hmm_summary(population, batch)
        if summary.shape[1] != HMM_COLUMNS:
            raise ValueError('Portable HMM summary width changed')
        hmm[selected] = summary
        model_keys.append(key)
    x, names = matrix(es, index, priors)
    local = np.array([[local_probability(e['q'], e['h'], priors[e['q'].episode_id], **c)
                       for c in local_configs] for e in es])
    return dict(x=x, hmm=hmm, local=local, names=np.array(names),
                ids=np.array([e['q'].query_id for e in es]),
                ep=np.array([e['q'].episode_id for e in es]),
                y=np.array([-1 if e['y'] is None else e['y'] for e in es]),
                weight=np.array([e['weight'] for e in es]),
                known=np.array([e['signal_known'] for e in es]),
                cutoff=np.array([e['q'].cutoff for e in es])), model_keys


def design(data, *, protocol=None, folds=None, models=None):
    """Return the training and validation tables used by the phase-two CatBoost runs."""
    data = Path(data)
    protocol = json.loads(Path(protocol or PHASE2 / 'protocol.json').read_text())
    folds = json.loads(Path(folds or PHASE2 / 'folds.json').read_text())
    train_queries, train_history = load_inputs(data / 'train')
    val_queries, val_history = load_inputs(data / 'validation')
    train_index = causal_index(train_queries, train_history)
    val_index = causal_index(val_queries, val_history)
    labels = read_labels(data / 'train/labels.csv', train_queries)
    train_examples = examples(train_queries, train_index, labels, historical=True)
    val_examples = examples(val_queries, val_index)
    all_episodes = set(folds['train'])
    # Each training episode's upstream components come from the other three folds.
    crossfit = {e: {a for a in all_episodes if folds['train'][a] != folds['train'][e]} for e in all_episodes}
    train_table, train_keys = produce(train_examples, train_index, crossfit, train_queries,
                                      train_index, protocol['local'], models)
    val_table, val_keys = produce(val_examples, val_index, {q.episode_id: all_episodes for q in val_queries},
                                  train_queries, train_index, protocol['local'], models)
    schema = json.loads((PHASE2 / 'feature_schema.json').read_text())
    for name, table in (('train', train_table), ('validation', val_table)):
        if table['names'].tolist() != schema['observable']:
            raise ValueError(f'{name} observable feature schema does not match the committed schema')
    return {'train': train_table, 'validation': val_table,
            'provenance': {'train_model_keys': sorted(train_keys), 'validation_model_keys': sorted(val_keys),
                           'observable_features': len(schema['observable']), 'hmm_features': len(schema['hmm']),
                           'train_rows': int(len(train_table['y'])), 'validation_rows': int(len(val_table['y']))}}


def features_for(table, variant):
    """`observable` is the 182-column table; `hmm` appends the 7 portable summary columns."""
    if variant == 'observable':
        return table['x']
    if variant == 'hmm':
        return np.column_stack((table['x'], table['hmm']))
    raise ValueError(f'Unknown feature variant {variant}')


def cached_design(data, cache, **kwargs):
    """Cache on the inputs and this module, so unchanged inputs are not rebuilt."""
    cache = Path(cache)
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / 'slot_model/phase2_features.py', ROOT / 'slot_model/features.py',
                 ROOT / 'slot_model/phase2_predict.py', ROOT / 'slot_model/portable.py',
                 ROOT / 'organizer/slot_forecasting/transition.py', PHASE2 / 'protocol.json', PHASE2 / 'folds.json'):
        digest.update(path.read_bytes())
    for name in ('train/queries.csv', 'train/history.csv', 'train/labels.csv',
                 'validation/queries.csv', 'validation/history.csv'):
        digest.update((Path(data) / name).read_bytes())
    key = digest.hexdigest()[:16]
    path = cache / f'design-{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            tables = {'train': {}, 'validation': {}}
            for field in stored.files:
                if field == 'provenance':
                    continue
                split, _, column = field.partition('.')
                tables[split][column] = stored[field]
        provenance = json.loads((cache / f'design-{key}.json').read_text())
        return {**tables, 'provenance': provenance, 'cache_key': key, 'cached': True}
    built = design(data, **kwargs)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f'{split}.{column}': value
                                for split in ('train', 'validation')
                                for column, value in built[split].items()})
    (cache / f'design-{key}.json').write_text(json.dumps(built['provenance'], indent=2, sort_keys=True) + '\n')
    return {**built, 'cache_key': key, 'cached': False}
