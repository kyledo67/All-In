"""Inference-only two-head CatBoost component.

Why this module exists. `scripts/dense_catboost.py` and `scripts/two_head_catboost.py`
are TRAINING scripts: both load the full released train and validation splits at import
time from a hard-coded absolute path, and `two_head_catboost.py` imports the other one,
so importing either of them runs training-time work and cannot be shipped. The feature
builder they share is therefore extracted here with no import-time side effect, no data
path, and no training code.

Equivalence with the training scripts, stated precisely so it can be checked:

* `scripts/dense_catboost.held_features(queries, fit_episodes, source)` builds its rows
  as `observable(q, sample_history(q, idx), source[q.episode_id], prior, True)`, where
  `source[q.episode_id]` is the per-episode slice of the causal index. `observable`
  consumes that argument only in `for (episode, machine), seq in index.items(): if
  episode == q.episode_id and machine != q.machine_id`, so passing the WHOLE causal
  index gives bitwise-identical features. That is what `phase2_features.matrix` does,
  and it is the call used here.
* `examples(queries, index)` with the default `historical=False` emits exactly one row
  per query, with `signal_known=True` and `sample_history(..., thin=False)`, matching
  `held_features`.
* `prior` is read from a saved 3x5 JSON table instead of being refitted by
  `fit_prior(train_queries, ...)`, so no train split needs to be mounted.
* The per-seed five-vector is composed BEFORE averaging over seeds, exactly as
  `two_head_catboost.train_predict` does; averaging the two heads separately would give
  a different answer.

Verification lives in scripts/win_head_package.py, which rebuilds
artifacts/two_head_01/validation.csv from the saved heads and reports the deviation.
"""
from pathlib import Path

import numpy as np

from .phase2_features import examples, matrix

SEEDS = (20261108, 20261109)
WIN_CLASSES = [0, 1]
SIZE_CLASSES = [0, 1, 2, 3]


def feature_table(queries, index, prior, *, schema=None):
    """Return (examples, x, names) for the 182-column observable table.

    `prior` is the saved 3x5 population prior (TYPES order). `schema`, when given, is
    asserted against the realized column names rather than assumed.
    """
    rows = examples(queries, index)
    priors = {e['q'].episode_id: prior for e in rows}
    x, names = matrix(rows, index, priors)
    if schema is not None and list(names) != list(schema):
        raise ValueError('Observable feature schema changed')
    if not np.isfinite(x).all():
        raise ValueError('Observable feature table holds non-finite values')
    return rows, x, names


def predict_proba(x, model_dir, *, seeds=SEEDS, threads=2):
    """Seed-averaged five-vectors from the saved win/size heads. Loads models only."""
    from catboost import CatBoostClassifier
    model_dir = Path(model_dir)
    per_seed = []
    for seed in seeds:
        win = CatBoostClassifier()
        win.load_model(str(model_dir / f'win_{seed}.cbm'))
        size = CatBoostClassifier()
        size.load_model(str(model_dir / f'size_{seed}.cbm'))
        if list(win.classes_) != WIN_CLASSES:
            raise ValueError(f'win_{seed}.cbm class order changed: {list(win.classes_)}')
        if list(size.classes_) != SIZE_CLASSES:
            raise ValueError(f'size_{seed}.cbm class order changed: {list(size.classes_)}')
        chance = win.predict_proba(x, thread_count=threads)[:, 1]
        conditional = size.predict_proba(x, thread_count=threads)
        # Compose per seed, then average: this is the order the measured component used.
        per_seed.append(np.column_stack((1.0 - chance, chance[:, None] * conditional)))
    out = np.mean(per_seed, axis=0)
    if not np.isfinite(out).all() or out.min() < 0 or np.max(abs(out.sum(1) - 1)) > 1e-12:
        raise ValueError('Two-head component produced invalid probabilities')
    return out
