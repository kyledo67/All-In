"""TabPFN component over the already-validated CatBoost feature table.

TabPFN is a prior-fitted transformer: `fit` only stores the training context and
`predict_proba` runs one forward pass with that context in-attention. There is no
gradient step here, so nothing is "trained" on our data in the usual sense and no
tuning budget is consumed beyond the seed and estimator count.

The feature table is reused unchanged from `phase2_features.matrix` (182 observable
columns, optionally plus the 7 portable HMM summary columns). That table is already
all-numeric and well under TabPFN's 500-feature ceiling, so no categorical encoding
or importance pruning adapter is needed; both facts are asserted rather than assumed.

Local `tabpfn` package only. `tabpfn_client` would send features to a remote API and
is never imported.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

# Payout indices for 0x, 1x, 2x, 5x, 20x. The fixed submission order.
CLASSES = (0, 1, 2, 3, 4)
# TabPFN v2 was pre-trained for tables up to these sizes.
PRETRAIN_ROWS = 10000
PRETRAIN_FEATURES = 500
# The task brief's adapter threshold: stay under this many columns.
MAX_FEATURES = 250
CHECKPOINTS = Path(__file__).resolve().parents[1] / 'artifacts/tabpfn/checkpoints'
FLOOR = 1e-6


def simplex(probabilities):
    """Clip to [1e-6, 1-1e-6] then renormalize, as the blend contract requires."""
    values = np.clip(np.asarray(probabilities, dtype=np.float64), FLOOR, 1 - FLOOR)
    values = values / values.sum(axis=1, keepdims=True)
    if not np.isfinite(values).all() or values.min() < 0 or np.max(abs(values.sum(1) - 1)) >= 1e-12:
        raise ValueError('TabPFN produced probabilities outside the simplex')
    return values


def episode_subsample(episodes, weight, budget, seed):
    """Deterministic whole-episode selection; never splits an episode across the cut.

    Episodes are ordered by a seeded hash of their ID, so the choice depends only on
    (seed, episode IDs) and not on row order, numpy version or platform.
    """
    order = sorted(set(episodes), key=lambda e: hashlib.sha256(f'{seed}|{e}'.encode()).hexdigest())
    rows = {e: int(np.sum(episodes == e)) for e in order}
    keep, total = [], 0
    for episode in order:
        if total + rows[episode] > budget:
            continue
        keep.append(episode)
        total += rows[episode]
    if not keep:
        raise ValueError('Row budget cannot hold a single episode')
    mask = np.isin(episodes, keep)
    return mask, {'kept_episodes': len(keep), 'dropped_episodes': len(order) - len(keep),
                  'kept_rows': int(mask.sum()), 'offered_rows': int(len(episodes)),
                  'budget': budget, 'seed': seed,
                  'weight_retained': float(np.sum(weight[mask]) / np.sum(weight)) if weight is not None else None}


class TabPFNComponent:
    """Same fit/predict contract as the CatBoost component: fit(x, y) then predict_proba(x)."""

    def __init__(self, *, seed=20261108, n_estimators=4, variant='observable',
                 max_train_rows=PRETRAIN_ROWS, device='cpu', checkpoints=CHECKPOINTS,
                 threads=2, ignore_pretraining_limits=False):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.variant = variant
        self.max_train_rows = int(max_train_rows)
        self.device = device
        self.checkpoints = Path(checkpoints)
        self.threads = int(threads)
        self.ignore_pretraining_limits = bool(ignore_pretraining_limits)
        self.x = self.y = None
        self.subsample = None
        self.feature_names = None

    # ---- construction -------------------------------------------------------
    def _classifier(self):
        """Import locally so a zero-weight blend never pulls torch into the path."""
        import os
        os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(self.checkpoints))
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
        os.environ.setdefault('OMP_NUM_THREADS', str(self.threads))
        # Our table is inside TabPFN's pre-training limits (rows < 10000, features < 500);
        # only its CPU speed guard at 1000 rows trips, so lift that guard specifically
        # rather than `ignore_pretraining_limits`, which would also lift the real ones.
        os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
        import torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        from tabpfn import TabPFNClassifier
        if 'tabpfn_client' in __import__('sys').modules:
            raise RuntimeError('tabpfn_client must never be imported; inference is offline')
        return TabPFNClassifier(device=self.device, random_state=self.seed,
                                n_estimators=self.n_estimators,
                                ignore_pretraining_limits=self.ignore_pretraining_limits)

    def _check_table(self, x, names=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError('Feature table must be two-dimensional')
        if x.shape[1] > MAX_FEATURES:
            raise ValueError(f'{x.shape[1]} features exceeds the {MAX_FEATURES} ceiling; add a pruning adapter')
        if x.shape[1] > PRETRAIN_FEATURES:
            raise ValueError(f'{x.shape[1]} features exceeds TabPFN pre-training limit {PRETRAIN_FEATURES}')
        if not np.isfinite(x).all():
            raise ValueError('Feature table holds non-finite values')
        if names is not None and len(names) != x.shape[1]:
            raise ValueError('Feature name count does not match the table')
        return x

    # ---- fit / predict ------------------------------------------------------
    def fit(self, x, y, *, episodes=None, weight=None, names=None):
        start = time.perf_counter()
        x = self._check_table(x, names)
        y = np.asarray(y, dtype=np.int64)
        if len(x) != len(y):
            raise ValueError('Feature and label row counts differ')
        if y.min() < 0 or y.max() > 4:
            raise ValueError('Labels must be payout indices 0..4')
        if len(x) > self.max_train_rows:
            if episodes is None:
                raise ValueError('Row budget exceeded and no episode grouping supplied')
            mask, self.subsample = episode_subsample(np.asarray(episodes), weight, self.max_train_rows, self.seed)
            x, y = x[mask], y[mask]
        else:
            self.subsample = {'kept_rows': int(len(x)), 'offered_rows': int(len(x)),
                              'budget': self.max_train_rows, 'seed': self.seed, 'subsampled': False}
        present = sorted(set(y.tolist()))
        if len(present) < 2:
            raise ValueError('TabPFN needs at least two observed payout classes')
        self.x, self.y, self.feature_names = x, y, (list(names) if names is not None else None)
        self.classes_present = present
        self.model = self._classifier()
        self.model.fit(self.x, self.y)
        self.fit_seconds = time.perf_counter() - start
        # TabPFN carries no sample weights; the episode/machine normalization the
        # CatBoost Pool applies is therefore not reproduced here. Recorded, not hidden.
        self.weight_handling = 'unweighted context rows; CatBoost Pool weights not applicable to TabPFN'
        return self

    def predict_proba(self, x):
        if self.x is None:
            raise RuntimeError('Call fit before predict_proba')
        x = self._check_table(x, self.feature_names)
        if x.shape[1] != self.x.shape[1]:
            raise ValueError('Prediction table width differs from the fitted context')
        raw = np.asarray(self.model.predict_proba(x), dtype=np.float64)
        # Expand to the five payout columns even when a class was absent from context.
        full = np.zeros((len(x), 5), dtype=np.float64)
        for column, label in enumerate(self.model.classes_):
            full[:, int(label)] = raw[:, column]
        return simplex(full)

    # ---- persistence --------------------------------------------------------
    def save(self, directory):
        """Store the context table and config; `load` refits, which is deterministic.

        We deliberately do not pickle the fitted estimator: the context plus seed plus
        checkpoint fully determine the forward pass, and a plain npz survives library
        upgrades that a pickle would not.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'context.npz', x=self.x, y=self.y)
        config = {'seed': self.seed, 'n_estimators': self.n_estimators, 'variant': self.variant,
                  'max_train_rows': self.max_train_rows, 'device': self.device,
                  'threads': self.threads, 'ignore_pretraining_limits': self.ignore_pretraining_limits,
                  'features': int(self.x.shape[1]), 'context_rows': int(len(self.x)),
                  'feature_names': self.feature_names, 'classes_present': self.classes_present,
                  'subsample': self.subsample, 'weight_handling': self.weight_handling,
                  'context_sha256': hashlib.sha256(
                      np.ascontiguousarray(self.x).tobytes() + np.ascontiguousarray(self.y).tobytes()).hexdigest()}
        (directory / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
        return config

    @classmethod
    def load(cls, directory, *, checkpoints=CHECKPOINTS):
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        with np.load(directory / 'context.npz', allow_pickle=False) as data:
            x, y = data['x'], data['y']
        digest = hashlib.sha256(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
        if digest != config['context_sha256']:
            raise ValueError('Saved TabPFN context does not match its recorded hash')
        component = cls(seed=config['seed'], n_estimators=config['n_estimators'],
                        variant=config['variant'], max_train_rows=max(config['max_train_rows'], len(x)),
                        device=config['device'], checkpoints=checkpoints, threads=config['threads'],
                        ignore_pretraining_limits=config['ignore_pretraining_limits'])
        component.fit(x, y, names=config['feature_names'])
        return component


class TabPFNEnsemble:
    """Seed average over saved components, matching how evaluation scores TabPFN.

    `directory` is either one saved component or a parent holding several. Averaging
    across seeds is what scripts/tabpfn_evaluate.py scores, so inference must do the
    same or the deployed component would not be the one that was measured.
    """

    def __init__(self, components):
        if not components:
            raise ValueError('No saved TabPFN components found')
        variants = {c.variant for c in components}
        if len(variants) != 1:
            raise ValueError(f'Cannot average TabPFN components across feature variants {sorted(variants)}')
        widths = {c.x.shape[1] for c in components}
        if len(widths) != 1:
            raise ValueError('Saved TabPFN components disagree on feature count')
        self.components = components
        self.variant = components[0].variant
        self.seeds = sorted(c.seed for c in components)

    @classmethod
    def load(cls, directory, *, checkpoints=CHECKPOINTS):
        directory = Path(directory)
        if (directory / 'config.json').exists():
            return cls([TabPFNComponent.load(directory, checkpoints=checkpoints)])
        children = sorted(p for p in directory.iterdir() if (p / 'config.json').exists())
        return cls([TabPFNComponent.load(p, checkpoints=checkpoints) for p in children])

    def predict_proba(self, x):
        return simplex(np.mean([c.predict_proba(x) for c in self.components], axis=0))
