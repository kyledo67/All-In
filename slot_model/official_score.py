"""Official-scorer helpers without a JAX import.

`scripts/experiment.py` already exposes `scoring_inputs`, `score` and
`paired_bootstrap`, but importing that module pulls in `slot_model.runtime` and
therefore JAX/Dynamax. The TabPFN work needs neither, so the three helpers are
reproduced here against the organizer's own scorer, unchanged in behaviour:
`score` still delegates to `slot_forecasting.experiments.subset_score`, and
`paired_bootstrap` keeps the same episode resampling, seed and repeat count, so
numbers are directly comparable with `artifacts/phase2/evaluation.json`.

The round maximum is read from the kit's `metadata.json` through
`slot_forecasting.rounds.round_metadata` rather than hardcoded, and a kit that is not
the 60-point main round is refused: pointing these helpers at the 40-point adaptation
kit would otherwise report points 1.5x too large with no error raised.
"""
from pathlib import Path

import numpy as np

from slot_forecasting.experiments import subset_score
from slot_forecasting.io import read_rows, read_submission, read_targets
from slot_forecasting.rounds import ROUND_POINTS, round_metadata
from slot_forecasting.schema import CONDITIONS, ValidationError

# Round two (adaptation) is worth 40 points and its validation split is shaped
# differently, so these helpers refuse it instead of reporting 60/40 = 1.5x too much.
ROUND = 'main'


class ScoringInputs(tuple):
    """(targets, groups, reference), plus the round metadata they were loaded with.

    Still a plain three-element tuple, so every caller's `targets, groups, reference =
    scoring_inputs(data)` and `subset_score(predictions, *scoring, ...)` are unchanged.
    """

    def __new__(cls, targets, groups, reference, metadata):
        inputs = super().__new__(cls, (targets, groups, reference))
        inputs.metadata = metadata
        return inputs


def max_points(scoring):
    """The round maximum from the kit's own metadata.json, never a module constant."""
    metadata = getattr(scoring, 'metadata', None)
    if metadata is None:
        raise ValidationError('Pass the ScoringInputs returned by scoring_inputs(data); the round '
                              'maximum is read from the kit metadata, not assumed.')
    return metadata['max_points']


def scoring_inputs(data):
    data = Path(data)
    metadata = round_metadata(data)
    if metadata['round'] != ROUND:
        raise ValidationError(
            f"slot_model.official_score scores the {ROUND} round only ({ROUND_POINTS[ROUND]} points); this "
            f"kit is round {metadata['round']} ({metadata['max_points']} points). Scoring it here would "
            'report the wrong maximum, so it is refused rather than converted.')
    targets = read_targets(data / 'validation/targets.csv')
    groups = {r['query_id']: r['condition']
              for r in read_rows(data / 'validation/conditions.csv', ('query_id', 'condition'))}
    reference = read_submission(data / 'validation/reference.csv')
    return ScoringInputs(targets, groups, reference, metadata)


def score(predictions, scoring, episodes):
    return subset_score(predictions, *scoring, set(episodes), max_points(scoring))


def paired_bootstrap(a, b, scoring, episodes, repeats=2000):
    """Resample the same whole episodes for both models AND the frozen reference."""
    targets, groups, reference = scoring
    selected = [t for t in targets if t.episode_id in set(episodes)]
    rows = {}
    for condition in CONDITIONS:
        es = sorted({t.episode_id for t in selected if groups[t.query_id] == condition})
        values = []
        for episode in es:
            ts = [t for t in selected if t.episode_id == episode]
            values.append([np.mean([np.sum((np.asarray(p[t.query_id]) - t.probabilities) ** 2) for t in ts])
                           for p in (a, b, reference)])
        rows[condition] = np.asarray(values)
    rng = np.random.default_rng(314159)
    totals = np.zeros(repeats)
    conditions = {}
    for condition, values in rows.items():
        draws = rng.integers(0, len(values), (repeats, len(values)))
        means = values[draws].mean(1)
        diff = (means[:, 0] - means[:, 1]) / np.maximum(means[:, 2], 1e-9)
        totals += diff / 4
        conditions[condition] = np.quantile(diff, [.025, .975]).tolist()
    return dict(delta_loss_ratio_ci95=np.quantile(totals, [.025, .975]).tolist(),
                by_condition_ci95=conditions, repeats=repeats, seed=314159,
                interpretation='negative favors first model',
                unit='whole episode, stratified by condition')
