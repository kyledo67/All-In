"""Prove official_score derives max_points from the kit and refuses a non-main round.

The adaptation kit is simulated in a scratch tree: metadata.json is rewritten to
round=adaptation/max_points=40 and the validation CSVs are symlinked, so nothing in
main-participants-60-40 is touched.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np  # noqa: E402,F401  (import order mirrors the production scripts)

from slot_forecasting.io import read_submission  # noqa: E402
from slot_forecasting.schema import ValidationError  # noqa: E402
from slot_model.official_score import ROUND, max_points, score, scoring_inputs  # noqa: E402

DATA = ROOT / 'main-participants-60-40/public'
FROZEN = ROOT / 'artifacts/phase2/predictions/oof_ensemble.csv'
CANDIDATE = ROOT / 'artifacts/audit/oof_candidate_raw.csv'
EXPECTED = 31.70233166770079


def main():
    out = {'round_constant': ROUND}
    scoring = scoring_inputs(DATA)
    targets, groups, reference = scoring
    episodes = {t.episode_id for t in targets}
    out['unpacks_as_three_tuple'] = len(scoring) == 3 and len(reference) == 768
    out['metadata'] = {k: scoring.metadata[k] for k in ('round', 'max_points', 'season_id', 'suite_version')}
    out['max_points_from_metadata'] = max_points(scoring)
    out['max_points_is_hardcoded_anywhere'] = 'MAX_POINTS' in (ROOT / 'slot_model/official_score.py').read_text()

    frozen = score(read_submission(FROZEN), scoring, episodes)
    out['frozen_blend'] = {'technical_points': frozen['technical_points'],
                           'repr': repr(frozen['technical_points']),
                           'max_points': frozen['max_points'],
                           'contest_version': frozen['contest_version'],
                           'matches_recorded_31_702332': frozen['technical_points'] == EXPECTED}
    candidate = score(read_submission(CANDIDATE), scoring, episodes)
    out['candidate'] = {'technical_points': candidate['technical_points'],
                        'repr': repr(candidate['technical_points']),
                        'mean_loss_ratio': candidate['mean_loss_ratio']}
    out['reference_scores_itself'] = score(read_submission(DATA / 'validation/reference.csv'),
                                           scoring, episodes)['technical_points']

    # A plain hand-built 3-tuple must be refused, not silently scored at 60.
    try:
        score(read_submission(FROZEN), (targets, groups, reference), episodes)
        out['plain_tuple_refused'] = False
    except ValidationError as exc:
        out['plain_tuple_refused'] = True
        out['plain_tuple_message'] = str(exc)

    # A simulated 40-point adaptation kit must be refused.
    with tempfile.TemporaryDirectory() as tmp:
        kit = Path(tmp) / 'public'
        (kit / 'validation').mkdir(parents=True)
        metadata = json.loads((DATA / 'metadata.json').read_text())
        metadata.update(round='adaptation', max_points=40)
        (kit / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        for name in ('targets.csv', 'conditions.csv', 'reference.csv', 'queries.csv', 'history.csv'):
            os.symlink(DATA / 'validation' / name, kit / 'validation' / name)
        try:
            scoring_inputs(kit)
            out['adaptation_kit_refused'] = False
        except ValidationError as exc:
            out['adaptation_kit_refused'] = True
            out['adaptation_kit_message'] = str(exc)
        # and the organizer agrees about what that kit is worth
        from slot_forecasting.rounds import round_metadata
        out['adaptation_kit_metadata'] = {k: round_metadata(kit)[k] for k in ('round', 'max_points')}

    (Path(__file__).parent / 'official_score_round.json').write_text(
        json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2, sort_keys=True))
    assert out['frozen_blend']['matches_recorded_31_702332'], out['frozen_blend']
    assert out['frozen_blend']['max_points'] == 60.0
    assert out['max_points_from_metadata'] == 60
    assert out['adaptation_kit_refused'] and out['plain_tuple_refused']
    assert out['reference_scores_itself'] == 30.0
    assert not out['max_points_is_hardcoded_anywhere']
    print('OFFICIAL SCORE ROUND DERIVATION: PASSED')


if __name__ == '__main__':
    main()
