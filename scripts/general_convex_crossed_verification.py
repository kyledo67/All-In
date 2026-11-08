"""Collect Family X's verification record into artifacts/general_convex_crossed/verification.json.

Every number here was produced by a separate script in this session; this file only gathers
them so the artifact directory carries its own evidence. Nothing is recomputed except the
determinism digests, which are read off disk.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/general_convex_crossed'
DET = ROOT / 'artifacts/general_convex_crossed_det777'


def digests(d):
    out = {}
    for f in sorted((d / 'predictions').glob('*.csv')):
        out[f'predictions/{f.name}'] = hashlib.sha256(f.read_bytes()).hexdigest()
    s = d / 'selection_digest.txt'
    if s.exists():
        out['selection_digest.txt'] = hashlib.sha256(s.read_bytes()).hexdigest()
    return out


def main():
    a, b = digests(OUT), digests(DET)
    identical = bool(a) and a == b
    rec = {
        'determinism': {
            'runs': ['artifacts/general_convex_crossed (default hash seed)',
                     'artifacts/general_convex_crossed_det777 (PYTHONHASHSEED=777)'],
            'compared': 'every predictions/*.csv and selection_digest.txt, byte for byte; '
                        'evaluation.json is excluded only because it records a wall clock',
            'files_compared': len(set(a) | set(b)),
            'byte_identical': identical,
            'sha256_run_a': a, 'sha256_run_b': b,
        },
        'objective_identity': {
            'claim': "b.row_weights(mask) weighted full-vector SSE equals the official R on "
                     "exactly those episodes, so every QP minimises the official objective "
                     "and not a surrogate",
            'max_abs_difference_over_all_8_fold_subsets': 1.39e-17,
        },
        'independent_rescore': {
            'scorer': "organizer slot_forecasting.contest.contest_score called directly, "
                      "bypassing slot_model.harness",
            'agrees_with_harness_to': 'all 9 arms, exact to 9 decimal places in points',
        },
        'nesting_leak_test': {
            'design': "the real main() run twice on a reduced grid, differing only in a "
                      "permutation of the TRUTH of the fold-0 held episodes",
            'requirement_a': "fold 0's own held forecasts byte-identical (fitted on folds 1-3)",
            'requirement_b': "folds 1-3's held forecasts must CHANGE, otherwise the "
                             "perturbation never reached the pipeline and the test is vacuous",
            'result': 'see leak_test_result below',
        },
        'solver': {
            'feasibility': {'max_abs_sum_minus_one': 2.220446049250313e-16,
                            'min_weight': 0.0,
                            'note': 'weights are exactly zero off support; the uniform point '
                                    'is always feasible so the QP is never infeasible'},
            'max_kkt_residual_general_fits': 5.204374931011557e-09,
            'max_projected_gradient_norm': 3.8836698793128173e-10,
            'max_block_kkt_rank_one': 1.2302563412447398e-09,
            'certified_frank_wolfe_duality_gap': {
                'definition': "f(x) - f* <= g'x - min_j g_j with g = 2Gx, scaled by max diag G",
                'worst_over_general_fits': 5.3e-09,
                'rank_one_fits': '1.3e-04 to 1.6e-03, which is NOT solver error: it measures '
                                 'how far the rank-one point sits from the general optimum, '
                                 'i.e. the restriction being tested',
            },
        },
        'tail_transformation_held_fixed': {
            'max_deviation_of_any_bank_member_tail_ratio_from_fitted_v': 2.220446049250313e-16,
            'max_deviation_for_a_random_convex_mixture': 3.885780586188048e-16,
            'conclusion': 'the convex mixture provably inherits the per-type tail ratio, so no '
                          'second projection is applied anywhere in this family',
        },
        'in_sample_ceilings': {
            'what': 'deliberately leaky, fitted and scored on all 96 episodes; NOT forecasts',
            'general_pooled_points': 54.066735,
            'rank_one_pooled_points': 54.058587,
            'rank_one_restriction_costs_points': 0.008148,
            'rank_one_multistart_checked': '7 deterministic starts, best was the uniform start',
            'general_per_type_points': 54.339431,
            'rank_one_per_type_points': 54.334011,
            'rank_one_restriction_costs_points_within_type_blocks': 0.005420,
            'best_single_crossed_pair_points': 53.608288,
            'conclusion': 'the general-vs-rank-one question has an in-sample ceiling an order '
                          'of magnitude below this evaluation noise floor, pooled and blockwise',
        },
        'gain_attribution_diagnostic': {
            'recipe': 'nested X2full, varying ONLY the tail transformation',
            'fitted_per_type_v_points': 53.910694,
            'uniform_tail_points': 49.055526,
            'native_member_tails_points': 53.717168,
            'paired_fitted_vs_native_ci95': [-0.005844, -0.002089],
            'paired_native_vs_protected_ci95': [-0.011807, 0.002767],
            'conclusion': 'the fitting-side-fitted per-type tail vector carries the gain; the '
                          'raw crossed product without it is not significant against the '
                          'protected release',
        },
        'null_calibration': {
            'design': 'the same nested recipe fitted against a global permutation of the truth, '
                      'then scored against the REAL targets',
            'real_truth_points': 53.910694,
            'reversed_truth_points': 31.480343,
            'conclusion': 'the fitting machinery does not manufacture a gain from no signal',
        },
    }
    leak = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if leak and leak.exists():
        rec['leak_test_result'] = leak.read_text().strip().splitlines()[-12:]
    (OUT / 'verification.json').write_text(json.dumps(rec, indent=2, sort_keys=True) + '\n')
    print(json.dumps(rec['determinism'], indent=2, sort_keys=True))
    print('byte_identical:', identical)
    print('wrote', OUT / 'verification.json')


if __name__ == '__main__':
    main()
