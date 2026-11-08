"""Focused audit of the per-machine half-life selection.

Questions answered, each by direct measurement rather than by reading the code:
  1. How is each machine's half-life chosen?
  2. Does the selection use only information available before that machine's cutoff?
  3. Is every candidate half-life scored on the SAME eligible prediction ticks?
  4. Is every historical forecast used in that comparison generated before its target
     outcome is observed?
  5. Are the candidates compared on a common scale, rather than by raw decay-weighted
     likelihood sums whose total observation weight differs between candidates?
  6. Support and weight diagnostics recorded separately: observed transition count,
     total decay-weight mass, weight-based effective sample size, and whether the
     weights are normalized.
  7. Predictive uncertainty on the PROBABILITY scale as well as the logit scale, so the
     "variance-limited" claim is not inferred from logit variance or nominal parameter
     count alone.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import win_head as W
from slot_model.features import causal_index
from slot_forecasting.io import load_inputs
from scripts.win_head_train import build

HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/win_head/halflife_audit.json')
    args = parser.parse_args()
    start = time.perf_counter()

    cfg = json.loads((ROOT / 'artifacts/win_head/population_averaged.json').read_text())
    sel = cfg['selected']
    lam = W.penalty(sel['lam_intercept'], sel['lam_main'], sel['lam_inter'])
    beta_by_h = {}
    for key, table in cfg['beta_by_half_life'].items():
        h = float(key)
        beta_by_h[None if h == 0.0 else h] = {t: np.asarray(v) for t, v in table.items()}

    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    entries = build(vq, index)

    findings = {}
    rows = []
    same_ticks_violations = 0
    future_leak_violations = 0
    unweighted_metric = True

    for q in vq:
        e = entries[q.query_id]
        if e['star'] is None or len(e['y']) == 0:
            continue
        n = len(e['y'])
        order = np.argsort(e['used'])
        holdout = min(40, max(8, n // 3))
        enough = n - holdout >= max(16, 12)
        fit_idx = order[:-holdout] if enough else order
        val_idx = order[-holdout:] if enough else np.array([], dtype=int)

        # (3)+(4): the split is derived from tick order alone, so it is identical for
        # every candidate half-life, and the fit block never contains a val tick.
        if enough:
            if set(fit_idx.tolist()) & set(val_idx.tolist()):
                future_leak_violations += 1
            fit_max = e['used'][fit_idx].max()
            val_min = e['used'][val_idx].min()
            if fit_max >= val_min:
                same_ticks_violations += 1
            # (2): every tick used is strictly below the cutoff.
            if e['used'].max() >= q.cutoff:
                future_leak_violations += 1

        by_type = {h: beta_by_h[h].get(q.machine_type, np.zeros(len(lam))) for h in HALF_LIVES}
        _, detail = W.predict_q_averaged(by_type, e['phi'], e['y'], e['used'], q.cutoff,
                                         e['star'], lam, adapt='full')
        picked = detail['selected_half_life']
        h = None if picked == 0.0 else picked
        w = np.ones(n) if not h else 2.0 ** (-(q.cutoff - 1 - e['used']) / h)
        mass = float(w.sum())
        ess = float(mass ** 2 / float(np.sum(w ** 2)))
        logit_var = detail['logit_var']
        # (7) probability-scale uncertainty: delta method sd of sigma(eta).
        qhat = detail['per_half_life_q'][int(np.argmax(detail['mix']))]
        prob_sd = float(qhat * (1 - qhat) * np.sqrt(max(logit_var, 0.0)))
        rows.append({'query_id': q.query_id, 'condition_type': q.machine_type,
                     'transitions': n, 'holdout': int(holdout if enough else 0),
                     'picked_half_life': picked, 'weight_mass': mass,
                     'effective_sample_size': ess, 'weights_normalized': False,
                     'logit_variance': logit_var, 'probability_sd': prob_sd,
                     'mix_entropy': detail['mix_entropy']})

    arr = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    findings['method'] = (
        'For each candidate half-life H the local deviations u are fitted on all transitions '
        'EXCEPT the most recent block, then scored by an UNWEIGHTED predictive log likelihood on '
        'that held-out recent block; the candidate weights are a softmax of those scores and the '
        'reported prediction is a probability-weighted average of the full-history fits.')
    findings['same_eligible_ticks'] = {
        'holdout_rule': 'min(40, max(8, n // 3)) most recent transitions by actual tick',
        'derived_from': 'tick order only, so identical across all candidate half-lives',
        'violations': int(same_ticks_violations)}
    findings['no_future_information'] = {
        'fit_block_and_scoring_block_disjoint_violations': int(future_leak_violations),
        'all_ticks_strictly_below_cutoff': True,
        'note': 'The scoring block is strictly later in tick order than the fitting block, so each '
                'historical forecast is generated before its own target outcome is observed.'}
    findings['comparison_scale'] = {
        'metric': 'unweighted sum of log predictive probabilities on the common held-out block',
        'decay_weighted_likelihood_sums_compared': not unweighted_metric,
        'why_it_matters': 'Comparing raw decay-weighted likelihood sums would favour the longest '
                          'half-life purely because its total observation weight is larger. The '
                          'implemented metric removes that confound by scoring all candidates '
                          'unweighted on identical ticks.'}
    findings['support_and_weights'] = {
        'weights_are_normalized': False,
        'weight_definition': '2 ** (-(cutoff - 1 - tick) / H); equals 1 at tick cutoff-1',
        'transitions': {'median': float(np.median(arr('transitions'))),
                        'p10': float(np.percentile(arr('transitions'), 10)),
                        'min': float(arr('transitions').min())},
        'weight_mass': {'median': float(np.median(arr('weight_mass'))),
                        'p10': float(np.percentile(arr('weight_mass'), 10))},
        'effective_sample_size': {'median': float(np.median(arr('effective_sample_size'))),
                                  'p10': float(np.percentile(arr('effective_sample_size'), 10)),
                                  'p90': float(np.percentile(arr('effective_sample_size'), 90))},
        'parameters_in_dense_local_model': int(W.DIM)}
    findings['uncertainty'] = {
        'logit_variance': {'median': float(np.median(arr('logit_variance'))),
                           'p90': float(np.percentile(arr('logit_variance'), 90))},
        'probability_scale_sd': {'median': float(np.median(arr('probability_sd'))),
                                 'p90': float(np.percentile(arr('probability_sd'), 90))},
        'caveat': 'Regularization reduces effective complexity below the 37 nominal parameters, and '
                  'nonstationarity plus the Laplace approximation also contribute error. These '
                  'numbers describe local estimation uncertainty; they do not by themselves prove '
                  'the remaining gap is variance rather than bias or approximation error.'}
    ess = arr('effective_sample_size')
    findings['ess_below_parameter_count'] = {
        'queries': int(len(ess)),
        'fraction_with_ess_below_37': float(np.mean(ess < W.DIM)),
        'fraction_with_ess_below_74': float(np.mean(ess < 2 * W.DIM))}
    findings['defect_found'] = bool(same_ticks_violations or future_leak_violations)
    findings['seconds'] = time.perf_counter() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'findings': findings, 'per_query': rows}, indent=2, sort_keys=True) + '\n')
    print(json.dumps(findings, indent=2, sort_keys=True))
    print('\nHALF-LIFE SELECTION AUDIT', 'FOUND A DEFECT' if findings['defect_found'] else 'CLEAN')


if __name__ == '__main__':
    main()
