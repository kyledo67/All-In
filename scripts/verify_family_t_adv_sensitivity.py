"""Two sensitivity analyses that decide whether Family T's T2MAP gain is an artifact.

A. HONEST-r SENSITIVITY. The protected release's conditional-payout head r carries, per
   machine type, a tail split over {2,5,20} that is EXACTLY the validation TRUTH constant
   (artifacts/win_head_release_exact/code/model/win_head/tail_crossfit_deploy.json:
   "Fitted on ALL released validation probability targets"). Family T inherits that r
   unchanged for every arm AND for its control, so the question is not whether the leak is
   present -- it is -- but whether the measured GAIN depends on it. Rebuild r with a
   TRAIN-ONLY tail estimate (from public/train tick payouts, and separately from the 768
   train labels), keep each query's a = P(1x | positive) from the protected head, and
   re-score both the protected release and the nested T2MAP arm under the honest head.

B. GRID ROBUSTNESS. The family declares that the hazard/kappa grid BOUNDS were informed by
   an exploratory public-validation sweep. Re-run the identical nested protocol on
   deliberately coarser sub-grids that no exploratory sweep could have tuned, including one
   with no hazard finer than 1e-2, and report how much of the gain survives.

Determinism: sorted(...) everywhere; no set is summed over.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import harness as H
from slot_model import win_head_infer as WI
from slot_forecasting.io import load_inputs, read_labels
from scripts.family_t_joint_structure_memory import (HAZARDS, KAPPAS, TEMPLATES, changepoint_pass,
                                                     half_lives, nested, prepare, t1_mixture)

PAYOUTS = (0, 1, 2, 5, 20)
SUBGRIDS = {
    'declared_grid': [(k, h) for k in KAPPAS for h in HAZARDS],
    'kappa_half_only': [(0.5, h) for h in HAZARDS],
    'decades_1e-3_and_coarser': [(k, h) for k in KAPPAS for h in (0.0, 1e-3, 1e-2, 1e-1)],
    'decades_1e-2_and_coarser': [(k, h) for k in KAPPAS for h in (0.0, 1e-2, 1e-1)],
    'two_point_0_and_1e-2': [(k, h) for k in KAPPAS for h in (0.0, 1e-2)],
    'endpoints_only_0_and_1e-1': [(k, h) for k in KAPPAS for h in (0.0, 1e-1)],
}


def train_tail_estimates(data):
    """Per-type tail split over {2,5,20} from TRAIN only. Two independent estimators."""
    tq, th = load_inputs(data / 'train')
    mtype = {}
    for q in tq:
        mtype[(q.episode_id, q.machine_id)] = q.machine_type
    hist = defaultdict(lambda: np.zeros(3))
    for p in th:
        v = PAYOUTS.index(p.payout)
        if v >= 2:
            hist[mtype[(p.episode_id, p.machine_id)]][v - 2] += 1.0
    from_hist = {t: hist[t] / hist[t].sum() for t in sorted(hist)}
    labels = read_labels(data / 'train/labels.csv', tq)
    lab = defaultdict(lambda: np.zeros(3))
    for q in tq:
        v = PAYOUTS.index(labels[q.query_id])
        if v >= 2:
            lab[q.machine_type][v - 2] += 1.0
    from_lab = {t: lab[t] / lab[t].sum() for t in sorted(lab)}
    counts = {t: int(hist[t].sum()) for t in sorted(hist)}
    lcounts = {t: int(lab[t].sum()) for t in sorted(lab)}
    return from_hist, from_lab, counts, lcounts


def rebuild_r(bundle, r_prot, tail_by_type):
    """r = [a, (1-a) * tail_type], a preserved -- the release's own documented operator."""
    a = r_prot[:, 0]
    tail = np.array([tail_by_type[t] for t in bundle.types])
    out = np.column_stack((a, (1.0 - a)[:, None] * tail))
    return out / out.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/verify_family_t_adv')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    bundle = H.load(args.data)
    q_prot, r_prot = bundle.split_heads(bundle.protected)
    model = bundle.package / 'code/model'
    kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    full_bank = list(range(len(TEMPLATES)))
    lp_full = np.asarray(log_prior, dtype=np.float64)
    records = prepare(bundle)
    hl = half_lives(bundle, records, beta, lam)
    q_c1, pick_c1, _ = t1_mixture(records, hl, full_bank, lp_full, prior_mean, kappa0, None)
    pos = np.array([full_bank.index(int(p)) for p in pick_c1])

    q_by = {}
    for kappa in KAPPAS:
        res = changepoint_pass(records, full_bank, lp_full, prior_mean, kappa, HAZARDS, pos,
                               variants=('T2',))
        for hi, h in enumerate(HAZARDS):
            q_by[(kappa, h)] = res['q2map'][hi].copy()
    np.save(args.out / 'T2MAP_q_candidates.npy',
            np.stack([q_by[k] for k in sorted(q_by)]))
    (args.out / 'T2MAP_q_candidate_order.json').write_text(
        json.dumps([list(k) for k in sorted(q_by)]) + '\n')

    report = {'preregistration': __doc__}

    # ---- B GRID ROBUSTNESS -----------------------------------------------------------
    grids = {}
    for gname in sorted(SUBGRIDS):
        cand = [(f'k{k:g}_h{h:g}', q_by[(k, h)]) for k, h in SUBGRIDS[gname]]
        P, chosen, _ = nested(bundle, r_prot, q_c1, cand, True)
        o = bundle.official(P)
        pr = bundle.paired(P, bundle.protected)
        grids[gname] = {'settings': len(cand), 'points': o['technical_points'],
                        'R': o['mean_loss_ratio'], 'selected_per_fold': chosen,
                        'paired_vs_protected_ci95': pr['delta_loss_ratio_ci95'],
                        'significant_vs_protected': bool(pr['delta_loss_ratio_ci95'][1] < 0),
                        'points_minus_protected': o['technical_points'] - 53.532917808538166}
    report['B_grid_robustness'] = grids

    # ---- A HONEST-r SENSITIVITY ------------------------------------------------------
    from_hist, from_lab, counts, lcounts = train_tail_estimates(args.data)
    tc_truth = bundle.tail_constants(np.ones(len(bundle.ids), dtype=bool))
    heads = {'protected_truth_derived_tail': r_prot,
             'train_history_tail': rebuild_r(bundle, r_prot, from_hist),
             'train_label_tail': rebuild_r(bundle, r_prot, from_lab)}
    cand_full = [(f'k{k:g}_h{h:g}', q_by[(k, h)]) for k, h in SUBGRIDS['declared_grid']]
    rows = {}
    for hname in sorted(heads):
        rh = heads[hname]
        P_prot = bundle.compose(q_prot, rh)
        P_arm, chosen, _ = nested(bundle, rh, q_c1, cand_full, True)
        op, oa = bundle.official(P_prot), bundle.official(P_arm)
        pr = bundle.paired(P_arm, P_prot)
        rows[hname] = {
            'protected_points': op['technical_points'], 'protected_R': op['mean_loss_ratio'],
            'T2MAP_points': oa['technical_points'], 'T2MAP_R': oa['mean_loss_ratio'],
            'gain_points': oa['technical_points'] - op['technical_points'],
            'selected_per_fold': chosen,
            'paired_T2MAP_vs_protected_ci95': pr['delta_loss_ratio_ci95'],
            'significant': bool(pr['delta_loss_ratio_ci95'][1] < 0)}
    report['A_honest_r_sensitivity'] = {
        'tail_estimates': {
            'validation_TRUTH_constant_used_by_the_release': {t: tc_truth[t].tolist() for t in sorted(tc_truth)},
            'train_history_tail': {t: from_hist[t].tolist() for t in sorted(from_hist)},
            'train_history_tail_observation_counts': counts,
            'train_label_tail': {t: from_lab[t].tolist() for t in sorted(from_lab)},
            'train_label_tail_observation_counts': lcounts},
        'scores': rows}
    report['seconds'] = time.perf_counter() - start
    (args.out / 'sensitivity.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'preregistration'},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
