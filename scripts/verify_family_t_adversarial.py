"""ADVERSARIAL VERIFICATION of Family T's T3 claim (54.906013 / R=0.092776).

Independent of the family's own report. Every number here is recomputed.

  1 REPRO        official() on the family's saved oof_T3_gated.csv and on my own re-run.
  2 PAIRED       b.paired(arm, protected) recomputed, all arms, with by-condition CIs.
  3 NESTING      my own re-derivation of the nested protocol from recomputed candidate q,
                 plus the oracle (outer-held argmin) for comparison.
  4 HELD-TRUTH   scramble the TRUTH of fold f's held episodes, redo fold f's inner
     INVARIANCE  selection: the chosen setting and the inner objectives must not move.
                 Power control: scrambling the FITTING truth must move them.
  5 FORECAST     scramble the whole truth vector, recompute the candidate q: must be
     TRUTH-INV   bit-identical, i.e. the forecast never reads a target.
  6 CAUSALITY    (a) every scored observation tick < cutoff and max tick == cutoff-1;
                 (b) y labels equal 1(payout>0) at the ACTUAL tick from the raw index;
                 (c) fabricated wins at cutoff..cutoff+5 change q by exactly 0;
                 (d) inverting all neighbour machines changes q by exactly 0;
                 (e) deleting one pre-cutoff tick DOES change q (test has power).
  7 GRID         re-run the nested selection on sub-grids, to bound the disclosed
     EXPOSURE    exposure from grid bounds informed by a public-validation sweep.
  8 PROVENANCE   classify() every file the arm reads.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import harness as H
from slot_model import win_head_infer as WI
from slot_forecasting.io import load_inputs, read_submission
from scripts import family_t_joint_structure_memory as FT
from scripts.stack_manifest import classify

KAPPAS = FT.KAPPAS
HAZARDS = FT.HAZARDS


def inner_objectives(b, look, names, fold):
    """Sum of the official weighted SSE over the 3 inner held partitions of the fitting set."""
    mask = b.outer_mask(fold)
    weights = [b.row_weights(he) for _, he in b.inner_folds(~mask, k=3)]
    return {n: float(sum(b.wloss(look[n], w) for w in weights)) for n in names}


def nested_from_lookup(b, look, order):
    held = np.empty((len(b.ids), 5))
    chosen, table = [], []
    for fold in range(4):
        mask = b.outer_mask(fold)
        obj = inner_objectives(b, look, order, fold)
        best = min(order, key=lambda n: (obj[n], order.index(n)))
        chosen.append(best)
        held[mask] = look[best][mask]
        table.append(obj)
    return held, chosen, table


def oracle_from_lookup(b, look, order):
    """Per-fold argmin on the HELD fold itself: the leak this protocol must NOT achieve."""
    held = np.empty((len(b.ids), 5))
    chosen = []
    for fold in range(4):
        mask = b.outer_mask(fold)
        w = b.row_weights(mask)
        best = min(order, key=lambda n: (b.wloss(look[n], w), order.index(n)))
        chosen.append(best)
        held[mask] = look[best][mask]
    return held, chosen


def t3_candidates(bundle, records, bank, lp, prior_mean, hazards=HAZARDS, kappas=KAPPAS):
    out = {}
    for kappa in kappas:
        res = FT.changepoint_pass(records, bank, lp, prior_mean, kappa, hazards,
                                  np.zeros(len(records), dtype=np.int64), variants=('T3',))
        for hi, h in enumerate(hazards):
            out[f'k{kappa:g}_h{h:g}'] = res['q3'][hi].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    ap.add_argument('--family', type=Path, default=ROOT / 'artifacts/family_t_joint')
    ap.add_argument('--mine', type=Path, default=ROOT / 'artifacts/verify_family_t_joint')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/verify_family_t_joint')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rep = {}

    b = H.load(args.data)
    q_prot, r_prot = b.split_heads(b.protected)
    model = b.package / 'code/model'
    kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    lp_full = np.asarray(log_prior, dtype=np.float64)
    full_bank = list(range(len(FT.TEMPLATES)))

    # ---- 1 REPRO -----------------------------------------------------------------
    rep['1_repro'] = {}
    for tag, d in (('family', args.family), ('my_rerun', args.mine)):
        f = d / 'predictions/oof_T3_gated.csv'
        if not f.exists():
            continue
        P = np.array([read_submission(f)[i] for i in b.ids])
        o = b.official(P)
        rep['1_repro'][tag] = {
            'points': o['technical_points'], 'R': o['mean_loss_ratio'],
            'by_condition': {c: o['by_condition'][c]['loss_ratio'] for c in sorted(o['by_condition'])}}
    rep['1_repro']['claimed_points'] = 54.90601318307562
    rep['1_repro']['abs_points_error_vs_claim'] = abs(
        rep['1_repro']['my_rerun']['points'] - 54.90601318307562)
    rep['1_repro']['protected'] = {'points': b.official(b.protected)['technical_points'],
                                   'R': b.official(b.protected)['mean_loss_ratio']}

    # ---- 2 PAIRED ----------------------------------------------------------------
    rep['2_paired'] = {}
    for name in ('T3_gated', 'T3_forced', 'TALL_gated', 'T2_gated', 'control'):
        f = args.mine / f'predictions/oof_{name}.csv'
        if not f.exists():
            continue
        P = np.array([read_submission(f)[i] for i in b.ids])
        pr = b.paired(P, b.protected)
        rep['2_paired'][name] = {'ci95_vs_protected': pr['delta_loss_ratio_ci95'],
                                 'by_condition_ci95': pr['by_condition_ci95'],
                                 'upper_bound_below_zero': bool(pr['delta_loss_ratio_ci95'][1] < 0)}
    Pc = np.array([read_submission(args.mine / 'predictions/oof_control.csv')[i] for i in b.ids])
    P3 = np.array([read_submission(args.mine / 'predictions/oof_T3_gated.csv')[i] for i in b.ids])
    rep['2_paired']['T3_vs_own_control_C1'] = b.paired(P3, Pc)['delta_loss_ratio_ci95']
    rep['2_paired']['control_C1_vs_protected_max_abs_diff'] = float(np.max(np.abs(Pc - b.protected)))

    # ---- recompute the T3 candidate grid myself ----------------------------------
    t0 = time.perf_counter()
    records = FT.prepare(b)
    cand = t3_candidates(b, records, full_bank, lp_full, prior_mean)
    rep['recompute_seconds'] = time.perf_counter() - t0
    look = {n: b.compose(q, r_prot) for n, q in cand.items()}
    look['control'] = b.compose(q_prot, r_prot)
    order_gated = ['control'] + [f'k{k:g}_h{h:g}' for k in KAPPAS for h in HAZARDS]
    order_forced = [n for n in order_gated if n != 'control']

    # ---- 3 NESTING ----------------------------------------------------------------
    rep['3_nesting'] = {}
    for tag, order in (('gated', order_gated), ('forced', order_forced)):
        held, chosen, table = nested_from_lookup(b, look, order)
        o = b.official(held)
        rep['3_nesting'][tag] = {
            'points': o['technical_points'], 'R': o['mean_loss_ratio'],
            'chosen_per_fold': chosen,
            'by_condition': {c: o['by_condition'][c]['loss_ratio'] for c in sorted(o['by_condition'])},
            'max_abs_diff_vs_family_saved': float(np.max(np.abs(held - P3)))
            if tag == 'gated' else None}
    oheld, ochosen = oracle_from_lookup(b, look, order_forced)
    oo = b.official(oheld)
    rep['3_nesting']['oracle_outer_held_argmin'] = {
        'points': oo['technical_points'], 'R': oo['mean_loss_ratio'], 'chosen_per_fold': ochosen}
    rep['3_nesting']['nested_equals_oracle_choice'] = bool(
        rep['3_nesting']['forced']['chosen_per_fold'] == ochosen)

    # ---- 4 HELD-TRUTH INVARIANCE --------------------------------------------------
    rng = np.random.default_rng(20261108)
    base = {f: inner_objectives(b, look, order_gated, f) for f in range(4)}
    truth0 = b.truth.copy()
    out4 = {}
    for fold in range(4):
        mask = b.outer_mask(fold)
        scr = truth0.copy()
        g = rng.dirichlet(np.ones(5), size=int(mask.sum()))
        scr[mask] = g
        b.truth = scr
        b._wcache = {}
        after = inner_objectives(b, look, order_gated, fold)
        out4[f'fold{fold}_held_scrambled'] = {
            'max_abs_inner_objective_change': max(abs(after[n] - base[fold][n]) for n in after),
            'argmin_unchanged': min(after, key=lambda n: (after[n], order_gated.index(n)))
            == min(base[fold], key=lambda n: (base[fold][n], order_gated.index(n)))}
    # power control: scramble the FITTING truth of fold 0
    mask0 = b.outer_mask(0)
    scr = truth0.copy()
    scr[~mask0] = rng.dirichlet(np.ones(5), size=int((~mask0).sum()))
    b.truth = scr
    b._wcache = {}
    after = inner_objectives(b, look, order_gated, 0)
    out4['power_control_fold0_fitting_scrambled'] = {
        'max_abs_inner_objective_change': max(abs(after[n] - base[0][n]) for n in after),
        'argmin_changed': min(after, key=lambda n: (after[n], order_gated.index(n)))
        != min(base[0], key=lambda n: (base[0][n], order_gated.index(n)))}
    b.truth = truth0
    b._wcache = {}
    rep['4_held_truth_invariance'] = out4

    # ---- 5 FORECAST TRUTH-INVARIANCE ---------------------------------------------
    saved = b.truth
    b.truth = rng.dirichlet(np.ones(5), size=len(b.ids))
    rec2 = FT.prepare(b)
    c2 = t3_candidates(b, rec2, full_bank, lp_full, prior_mean, hazards=(1e-5,), kappas=(1.0,))
    b.truth = saved
    rep['5_forecast_truth_invariance'] = {
        'setting': 'k1_h1e-05',
        'max_abs_q_change_when_all_targets_randomised':
            float(np.max(np.abs(c2['k1_h1e-05'] - cand['k1_h1e-05'])))}

    # ---- 6 CAUSALITY --------------------------------------------------------------
    queries, _ = load_inputs(b.data / 'validation')
    byq = {q.query_id: q for q in queries}
    worst_tick, worst_last, label_err, nobs = -10 ** 9, [], 0, 0
    for r, qid in zip(records, b.ids):
        q = byq[qid]
        ticks = b.index[(q.episode_id, q.machine_id)]
        worst_tick = max(worst_tick, float(np.max(r['ticks'])) - q.cutoff)
        worst_last.append(q.cutoff - float(np.max(r['ticks'])))
        for t, y in zip(r['ticks'].tolist(), r['y'].tolist()):
            nobs += 1
            if (1.0 if ticks[int(t)] > 0 else 0.0) != y:
                label_err += 1
    one = {'k1_h1e-05': cand['k1_h1e-05']}

    def recompute(index, tag):
        keep = b.index
        b.index = index
        try:
            rr = FT.prepare(b)
        finally:
            b.index = keep
        c = t3_candidates(b, rr, full_bank, lp_full, prior_mean, hazards=(1e-5,), kappas=(1.0,))
        return float(np.max(np.abs(c['k1_h1e-05'] - one['k1_h1e-05'])))

    inj = copy.deepcopy(b.index)
    for qid in b.ids:
        q = byq[qid]
        for d in range(0, 6):
            inj[(q.episode_id, q.machine_id)][q.cutoff + d] = 4
    d_inject = recompute(inj, 'inject')

    flip = copy.deepcopy(b.index)
    machines = {}
    for (e, m) in sorted(flip):
        machines.setdefault(e, []).append(m)
    for e in sorted(machines):
        for m in machines[e]:
            tk = flip[(e, m)]
            for t in sorted(tk):
                tk[t] = 0 if tk[t] > 0 else 2
    for qid in b.ids:
        q = byq[qid]
        flip[(q.episode_id, q.machine_id)] = dict(b.index[(q.episode_id, q.machine_id)])
    d_flip = recompute(flip, 'flip')

    drop = copy.deepcopy(b.index)
    for qid in b.ids:
        q = byq[qid]
        tk = drop[(q.episode_id, q.machine_id)]
        t = max(tk)
        tk[t] = 0 if tk[t] > 0 else 2          # flip the machine's OWN last pre-cutoff tick
    d_own = recompute(drop, 'own')

    rep['6_causality'] = {
        'max_tick_minus_cutoff': worst_tick,
        'cutoff_minus_last_tick_histogram': {str(k): int(v) for k, v in
                                             zip(*np.unique(np.array(worst_last), return_counts=True))},
        'scored_observations_checked': nobs,
        'label_mismatches_vs_raw_index': label_err,
        'max_abs_q_change_post_cutoff_injection_cutoff_to_plus5': d_inject,
        'max_abs_q_change_all_neighbours_inverted': d_flip,
        'max_abs_q_change_own_last_pre_cutoff_tick_flipped_POWER': d_own}

    # ---- 7 GRID EXPOSURE ----------------------------------------------------------
    sub = {
        'declared_full_grid': [f'k{k:g}_h{h:g}' for k in KAPPAS for h in HAZARDS],
        'drop_the_two_selected_decades_1e-5_1e-4': [f'k{k:g}_h{h:g}' for k in KAPPAS
                                                    for h in HAZARDS if h not in (1e-5, 1e-4)],
        'coarse_decades_only_0_1e-3_1e-2_1e-1': [f'k{k:g}_h{h:g}' for k in KAPPAS
                                                 for h in (0.0, 1e-3, 1e-2, 1e-1)],
        'endpoints_only_0_1e-1': [f'k{k:g}_h{h:g}' for k in KAPPAS for h in (0.0, 1e-1)],
        'single_kappa_0.5_full_hazards': [f'k0.5_h{h:g}' for h in HAZARDS],
    }
    rep['7_grid_exposure'] = {}
    for tag, order in sub.items():
        held, chosen, _ = nested_from_lookup(b, look, ['control'] + order)
        o = b.official(held)
        pr = b.paired(held, b.protected)
        rep['7_grid_exposure'][tag] = {
            'points': o['technical_points'], 'R': o['mean_loss_ratio'], 'chosen_per_fold': chosen,
            'ci95_vs_protected': pr['delta_loss_ratio_ci95'],
            'significant': bool(pr['delta_loss_ratio_ci95'][1] < 0)}

    # ---- 8 PROVENANCE -------------------------------------------------------------
    inputs = ['artifacts/win_head_release_exact/validation_submission.csv',
              'artifacts/win_head_release_exact/code/model/win_head/beta_templates_deploy.json',
              'artifacts/win_head_release_exact/code/model/win_head/population_averaged.json']
    rep['8_provenance'] = {p: classify(p) for p in inputs}
    rep['8_provenance']['note'] = ('r head and frozen constants all come from the protected '
                                   'release package; no stack_manifest RAW pool member is read.')
    rep['seconds'] = time.perf_counter() - start
    (args.out / 'adversarial_verification.json').write_text(
        json.dumps(rep, indent=2, sort_keys=True, default=float) + '\n')
    print(json.dumps(rep, indent=2, sort_keys=True, default=float))


if __name__ == '__main__':
    main()
