"""ADVERSARIAL VERIFICATION of Family T's T2 arm (bounded changepoint, structure fixed).

Independent of scripts/family_t_joint_structure_memory.py for the parts under test:
the T2 run-length recursion is re-derived here from scratch (explicit Adams-MacKay loop,
no shared code), the nested assembly is re-derived, and the score and paired interval are
recomputed with the harness. The only shared code used is input preparation (prepare /
half_lives / t1_mixture) whose causality is checked separately.

Checks
  R1  reproduce the saved T2 held-fold forecast from an independent recursion
  R2  official() on the concatenated held folds == reported points
  R3  b.paired(T2, protected) upper bound < 0
  R4  r frozen: r(T2 oof) == r(protected) row by row
  R5  independent re-derivation of the per-fold inner selection, fitting episodes only
  R6  HELD-TARGET INVARIANCE: corrupt truth on fold f's held episodes, selection for f
      must not move (repeat for all 4 folds, several corruptions)
  R7  non-vacuity: flipping the outcome at tick cutoff-1 MUST move q
  R8  clean-r robustness: recompose the T2 and protected q heads with a RAW r head
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
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
from scripts.family_t_joint_structure_memory import (TEMPLATES, HAZARDS, KAPPAS, prepare,
                                                     half_lives, t1_mixture)
from slot_model import win_head_infer as _WI

OUT = ROOT / 'artifacts/verify_family_t2_independent'
RUN = ROOT / 'artifacts/family_t_joint'


# ---------------------------------------------------------------- independent recursion
def cells_for(signs, star_signs, subset, template):
    c, width = _WI._template_cells(signs, subset, template)
    s, _ = _WI._template_cells(star_signs, subset, template)
    return np.asarray(c, dtype=int), int(s[0]), int(width)


def t2_q_slow(record, subset, template, kappa, mu, h):
    """Exact Adams-MacKay run-length posterior at ONE fixed template.

    Pure python, O(n^3), written straight from the model statement with no prefix-sum
    trick and no shared code. Used as the ground truth for the vectorised version below.
    """
    y = [float(v) for v in record['y']]
    ticks = [float(v) for v in record['ticks']]
    n = len(y)
    c, cstar, _ = cells_for(record['signs'], record['star_signs'], subset, template)
    c = [int(v) for v in c]
    a0, b0 = kappa * mu, kappa * (1.0 - mu)
    w = [1.0] + [0.0] * n
    for k in range(n):
        keep = 1.0 if k == 0 else (1.0 - h) ** (ticks[k] - ticks[k - 1])
        a = [keep * w[r] for r in range(k + 1)]
        if k > 0:
            a[0] += (1.0 - keep) * sum(w[: k + 1])
        theta = []
        for r in range(k + 1):
            win = cnt = 0.0
            for t in range(k - r, k):
                if c[t] == c[k]:
                    cnt += 1.0
                    win += y[t]
            theta.append((a0 + win) / (a0 + b0 + cnt))
        lik = [th if y[k] > 0 else 1.0 - th for th in theta]
        m = [a[r] * lik[r] for r in range(k + 1)]
        tot = sum(m)
        w = [0.0] + [v / tot for v in m] + [0.0] * (n - k - 1)
    keep = (1.0 - h) ** (record['cutoff'] - ticks[-1])
    a = [keep * w[r] for r in range(n + 1)]
    a[0] += (1.0 - keep) * sum(w)
    tot = sum(a)
    a = [v / tot for v in a]
    out = 0.0
    for r in range(n + 1):
        win = cnt = 0.0
        for t in range(n - r, n):
            if c[t] == cstar:
                cnt += 1.0
                win += y[t]
        out += a[r] * (a0 + win) / (a0 + b0 + cnt)
    return float(out)


def t2_q(record, subset, template, kappa, mu, h):
    """Same recursion, vectorised over run length with my own cumulative-count tables."""
    y = np.asarray(record['y'], dtype=float)
    ticks = np.asarray(record['ticks'], dtype=float)
    n = len(y)
    c, cstar, width = cells_for(record['signs'], record['star_signs'], subset, template)
    a0, b0 = kappa * mu, kappa * (1.0 - mu)
    # cw[j, k] / cn[j, k]: wins / observations in cell j among observations with index < k
    cw = np.zeros((width, n + 1))
    cn = np.zeros((width, n + 1))
    for j in range(width):
        hit = (c == j).astype(float)
        cw[j, 1:] = np.cumsum(hit * y)
        cn[j, 1:] = np.cumsum(hit)
    w = np.zeros(n + 1)
    w[0] = 1.0
    for k in range(n):
        keep = 1.0 if k == 0 else (1.0 - h) ** float(ticks[k] - ticks[k - 1])
        a = keep * w[: k + 1]
        if k > 0:
            a = a.copy()
            a[0] += (1.0 - keep) * w[: k + 1].sum()
        lo = k - np.arange(k + 1)
        j = c[k]
        win = cw[j, k] - cw[j, lo]
        cnt = cn[j, k] - cn[j, lo]
        theta = (a0 + win) / (a0 + b0 + cnt)
        lik = theta if y[k] > 0 else 1.0 - theta
        m = a * lik
        nxt = np.zeros(n + 1)
        nxt[1: k + 2] = m / m.sum()
        w = nxt
    keep = (1.0 - h) ** float(record['cutoff'] - ticks[-1])
    a = keep * w.copy()
    a[0] += (1.0 - keep) * w.sum()
    a = a / a.sum()
    lo = n - np.arange(n + 1)
    win = cw[cstar, n] - cw[cstar, lo]
    cnt = cn[cstar, n] - cn[cstar, lo]
    return float(np.dot(a, (a0 + win) / (a0 + b0 + cnt)))


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    b = H.load(ROOT / 'main-participants-60-40/public')
    q_prot, r_prot = b.split_heads(b.protected)
    model = b.package / 'code/model'
    kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    lp_full = np.asarray(log_prior, dtype=np.float64)
    full_bank = list(range(len(TEMPLATES)))

    records = prepare(b)
    hl = half_lives(b, records, beta, lam)
    q_c1, pick_c1, _ = t1_mixture(records, hl, full_bank, lp_full, prior_mean, kappa0, None)
    report['C1_max_abs_q_dev_from_protected'] = float(np.max(np.abs(q_c1 - q_prot)))

    saved = read_submission(RUN / 'predictions/oof_T2_gated.csv')
    P_saved = np.array([saved[i] for i in b.ids])
    P_saved = P_saved / P_saved.sum(axis=1, keepdims=True)
    q_saved, r_saved = b.split_heads(P_saved)

    # R4 -- r frozen ------------------------------------------------------------------
    report['R4_max_abs_r_dev_T2_vs_protected'] = float(np.max(np.abs(r_saved - r_prot)))

    # R2 -- score the concatenated held folds once ------------------------------------
    o = b.official(P_saved)
    report['R2_official_on_saved_oof'] = {
        'points': o['technical_points'], 'R': o['mean_loss_ratio'],
        'by_condition': {c: o['by_condition'][c]['loss_ratio'] for c in sorted(o['by_condition'])}}
    rep = json.loads((RUN / 'evaluation.json').read_text())
    report['R2_reported_points'] = rep['arms']['T2']['gated']['points']
    report['R2_abs_diff'] = abs(o['technical_points'] - rep['arms']['T2']['gated']['points'])

    # R3 -- paired interval -----------------------------------------------------------
    pv = b.paired(P_saved, b.protected)
    report['R3_paired_vs_protected_ci95'] = pv['delta_loss_ratio_ci95']
    report['R3_by_condition_ci95'] = pv['by_condition_ci95']
    report['R3_upper_below_zero'] = bool(pv['delta_loss_ratio_ci95'][1] < 0)
    pc = b.paired(P_saved, b.compose(q_c1, r_prot))
    report['R3_paired_vs_control_C1_ci95'] = pc['delta_loss_ratio_ci95']

    # R1/R5 -- independent recursion on the full (kappa, hazard) grid -----------------
    tpl = [TEMPLATES[int(p)] for p in pick_c1]
    # agreement between the pure-python O(n^3) reference and the vectorised version
    nobs = np.array([len(r['y']) for r in records])
    probe = list(np.argsort(nobs, kind='stable')[:6]) + list(np.argsort(nobs, kind='stable')[-2:])
    worst = 0.0
    for i in probe:
        i = int(i)
        mu = float(prior_mean.get(records[i]['type'], 0.44))
        for kappa, h in ((0.5, 1e-4), (1.0, 1e-5), (2.0, 1e-2)):
            s1 = t2_q_slow(records[i], tpl[i][0], tpl[i][1], kappa, mu, h)
            s2 = t2_q(records[i], tpl[i][0], tpl[i][1], kappa, mu, h)
            worst = max(worst, abs(s1 - s2))
    report['R1_slow_vs_vectorised_max_abs'] = worst
    report['R1_probe_n_observations'] = [int(nobs[int(i)]) for i in probe]
    grid = [(k, h) for k in KAPPAS for h in HAZARDS]
    qgrid = {}
    for kappa, h in grid:
        qq = np.empty(len(records))
        for i, r in enumerate(records):
            mu = float(prior_mean.get(r['type'], 0.44))
            qq[i] = t2_q(r, tpl[i][0], tpl[i][1], kappa, mu, h)
        qgrid[f'k{kappa:g}_h{h:g}'] = qq
        print(f'  independent T2 k={kappa:g} h={h:g} done '
              f'({time.perf_counter()-start:.0f}s)', flush=True)

    # independent nested assembly, fitting episodes only
    look = {t: b.compose(q, r_prot) for t, q in qgrid.items()}
    look_ctl = b.compose(q_c1, r_prot)
    order = ['control'] + [f'k{k:g}_h{h:g}' for k in KAPPAS for h in HAZARDS]
    allp = dict(look); allp['control'] = look_ctl

    def select(truth=None):
        """Per-fold argmin of the inner-held objective. `truth` optionally overrides."""
        bb = b
        if truth is not None:
            bb = copy.copy(b)
            bb.truth = truth
            bb._wcache = {}
        picks = []
        for fold in range(4):
            mask = bb.outer_mask(fold)
            ws = [bb.row_weights(he) for _, he in bb.inner_folds(~mask, k=3)]
            best = None
            for name in order:
                tot = float(sum(bb.wloss(allp[name], w) for w in ws))
                if best is None or tot < best[0]:
                    best = (tot, name)
            picks.append(best[1])
        return picks

    picks = select()
    report['R5_independent_selected_per_fold'] = picks
    report['R5_reported_selected_per_fold'] = rep['arms']['T2']['gated']['selected_per_fold']
    held = np.empty((len(b.ids), 5))
    for fold in range(4):
        m = b.outer_mask(fold)
        held[m] = allp[picks[fold]][m]
    report['R1_max_abs_dev_independent_vs_saved_oof'] = float(np.max(np.abs(held - P_saved)))
    oi = b.official(held)
    report['R1_independent_points'] = oi['technical_points']
    report['R1_independent_R'] = oi['mean_loss_ratio']

    # R6 -- held-target invariance -----------------------------------------------------
    rng = np.random.default_rng(20261108)
    inv = {}
    for fold in range(4):
        m = b.outer_mask(fold)
        ok = True
        for trial in range(3):
            t2 = b.truth.copy()
            draw = rng.dirichlet(np.ones(5), size=int(m.sum()))
            t2[m] = draw
            p = select(truth=t2)
            inv[f'fold{fold}_trial{trial}'] = {'picks': p, 'fold_pick_unchanged':
                                               bool(p[fold] == picks[fold]),
                                               'other_folds_unchanged':
                                               bool([p[j] for j in range(4) if j != fold]
                                                    == [picks[j] for j in range(4) if j != fold])}
            ok = ok and p[fold] == picks[fold]
        inv[f'fold{fold}_all_trials_pick_unchanged'] = ok
    report['R6_held_target_invariance'] = inv
    report['R6_all_folds_invariant'] = bool(
        all(inv[f'fold{f}_all_trials_pick_unchanged'] for f in range(4)))
    moved = []
    for trial in range(3):
        t2 = b.truth.copy()
        fitm = ~b.outer_mask(0)
        t2[fitm] = rng.dirichlet(np.ones(5), size=int(fitm.sum()))
        moved.append(select(truth=t2))
    report['R6_nonvacuity_corrupt_FITTING_truth_picks'] = moved
    report['R6_nonvacuity_fold0_pick_moved'] = bool(
        any(p[0] != picks[0] for p in moved))

    # R7 -- non-vacuity: the forecast MUST react to the last pre-cutoff tick ----------
    queries, _ = load_inputs(b.data / 'validation')
    by_query = {q.query_id: q for q in queries}
    flip = copy.deepcopy(b.index)
    for qid in b.ids:
        q = by_query[qid]
        t = flip[(q.episode_id, q.machine_id)]
        key = q.cutoff - 1
        if key in t:
            t[key] = 0 if t[key] > 0 else 2
    savedix = b.index
    b.index = flip
    try:
        rec2 = prepare(b)
    finally:
        b.index = savedix
    k_, h_ = 1.0, 1e-4
    qa = np.array([t2_q(records[i], tpl[i][0], tpl[i][1], k_,
                        float(prior_mean.get(records[i]['type'], 0.44)), h_)
                   for i in range(len(records))])
    qb = np.array([t2_q(rec2[i], tpl[i][0], tpl[i][1], k_,
                        float(prior_mean.get(rec2[i]['type'], 0.44)), h_)
                   for i in range(len(rec2))])
    report['R7_lag1_flip_max_abs_q_change'] = float(np.max(np.abs(qa - qb)))
    report['R7_lag1_flip_mean_abs_q_change'] = float(np.mean(np.abs(qa - qb)))
    report['R7_changed'] = bool(np.max(np.abs(qa - qb)) > 1e-6)

    # R8 -- clean-r robustness ---------------------------------------------------------
    clean = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    C = np.array([clean[i] for i in b.ids]); C = C / C.sum(axis=1, keepdims=True)
    _, r_clean = b.split_heads(C)
    P_prot_clean = b.compose(q_prot, r_clean)
    P_t2_clean = b.compose(q_saved, r_clean)
    op, ot = b.official(P_prot_clean), b.official(P_t2_clean)
    pr = b.paired(P_t2_clean, P_prot_clean)
    report['R8_clean_r'] = {
        'source': 'artifacts/two_head_01/validation.csv (RAW_SOURCES, classify()==RAW)',
        'protected_q_with_clean_r_points': op['technical_points'],
        'protected_q_with_clean_r_R': op['mean_loss_ratio'],
        'T2_q_with_clean_r_points': ot['technical_points'],
        'T2_q_with_clean_r_R': ot['mean_loss_ratio'],
        'paired_T2_vs_protected_ci95': pr['delta_loss_ratio_ci95']}

    report['seconds'] = time.perf_counter() - start
    (OUT / 'adversarial_verification.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
