"""Why is this approach at ~53/60? Identity checks, then the three-part error decomposition.

No model is fitted here. Everything is measured from saved aligned forecasts.

(1) IDENTITY. Establish what each reported number actually compares, since two of the
    previous claims depended on labels rather than arrays:
      - was r genuinely frozen across the memory policies, or did changing H move r too
        (the Dirichlet r inherits a half-life, so this must be checked, not assumed)
      - which control produced the reported H=16 gain of 0.300463, and is it matched
(2) DECOMPOSITION. With r fixed, the official query loss is
        L(q) = (q - u)^2 + ||q r - t_pos||^2
    minimised at q_star = clip((u + r.t_pos) / (1 + r.r), 0, 1). Split the current loss into
        A = L(q_star)                    unreachable by changing q alone, given this r
        B = L(q_range) - L(q_star)       the existing experts do not span the needed q
        C = L(q_current) - L(q_range)    recoverable by choosing better among current q
    with q_range = clip(q_star, min_H q_H, max_H q_H). Aggregate with the exact official
    query/episode/condition weights and require A+B+C to reproduce R.
(3) ORACLE REFRESH. The previously quoted +3.7503 / +2.5219 belong to the older 53.191542
    parity baseline. Recompute both substitutions against the current candidate.

All substitutions use organizer true vectors and are PRIVILEGED diagnostics. They do not
measure irreducible hidden-state uncertainty and are not achievable scores.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model.official_score import score, scoring_inputs
from slot_forecasting.io import read_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/plateau')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    cond = np.array([groups[q] for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())

    def load(p):
        d = read_submission(p)
        missing = set(ids) - set(d)
        if missing:
            raise SystemExit(f'{p} is missing {len(missing)} query ids')
        return np.array([d[q] for q in ids])

    def official(a):
        return score(dict(zip(ids, a.tolist())), scoring, episodes)

    findings = {}

    # ---------- (1) identity ----------
    dep = load(ROOT / 'artifacts/win_head_release_dirichlet/validation_submission.csv')
    q_dep = 1.0 - dep[:, 0]
    r_dep = dep[:, 1:] / np.maximum(q_dep, 1e-12)[:, None]
    r_dep /= r_dep.sum(axis=1, keepdims=True)
    glob = load(ROOT / 'artifacts/memory_policies/predictions/oof_global_heldfold.csv')
    q_glob = 1.0 - glob[:, 0]
    r_glob = glob[:, 1:] / np.maximum(q_glob, 1e-12)[:, None]
    r_glob /= r_glob.sum(axis=1, keepdims=True)
    proc = load(ROOT / 'artifacts/dirichlet_r/predictions/oof_r_blend.csv')
    q_proc = 1.0 - proc[:, 0]
    r_proc = proc[:, 1:] / np.maximum(q_proc, 1e-12)[:, None]
    r_proc /= r_proc.sum(axis=1, keepdims=True)

    findings['r_frozen_check'] = {
        'max_abs_r_difference_deployed_vs_global_H16': float(np.max(abs(r_dep - r_glob))),
        'r_was_frozen_across_memory_policies': bool(np.max(abs(r_dep - r_glob)) < 1e-12),
        'max_abs_q_difference_deployed_vs_global_H16': float(np.max(abs(q_dep - q_glob))),
        'interpretation':
            'If the r difference is zero, the H=16 comparison is a genuine q-only contrast and '
            'the Dirichlet r did NOT inherit the varied H. The r half-life comes from the dense '
            'averaged selector inside dirichlet_proportions, not from the q expert being varied.'}
    findings['matched_control'] = {
        'deployed_fixed_replay_points': official(dep)['technical_points'],
        'global_H16_points': official(glob)['technical_points'],
        'difference': official(glob)['technical_points'] - official(dep)['technical_points'],
        'heldfold_procedure_points': official(proc)['technical_points'],
        'max_abs_r_difference_deployed_vs_heldfold_procedure': float(np.max(abs(r_dep - r_proc))),
        'note':
            'The reported 0.300463 compares global_H16 against the DEPLOYED FIXED REPLAY. Both '
            'share the identical w=0.42 deployment r, so the contrast is matched on r and isolates '
            'q. But global_H16 therefore inherits a deployment-replay r and is NOT a pure '
            'held-fold procedure, which is how it was previously labelled. Comparing it against '
            'the 53.359683 held-fold procedure would be UNMATCHED, because that prediction uses '
            'fold-specific r weights.'}

    # ---------- (2) three-part decomposition ----------
    experts = np.load(ROOT / 'artifacts/memory_envelope/experts.npy')
    u = 1.0 - truth[:, 0]
    t_pos = truth[:, 1:]
    rr = np.sum(r_dep ** 2, axis=1)
    q_star = np.clip((u + np.sum(r_dep * t_pos, axis=1)) / (1.0 + rr), 0.0, 1.0)

    def loss_of(q):
        return (q - u) ** 2 + np.sum((q[:, None] * r_dep - t_pos) ** 2, axis=1)

    lo = np.nanmin(experts, axis=1)
    hi = np.nanmax(experts, axis=1)
    inside = (q_dep >= lo - 1e-12) & (q_dep <= hi + 1e-12)
    # Include current q as a control expert so term C cannot be negative by construction.
    lo_c = np.minimum(lo, q_dep)
    hi_c = np.maximum(hi, q_dep)
    q_range = np.clip(q_star, lo_c, hi_c)

    # official weights: contribution of query i is L_i / (4 N_c n_e B_c)
    weight = np.zeros(len(ids))
    B = {}
    for c in sorted(set(cond.tolist())):
        es = sorted({e for e, cc in zip(ep, cond) if cc == c})
        B[c] = float(np.mean([np.mean(np.sum((ref_arr[ep == e] - truth[ep == e]) ** 2, axis=1))
                              for e in es]))
        for e in es:
            m = ep == e
            weight[m] = 1.0 / (4 * len(es) * int(m.sum()) * max(B[c], 1e-9))

    L_cur, L_rng, L_star = loss_of(q_dep), loss_of(q_range), loss_of(q_star)
    A = float(np.sum(weight * L_star))
    Bterm = float(np.sum(weight * (L_rng - L_star)))
    C = float(np.sum(weight * (L_cur - L_rng)))
    R_official = official(dep)['mean_loss_ratio']
    findings['decomposition'] = {
        'R_official': R_official, 'A_unreachable_given_this_r': A,
        'B_experts_do_not_span': Bterm, 'C_better_choice_among_experts': C,
        'sum': A + Bterm + C, 'identity_residual': abs(A + Bterm + C - R_official),
        'shares': {'A': A / R_official, 'B': Bterm / R_official, 'C': C / R_official},
        'current_q_inside_expert_interval': {'queries': int(inside.sum()), 'of': len(ids),
                                             'fraction': float(inside.mean())},
        'note': 'A is not irreducible uncertainty. It is the error that cannot be removed by '
                'changing q alone while holding THIS r fixed, so improving r moves it.'}
    per_cond = {}
    for c in sorted(set(cond.tolist())):
        m = cond == c
        per_cond[c] = {'A': float(np.sum(weight[m] * L_star[m]) * 4),
                       'B': float(np.sum(weight[m] * (L_rng - L_star)[m]) * 4),
                       'C': float(np.sum(weight[m] * (L_cur - L_rng)[m]) * 4),
                       'ratio': official(dep)['by_condition'][c]['loss_ratio']}
    findings['decomposition_by_condition'] = per_cond

    # independent scalar check of q_star on a tiny example
    rt = np.array([0.5, 0.3, 0.15, 0.05])
    tt = np.array([0.2, 0.1, 0.05, 0.05])
    uu = tt.sum()
    grid = np.linspace(0, 1, 2000001)
    brute = grid[int(np.argmin((grid - uu) ** 2
                               + np.sum((grid[:, None] * rt - tt) ** 2, axis=1)))]
    closed = float(np.clip((uu + rt @ tt) / (1 + rt @ rt), 0, 1))
    findings['scalar_optimum_check'] = {'closed_form': closed, 'brute_force_grid': float(brute),
                                        'abs_difference': abs(closed - float(brute))}

    # ---------- (3) oracle refresh on the CURRENT candidate ----------
    sub_q = np.column_stack((1 - u, u[:, None] * r_dep))
    sub_r = np.column_stack((1 - q_dep, q_dep[:, None] * (t_pos / np.maximum(
        t_pos.sum(axis=1, keepdims=True), 1e-12))))
    both = truth.copy()
    cur_pts = official(dep)['technical_points']
    findings['oracle_refresh'] = {
        'current_candidate_points': cur_pts,
        'true_q_with_current_r': official(sub_q / sub_q.sum(1, keepdims=True))['technical_points'],
        'true_r_with_current_q': official(sub_r / sub_r.sum(1, keepdims=True))['technical_points'],
        'both_true': official(both)['technical_points'],
        'stale_values_belonged_to': '53.191542 parity baseline (true_q 56.9418, true_r 55.7135)',
        'note': 'PRIVILEGED. Gains do not add and are not ceilings.'}
    findings['envelope_share_check'] = {
        'claimed_fraction_of_R_removed': 0.274,
        'measured_fraction_removed_by_q_range': float((Bterm + C) / R_official)
        if False else float(C / R_official + 0.0),
        'envelope_points': 55.042506,
        'measured_R_after_q_range': float(A + Bterm),
        'measured_fraction_of_R_remaining': float((A + Bterm) / R_official)}

    (args.out / 'diagnosis.json').write_text(json.dumps(findings, indent=2, sort_keys=True) + '\n')

    f = findings
    print('--- (1) identity ---')
    print(f"r frozen across memory policies: {f['r_frozen_check']['r_was_frozen_across_memory_policies']} "
          f"(max |dr| = {f['r_frozen_check']['max_abs_r_difference_deployed_vs_global_H16']:.2e}, "
          f"max |dq| = {f['r_frozen_check']['max_abs_q_difference_deployed_vs_global_H16']:.4f})")
    mc = f['matched_control']
    print(f"deployed fixed replay {mc['deployed_fixed_replay_points']:.6f} | "
          f"global H=16 {mc['global_H16_points']:.6f} | diff {mc['difference']:+.6f}")
    print(f"held-fold procedure  {mc['heldfold_procedure_points']:.6f}  "
          f"(its r differs from deployed by {mc['max_abs_r_difference_deployed_vs_heldfold_procedure']:.2e})")
    print('\n--- (2) decomposition of R ---')
    d = f['decomposition']
    print(f"R = {d['R_official']:.6f}; A+B+C = {d['sum']:.6f} (residual {d['identity_residual']:.2e})")
    print(f"  A unreachable by q alone, given this r : {d['A_unreachable_given_this_r']:.6f}  "
          f"{100*d['shares']['A']:5.1f}%")
    print(f"  B experts do not span needed q         : {d['B_experts_do_not_span']:.6f}  "
          f"{100*d['shares']['B']:5.1f}%")
    print(f"  C better choice among current experts  : {d['C_better_choice_among_experts']:.6f}  "
          f"{100*d['shares']['C']:5.1f}%")
    print(f"  current q inside expert interval: {d['current_q_inside_expert_interval']['queries']}"
          f"/{d['current_q_inside_expert_interval']['of']}")
    print('  by condition (A/B/C, condition-normalised):')
    for c, v in per_cond.items():
        print(f"    {c:<14} A={v['A']:.4f} B={v['B']:.4f} C={v['C']:.4f}  ratio={v['ratio']:.4f}")
    s = f['scalar_optimum_check']
    print(f"\nscalar optimum check: closed {s['closed_form']:.9f} vs brute {s['brute_force_grid']:.9f} "
          f"(diff {s['abs_difference']:.2e})")
    o = f['oracle_refresh']
    print('\n--- (3) oracle refresh on the CURRENT candidate ---')
    print(f"  current {o['current_candidate_points']:.4f} | true q -> {o['true_q_with_current_r']:.4f} "
          f"(+{o['true_q_with_current_r']-o['current_candidate_points']:.4f}) | "
          f"true r -> {o['true_r_with_current_q']:.4f} "
          f"(+{o['true_r_with_current_q']-o['current_candidate_points']:.4f}) | "
          f"both -> {o['both_true']:.4f}")
    print(f"wrote {args.out / 'diagnosis.json'}")


if __name__ == '__main__':
    main()
