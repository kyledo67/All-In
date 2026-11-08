"""FAMILY D, STEP 2: score the pair-sufficient support recovery under frozen nesting.

Arms (q head only; the conditional payout head r is the protected release's own r,
recovered with Bundle.split_heads and never refitted, so every arm differs ONLY in q):

  D0  no change / zero correction. The selected template's counts from the
      complete-eight-lag dense design. Verified bit-identical to the protected release.
  D1  identical selected structure, memory, prior and r, but the SELECTED template is
      updated on EVERY strictly causal transition whose required pair lags and outcome
      are observed.
  D2  probability blend q = (1-lam) q0 + lam q1, lam chosen per outer fold by 3-fold
      inner CV on the fitting episodes only; lam = 0 is in the grid.
  D3  sufficient-statistic shrinkage: the RECOVERED pseudo-counts enter the conjugate
      update with weight rho, chosen per outer fold by inner CV; rho = 0 is in the grid.
  D4  support-gated: apply D1 only where the dense row count is <= T, T chosen per outer
      fold by inner CV; T = 0 (never) is in the grid. n_dense is observable at inference.
  D5  cell-support-gated: apply D1 only where the QUERIED cell's dense effective
      (recency-weighted) support is <= C, C chosen per outer fold by inner CV; C = 0
      (never) is in the grid. That statistic is observable at inference too.
  DB  the brief's explicit "decision to use the method at all": a two-point nested
      choice between D0 and D1 inside each outer fold, nothing else tuned.
  DS  honest super-selection. One inner CV over the WHOLE declared recipe family
      {D0, D1, every D2 lam, every D3 rho, every D4 T, every D5 C}, so the cost of
      choosing among the family members is paid inside each outer fold instead of being
      hidden by quoting the best outer-table row. This is the number to believe.
  DT  DIAGNOSTIC TRAP, reported to be rejected, not shipped: re-select the template by
      comparing conjugate log evidences each computed over its OWN pair-sufficient
      response set. Those evidences are integrals over DIFFERENT response sets and their
      magnitudes are NOT comparable; a candidate that scores more observations gets a
      systematically different scale. This arm exists to measure how far that invalid
      comparison moves the score, never to justify it.

Nesting. The four frozen complete-episode outer folds come from the harness. Every
target-dependent quantity in D2/D3/D4 -- the blend weight, the shrinkage, the gate and
the decision to apply the method at all -- is chosen inside each outer fold using only
that fold's fitting episodes via b.inner_folds, with the official fitting-side row
weights recomputed on each inner-held partition. Held-fold forecasts are concatenated
and scored ONCE. q0/q1/qT themselves contain no validation target: beta, kappa, the
template log prior and the per-type prior means all come from the frozen release model
directory, and r comes from the protected submission.

Determinism: every reduction iterates sorted(...); grids are explicit tuples; ties go to
the first (smallest) grid entry.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path.insert(0, str(ROOT))

from slot_model import harness as H                      # noqa: E402
from slot_model import pair_dict as PD                   # noqa: E402
from slot_model import win_head as W                     # noqa: E402
from slot_model import win_head_infer as WI              # noqa: E402

DATA = ROOT / 'main-participants-60-40/public'
MODEL = ROOT / 'artifacts/win_head_release_exact/code/model/win_head'
OUT = ROOT / 'artifacts/family_d_pair_support'
TEMPLATES = [(None, 'null')] + [(p, k) for p in PD.PAIRS for k in ('full', 'parity')]
LAM_GRID = tuple(round(0.1 * k, 2) for k in range(11))
RHO_GRID = tuple(round(0.1 * k, 2) for k in range(11))
T_GRID = (0, 8, 16, 24, 32, 48, 64, 128, 10 ** 9)
C_GRID = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 1e9)
REGIMES = ((0, 16), (17, 32), (33, 64), (65, 128), (129, 10 ** 9))
KAPPA = 0.5   # frozen release value; asserted against the loaded deploy file below


def regime_of(n):
    for lo, hi in REGIMES:
        if lo <= n <= hi:
            return f'{lo}-{hi}' if hi < 10 ** 8 else f'{lo}+'
    raise AssertionError


def trap_heads(b, beta_by_half_life, lam, kappa, prior_mean, log_prior):
    """DIAGNOSTIC ONLY. Re-select the template using per-candidate pair-sufficient rows.

    Returns (q, chosen_index). The log evidences being argmaxed here are computed over
    different response sets, so this comparison is invalid by construction.
    """
    from scipy.special import betaln
    keys = [(b.ep[i], q.rsplit('-', 1)[0]) for i, q in enumerate(b.ids)]
    q = np.zeros(len(b.ids))
    chosen, changed = [], 0
    for i, qid in enumerate(b.ids):
        ticks = b.index[keys[i]]
        cutoff = int(b.cutoffs[i])
        mtype = b.types[i]
        phi, labels, _, used = W.design(ticks, cutoff, half_life=None)
        star = W.query_row(ticks, cutoff)
        by_type = {h: table.get(mtype, np.zeros(W.DIM))
                   for h, table in beta_by_half_life.items()}
        _, mix = W.predict_q_averaged(by_type, phi, labels, np.asarray(used), cutoff,
                                      star, lam, adapt='full')
        picked = mix['selected_half_life']
        half_life = None if picked == 0.0 else picked
        mu = float(prior_mean.get(mtype, 0.44))
        a0, b0 = kappa * mu, kappa * (1.0 - mu)
        ts = np.asarray(sorted(t for t in ticks if t < cutoff), dtype=np.int64)
        present = np.asarray(sorted(ticks), dtype=np.int64)
        y_all = np.asarray([1.0 if ticks[int(t)] > 0 else 0.0 for t in ts])
        w_all = (np.ones(len(ts)) if not half_life
                 else 2.0 ** (-(cutoff - 1 - ts.astype(float)) / half_life))
        star_signs = star[1:1 + W.HORIZON]
        best = (-np.inf, 0, float('nan'))
        for it, (subset, template) in enumerate(TEMPLATES):
            if subset is None:
                keep = np.ones(len(ts), dtype=bool)
            else:
                keep = np.ones(len(ts), dtype=bool)
                for l in subset:
                    keep &= np.isin(ts - l, present)
            if not keep.any():
                continue
            if subset is None:
                sg = np.zeros((int(keep.sum()), W.HORIZON))
            else:
                sg = np.zeros((int(keep.sum()), W.HORIZON))
                for l in subset:
                    sg[:, l - 1] = [1.0 if ticks[int(t) - l] > 0 else -1.0
                                    for t in ts[keep]]
            idx, ncell = WI._template_cells(sg, subset, template)
            sidx, _ = WI._template_cells(np.asarray(star_signs)[None, :], subset, template)
            wins = np.bincount(idx, weights=w_all[keep] * y_all[keep], minlength=ncell)
            losses = np.bincount(idx, weights=w_all[keep] * (1.0 - y_all[keep]),
                                 minlength=ncell)
            ev = float(np.sum(betaln(a0 + wins, b0 + losses) - betaln(a0, b0)))
            sc = ev + float(log_prior[it])
            if np.isfinite(sc) and sc > best[0]:
                c = int(sidx[0])
                best = (sc, it, float((a0 + wins[c]) / (a0 + b0 + wins[c] + losses[c])))
        q[i] = best[2]
        chosen.append(best[1])
    return q, np.asarray(chosen)


def main():
    t0 = time.time()
    np.seterr(all='ignore')          # the frozen win-head solve emits benign fp warnings
    b = H.load(DATA)
    d = np.load(OUT / 'heads.npz', allow_pickle=True)
    assert list(d['ids']) == list(b.ids)
    q0, q1 = d['q0'], d['q1']
    stats = d['stats']               # a_dense, b_dense, a_pair, b_pair for the queried cell
    n_dense = d['n_dense']
    a_d, b_d, a_p, b_p = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    wins_rec, losses_rec = a_p - a_d, b_p - b_d
    _, r_prot = b.split_heads(b.protected)

    # D0 must be the protected release exactly.
    P0 = b.compose(q0, r_prot)
    rep = float(np.max(np.abs(P0 - b.protected)))

    def q_rho(rho):
        return (a_d + rho * wins_rec) / (a_d + b_d + rho * (wins_rec + losses_rec))

    def q_lam(lmb):
        return (1.0 - lmb) * q0 + lmb * q1

    def q_gate(T):
        return np.where(n_dense <= T, q1, q0)

    cell_eff_dense = a_d + b_d - KAPPA

    def q_cell_gate(C):
        return np.where(cell_eff_dense <= C, q1, q0)

    recipes = {
        'D2': ('lam', LAM_GRID, q_lam),
        'D3': ('rho', RHO_GRID, q_rho),
        'D4': ('T', T_GRID, q_gate),
        'D5': ('C', C_GRID, q_cell_gate),
        # The brief's "decision to use your method at all", nested: a two-point choice
        # between no change and full recovery, made inside each outer fold.
        'DB': ('use_d1', (0.0, 1.0), q_lam),
    }
    # The whole declared family, in one fixed deterministic order, for the DS arm.
    family = ([('D0', 0.0, q_lam)]
              + [('D2', g, q_lam) for g in LAM_GRID]
              + [('D3', g, q_rho) for g in RHO_GRID]
              + [('D4', g, q_gate) for g in T_GRID]
              + [('D5', g, q_cell_gate) for g in C_GRID]
              + [('D1', 1.0, q_lam)])

    beta_by_half_life, lam, _ = WI.load_win_head(MODEL / 'population_averaged.json')
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        MODEL / 'beta_templates_deploy.json')
    assert abs(kappa - KAPPA) < 1e-12, 'frozen kappa moved; cell support formula invalid'

    t_trap = time.time()
    qT, trap_choice = trap_heads(b, beta_by_half_life, lam, kappa, prior_mean, log_prior)
    trap_seconds = time.time() - t_trap

    # ---------------- nested evaluation -----------------------------------------
    folds = sorted(set(b.folds.values()))
    picks = {k: {} for k in recipes}
    inner_tables = {k: {} for k in recipes}
    Pn = {k: np.zeros_like(b.protected) for k in recipes}
    ds_pick, ds_table = {}, {}
    P_ds = np.zeros_like(b.protected)
    for f in folds:
        held = b.outer_mask(f)
        fit = ~held
        inner = b.inner_folds(fit, k=3)
        ws = [b.row_weights(he) for _, he in inner]
        for name, (pname, grid, fn) in recipes.items():
            table = []
            for g in grid:
                P = b.compose(fn(g), r_prot)
                table.append(float(np.mean([b.wloss(P, w) for w in ws])))
            table = np.asarray(table)
            best = int(np.argmin(table))        # argmin: first (smallest grid entry) wins
            picks[name][f] = grid[best]
            inner_tables[name][f] = {str(g): float(v) for g, v in zip(grid, table)}
            Pn[name][held] = b.compose(fn(grid[best]), r_prot)[held]
        ftab = [float(np.mean([b.wloss(b.compose(fn(g), r_prot), w) for w in ws]))
                for _, g, fn in family]
        fbest = int(np.argmin(np.asarray(ftab)))
        ds_pick[f] = f'{family[fbest][0]}@{family[fbest][1]}'
        ds_table[f] = {f'{lbl}@{g}': float(v)
                       for (lbl, g, _), v in zip(family, ftab)}
        _, gsel, fnsel = family[fbest]
        P_ds[held] = b.compose(fnsel(gsel), r_prot)[held]

    arms = {
        'protected': b.protected,
        'D0': P0,
        'D1': b.compose(q1, r_prot),
        'D2': Pn['D2'],
        'D3': Pn['D3'],
        'D4': Pn['D4'],
        'D5': Pn['D5'],
        'DB': Pn['DB'],
        'DS': P_ds,
        'DT': b.compose(qT, r_prot),
    }

    w_all = b.row_weights(np.ones(len(b.ids), dtype=bool))
    reg = np.array([regime_of(int(n)) for n in n_dense])
    cond = np.array([b.groups[i] for i in b.ids])

    results = {}
    for name, P in sorted(arms.items()):
        o = b.official(P)
        per_row = w_all * np.sum((P - b.truth) ** 2, axis=1)
        base = w_all * np.sum((P0 - b.truth) ** 2, axis=1)
        results[name] = {
            'points': o['technical_points'], 'R': o['mean_loss_ratio'],
            'by_condition': {k: v['loss_ratio'] for k, v in sorted(o['by_condition'].items())},
            'dR_vs_D0_total': float(per_row.sum() - base.sum()),
            'dR_vs_D0_by_regime': {g: float(per_row[reg == g].sum() - base[reg == g].sum())
                                   for g in sorted(set(reg.tolist()))},
            'dR_vs_D0_by_condition': {g: float(per_row[cond == g].sum() - base[cond == g].sum())
                                      for g in sorted(set(cond.tolist()))},
            'paired_vs_protected': b.paired(P, b.protected)['delta_loss_ratio_ci95'],
            'paired_vs_D0': b.paired(P, P0)['delta_loss_ratio_ci95'],
            'mean_abs_dq_vs_D0': float(np.mean(np.abs((1 - P[:, 0]) - q0))),
        }

    # In-sample (NOT a claim; an upper bound on what any nested rule could have reached).
    oracle = {}
    for name, (pname, grid, fn) in sorted(recipes.items()):
        vals = [(float(b.wloss(b.compose(fn(g), r_prot), w_all)), g) for g in grid]
        bestv, bestg = min(vals)
        oracle[name] = {'param': pname, 'in_sample_best': bestg,
                        'in_sample_R': bestv,
                        'in_sample_points': 60.0 / (1.0 + bestv)}

    payload = {
        'd0_equals_protected_max_abs': rep,
        'picks_per_outer_fold': {k: {str(f): v for f, v in sorted(vv.items())}
                                 for k, vv in sorted(picks.items())},
        'ds_pick_per_outer_fold': {str(f): v for f, v in sorted(ds_pick.items())},
        'ds_inner_cv_table': {str(f): v for f, v in sorted(ds_table.items())},
        'inner_cv_tables': inner_tables,
        'in_sample_oracle': oracle,
        'trap_template_changed_queries': int((trap_choice != -1).sum()),
        'trap_seconds': trap_seconds,
        'results': results,
        'runtime_seconds': time.time() - t0,
    }
    (OUT / 'arms.json').write_text(json.dumps(payload, indent=2, sort_keys=True))

    import csv
    with open(OUT / 'arm_scores.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['arm', 'points', 'R', 'stable', 'drift', 'shared_change',
                     'short_history', 'dR_vs_D0', 'ci_lo_vs_protected',
                     'ci_hi_vs_protected', 'ci_lo_vs_D0', 'ci_hi_vs_D0'])
        for name in sorted(results):
            r = results[name]
            c = r['by_condition']
            wr.writerow([name, f"{r['points']:.6f}", f"{r['R']:.12f}",
                         f"{c['stable']:.6f}", f"{c['drift']:.6f}",
                         f"{c['shared_change']:.6f}", f"{c['short_history']:.6f}",
                         f"{r['dR_vs_D0_total']:+.8f}",
                         f"{r['paired_vs_protected'][0]:+.8f}",
                         f"{r['paired_vs_protected'][1]:+.8f}",
                         f"{r['paired_vs_D0'][0]:+.8f}",
                         f"{r['paired_vs_D0'][1]:+.8f}"])

    print(f"D0 == protected, max abs coordinate difference {rep:.3e}")
    print(f"picks per outer fold: {json.dumps(payload['picks_per_outer_fold'], sort_keys=True)}")
    print(f"DS picks per outer fold: {json.dumps(payload['ds_pick_per_outer_fold'], sort_keys=True)}")
    print(f"in-sample oracle: {json.dumps(oracle, sort_keys=True)}")
    for name in sorted(results):
        r = results[name]
        print(f"{name:10s} pts={r['points']:.6f} R={r['R']:.9f} "
              f"dR={r['dR_vs_D0_total']:+.8f} "
              f"vsProt=[{r['paired_vs_protected'][0]:+.6f},{r['paired_vs_protected'][1]:+.6f}] "
              f"vsD0=[{r['paired_vs_D0'][0]:+.6f},{r['paired_vs_D0'][1]:+.6f}]")
        print(f"           by_regime dR " +
              ' '.join(f"{k}={v:+.6f}" for k, v in sorted(r['dR_vs_D0_by_regime'].items())))
        print(f"           by_cond   dR " +
              ' '.join(f"{k}={v:+.6f}" for k, v in sorted(r['dR_vs_D0_by_condition'].items())))
        print(f"           ratios       " +
              ' '.join(f"{k}={v:.6f}" for k, v in sorted(r['by_condition'].items())))
    print(f"total runtime {payload['runtime_seconds']:.1f}s")


if __name__ == '__main__':
    main()
