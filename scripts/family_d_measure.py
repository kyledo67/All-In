"""FAMILY D, STEP 1: measure how much strictly causal support a SELECTED pair template
recovers over the complete-eight-lag dense design.

The dense design (slot_model.win_head.design) emits a row only when the target tick and
ALL of t-1..t-8 are observed. The protected release's local win model is a Beta-Bernoulli
template dictionary whose selected candidate needs only its own two lag positions plus the
outcome. So there may be strictly causal transitions that the chosen template could score
but the dense design throws away.

Structural selection (half-life AND template identity) is frozen on the shared
complete-eight anchor rows, exactly as the protected release computes it, so it cannot move
between arms. Nothing here touches validation truth: beta, kappa, the template log prior and
the per-type prior means are all read from the frozen release model directory.

Outputs (fresh directory, nothing under artifacts/win_head_release_* is read-write):
  per_query_support.csv   one row per query
  support_summary.json    aggregates overall / by condition / by support regime
  heads.npz               q0, q1 and the per-query sufficient statistics for STEP 2
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
REGIMES = ((0, 16), (17, 32), (33, 64), (65, 128), (129, 10 ** 9))


def regime_of(n):
    for lo, hi in REGIMES:
        if lo <= n <= hi:
            return f'{lo}-{hi}' if hi < 10 ** 8 else f'{lo}+'
    raise AssertionError


def cell_index(signs_row, subset, template):
    """Cell of ONE observation given its (HORIZON,) +/-1 sign row."""
    idx, n = WI._template_cells(np.asarray(signs_row, dtype=float)[None, :], subset, template)
    return int(idx[0]), n


def pair_rows(ticks, cutoff, subset, half_life):
    """Every strictly causal transition whose REQUIRED lags and outcome are observed.

    Required lags are exactly the selected subset's two positions (none for the null
    template). Lags are ACTUAL tick differences: `(t - l) in ticks`, never a row offset.
    A missing tick is simply not a usable transition; it is never imputed as a loss.
    Weights use the same frozen recency rule 2**(-(cutoff-1-t)/half_life).
    """
    need = () if subset is None else tuple(subset)
    ys, ws, used = [], [], []
    for t in sorted(ticks):                      # sorted: determinism, never a set order
        if t >= cutoff:
            continue
        if any((t - l) not in ticks for l in need):
            continue
        ys.append(1.0 if ticks[t] > 0 else 0.0)
        ws.append(2.0 ** (-(cutoff - 1 - t) / half_life) if half_life else 1.0)
        used.append(t)
    return (np.asarray(ys, dtype=float), np.asarray(ws, dtype=float),
            np.asarray(used, dtype=np.int64))


def pair_signs(ticks, used, subset):
    """Sign rows restricted to the two required lag positions, padded elsewhere.

    Padding is never read: _template_cells only indexes positions i-1 and j-1 of the
    selected subset, so the unobserved positions are filled with nan to make any
    accidental use blow up rather than silently become an observed loss.
    """
    out = np.full((len(used), W.HORIZON), np.nan)
    if subset is None:
        return out
    for col, l in enumerate(subset):
        out[:, l - 1] = [1.0 if ticks[t - l] > 0 else -1.0 for t in used]
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    b = H.load(DATA)
    beta_by_half_life, lam, selected = WI.load_win_head(MODEL / 'population_averaged.json')
    kappa, prior_mean, log_prior, tpl_payload = WI.load_beta_templates(
        MODEL / 'beta_templates_deploy.json')
    from scipy.special import betaln

    q_prot = 1.0 - b.protected[:, 0]
    keys = [(b.ep[i], q.rsplit('-', 1)[0]) for i, q in enumerate(b.ids)]

    rows = []
    q0 = np.zeros(len(b.ids))
    q1 = np.zeros(len(b.ids))
    stats = []        # per query: (a_cell_dense, b_cell_dense, a_cell_pair, b_cell_pair)
    for i, qid in enumerate(b.ids):
        ticks = b.index[keys[i]]
        cutoff = int(b.cutoffs[i])
        mtype = b.types[i]
        phi, labels, _, used = W.design(ticks, cutoff, half_life=None)
        star = W.query_row(ticks, cutoff)
        assert star is not None and len(labels) > 0, f'unexpected fallback at {qid}'

        by_type = {h: table.get(mtype, np.zeros(W.DIM))
                   for h, table in beta_by_half_life.items()}
        _, mix = W.predict_q_averaged(by_type, phi, labels, np.asarray(used), cutoff,
                                      star, lam, adapt='full')
        picked = mix['selected_half_life']
        half_life = None if picked == 0.0 else picked
        wts = (np.ones(len(labels)) if not half_life
               else 2.0 ** (-(cutoff - 1 - np.asarray(used, dtype=float)) / half_life))

        mu = float(prior_mean.get(mtype, 0.44))
        a0, b0 = kappa * mu, kappa * (1.0 - mu)
        signs = PD.signs_from(phi)
        star_signs = star[1:1 + W.HORIZON]

        # ---- frozen structural selection on the complete-eight ANCHOR rows ----------
        best = (-np.inf, None, None, None)
        for it, (subset, template) in enumerate(TEMPLATES):
            idx, ncell = WI._template_cells(signs, subset, template)
            wins = np.bincount(idx, weights=wts * labels, minlength=ncell)
            losses = np.bincount(idx, weights=wts * (1.0 - labels), minlength=ncell)
            ev = float(np.sum(betaln(a0 + wins, b0 + losses) - betaln(a0, b0)))
            sc = ev + float(log_prior[it])
            if np.isfinite(sc) and sc > best[0]:
                best = (sc, it, wins, losses)
        score_sel, it_sel, wins0, losses0 = best
        subset, template = TEMPLATES[it_sel]
        c, ncell = cell_index(star_signs, subset, template)
        q0[i] = (a0 + wins0[c]) / (a0 + b0 + wins0[c] + losses0[c])

        # ---- D1 support: every transition sufficient for THAT template --------------
        ys, ws, pused = pair_rows(ticks, cutoff, subset, half_life)
        psigns = pair_signs(ticks, pused, subset)
        pidx, _ = WI._template_cells(psigns, subset, template)
        wins1 = np.bincount(pidx, weights=ws * ys, minlength=ncell)
        losses1 = np.bincount(pidx, weights=ws * (1.0 - ys), minlength=ncell)
        q1[i] = (a0 + wins1[c]) / (a0 + b0 + wins1[c] + losses1[c])

        dense_cell_raw = int((WI._template_cells(signs, subset, template)[0] == c).sum())
        pair_cell_raw = int((pidx == c).sum())
        rows.append({
            'query_id': qid, 'condition': b.groups[qid], 'machine_type': mtype,
            'fold': int(b.folds[b.ep[i]]), 'half_life': picked,
            'template': template, 'subset': '' if subset is None else f'{subset[0]}_{subset[1]}',
            'n_dense': int(len(labels)), 'n_pair': int(len(ys)),
            'recovered': int(len(ys) - len(labels)),
            'eff_dense': float(wts.sum()), 'eff_pair': float(ws.sum()),
            'cell': c, 'n_cells': ncell,
            'cell_dense_raw': dense_cell_raw, 'cell_pair_raw': pair_cell_raw,
            'cell_eff_dense': float(wins0[c] + losses0[c]),
            'cell_eff_pair': float(wins1[c] + losses1[c]),
            'q0': float(q0[i]), 'q1': float(q1[i]), 'q_protected': float(q_prot[i]),
            'regime': regime_of(int(len(labels))),
        })
        stats.append([a0 + wins0[c], b0 + losses0[c], a0 + wins1[c], b0 + losses1[c]])

    import csv
    fields = list(rows[0])
    with open(OUT / 'per_query_support.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    def agg(sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return None
        nd = sum(r['n_dense'] for r in sub)
        npr = sum(r['n_pair'] for r in sub)
        ed = sum(r['eff_dense'] for r in sub)
        ep = sum(r['eff_pair'] for r in sub)
        cd = sum(r['cell_eff_dense'] for r in sub)
        cp = sum(r['cell_eff_pair'] for r in sub)
        return {
            'queries': len(sub),
            'rows_dense': nd, 'rows_pair': npr, 'rows_recovered': npr - nd,
            'rows_recovered_pct': 100.0 * (npr - nd) / max(nd, 1),
            'eff_dense': ed, 'eff_pair': ep,
            'eff_recovered_pct': 100.0 * (ep - ed) / max(ed, 1e-12),
            'queried_cell_eff_dense': cd, 'queried_cell_eff_pair': cp,
            'queried_cell_eff_recovered_pct': 100.0 * (cp - cd) / max(cd, 1e-12),
            'median_recovered_rows': float(np.median([r['recovered'] for r in sub])),
            'queries_with_zero_recovery': sum(1 for r in sub if r['recovered'] == 0),
            'median_cell_eff_dense': float(np.median([r['cell_eff_dense'] for r in sub])),
            'mean_abs_dq': float(np.mean([abs(r['q1'] - r['q0']) for r in sub])),
            'max_abs_dq': float(np.max([abs(r['q1'] - r['q0']) for r in sub])),
        }

    summary = {
        'replication': {
            'max_abs_q0_minus_protected': float(np.max(np.abs(q0 - q_prot))),
            'mean_abs_q0_minus_protected': float(np.mean(np.abs(q0 - q_prot))),
        },
        'template_selection_counts': {
            t: sum(1 for r in rows if r['template'] == t) for t in ('null', 'full', 'parity')},
        'subset_counts': {},
        'overall': agg(lambda r: True),
        'by_condition': {c: agg(lambda r, c=c: r['condition'] == c)
                         for c in sorted({r['condition'] for r in rows})},
        'by_regime': {g: agg(lambda r, g=g: r['regime'] == g)
                      for g in sorted({r['regime'] for r in rows})},
        'by_half_life': {str(h): agg(lambda r, h=h: r['half_life'] == h)
                         for h in sorted({r['half_life'] for r in rows})},
        'runtime_seconds': time.time() - t0,
    }
    for r in rows:
        k = f"{r['template']}:{r['subset']}"
        summary['subset_counts'][k] = summary['subset_counts'].get(k, 0) + 1
    summary['subset_counts'] = dict(sorted(summary['subset_counts'].items()))
    (OUT / 'support_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    np.savez(OUT / 'heads.npz', q0=q0, q1=q1, q_protected=q_prot,
             stats=np.asarray(stats, dtype=float),
             n_dense=np.asarray([r['n_dense'] for r in rows]),
             n_pair=np.asarray([r['n_pair'] for r in rows]),
             ids=np.asarray(b.ids))
    print(json.dumps({k: summary[k] for k in
                      ('replication', 'template_selection_counts', 'overall',
                       'runtime_seconds')}, indent=2, sort_keys=True))
    for name in ('by_condition', 'by_regime'):
        print(f'--- {name}')
        for k, v in summary[name].items():
            print(f"  {k:14s} n={v['queries']:4d} dense={v['rows_dense']:7d} "
                  f"recov={v['rows_recovered']:6d} ({v['rows_recovered_pct']:6.2f}%) "
                  f"effrec={v['eff_recovered_pct']:7.2f}% "
                  f"cellrec={v['queried_cell_eff_recovered_pct']:7.2f}% "
                  f"meanabsdq={v['mean_abs_dq']:.5f}")


if __name__ == '__main__':
    main()
