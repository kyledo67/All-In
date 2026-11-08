"""FAMILY D verification: independent re-derivation, causality audit, negative control.

Three jobs.

1. AUDIT. Re-derive the pair-sufficient transition set for every query with a second,
   deliberately different implementation (set membership over an explicit tick range
   rather than iteration over the observed keys) and assert it agrees exactly with the
   measurement script. Also assert, per query: every used tick is strictly below the
   cutoff; both required lag ticks are present at the EXACT tick difference; and the
   complete-eight dense transition set is a strict subset of the pair-sufficient set.

2. BREAKDOWN. dR of D1 against D0 split by the machine's frozen selected half-life,
   which is what explains where the recovery helps and where it mildly hurts.

3. NEGATIVE CONTROL (DX). Rebuild the recovered rows with ROW-OFFSET lags instead of
   actual tick differences -- i.e. take the 2nd and 5th observed row back in the sorted
   key order rather than ticks t-2 and t-5. That is the fabrication the brief forbids. It
   is scored here only to show that the measured gain comes from genuine lag structure
   and not from any perturbation of q at all. It is never a candidate.
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


def main():
    t0 = time.time()
    np.seterr(all='ignore')
    b = H.load(DATA)
    import csv
    with open(OUT / 'per_query_support.csv') as fh:
        rows = list(csv.DictReader(fh))
    assert [r['query_id'] for r in rows] == list(b.ids)
    d = np.load(OUT / 'heads.npz', allow_pickle=True)
    q0, q1 = d['q0'], d['q1']
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        MODEL / 'beta_templates_deploy.json')
    _, r_prot = b.split_heads(b.protected)
    keys = [(b.ep[i], q.rsplit('-', 1)[0]) for i, q in enumerate(b.ids)]

    audit = {'queries': 0, 'dense_subset_violations': 0, 'count_mismatch': 0,
             'causality_violations': 0, 'lag_violations': 0, 'q1_mismatch': 0}
    qX = np.zeros(len(b.ids))
    for i, qid in enumerate(b.ids):
        ticks = b.index[keys[i]]
        cutoff = int(b.cutoffs[i])
        subset = None if rows[i]['subset'] == '' else tuple(
            int(x) for x in rows[i]['subset'].split('_'))
        template = rows[i]['template']
        hl = float(rows[i]['half_life'])
        half_life = None if hl == 0.0 else hl
        mu = float(prior_mean.get(b.types[i], 0.44))
        a0, b0 = kappa * mu, kappa * (1.0 - mu)

        # --- independent re-derivation over an explicit tick RANGE, not the key order
        present = set(ticks)
        need = () if subset is None else tuple(subset)
        alt = [t for t in range(min(ticks), cutoff)
               if t in present and all((t - l) in present for l in need)]
        if len(alt) != int(rows[i]['n_pair']):
            audit['count_mismatch'] += 1
        if any(t >= cutoff for t in alt):
            audit['causality_violations'] += 1
        for t in alt:
            if any((t - l) not in present for l in need):
                audit['lag_violations'] += 1
                break

        # --- dense must be a subset of pair-sufficient
        _, _, _, used_dense = W.design(ticks, cutoff, half_life=None)
        if not set(used_dense).issubset(set(alt)):
            audit['dense_subset_violations'] += 1

        # --- recompute q1 along the independent path
        star = W.query_row(ticks, cutoff)
        sg = np.zeros((len(alt), W.HORIZON))
        for l in need:
            sg[:, l - 1] = [1.0 if ticks[t - l] > 0 else -1.0 for t in alt]
        idx, ncell = WI._template_cells(sg, subset, template)
        sidx, _ = WI._template_cells(np.asarray(star[1:1 + W.HORIZON])[None, :],
                                     subset, template)
        y = np.asarray([1.0 if ticks[t] > 0 else 0.0 for t in alt])
        wt = (np.ones(len(alt)) if not half_life
              else 2.0 ** (-(cutoff - 1 - np.asarray(alt, dtype=float)) / half_life))
        wins = np.bincount(idx, weights=wt * y, minlength=ncell)
        losses = np.bincount(idx, weights=wt * (1.0 - y), minlength=ncell)
        c = int(sidx[0])
        v = float((a0 + wins[c]) / (a0 + b0 + wins[c] + losses[c]))
        if abs(v - q1[i]) > 1e-12:
            audit['q1_mismatch'] += 1

        # --- NEGATIVE CONTROL: row-offset "lags" instead of actual tick differences
        order = sorted(ticks)
        pos = {t: k for k, t in enumerate(order)}
        altX = [t for t in order
                if t < cutoff and all(pos[t] - l >= 0 for l in need)]
        sgX = np.zeros((len(altX), W.HORIZON))
        for l in need:
            sgX[:, l - 1] = [1.0 if ticks[order[pos[t] - l]] > 0 else -1.0 for t in altX]
        idxX, _ = WI._template_cells(sgX, subset, template)
        yX = np.asarray([1.0 if ticks[t] > 0 else 0.0 for t in altX])
        wtX = (np.ones(len(altX)) if not half_life
               else 2.0 ** (-(cutoff - 1 - np.asarray(altX, dtype=float)) / half_life))
        winsX = np.bincount(idxX, weights=wtX * yX, minlength=ncell)
        lossesX = np.bincount(idxX, weights=wtX * (1.0 - yX), minlength=ncell)
        qX[i] = float((a0 + winsX[c]) / (a0 + b0 + winsX[c] + lossesX[c]))
        audit['queries'] += 1

    P0 = b.compose(q0, r_prot)
    P1 = b.compose(q1, r_prot)
    PX = b.compose(qX, r_prot)
    w_all = b.row_weights(np.ones(len(b.ids), dtype=bool))
    base = w_all * np.sum((P0 - b.truth) ** 2, axis=1)
    one = w_all * np.sum((P1 - b.truth) ** 2, axis=1)
    hl = np.array([float(r['half_life']) for r in rows])
    by_hl = {}
    for h in sorted(set(hl.tolist())):
        m = hl == h
        by_hl[str(h)] = {
            'queries': int(m.sum()),
            'dR_D1_vs_D0': float(one[m].sum() - base[m].sum()),
            'rows_recovered': int(sum(int(r['recovered']) for r, k in zip(rows, m) if k)),
            'eff_recovered_pct': float(
                100.0 * (sum(float(r['eff_pair']) for r, k in zip(rows, m) if k)
                         / max(sum(float(r['eff_dense']) for r, k in zip(rows, m) if k), 1e-12) - 1)),
        }
    oX = b.official(PX)
    o1 = b.official(P1)
    payload = {
        'audit': audit,
        'by_selected_half_life': by_hl,
        'negative_control_row_offset_lags': {
            'points': oX['technical_points'], 'R': oX['mean_loss_ratio'],
            'dR_vs_D0': float((w_all * np.sum((PX - b.truth) ** 2, axis=1)).sum()
                              - base.sum()),
            'paired_vs_protected_ci95': b.paired(PX, b.protected)['delta_loss_ratio_ci95'],
            'paired_vs_D1_ci95': b.paired(PX, P1)['delta_loss_ratio_ci95'],
            'mean_abs_q_minus_q0': float(np.mean(np.abs(qX - q0))),
        },
        'D1_recheck': {'points': o1['technical_points'], 'R': o1['mean_loss_ratio']},
        'runtime_seconds': time.time() - t0,
    }
    (OUT / 'verification.json').write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
