"""Cache the template-dictionary evidence so any prior can be swept without refitting.

`log_prior` enters inference at exactly one place: inside
slot_model/win_head_infer.beta_template_win_probabilities the per-candidate score is
`evidence + log_prior[t]`, and the forecast reads the queried cell's posterior mean of the
argmax candidate. The evidence term, the decay weights, the eligible ticks, the selected
half-life, the reference q and the whole r path are all independent of log_prior.

So this script runs the expensive part once per split -- features, the two-head CatBoost
reference, the dense half-life selector, the Beta cell counts -- and stores, per query, the
57-vector of candidate evidence and the 57-vector of candidate queried-cell probabilities.
After that, changing the prior is an argmax over a cached matrix.

Everything here reproduces the packaged path; `--verify` checks that the arm-A endpoint
matches the package's own forecast before any arm is trusted.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')

ROOT = Path('/Users/ayan/Downloads/All-In')
PKG = ROOT / 'artifacts/r2_release/code'
sys.path.insert(0, str(PKG))

import numpy as np
from scipy.special import betaln

from slot_forecasting.io import load_inputs
from slot_model.features import causal_index
from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model.win_head_infer import (load_win_head, load_beta_templates, load_dirichlet_r,
                                       load_tail_projection, dirichlet_proportions,
                                       project_tail, proportions, reference_vectors,
                                       feature_table, _template_cells)

TEMPLATES = [(None, 'null')] + [(p, k) for p in PD.PAIRS for k in ('full', 'parity')]
NT = len(TEMPLATES)


def build(data, split, model, threads=2, need_reference=True, force_half_life=None):
    t0 = time.perf_counter()
    queries, history = load_inputs(Path(data) / split)
    index = causal_index(queries, history)
    deployment = json.loads((model / 'deployment.json').read_text())
    prior_pop = json.loads((model / 'population_prior.json').read_text())
    schema = json.loads((model / 'feature_schema.json').read_text())['observable']
    rows, x, _ = feature_table(queries, index, prior_pop, schema=schema)
    ordered = [e['q'] for e in rows]
    ids = [q.query_id for q in ordered]

    ref_q = np.zeros(len(ids))
    r_final = np.full((len(ids), 4), 0.25)
    if need_reference:
        reference, _ = reference_vectors(deployment['reference_variant'],
                                         deployment.get('weights', {}), rows, x,
                                         queries, history, model, threads=threads)
        ref_q = 1.0 - reference[:, 0]
        r_pool = proportions(reference)
        beta_, lam_, _ = load_win_head(model / 'win_head' / deployment['win_head'])
        tau, dprior, weight, _ = load_dirichlet_r(
            model / 'win_head' / deployment['dirichlet_r'])
        r_final, _ = dirichlet_proportions(ordered, index, beta_, lam_, tau, dprior,
                                           r_pool, weight)
        tail_by_type, _ = load_tail_projection(
            model / 'win_head' / deployment['tail_projection'])
        r_final = project_tail(ordered, r_final, tail_by_type)
    ref_seconds = time.perf_counter() - t0

    beta, lam, _ = load_win_head(model / 'win_head' / deployment['win_head'])
    kappa, prior_mean, log_prior_saved, payload = load_beta_templates(
        model / 'win_head' / deployment['beta_templates'])

    ev = np.full((len(ids), NT), -np.inf)
    qc = np.zeros((len(ids), NT))
    halflife = np.zeros(len(ids))
    fallback = np.zeros(len(ids), dtype=bool)
    nobs = np.zeros(len(ids), dtype=np.int64)
    mtype = []
    episode = []
    for i, query in enumerate(ordered):
        ticks = index[query.episode_id, query.machine_id]
        nobs[i] = len(ticks)
        mtype.append(query.machine_type)
        episode.append(query.episode_id)
        phi, labels, _, used = W.design(ticks, query.cutoff, half_life=None)
        star = W.query_row(ticks, query.cutoff)
        if star is None or len(labels) == 0:
            fallback[i] = True
            continue
        by_type = {h: table.get(query.machine_type, np.zeros(W.DIM))
                   for h, table in beta.items()}
        _, mix = W.predict_q_averaged(by_type, phi, labels, used, query.cutoff, star,
                                      lam, adapt='full')
        picked = mix['selected_half_life'] if force_half_life is None else force_half_life
        halflife[i] = picked
        hl = None if picked == 0.0 else picked
        weights = (np.ones(len(labels)) if not hl else
                   2.0 ** (-(query.cutoff - 1 - np.asarray(used, dtype=np.float64)) / hl))
        mu = float(prior_mean.get(query.machine_type, 0.44))
        a0, b0 = kappa * mu, kappa * (1.0 - mu)
        signs = PD.signs_from(phi)
        star_signs = star[1:1 + W.HORIZON][None, :]
        for t, (subset, template) in enumerate(TEMPLATES):
            idx, ncell = _template_cells(signs, subset, template)
            sidx, _ = _template_cells(star_signs, subset, template)
            wins = np.bincount(idx, weights=weights * labels, minlength=ncell)
            losses = np.bincount(idx, weights=weights * (1.0 - labels), minlength=ncell)
            e = float(np.sum(betaln(a0 + wins, b0 + losses) - betaln(a0, b0)))
            ev[i, t] = e if np.isfinite(e) else -np.inf
            c = int(sidx[0])
            qc[i, t] = float((a0 + wins[c]) / (a0 + b0 + wins[c] + losses[c]))
    return {'ids': np.array(ids), 'evidence': ev, 'qcand': qc, 'halflife': halflife,
            'fallback': fallback, 'nobs': nobs, 'mtype': np.array(mtype),
            'episode': np.array(episode), 'ref_q': ref_q, 'r': r_final,
            'log_prior_saved': log_prior_saved,
            'candidates': np.array(payload['candidates']),
            'train_argmax_counts': np.array(payload['train_argmax_counts'], dtype=float),
            'prior_smoothing': float(payload['prior_smoothing']),
            'kappa': float(kappa),
            'seconds': time.perf_counter() - t0, 'ref_seconds': ref_seconds}


def select(cache, log_prior):
    """Hard selection: argmax over finite evidence + log_prior, first index wins ties."""
    score = cache['evidence'] + np.asarray(log_prior)[None, :]
    pick = np.argmax(np.where(np.isfinite(score), score, -np.inf), axis=1)
    q = cache['qcand'][np.arange(len(pick)), pick]
    allbad = ~np.isfinite(score).any(axis=1)
    fb = cache['fallback'] | allbad
    q = np.where(fb, cache['ref_q'], q)
    pick = np.where(fb, -1, pick)
    return q, pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('/Users/ayan/Downloads/adaptation-participants/public'))
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', type=Path, default=PKG / 'model')
    ap.add_argument('--no-reference', action='store_true')
    ap.add_argument('--verify', type=Path, help='packaged submission CSV to reproduce')
    ap.add_argument('--half-life', type=float, default=None)
    args = ap.parse_args()

    c = build(args.data, args.split, args.model, need_reference=not args.no_reference,
              force_half_life=args.half_life)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{k: v for k, v in c.items()
                                     if k not in ('log_prior_saved', 'candidates',
                                                  'train_argmax_counts')},
                        log_prior_saved=c['log_prior_saved'],
                        candidates=c['candidates'],
                        train_argmax_counts=c['train_argmax_counts'])
    print(f"{args.split}: {len(c['ids'])} queries, {c['seconds']:.1f}s "
          f"(reference {c['ref_seconds']:.1f}s), {int(c['fallback'].sum())} fallbacks")

    if args.verify:
        import csv
        q, pick = select(c, c['log_prior_saved'])
        dep = json.loads((args.model / 'deployment.json').read_text())
        a = dep['alpha']
        if isinstance(a, dict):
            av = np.where(c['nobs'] < a['min_observations'], a['short'], a['long'])
        else:
            av = np.full(len(q), float(a))
        blended = (1 - av) * c['ref_q'] + av * q
        mine = np.column_stack([1 - blended, blended[:, None] * c['r']])
        got = {}
        with open(args.verify, newline='') as fh:
            for row in csv.DictReader(fh):
                got[row['query_id']] = [float(row[k]) for k in
                                        ('p_0', 'p_1', 'p_2', 'p_5', 'p_20')]
        pkg = np.array([got[i] for i in c['ids']])
        d = np.abs(mine - pkg).max()
        print(f'  ARM-A ENDPOINT CHECK vs packaged forecast: max abs diff {d:.3e} '
              f'{"PASS" if d < 1e-9 else "FAIL"}')
        rr = pkg[:, 1:] / np.maximum(1 - pkg[:, 0], 1e-300)[:, None]
        print(f'  r from package vs cached r: max abs diff '
              f'{np.abs(rr - c["r"]).max():.3e}  (r must be prior-independent)')


if __name__ == '__main__':
    main()
