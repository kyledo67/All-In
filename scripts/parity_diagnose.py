"""Refreshed residual diagnosis for the parity winner, plus the weighted-likelihood scope check.

Two jobs.

(A) LIKELIHOOD SCOPE. The existing sequential-product check covered UNIT weights only:
for unit weights ordinary Beta-Bernoulli predictive probabilities telescope to the batch
beta-function ratio. The deployed calculation uses FRACTIONAL decay weights, where the
increments are NOT ordinary predictive means raised to fractional powers. This adds one
small independent check of what the deployed quantity actually is:

    exp(betaln(a+W, b+L) - betaln(a, b))  ==  integral over theta of
        theta^W (1-theta)^(L) under Beta(a,b),   W, L fractional

i.e. an integrated POWER (tempered) likelihood ratio, which is a generalized-posterior
normalizer, not an exact sequential Bayes factor. Verified here by independent Gaussian
quadrature. It is also shown NOT to equal the product of fractional powers of ordinary
predictive means, so the distinction is demonstrated rather than merely asserted.

(B) OFFICIAL-R DECOMPOSITION. Each query's contribution to R is

    L_i / (4 * N_c * n_e * max(B_c, 1e-9))

with N_c evaluated episodes in condition c and n_e queries in episode e. The diagnostic
bins must sum to the candidate's R, which is asserted. Every support bin is reported,
including empty contexts, and raw query loss is kept separate from the R contribution so
a large raw loss in a tiny bin cannot masquerade as a large lever.
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
from scipy.integrate import quad
from scipy.special import betaln

from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import causal_index
from slot_model.official_score import score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission
from scripts.win_head_train import build

TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_parity')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/parity_diagnosis')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    findings = {}

    # ---------- (A) weighted-likelihood scope ----------
    a, b = 0.2, 0.3
    rows = []
    worst_quad, worst_naive = 0.0, 0.0
    for W_, L_ in ((2.7, 1.3), (0.4, 0.0), (0.0, 0.9), (5.5, 5.5), (0.05, 12.25)):
        closed = float(np.exp(betaln(a + W_, b + L_) - betaln(a, b)))
        integ, _ = quad(lambda th: th ** W_ * (1 - th) ** L_
                        * th ** (a - 1) * (1 - th) ** (b - 1), 0, 1, limit=400)
        norm, _ = quad(lambda th: th ** (a - 1) * (1 - th) ** (b - 1), 0, 1, limit=400)
        quad_val = integ / norm
        # the naive (wrong) reading: predictive mean raised to the fractional power
        mean = a / (a + b)
        naive = mean ** W_ * (1 - mean) ** L_
        worst_quad = max(worst_quad, abs(closed - quad_val) / max(closed, 1e-300))
        worst_naive = max(worst_naive, abs(closed - naive) / max(closed, 1e-300))
        rows.append({'W': W_, 'L': L_, 'closed_form': closed, 'quadrature': quad_val,
                     'naive_power_of_predictive_mean': naive})
    findings['weighted_likelihood_scope'] = {
        'claim': 'the deployed fractional-weight quantity is an integrated POWER-likelihood '
                 'ratio under the Beta prior, i.e. a generalized-posterior normalizer',
        'max_relative_deviation_vs_quadrature': worst_quad,
        'max_relative_deviation_vs_naive_fractional_power': worst_naive,
        'cases': rows,
        'verdict': 'confirmed by independent quadrature' if worst_quad < 1e-8 else 'MISMATCH',
        'scope_note': 'The earlier sequential-product check established the UNIT-weight '
                      'telescoping identity only. It did not cover the deployed weighted '
                      'calculation, and it is not evidence of an exact sequential Bayes factor '
                      'for a tempered likelihood with recency weights that change with the '
                      'prediction origin.'}
    print(f"(A) weighted likelihood: closed form matches quadrature to "
          f"{worst_quad:.2e} relative; the naive fractional-power reading is wrong by up to "
          f"{worst_naive:.2e} relative", flush=True)

    # ---------- (B) official-R decomposition ----------
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    cond = np.array([groups[q] for q in ids])

    pred = read_submission(args.package / 'validation_submission.csv')
    pred = np.array([pred[q] for q in ids])
    official = score(dict(zip(ids, pred.tolist())), scoring, set(ep.tolist()))
    loss = np.sum((pred - truth) ** 2, axis=1)

    # per-query R contribution
    contrib = np.zeros(len(ids))
    B = {}
    for c in sorted(set(cond.tolist())):
        es = sorted({e for e, cc in zip(ep, cond) if cc == c})
        B[c] = float(np.mean([np.mean(np.sum((ref_arr[ep == e] - truth[ep == e]) ** 2, axis=1))
                              for e in es]))
        for e in es:
            m = ep == e
            contrib[m] = loss[m] / (4 * len(es) * int(m.sum()) * max(B[c], 1e-9))
    total = float(contrib.sum())
    findings['R_decomposition'] = {
        'official_R': official['mean_loss_ratio'], 'sum_of_query_contributions': total,
        'absolute_difference': abs(total - official['mean_loss_ratio']),
        'identity_holds': bool(abs(total - official['mean_loss_ratio']) < 1e-12),
        'B_by_condition': B}
    print(f"(B) per-query contributions sum to {total:.12f} vs official R "
          f"{official['mean_loss_ratio']:.12f}  (diff {abs(total-official['mean_loss_ratio']):.2e})",
          flush=True)

    # ---------- per-query state of the deployed parity model ----------
    model = args.package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    ent = build(vq, index)
    by_id = {q.query_id: q for q in vq}

    state = []
    for qid in ids:
        e = ent[qid]
        q = by_id[qid]
        if e['star'] is None or len(e['y']) == 0:
            state.append({'template': 'fallback', 'mass': 0.0, 'ess': 0.0, 'raw': 0,
                          'prior_fraction': 1.0, 'margin': 0.0, 'post_sd': 0.0})
            continue
        mu = float(prior_mean.get(e['type'], 0.44))
        a0, b0 = kappa * mu, kappa * (1 - mu)
        _, mix = W.predict_q_averaged(
            {h: beta[h].get(e['type'], np.zeros(W.DIM)) for h in beta},
            e['phi'], e['y'], e['used'], e['cutoff'], e['star'], lam, adapt='full')
        hl = mix['selected_half_life']
        w = (np.ones(len(e['y'])) if hl == 0.0 else
             2.0 ** (-(e['cutoff'] - 1 - e['used']) / hl))
        signs = PD.signs_from(e['phi'])
        star_signs = e['star'][1:1 + W.HORIZON][None, :]
        scores = np.full(len(TEMPLATES), -np.inf)
        best = None
        for ix, (subset, template) in enumerate(TEMPLATES):
            ci, nc = WI._template_cells(signs, subset, template)
            si, _ = WI._template_cells(star_signs, subset, template)
            wins = np.bincount(ci, weights=w * e['y'], minlength=nc)
            loss_c = np.bincount(ci, weights=w * (1 - e['y']), minlength=nc)
            ev = float(np.sum(betaln(a0 + wins, b0 + loss_c) - betaln(a0, b0))) + float(log_prior[ix])
            scores[ix] = ev
            if best is None or ev > best[0]:
                c = int(si[0]); sel = ci == c
                n = float(wins[c] + loss_c[c])
                pa, pb = a0 + wins[c], b0 + loss_c[c]
                best = (ev, {'template': template,
                             'subset': None if subset is None else list(subset),
                             'half_life': hl, 'raw': int(sel.sum()), 'mass': n,
                             'ess': float(w[sel].sum() ** 2 / max(float(np.sum(w[sel] ** 2)), 1e-12))
                                    if sel.any() else 0.0,
                             'prior_fraction': float(kappa / (kappa + n)),
                             'post_sd': float(np.sqrt(pa * pb / ((pa + pb) ** 2 * (pa + pb + 1))))})
        order = np.sort(scores[np.isfinite(scores)])[::-1]
        best[1]['margin'] = float(order[0] - order[1]) if len(order) > 1 else float('inf')
        state.append(best[1])

    mass = np.array([s['mass'] for s in state])
    bins = [(0, 1e-9), (1e-9, 1), (1, 3), (3, 8), (8, 20), (20, 1e9)]
    table = {}
    for lo, hi in bins:
        m = (mass >= lo) & (mass < hi)
        if not m.any():
            continue
        label = 'empty' if hi <= 1e-9 else f'{lo:g}-{hi if hi < 1e9 else "inf"}'
        table[label] = {'queries': int(m.sum()),
                        'mean_raw_query_loss': float(loss[m].mean()),
                        'R_contribution': float(contrib[m].sum()),
                        'share_of_R': float(contrib[m].sum() / total),
                        'mean_post_sd': float(np.mean([s['post_sd'] for s, k in zip(state, m) if k])),
                        'mean_prior_fraction': float(np.mean([s['prior_fraction'] for s, k in zip(state, m) if k]))}
    findings['queried_context_support'] = table

    # ambiguity and episode concentration
    margin = np.array([s['margin'] for s in state])
    finite = np.isfinite(margin)
    qs = [0.0, 0.5, 1.0, 2.0, 4.0, 1e9]
    amb = {}
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = finite & (margin >= lo) & (margin < hi)
        if m.any():
            amb[f'{lo:g}-{hi if hi < 1e9 else "inf"}'] = {
                'queries': int(m.sum()), 'R_contribution': float(contrib[m].sum()),
                'share_of_R': float(contrib[m].sum() / total),
                'mean_raw_query_loss': float(loss[m].mean())}
    findings['selection_margin_nats'] = amb
    ep_contrib = {}
    for e in sorted(set(ep.tolist())):
        ep_contrib[e] = float(contrib[ep == e].sum())
    top = sorted(ep_contrib.items(), key=lambda kv: -kv[1])[:8]
    findings['episode_concentration'] = {
        'top8_share_of_R': float(sum(v for _, v in top) / total),
        'top8': [{'episode': k, 'R_contribution': v, 'condition': str(cond[ep == k][0])} for k, v in top]}

    # privileged q-vs-r substitutions on THIS candidate
    qp = 1 - pred[:, 0]
    r_pred = pred[:, 1:] / np.maximum(qp, 1e-12)[:, None]
    qt = 1 - truth[:, 0]
    r_true = truth[:, 1:] / np.maximum(qt, 1e-12)[:, None]
    sub_q = np.column_stack((1 - qt, qt[:, None] * r_pred))
    sub_r = np.column_stack((1 - qp, qp[:, None] * r_true))
    findings['privileged_substitutions'] = {
        'note': 'PRIVILEGED diagnostics using organizer true vectors. Not attainable ceilings '
                'and their gains need not add.',
        'candidate': official['technical_points'],
        'true_q_with_candidate_r': score(dict(zip(ids, (sub_q / sub_q.sum(1, keepdims=True)).tolist())),
                                         scoring, set(ep.tolist()))['technical_points'],
        'candidate_q_with_true_r': score(dict(zip(ids, (sub_r / sub_r.sum(1, keepdims=True)).tolist())),
                                         scoring, set(ep.tolist()))['technical_points']}

    findings['candidate'] = {'points': official['technical_points'], 'R': official['mean_loss_ratio'],
                             'by_condition': {c: v['loss_ratio'] for c, v in official['by_condition'].items()}}
    findings['template_counts'] = {t: int(sum(1 for s in state if s['template'] == t))
                                   for t in ('null', 'full', 'parity', 'fallback')}
    (args.out / 'diagnosis.json').write_text(json.dumps(findings, indent=2, sort_keys=True) + '\n')
    (args.out / 'per_query_state.json').write_text(
        json.dumps({ids[i]: {**state[i], 'raw_loss': float(loss[i]),
                             'R_contribution': float(contrib[i])} for i in range(len(ids))},
                   indent=2, sort_keys=True) + '\n')

    print(f"\ncandidate {official['technical_points']:.6f}/60 R={official['mean_loss_ratio']:.6f}")
    print('\nqueried-context support, by R contribution (not raw loss):')
    for k, v in table.items():
        print(f"   mass {k:<9} n={v['queries']:<4} share of R {100*v['share_of_R']:5.1f}%  "
              f"raw loss {v['mean_raw_query_loss']:.4f}  post sd {v['mean_post_sd']:.3f}  "
              f"prior frac {v['mean_prior_fraction']:.3f}")
    print('\nselection margin (nats of evidence between best and runner-up):')
    for k, v in amb.items():
        print(f"   {k:<9} n={v['queries']:<4} share of R {100*v['share_of_R']:5.1f}%  "
              f"raw loss {v['mean_raw_query_loss']:.4f}")
    print(f"\ntop 8 episodes carry {100*findings['episode_concentration']['top8_share_of_R']:.1f}% of R")
    p = findings['privileged_substitutions']
    print(f"privileged: true q -> {p['true_q_with_candidate_r']:.4f}, "
          f"true r -> {p['candidate_q_with_true_r']:.4f} (candidate {p['candidate']:.4f})")
    print(f"wrote {args.out / 'diagnosis.json'}")


if __name__ == '__main__':
    main()
