"""Matched memory-policy comparison over the cached parity experts, r held fixed.

PREREGISTERED. The capacity envelope passed its 0.25-point gate (privileged bound
55.042506 against the deployed 53.376905), so the expert family can repay a weighting
experiment. The single-expert table also showed H=16 scoring above the deployed
per-machine selector, but that table is public validation, so a global half-life must be
SELECTED fitting-side per outer fold rather than read off it.

Policies compared, all with the deployed r frozen and the structural choice held common:
  deployed          per-machine dense half-life averaging (the shipped policy)
  global_heldfold   one global half-life, selected on each outer fold's FITTING episodes
                    by the exact official weighted objective, applied to the held fold
  fixed_share       causal reset-to-prior fixed-share mixture over the experts, run along
                    each machine's own transition sequence, with rho = 0 as the no-share
                    control and two predeclared positive rates

Fixed-share definition used, stated precisely. Experts are the half-life grid with a
uniform prior pi. Weights v are those immediately after the last eligible observed
outcome, before sharing. At a prediction origin Delta ACTUAL ticks later,
    a = (1 - rho)^Delta * v + (1 - (1 - rho)^Delta) * pi
and the forecast is sum_H a_H q_H. The gap transition is applied once; no extra share
step follows the preceding outcome. After an eligible observed binary outcome, v is
updated by that outcome's predictive likelihood with learning rate 1. Unobserved ticks
produce no likelihood update. Predictions always precede the update.

Advancement: a policy replaces the deployed one only if its held-fold points improve AND
its paired whole-episode interval against the deployed forecast lies below zero.

Exposure: repeatedly inspected public validation; exploratory research round.
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
from scipy.special import betaln

from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.win_head_train import build

HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)
TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
RHOS = (0.0, 0.01, 0.05)
LABELS = ['inf' if h is None else str(int(h)) for h in HALF_LIVES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/memory_policies')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    model = args.package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    experts = np.load(ROOT / 'artifacts/memory_envelope/experts.npy')

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    deployed = read_submission(args.package / 'validation_submission.csv')
    deployed = np.array([deployed[q] for q in ids])
    q_dep = 1.0 - deployed[:, 0]
    r_dep = deployed[:, 1:] / np.maximum(q_dep, 1e-12)[:, None]
    r_dep /= r_dep.sum(axis=1, keepdims=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(q):
        out = np.column_stack((1.0 - q, q[:, None] * r_dep))
        return out / out.sum(axis=1, keepdims=True)

    def official(a, es=episodes):
        return score(mapping(a), scoring, es)

    def row_weights(es):
        w = np.zeros(len(ids))
        for c in sorted(set(groups.values())):
            ces = {x for i, x in enumerate(ep) if x in es and groups[ids[i]] == c}
            b = np.mean([np.mean(np.sum((ref_arr[ep == x] - truth[ep == x]) ** 2, axis=1)) for x in sorted(ces)])
            for x in ces:
                w[ep == x] = 1 / (4 * len(ces) * sum(ep == x) * max(b, 1e-9))
        return w

    def wloss(q, w):
        return float(np.sum(w * np.sum((compose(q) - truth) ** 2, axis=1)))

    # ---- policy 2: one global half-life, selected fitting-side ----------
    held_global = np.empty(len(ids))
    chosen = []
    for fold in range(4):
        m = np.array([folds[x] == fold for x in ep])
        fit = {x for x in episodes if folds[x] != fold}
        w = row_weights(fit)
        scores = [wloss(experts[:, j], w) for j in range(len(HALF_LIVES))]
        j = int(np.argmin(scores))
        chosen.append(LABELS[j])
        held_global[m] = experts[:, j][m]

    # ---- policy 3: causal reset-to-prior fixed-share --------------------
    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)
    pi = np.full(len(HALF_LIVES), 1.0 / len(HALF_LIVES))

    def fixed_share(rho):
        out = np.empty(len(ids))
        diag = []
        for i, qid in enumerate(ids):
            e = vent[qid]
            if e['star'] is None or len(e['y']) == 0:
                out[i] = q_dep[i]; diag.append({'fallback': True}); continue
            mu = float(prior_mean.get(e['type'], 0.44))
            a0, b0 = kappa * mu, kappa * (1 - mu)
            phi, y, used, cutoff, star = e['phi'], e['y'], e['used'], e['cutoff'], e['star']
            order = np.argsort(used)
            ticks = used[order]
            labels = y[order]
            signs = PD.signs_from(phi[order])
            # Structure chosen once from the strict prefix available at the FIRST origin we
            # score from, so no later outcome informs it. Held common across experts.
            warm = max(8, len(order) // 4)
            ix = None
            v = pi.copy()
            last_tick = None
            for step in range(len(order)):
                if step < warm:
                    continue
                if ix is None:
                    rows = np.arange(step)
                    best = (-np.inf, 0)
                    for t_ix, (subset, template) in enumerate(TEMPLATES):
                        ci, nc = WI._template_cells(signs[rows], subset, template)
                        wn = np.bincount(ci, weights=labels[rows], minlength=nc)
                        ls = np.bincount(ci, weights=1 - labels[rows], minlength=nc)
                        ev = float(np.sum(betaln(a0 + wn, b0 + ls) - betaln(a0, b0))) \
                            + float(log_prior[t_ix])
                        if np.isfinite(ev) and ev > best[0]:
                            best = (ev, t_ix)
                    ix = best[1]
                subset, template = TEMPLATES[ix]
                origin = float(ticks[step])
                # gap transition, applied once, in ACTUAL ticks
                if last_tick is not None and rho > 0:
                    delta = max(1.0, origin - last_tick)
                    keep = (1.0 - rho) ** delta
                    a = keep * v + (1.0 - keep) * pi
                else:
                    a = v.copy()
                rows = np.arange(step)
                ci, nc = WI._template_cells(signs[rows], subset, template)
                cur, _ = WI._template_cells(signs[step:step + 1], subset, template)
                c = int(cur[0])
                qh = np.empty(len(HALF_LIVES))
                for j, h in enumerate(HALF_LIVES):
                    ww = (np.ones(step) if not h else
                          2.0 ** (-(origin - 1 - ticks[rows]) / h))
                    wn = np.bincount(ci, weights=ww * labels[rows], minlength=nc)
                    ls = np.bincount(ci, weights=ww * (1 - labels[rows]), minlength=nc)
                    qh[j] = (a0 + wn[c]) / (a0 + b0 + wn[c] + ls[c])
                # predict BEFORE observing, then update from the observed outcome
                yt = labels[step]
                like = np.clip(np.where(yt > 0, qh, 1 - qh), 1e-12, 1.0)
                v = a * like
                v = v / v.sum()
                last_tick = origin
            # final forecast at the cutoff
            subset, template = TEMPLATES[ix if ix is not None else 0]
            rows = np.arange(len(order))
            ci, nc = WI._template_cells(signs, subset, template)
            si, _ = WI._template_cells(star[1:1 + W.HORIZON][None, :], subset, template)
            c = int(si[0])
            qh = np.empty(len(HALF_LIVES))
            for j, h in enumerate(HALF_LIVES):
                ww = (np.ones(len(order)) if not h else
                      2.0 ** (-(cutoff - 1 - ticks) / h))
                wn = np.bincount(ci, weights=ww * labels, minlength=nc)
                ls = np.bincount(ci, weights=ww * (1 - labels), minlength=nc)
                qh[j] = (a0 + wn[c]) / (a0 + b0 + wn[c] + ls[c])
            if last_tick is not None and rho > 0:
                delta = max(1.0, cutoff - last_tick)
                keep = (1.0 - rho) ** delta
                a = keep * v + (1.0 - keep) * pi
            else:
                a = v.copy()
            out[i] = float(np.dot(a, qh))
            diag.append({'fallback': False, 'max_weight': float(a.max()),
                         'argmax': LABELS[int(np.argmax(a))]})
        return out, diag

    results = {'deployed_per_machine': official(compose(q_dep)),
               'global_heldfold': official(compose(held_global))}
    share_q, share_diag = {}, {}
    for rho in RHOS:
        qs, dg = fixed_share(rho)
        share_q[rho] = qs
        share_diag[rho] = dg
        results[f'fixed_share_rho{rho:g}'] = official(compose(qs))
        print(f'  rho={rho:<5} {results[f"fixed_share_rho{rho:g}"]["technical_points"]:.6f}', flush=True)

    best_name = max(results, key=lambda k: results[k]['technical_points'])
    best_q = {'deployed_per_machine': q_dep, 'global_heldfold': held_global,
              **{f'fixed_share_rho{r:g}': share_q[r] for r in RHOS}}[best_name]
    paired = paired_bootstrap(mapping(compose(best_q)), mapping(compose(q_dep)), scoring, episodes)

    report = {'preregistration': __doc__,
              'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                              'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                          for k, v in results.items()},
              'global_half_life_per_fold': chosen,
              'fixed_share_argmax_histogram': {
                  f'rho{r:g}': {lab: int(sum(1 for d in share_diag[r] if d.get('argmax') == lab))
                                for lab in LABELS} for r in RHOS},
              'fixed_share_mean_max_weight': {
                  f'rho{r:g}': float(np.mean([d['max_weight'] for d in share_diag[r]
                                              if not d.get('fallback')])) for r in RHOS},
              'best': best_name, 'paired_best_vs_deployed': paired,
              'r_held_fixed': True,
              'exposure': 'Repeatedly inspected public validation; exploratory research round.',
              'seconds': time.perf_counter() - start}
    for k, q in (('global_heldfold', held_global), (best_name, best_q)):
        write_submission(args.out / 'predictions' / f'oof_{k}.csv', mapping(compose(q)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(results['deployed_per_machine']['by_condition'])
    print(f"\n{'policy':<30}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in sorted(results, key=lambda k: -results[k]['technical_points']):
        v = results[k]
        print(f"{k:<30}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\nglobal half-life selected per fold: {chosen}")
    print(f"best {best_name}; paired 95% vs deployed [{lo:+.5f}, {hi:+.5f}] "
          f"{'significant' if hi < 0 else 'overlaps zero'}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
