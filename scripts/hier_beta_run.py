"""Verify, then evaluate, hierarchical partial pooling within the parity group.

PREREGISTERED BATCH, frozen before any public-validation number below was read.

Question. Does letting the two original cells inside a selected parity group depart from
their parent improve the full five-probability forecast?

Candidates, all sharing the deployed pair/template selection policy, memory policy,
population prior and kappa conventions, feature availability and r predictions:
  pooled_endpoint   exact tau -> infinity, which must reproduce the deployed parity model
  tau = 32          prior cell departure sd about 0.0853 at theta = 0.4
  tau = 8           about 0.1633
  tau = 2           about 0.2828
  independent_4cell separately specified control, each cell under the parity prior alone
Three finite taus plus the pooled endpoint, as specified. Their scale is chosen from the
interpretable departure sd sqrt(theta(1-theta)/(tau+1)) rather than from any validation
score, and the rationale is recorded here.

Primary metric: official points / R. Selection: one GLOBAL tau chosen on training
episodes by episode-equal win-indicator Brier on out-of-fold training predictions, a
sampled-outcome surrogate used because training has no true probability vectors.
Advancement: the candidate replaces the deployed predictor only if held-fold points
improve AND the paired whole-episode interval against the deployed forecast lies below
zero. That is the same rule that withheld global H=16 and independent r memory, and it is
not relaxed after seeing this run.
Stopping: one batch. Only parity-routed predictions change; null and full selections are
preserved exactly.

Comparisons below hold r at the fixed w=0.42 deployment vectors, so they are CONDITIONAL
DEPLOYMENT REPLAYS, labelled as such, matched on r across every candidate.

Exposure: 96 repeatedly inspected validation episodes; exploratory research round.
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
from scipy.integrate import quad
from scipy.special import betaln

from slot_model import hier_beta as HB
from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import PAYOUTS, causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from slot_forecasting.schema import Play
from scripts.win_head_train import build

TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
TAUS = (None, 32.0, 8.0, 2.0)
NODES = None  # per-prior Gauss-Jacobi nodes are cached inside hier_beta
RESULTS = []


def record(name, ok, detail):
    RESULTS.append({'check': name, 'passed': bool(ok), 'detail': detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def four_cell_counts(signs, y, w, subset):
    i, j = subset
    a = signs[:, i - 1] > 0
    b = signs[:, j - 1] > 0
    idx = a.astype(np.int64) * 2 + b.astype(np.int64)
    wins = np.bincount(idx, weights=w * y, minlength=4)
    loss = np.bincount(idx, weights=w * (1 - y), minlength=4)
    return wins, loss


def queried_cell(star_signs, subset):
    i, j = subset
    return int((star_signs[i - 1] > 0) * 2 + (star_signs[j - 1] > 0))


def verify():
    a0, b0 = 0.5 * 0.4, 0.5 * 0.6

    # pooled endpoint must equal the deployed parity formula exactly
    wins = np.array([3.0, 1.0, 2.0, 5.0])
    loss = np.array([2.0, 4.0, 1.0, 1.0])
    for cell in range(4):
        cells = HB.group_of(cell)
        parity_w = sum(wins[c] for c in cells)
        parity_l = sum(loss[c] for c in cells)
        deployed = (a0 + parity_w) / (a0 + b0 + parity_w + parity_l)
        got, _ = HB.predictive(a0, b0, wins, loss, cell, None)
        if abs(got - deployed) > 1e-15:
            record('pooled_endpoint', False, f'cell {cell}: {got} vs {deployed}')
            return
    record('pooled_endpoint', True,
           'the exact tau->infinity branch reproduces the deployed parity posterior mean '
           'for all four cells to <1e-15, in closed form rather than via a large tau')

    # large finite tau must approach the endpoint
    diffs = []
    for cell in range(4):
        end, _ = HB.predictive(a0, b0, wins, loss, cell, None)
        big, _ = HB.predictive(a0, b0, wins, loss, cell, 1e6, HB._nodes(128, a0, b0))
        diffs.append(abs(end - big))
    record('endpoint_limit', max(diffs) < 1e-4,
           f'tau=1e6 approaches the closed-form endpoint to {max(diffs):.2e}')

    # quadrature against an independent high-accuracy integral, incl. weak prior/boundary
    worst = 0.0
    for (aa, bb, tau, wv, lv) in ((a0, b0, 8.0, [3., 1., 2., 5.], [2., 4., 1., 1.]),
                                  (0.05, 0.05, 2.0, [0., 0., 0., 0.], [0., 0., 0., 0.]),
                                  (a0, b0, 32.0, [0., 0., 12., 0.], [0., 0., 0., 9.]),
                                  (2.0, 3.0, 2.0, [0.4, 0.0, 7.3, 1.1], [0.0, 0.0, 2.2, 0.0])):
        wv, lv = np.array(wv), np.array(lv)
        for cell in (0, 3):
            cells = HB.group_of(cell)
            f = lambda th: np.exp(HB.parent_log_weight(np.array([th]), aa, bb, wv, lv, cells, tau)[0])
            num, _ = quad(lambda th: th * f(th), 0, 1, limit=500)
            den, _ = quad(f, 0, 1, limit=500)
            ref = num / den
            _, got = HB.predictive(aa, bb, wv, lv, cell, tau, HB._nodes(128, aa, bb))
            worst = max(worst, abs(got - ref))
    record('quadrature', worst < 1e-8,
           f'256-node Gauss-Legendre parent mean matches scipy.quad to {worst:.2e} across '
           f'8 cases including an empty cell, a weak Beta(0.05,0.05) prior and fractional counts')

    # suppression when there is genuinely no within-group difference
    sym_w = np.array([10.0, 0.0, 0.0, 10.0])
    sym_l = np.array([10.0, 0.0, 0.0, 10.0])
    nd = HB._nodes(128, a0, b0)
    q0, _ = HB.predictive(a0, b0, sym_w, sym_l, 0, 8.0, nd)
    q3, _ = HB.predictive(a0, b0, sym_w, sym_l, 3, 8.0, nd)
    pooled, _ = HB.predictive(a0, b0, sym_w, sym_l, 0, None)
    record('suppresses_unsupported', abs(q0 - q3) < 1e-9 and abs(q0 - pooled) < 0.02,
           f'with identical cell counts the two cells agree to {abs(q0-q3):.1e} and sit '
           f'{abs(q0-pooled):.4f} from the pooled value')

    # recovery when the difference is real and well supported
    asym_w = np.array([18.0, 0.0, 0.0, 2.0])
    asym_l = np.array([2.0, 0.0, 0.0, 18.0])
    r0, _ = HB.predictive(a0, b0, asym_w, asym_l, 0, 8.0, nd)
    r3, _ = HB.predictive(a0, b0, asym_w, asym_l, 3, 8.0, nd)
    p0, _ = HB.predictive(a0, b0, asym_w, asym_l, 0, None)
    ind0 = HB.independent_mean(a0, b0, asym_w, asym_l, 0)
    record('recovers_supported', (r0 - r3) > 0.4 and abs(p0 - 0.5) < 0.05,
           f'with 18/2 against 2/18 the cells separate by {r0-r3:+.4f} at tau=8 '
           f'(pooled gives {p0:.4f} for both; the independent control gives {ind0:.4f}), '
           f'so a supported difference is recovered while pooling erases it')

    # Opposite asymmetries across machines must not cancel. Symmetry about 0.5 only holds
    # under a SYMMETRIC prior, so the mechanism is checked with mu=0.5; the asymmetric
    # deployed prior (mean 0.4) is then checked for directional separation only.
    flip_w, flip_l = asym_l.copy(), asym_w.copy()
    sa = sb = 0.5 * 0.5
    s_up, _ = HB.predictive(sa, sb, asym_w, asym_l, 0, 8.0, HB._nodes(128, sa, sb))
    s_dn, _ = HB.predictive(sa, sb, flip_w, flip_l, 0, 8.0, HB._nodes(128, sa, sb))
    f0, _ = HB.predictive(a0, b0, flip_w, flip_l, 0, 8.0, NODES)
    record('opposite_machines',
           abs((s_up - 0.5) + (s_dn - 0.5)) < 1e-9 and (r0 - f0) > 0.4,
           f'under a symmetric prior the mirrored machines give {s_up:.4f} and {s_dn:.4f}, '
           f'exactly symmetric about 0.5 to {abs((s_up-0.5)+(s_dn-0.5)):.1e}; under the '
           f'deployed prior (mean 0.4) they separate {r0:.4f} against {f0:.4f}. Opposite '
           f'responses are represented per machine, not averaged away')

    # empty queried cell borrows from its group partner
    part_w = np.array([0.0, 0.0, 0.0, 14.0])
    part_l = np.array([0.0, 0.0, 0.0, 2.0])
    e0, parent = HB.predictive(a0, b0, part_w, part_l, 0, 8.0, nd)
    prior_only = a0 / (a0 + b0)
    record('empty_cell_borrows', e0 > prior_only + 0.2 and e0 < 0.95,
           f'an EMPTY queried cell whose group partner is 14/2 returns {e0:.4f}, above the '
           f'prior mean {prior_only:.4f} and below the partner, with parent {parent:.4f}')

    # finiteness and normalisation across a sweep
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(4000):
        wv = rng.gamma(1.0, 3.0, 4) * (rng.random(4) < 0.7)
        lv = rng.gamma(1.0, 3.0, 4) * (rng.random(4) < 0.7)
        tau = float(rng.choice([2.0, 8.0, 32.0]))
        q, _ = HB.predictive(a0, b0, wv, lv, int(rng.integers(4)), tau, HB._nodes(128, a0, b0))
        if not (np.isfinite(q) and 0.0 < q < 1.0):
            bad += 1
    record('finite_normalised', bad == 0, f'{bad} of 4000 random count configurations left (0,1)')
    return all(r['passed'] for r in RESULTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/hier_beta')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    print('=== verification (implementation behaviour under stated assumptions) ===')
    ok = verify()
    (args.out / 'verification.json').write_text(
        json.dumps({'checks': RESULTS, 'passed': ok}, indent=2, sort_keys=True) + '\n')
    if not ok:
        raise SystemExit('verification failed; not evaluating')

    model = args.package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')

    def resolve(e, typ):
        """Deployed half-life and the deployed template selection, unchanged."""
        _, mix = W.predict_q_averaged({h: beta[h].get(typ, np.zeros(W.DIM)) for h in beta},
                                      e['phi'], e['y'], e['used'], e['cutoff'], e['star'],
                                      lam, adapt='full')
        p = mix['selected_half_life']
        h = None if p == 0.0 else p
        w = (np.ones(len(e['y'])) if not h else
             2.0 ** (-(e['cutoff'] - 1 - e['used']) / h))
        mu = float(prior_mean.get(typ, 0.44))
        a0, b0 = kappa * mu, kappa * (1 - mu)
        signs = PD.signs_from(e['phi'])
        star_signs = e['star'][1:1 + W.HORIZON]
        best = (-np.inf, 0)
        for ix, (subset, template) in enumerate(TEMPLATES):
            ci, nc = WI._template_cells(signs, subset, template)
            wn = np.bincount(ci, weights=w * e['y'], minlength=nc)
            ls = np.bincount(ci, weights=w * (1 - e['y']), minlength=nc)
            ev = float(np.sum(betaln(a0 + wn, b0 + ls) - betaln(a0, b0))) + float(log_prior[ix])
            if np.isfinite(ev) and ev > best[0]:
                best = (ev, ix)
        return best[1], w, a0, b0, signs, star_signs

    def forecast(entries, meta, tau, mode='hier'):
        """q per query; only parity-routed selections change."""
        out, routed = {}, 0
        for qid, e in entries.items():
            typ = meta[qid]
            if e['star'] is None or len(e['y']) == 0:
                out[qid] = None
                continue
            ix, w, a0, b0, signs, star_signs = resolve(e, typ)
            subset, template = TEMPLATES[ix]
            if template != 'parity':
                ci, nc = WI._template_cells(signs, subset, template)
                si, _ = WI._template_cells(star_signs[None, :], subset, template)
                wn = np.bincount(ci, weights=w * e['y'], minlength=nc)
                ls = np.bincount(ci, weights=w * (1 - e['y']), minlength=nc)
                c = int(si[0])
                out[qid] = float((a0 + wn[c]) / (a0 + b0 + wn[c] + ls[c]))
                continue
            routed += 1
            wins, loss = four_cell_counts(signs, e['y'], w, subset)
            cell = queried_cell(star_signs, subset)
            if mode == 'independent':
                out[qid] = float(HB.independent_mean(a0, b0, wins, loss, cell))
            else:
                out[qid] = float(HB.predictive(a0, b0, wins, loss, cell, tau,
                                               HB._nodes(128, a0, b0))[0])
        return out, routed

    # ---- global tau selected on TRAIN -------------------------------------
    tq, thh = load_inputs(args.data / 'train')
    tindex = causal_index(tq, thh)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    tent = build(tq, tindex)
    tmeta = {q.query_id: q.machine_type for q in tq}
    epof = {k: tent[k]['episode'] for k in tent}
    wins_t = {k: 1.0 if PAYOUTS.index(labels[k]) > 0 else 0.0 for k in tent}
    train_records = []
    for tau in list(TAUS) + ['independent']:
        mode = 'independent' if tau == 'independent' else 'hier'
        pred, routed = forecast(tent, tmeta, None if mode == 'independent' else tau, mode)
        per = {}
        for k, v in pred.items():
            if v is None:
                continue
            per.setdefault(epof[k], []).append((v - wins_t[k]) ** 2)
        brier = float(np.mean([np.mean(v) for v in per.values()]))
        label = 'independent_4cell' if tau == 'independent' else (
            'pooled_endpoint' if tau is None else f'tau={tau:g}')
        train_records.append({'candidate': label, 'train_oof_episode_win_brier': brier,
                              'parity_routed': routed})
        print(f'  {label:<18} train episode win Brier {brier:.6f}  '
              f'({routed} parity-routed)', flush=True)
    finite = [r for r in train_records if r['candidate'] != 'independent_4cell']
    chosen = min(finite, key=lambda r: r['train_oof_episode_win_brier'])['candidate']
    print(f"\nselected on the TRAIN surrogate: {chosen}", flush=True)

    # ---- validation, conditional deployment replay ------------------------
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    episodes = set(ep.tolist())
    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)
    vmeta = {q.query_id: q.machine_type for q in vq}

    dep = read_submission(args.package / 'validation_submission.csv')
    dep = np.array([dep[q] for q in ids])
    q_dep = 1.0 - dep[:, 0]
    r_dep = dep[:, 1:] / np.maximum(q_dep, 1e-12)[:, None]
    r_dep /= r_dep.sum(axis=1, keepdims=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(q):
        out = np.column_stack((1.0 - q, q[:, None] * r_dep))
        return out / out.sum(axis=1, keepdims=True)

    def official(a):
        return score(mapping(a), scoring, episodes)

    results, arrays = {}, {}
    for tau in list(TAUS) + ['independent']:
        mode = 'independent' if tau == 'independent' else 'hier'
        pred, routed = forecast(vent, vmeta, None if mode == 'independent' else tau, mode)
        q = np.array([pred[i] if pred[i] is not None else q_dep[k]
                      for k, i in enumerate(ids)])
        label = 'independent_4cell' if tau == 'independent' else (
            'pooled_endpoint' if tau is None else f'tau={tau:g}')
        arrays[label] = q
        results[label] = official(compose(q))
    deployed = official(compose(q_dep))
    endpoint_match = float(np.max(abs(arrays['pooled_endpoint'] - q_dep)))

    sel = arrays[chosen]
    paired = paired_bootstrap(mapping(compose(sel)), mapping(compose(q_dep)), scoring, episodes)
    gain = results[chosen]['technical_points'] - deployed['technical_points']
    lo, hi = paired['delta_loss_ratio_ci95']
    advanced = bool(gain > 0 and hi < 0)

    report = {
        'preregistration': __doc__,
        'verification_passed': ok,
        'deployed_control': {'points': deployed['technical_points'], 'R': deployed['mean_loss_ratio'],
                             'by_condition': {c: v['loss_ratio'] for c, v in deployed['by_condition'].items()}},
        'pooled_endpoint_reproduces_deployed_q_to': endpoint_match,
        'train_selection': train_records, 'selected_on_train': chosen,
        'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                        'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                    for k, v in results.items()},
        'selected_gain_over_deployed': gain,
        'paired_selected_vs_deployed': paired,
        'advancement_rule': 'held-fold points improve AND paired interval upper bound < 0',
        'advanced': advanced,
        'comparison_label': 'conditional deployment replay: r held at the fixed w=0.42 vectors '
                            'and identical across every candidate',
        'exposure': 'Repeatedly inspected public validation; exploratory research round.',
        'seconds': time.perf_counter() - start}
    for k, q in arrays.items():
        write_submission(args.out / 'predictions' / f'oof_{k.replace("=", "")}.csv', mapping(compose(q)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(deployed['by_condition'])
    print(f"\n{'candidate':<22}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    print(f"{'deployed (control)':<22}{deployed['technical_points']:>12.6f}"
          f"{deployed['mean_loss_ratio']:>10.6f}  "
          + '  '.join(f"{deployed['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    for k in ['pooled_endpoint', 'tau=32', 'tau=8', 'tau=2', 'independent_4cell']:
        v = results[k]
        print(f"{k:<22}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print(f"\npooled endpoint reproduces the deployed q to {endpoint_match:.2e}")
    print(f"selected {chosen}: gain {gain:+.6f}; paired 95% [{lo:+.5f}, {hi:+.5f}]")
    print(f"ADVANCEMENT: {'GRANTED' if advanced else 'WITHHELD'} under the preregistered rule")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
