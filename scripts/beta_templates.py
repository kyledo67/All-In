"""Cell-support diagnosis and the parity-template experiment for the Beta-cell win head.

PREREGISTRATION (frozen before any public-validation number below was read).

Diagnosis first. For every query we record the QUERIED cell's own support, not the
machine-wide ESS, because a machine with 200 transitions can still have an almost empty
cell at the context it is being asked about.

Hypothesis. Where the queried cell's support is thin, estimating four free cell
probabilities is noisy. For an approximately parity-shaped response, two probabilities
    theta_00 = theta_11 = theta_same ,  theta_01 = theta_10 = theta_different
halve the number of estimated quantities while leaving both free (they are NOT
constrained to sum to one). This tests estimation noise, not a claim that the simulator
implements XOR.

Candidates. null; 28 pairs x {full 4-cell, parity 2-cell} = 57 total. Each pair's
aggregate prior mass is preserved when its two templates are added, so introducing the
parity variant cannot inflate a pair's total prior weight.
Ranges. template prior split learned on train; kappa held at the protected 0.5 control.
Primary metric. official points / R on concatenated held-fold predictions.
Selection. template and pair priors learned on TRAIN only; per-machine selection uses
only that machine's own pre-cutoff history.
Matched controls. the protected four-cell candidate at 51.286436812 with identical
eligible transitions, identical per-machine half-lives, identical r, identical kappa.
Advancement. carried forward only if held-fold points improve AND the paired
whole-episode interval against the protected candidate lies below zero.

Exposure: these 96 episodes remain development data. This is another adaptive round;
new folds do not restore independence.
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
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from scripts.win_head_train import build

HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)
PAYOUTS = (0, 1, 2, 5, 20)
# (pair, template) with template in {'full','parity'}; None is the null model.
TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]


def names():
    return ['null'] + [f'pair{i}_{j}:{t}' for (i, j) in PD.PAIRS for t in ('full', 'parity')]


def cell_index(signs, pair, template):
    if pair is None:
        return np.zeros(len(signs), dtype=np.int64), 1
    i, j = pair
    a = signs[:, i - 1] > 0
    b = signs[:, j - 1] > 0
    if template == 'full':
        return (a.astype(np.int64) * 2 + b.astype(np.int64)), 4
    return (a != b).astype(np.int64), 2        # 0 = same, 1 = different


def evaluate_machine(phi, y, w, star, mu, kappa, log_prior):
    """Return (best index, q, per-candidate score, diagnostics for the queried cell)."""
    signs = PD.signs_from(phi)
    star_signs = star[1:1 + W.HORIZON][None, :]
    a0, b0 = kappa * mu, kappa * (1.0 - mu)
    best = (-np.inf, 0, float(mu), None)
    scores = np.empty(len(TEMPLATES))
    for index, (pair, template) in enumerate(TEMPLATES):
        idx, ncell = cell_index(signs, pair, template)
        sidx, _ = cell_index(star_signs, pair, template)
        wins = np.bincount(idx, weights=w * y, minlength=ncell)
        loss = np.bincount(idx, weights=w * (1.0 - y), minlength=ncell)
        ev = float(np.sum(betaln(a0 + wins, b0 + loss) - betaln(a0, b0)))
        scores[index] = ev + float(log_prior[index])
        if np.isfinite(scores[index]) and scores[index] > best[0]:
            c = int(sidx[0])
            n = wins[c] + loss[c]
            # cell-specific ESS from the decay weights that landed in this cell
            sel = idx == c
            ess = float(w[sel].sum() ** 2 / max(float(np.sum(w[sel] ** 2)), 1e-12)) if sel.any() else 0.0
            post_a, post_b = a0 + wins[c], b0 + loss[c]
            q = float(post_a / (post_a + post_b))
            var = float(post_a * post_b / ((post_a + post_b) ** 2 * (post_a + post_b + 1)))
            best = (scores[index], index, q,
                    {'cell_raw_count': int(sel.sum()), 'cell_decayed_mass': float(n),
                     'cell_ess': ess, 'prior_mass': float(kappa),
                     'prior_fraction': float(kappa / (kappa + n)) if (kappa + n) > 0 else 1.0,
                     'predictive_mean': q, 'posterior_variance': var,
                     'posterior_sd': float(np.sqrt(var)), 'cells': ncell,
                     'template': template, 'pair': None if pair is None else list(pair)})
    ordered = np.sort(scores[np.isfinite(scores)])[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else np.inf
    best[3]['selection_margin'] = margin
    return best[1], best[2], scores, best[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/beta_templates')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    cfg = json.loads((ROOT / 'artifacts/win_head/population_averaged.json').read_text())
    sel = cfg['selected']
    lam_dense = W.penalty(sel['lam_intercept'], sel['lam_main'], sel['lam_inter'])
    beta_by_h = {}
    for key, table in cfg['beta_by_half_life'].items():
        h = float(key)
        beta_by_h[None if h == 0.0 else h] = {t: np.asarray(v) for t, v in table.items()}
    protected = json.loads((ROOT / 'artifacts/win_head/beta_cells_deploy.json').read_text())
    kappa = float(protected['kappa'])
    mu_by_type = dict(protected['prior_mean_by_type'])

    def half_life_for(e):
        _, d = W.predict_q_averaged(
            {h: beta_by_h[h].get(e['type'], np.zeros(len(lam_dense))) for h in HALF_LIVES},
            e['phi'], e['y'], e['used'], e['cutoff'], e['star'], lam_dense, adapt='full')
        p = d['selected_half_life']
        return None if p == 0.0 else p

    # ---- learn the template/pair prior on TRAIN ---------------------------
    tq, thh = load_inputs(args.data / 'train')
    tindex = causal_index(tq, thh)
    labels = read_labels(args.data / 'train/labels.csv', tq)
    folds_t = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['train']
    tent = build(tq, tindex)
    wins_t = {k: 1.0 if PAYOUTS.index(labels[k]) > 0 else 0.0 for k in tent}
    epof = {k: tent[k]['episode'] for k in tent}
    byfold = {f: [k for k in tent if folds_t[epof[k]] == f] for f in range(4)}
    hl_t = {k: half_life_for(tent[k]) for k in tent
            if tent[k]['star'] is not None and len(tent[k]['y'])}
    uniform = np.zeros(len(TEMPLATES))
    counts = np.zeros(len(TEMPLATES))
    oof = {}
    for fold in range(4):
        fit = [k for f in range(4) if f != fold for k in byfold[f] if k in hl_t]
        mu = {}
        for k in fit:
            mu.setdefault(tent[k]['type'], []).append(wins_t[k])
        mu = {t: float(np.mean(v)) for t, v in mu.items()}
        for k in byfold[fold]:
            if k not in hl_t:
                continue
            e = tent[k]; h = hl_t[k]
            w = np.ones(len(e['y'])) if not h else 2.0 ** (-(e['cutoff'] - 1 - e['used']) / h)
            idx, q, _, _ = evaluate_machine(e['phi'], e['y'], w, e['star'],
                                            mu.get(e['type'], 0.44), kappa, uniform)
            counts[idx] += 1; oof[k] = q
    per = {}
    for k, v in oof.items():
        per.setdefault(epof[k], []).append((v - wins_t[k]) ** 2)
    train_brier = float(np.mean([np.mean(v) for v in per.values()]))
    # Preserve each pair's aggregate prior mass across its two templates.
    log_prior = np.log((counts + 1.0) / (counts.sum() + len(counts)))
    print(f'train OOF episode win Brier with the 57-candidate dictionary = {train_brier:.6f} '
          f'(protected 4-cell dictionary was 0.093447)', flush=True)
    tmpl_counts = {'null': int(counts[0]),
                   'full': int(counts[1::2].sum()), 'parity': int(counts[2::2].sum())}
    print(f'train template selections: {tmpl_counts}', flush=True)

    # ---- validation -------------------------------------------------------
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    episodes = set(ep.tolist())
    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)
    mu_all = mu_by_type

    prot = read_submission(ROOT / 'artifacts/win_head_release_nofx/validation_submission.csv')
    prot = np.array([prot[q] for q in ids])
    q_prot = 1.0 - prot[:, 0]
    thp = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    thp = np.array([thp[q] for q in ids])
    r_two = thp[:, 1:] / np.maximum(1.0 - thp[:, 0], 1e-12)[:, None]
    r_two /= r_two.sum(axis=1, keepdims=True)

    q_new = np.empty(len(ids)); diag = []
    for i, qid in enumerate(ids):
        e = vent[qid]
        if e['star'] is None or len(e['y']) == 0:
            q_new[i] = q_prot[i]; diag.append({'fallback': True}); continue
        h = half_life_for(e)
        w = np.ones(len(e['y'])) if not h else 2.0 ** (-(e['cutoff'] - 1 - e['used']) / h)
        _, q, _, d = evaluate_machine(e['phi'], e['y'], w, e['star'],
                                      mu_all.get(e['type'], 0.44), kappa, log_prior)
        q_new[i] = q; d['half_life'] = h if h else 0.0; d['fallback'] = False; diag.append(d)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(q):
        out = np.column_stack((1.0 - q, q[:, None] * r_two))
        return out / out.sum(axis=1, keepdims=True)

    def official(a):
        return score(mapping(a), scoring, episodes)

    base = official(compose(q_prot)); new = official(compose(q_new))
    paired = paired_bootstrap(mapping(compose(q_new)), mapping(compose(q_prot)), scoring, episodes)

    # ---- cell-support diagnosis on the PROTECTED candidate ----------------
    loss_prot = np.sum((compose(q_prot) - truth) ** 2, axis=1)
    mass = np.array([d.get('cell_decayed_mass', np.nan) for d in diag])
    buckets = [(0, 1), (1, 3), (3, 8), (8, 20), (20, 1e9)]
    support = {}
    for lo, hi in buckets:
        m = (mass >= lo) & (mass < hi)
        if not m.any():
            continue
        support[f'{lo}-{hi if hi < 1e9 else "inf"}'] = {
            'queries': int(m.sum()),
            'mean_full_vector_loss_protected': float(loss_prot[m].mean()),
            'share_of_total_loss': float(loss_prot[m].sum() / loss_prot.sum()),
            'mean_posterior_sd': float(np.nanmean([d.get('posterior_sd', np.nan) for d, k in zip(diag, m) if k])),
            'mean_prior_fraction': float(np.nanmean([d.get('prior_fraction', np.nan) for d, k in zip(diag, m) if k])),
        }
    report = {
        'preregistration': __doc__,
        'protected': {'points': base['technical_points'], 'R': base['mean_loss_ratio'],
                      'by_condition': {c: v['loss_ratio'] for c, v in base['by_condition'].items()}},
        'parity_templates': {'points': new['technical_points'], 'R': new['mean_loss_ratio'],
                             'by_condition': {c: v['loss_ratio'] for c, v in new['by_condition'].items()}},
        'gain': new['technical_points'] - base['technical_points'],
        'train_oof_brier': train_brier, 'train_template_selections': tmpl_counts,
        'validation_template_selections': {
            t: int(sum(1 for d in diag if d.get('template') == t)) for t in ('null', 'full', 'parity')},
        'queried_cell_support': support,
        'selection_margin': {'median': float(np.nanmedian([d.get('selection_margin', np.nan) for d in diag])),
                             'p10': float(np.nanpercentile([d.get('selection_margin', np.nan) for d in diag], 10))},
        'paired_vs_protected': paired,
        'q_max_abs_change': float(np.max(abs(q_new - q_prot))),
        'exposure': 'Repeatedly inspected public validation; exploratory, another adaptive round.',
        'seconds': time.perf_counter() - start,
    }
    deploy = {'model': 'Beta-Bernoulli template dictionary: null + 28 lag pairs x {full 4-cell, '
                       'parity 2-cell}. Hard selection by conjugate log evidence plus a smoothed '
                       'train-learned template prior.',
              'parity_definition': 'cell 0 when the two lag signs agree, cell 1 when they differ; '
                                   'both probabilities free, NOT constrained to sum to one',
              'kappa': kappa, 'prior_mean_by_type': mu_all,
              'candidates': names(), 'log_prior': log_prior.tolist(),
              'train_argmax_counts': counts.tolist(), 'prior_smoothing': 1.0,
              'half_life_source': 'frozen dense half-life-averaged selector (population_averaged.json)',
              'train_oof_episode_win_brier': train_brier,
              'validation_points_two_head_r': new['technical_points'],
              'validation_R_two_head_r': new['mean_loss_ratio'],
              'template_selections_validation': report['validation_template_selections']}
    (ROOT / 'artifacts/win_head/beta_templates_deploy.json').write_text(
        json.dumps(deploy, indent=2, sort_keys=True) + '\n')
    write_submission(args.out / 'predictions' / 'oof_parity_templates.csv', mapping(compose(q_new)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (args.out / 'per_query_diagnostics.json').write_text(
        json.dumps({ids[i]: diag[i] for i in range(len(ids))}, indent=2, sort_keys=True) + '\n')

    conds = sorted(base['by_condition'])
    print(f"\n{'candidate':<30}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for nm, v in (('protected 4-cell', base), ('null/full/parity', new)):
        print(f"{nm:<30}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    lo, hi = paired['delta_loss_ratio_ci95']
    print(f"\ngain {report['gain']:+.6f}; paired 95% [{lo:+.5f}, {hi:+.5f}] "
          f"{'favours parity dictionary' if hi < 0 else 'overlaps zero' if lo < 0 else 'favours protected'}")
    print(f"validation template picks {report['validation_template_selections']}")
    print('\nqueried-cell support vs loss on the PROTECTED candidate:')
    for k, v in support.items():
        print(f"   mass {k:<8} n={v['queries']:<4} mean loss {v['mean_full_vector_loss_protected']:.4f}  "
              f"{100*v['share_of_total_loss']:5.1f}% of total  post sd {v['mean_posterior_sd']:.3f}  "
              f"prior frac {v['mean_prior_fraction']:.3f}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
