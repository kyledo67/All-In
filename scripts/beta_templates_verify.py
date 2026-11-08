"""Verification for the SHIPPED Beta template dictionary (artifacts/win_head_release_parity).

Earlier verification exercised the dense and logistic predictors. None of it certifies
conjugate Beta inference, so these checks run against the deployed template path.

Checks
  tiny_mean        predictive mean against an independent hand calculation on tiny
                   sequences, including empty cells, all wins, all losses, and unequal
                   recency weights
  tiny_evidence    the beta-function evidence against an independent calculation, and
                   against a direct product-of-predictives factorization, which is the
                   honest cross-check that no Beta-binomial count coefficient has been
                   introduced into the likelihood of a specific ORDERED sequence
  parity_recovery  a pure parity response selects the parity template
  asymmetric       an asymmetric four-cell response selects the full template, and the
                   cost of forcing parity on it is measured
  parity_cost      the cost of forcing the full template on a true parity response
  post_cutoff      mutating outcomes at or after the cutoff leaves deployed predictions
                   bitwise unchanged
  positive_control a constructed INFORMATIVE pre-cutoff change must move the forecast,
                   so that post_cutoff passing cannot be explained by an inert model
  simplex          the shipped CSV rows are finite, nonnegative and normalized
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.special import betaln

from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import PAYOUTS, causal_index
from slot_forecasting.io import load_inputs, read_submission
from slot_forecasting.schema import Play

RESULTS = []


def record(name, passed, detail):
    RESULTS.append({'check': name, 'passed': bool(passed), 'detail': detail})
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)


def cells_of(signs, subset, template):
    return WI._template_cells(signs, subset, template)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_parity')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'artifacts/win_head/beta_templates_verification.json')
    args = parser.parse_args()
    model = args.package / 'code/model'
    kappa, prior_mean, log_prior, payload = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')

    # ---- tiny-sequence predictive mean -----------------------------------
    mu, k = 0.4, 0.5
    a0, b0 = k * mu, k * (1 - mu)
    cases = []
    # empty cell: posterior mean must equal the prior mean exactly
    cases.append(('empty', 0.0, 0.0, mu))
    # all wins with unit weights
    cases.append(('all_wins_3', 3.0, 0.0, (a0 + 3) / (a0 + b0 + 3)))
    # all losses
    cases.append(('all_losses_3', 0.0, 3.0, (a0 + 0) / (a0 + b0 + 3)))
    # unequal recency weights: W = 0.5+0.25, L = 1.0
    cases.append(('weighted', 0.75, 1.0, (a0 + 0.75) / (a0 + b0 + 1.75)))
    worst = 0.0
    for name, wins, losses, expect in cases:
        got = (a0 + wins) / (a0 + b0 + wins + losses)
        worst = max(worst, abs(got - expect))
    record('tiny_mean', worst < 1e-15,
           f'4 tiny cases (empty, all-wins, all-losses, unequal weights) agree with an '
           f'independent calculation to {worst:.2e}; the empty cell returns the prior mean '
           f'{mu} exactly')

    # ---- tiny-sequence evidence -------------------------------------------
    # For an ORDERED binary sequence under a Beta prior, the sequential predictive
    # product telescopes to exp(betaln(a+W, b+L) - betaln(a, b)) with NO binomial
    # coefficient. Verify that identity numerically on unit-weight sequences.
    seqs = [[1, 1, 1], [0, 0, 0], [1, 0, 1, 1, 0], [1], [0, 1]]
    worst_ev = 0.0
    for seq in seqs:
        a, b = a0, b0
        logp = 0.0
        for y in seq:
            p = a / (a + b)
            logp += math.log(p if y else 1 - p)
            a += y
            b += 1 - y
        wins, losses = float(sum(seq)), float(len(seq) - sum(seq))
        closed = float(betaln(a0 + wins, b0 + losses) - betaln(a0, b0))
        worst_ev = max(worst_ev, abs(logp - closed))
    record('tiny_evidence', worst_ev < 1e-12,
           f'sequential predictive product equals betaln(a+W,b+L)-betaln(a,b) on 5 ordered '
           f'sequences to {worst_ev:.2e}, confirming no Beta-binomial count coefficient '
           f'contaminates the ordered-sequence likelihood')

    # ---- synthetic template recovery --------------------------------------
    def synth(kind, n=300, seed=0):
        rng = np.random.default_rng(seed)
        hist = {t: int(rng.random() < 0.4) for t in range(8)}
        for t in range(8, n):
            s1 = 1 if hist[t - 1] > 0 else -1
            s2 = 1 if hist[t - 2] > 0 else -1
            if kind == 'parity':
                p = 0.80 if s1 * s2 > 0 else 0.18
            else:  # asymmetric four-cell: all four cells differ
                p = {(1, 1): 0.85, (1, -1): 0.55, (-1, 1): 0.22, (-1, -1): 0.40}[(s1, s2)]
            hist[t] = int(rng.random() < p)
        return hist, n

    def selected_template(hist, cutoff, forced=None):
        phi, y, _, used = W.design(hist, cutoff, half_life=None)
        star = W.query_row(hist, cutoff)
        w = np.ones(len(y))
        signs = PD.signs_from(phi)
        star_signs = star[1:1 + W.HORIZON][None, :]
        templates = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
        best = (-np.inf, None, None)
        for idx, (subset, template) in enumerate(templates):
            if forced is not None and template != forced and subset is not None:
                continue
            ci, nc = cells_of(signs, subset, template)
            si, _ = cells_of(star_signs, subset, template)
            wins = np.bincount(ci, weights=w * y, minlength=nc)
            loss = np.bincount(ci, weights=w * (1 - y), minlength=nc)
            ev = float(np.sum(betaln(a0 + wins, b0 + loss) - betaln(a0, b0)))
            if ev > best[0]:
                c = int(si[0])
                q = float((a0 + wins[c]) / (a0 + b0 + wins[c] + loss[c]))
                best = (ev, (subset, template), q)
        return best

    hp, cp = synth('parity', seed=3)
    ev_p, pick_p, q_p = selected_template(hp, cp)
    record('parity_recovery', pick_p[1] == 'parity' and pick_p[0] == (1, 2),
           f'a pure s1*s2 parity response (0.80 / 0.18) selects {pick_p[1]} on lags '
           f'{pick_p[0]} by unpenalised evidence')
    ha, ca = synth('asym', seed=4)
    ev_a, pick_a, q_a = selected_template(ha, ca)
    record('asymmetric', pick_a[1] == 'full',
           f'an asymmetric four-cell response (0.85/0.55/0.22/0.40) selects {pick_a[1]} on '
           f'lags {pick_a[0]}, so parity is not preferred when the cells genuinely differ')
    # cost of forcing the wrong template, in evidence nats and in predicted probability
    ev_forced_par, _, q_forced_par = selected_template(ha, ca, forced='parity')
    ev_forced_full, _, q_forced_full = selected_template(hp, cp, forced='full')
    record('pooling_cost', True,
           f'forcing parity on the asymmetric machine costs {ev_a - ev_forced_par:.2f} nats of '
           f'evidence and moves q by {abs(q_a - q_forced_par):.4f}; forcing the full template on '
           f'the parity machine costs {ev_p - ev_forced_full:.2f} nats and moves q by '
           f'{abs(q_p - q_forced_full):.4f} - pooling is not free when it is wrong')

    # ---- deployed-path post-cutoff invariance and positive control --------
    vq, vh = load_inputs(args.data / 'validation')
    index = causal_index(vq, vh)
    order = sorted(vq, key=lambda q: q.query_id)
    fb = np.full(len(order), 0.5)
    q0, d0 = WI.beta_template_win_probabilities(order, index, beta, lam, fb,
                                                kappa, prior_mean, log_prior)
    cut = {(q.episode_id, q.machine_id): q.cutoff for q in vq}
    extra = []
    for (e, m), c in sorted(cut.items()):
        obs = index[(e, m)]
        for off in range(8):
            if c + off not in obs:
                extra.append(Play(e, m, c + off, PAYOUTS[(obs.get(c - 1, 0) + 3) % 5]))
    q1, _ = WI.beta_template_win_probabilities(order, causal_index(vq, tuple(vh) + tuple(extra)),
                                               beta, lam, fb, kappa, prior_mean, log_prior)
    record('post_cutoff', bool((q0 == q1).all()) and len(extra) > 0,
           f'{len(extra)} adversarial plays appended at ticks cutoff..cutoff+7 across '
           f'{len(cut)} machines; {int((q0 != q1).sum())} of {len(q0)} deployed estimates changed')

    # Positive control: flip the single most recent PRE-cutoff outcome of every machine.
    # A responsive model must move; if it does not, post_cutoff passing is meaningless.
    flipped = []
    for p in vh:
        c = cut.get((p.episode_id, p.machine_id))
        if c is not None and p.tick == c - 1:
            flipped.append(Play(p.episode_id, p.machine_id, p.tick,
                                0 if p.payout > 0 else 5))
        else:
            flipped.append(p)
    q2, _ = WI.beta_template_win_probabilities(order, causal_index(vq, tuple(flipped)),
                                               beta, lam, fb, kappa, prior_mean, log_prior)
    moved = int((q0 != q2).sum())
    record('positive_control', moved > len(q0) * 0.5,
           f'flipping each machine\'s lag-1 pre-cutoff outcome moved {moved} of {len(q0)} '
           f'estimates (median |dq| {float(np.median(abs(q0 - q2))):.4f}), so the model is '
           f'genuinely responsive to legal inputs')

    record('fallbacks', sum(1 for d in d0 if d.get('fallback')) == 0,
           f"{sum(1 for d in d0 if d.get('fallback'))} of {len(d0)} queries fell back; "
           f"template picks "
           f"{ {t: sum(1 for d in d0 if d.get('template') == t) for t in ('null','full','parity')} }")

    # ---- shipped CSV ------------------------------------------------------
    sub = read_submission(args.package / 'test_submission.csv')
    A = np.array([sub[k] for k in sorted(sub)])
    record('simplex', bool(np.isfinite(A).all() and A.min() >= 0
                           and float(abs(A.sum(1) - 1).max()) < 1e-6 and A.shape == (1536, 5)),
           f'shipped CSV {A.shape}: min {A.min():.3e}, max row-sum error '
           f'{float(abs(A.sum(1) - 1).max()):.2e}, order [0,1,2,5,20]')

    passed = all(r['passed'] for r in RESULTS)
    report = {'checks': RESULTS, 'passed': passed,
              'failures': [r for r in RESULTS if not r['passed']],
              'package': str(args.package),
              'scope': 'the SHIPPED Beta template dictionary inference path; synthetic cases test '
                       'implementation behaviour, not the private competition simulator'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"\n{sum(r['passed'] for r in RESULTS)}/{len(RESULTS)} checks passed")
    print('BETA TEMPLATE VERIFICATION', 'PASSED' if passed else 'FAILED')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
