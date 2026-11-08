"""Focused verification of the hierarchical logistic win head.

These are Python test outputs about the implementation, kept separate from the
competition scores. Synthetic success establishes implementation capability, not
competition improvement. Nothing here is a forecasting score.

Checks:
  gradient        central finite differences against the analytic gradient, inside the
                  region where the two are gradients of the same function
  stationary_mode the solver reaches a stationary, positive-definite mode
  clip_consistency `objective` differentiates through the ETA_CLIP saturation and
                  `gradient` does not, so they disagree outside |eta| < ETA_CLIP; this
                  asserts the property that makes that inert, namely that no fit on
                  these inputs gets anywhere near the clip
  opposite        two same-type machines with OPPOSITE lag-1 responses are separated by
                  the local channel, while the pooled fit cancels the effect entirely
  xor             a pairwise XOR response is recovered by the interaction model and
                  missed by the main-effects-only model, whose interaction block is
                  verified to be frozen at zero
  shrinkage       an empty local history returns the population prediction exactly, and
                  a short history widens the Laplace logit variance, which pulls the
                  predictive probability toward 0.5
  post_cutoff     altering outcomes at or after the cutoff leaves forecasts bitwise
                  equal, checked both end to end and at the win-head boundary
  missing         a missing lag drops exactly the transitions that need it instead of
                  becoming an observed loss
  coverage        every scored query really is predicted by the win head, with no silent
                  fallback to the reference
  outer_fold      altering only a held fold's target vectors leaves the fitted blend
                  coefficient bitwise equal, compared BEFORE the [0,1] clip
  simplex         all five-class outputs finite, nonnegative and summing to one
  signal          the win head source contains no reference to any signal field
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import win_head as W
from slot_model.features import causal_index
from slot_forecasting.io import load_inputs, read_submission
from slot_forecasting.schema import Play

RESULTS = []


def record(name, passed, detail):
    RESULTS.append({'check': name, 'passed': bool(passed), 'detail': detail})
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)


def synth(coef, n=400, seed=0):
    """Generate a machine whose win probability follows `coef` on the 37-dim basis.

    `coef[0]` multiplies the bias column of `_row`, so it is the intercept.
    """
    rng = np.random.default_rng(seed)
    hist = {}
    # Seed the first 8 ticks arbitrarily, then generate forward from actual lags.
    for t in range(8):
        hist[t] = 1 if rng.random() < 0.5 else 0
    for t in range(8, n):
        signs = np.array([1.0 if hist[t - l] > 0 else -1.0 for l in range(1, 9)])
        eta = float(coef @ W._row(signs))
        hist[t] = 1 if rng.random() < 1 / (1 + np.exp(-eta)) else 0
    return hist, n


def force(hist, cutoff, pattern, first_lag=1):
    """Copy of `hist` with ticks cutoff-first_lag.. set from `pattern`."""
    out = dict(hist)
    for offset, value in enumerate(pattern):
        out[cutoff - (first_lag + offset)] = int(value)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/win_head/verification.json')
    args = parser.parse_args()
    lam = W.penalty(0.01, 4.0, 1.0)
    beta_zero = np.zeros(W.DIM)

    # ---- gradient ----------------------------------------------------------
    rng = np.random.default_rng(7)
    hist, cutoff = synth(np.zeros(W.DIM), n=160, seed=1)
    phi, y, w, used = W.design(hist, cutoff, half_life=64.0)
    theta = rng.normal(scale=0.3, size=W.DIM)
    offset = rng.normal(scale=0.2, size=len(y))
    analytic = W.gradient(theta, phi, y, w, offset, lam)
    numeric = np.empty(W.DIM)
    h = 1e-6
    for i in range(W.DIM):
        up, dn = theta.copy(), theta.copy()
        up[i] += h; dn[i] -= h
        numeric[i] = (W.objective(up, phi, y, w, offset, lam)
                      - W.objective(dn, phi, y, w, offset, lam)) / (2 * h)
    rel = float(np.max(abs(analytic - numeric)) / max(np.max(abs(numeric)), 1e-12))
    # The finite-difference identity only holds where `objective` is not saturated,
    # so assert the probe stays in that region rather than assuming it.
    probe_eta = float(np.max(abs(offset + phi @ theta)))
    record('gradient', rel < 1e-6 and probe_eta < W.ETA_CLIP,
           f'max relative deviation {rel:.2e} over {W.DIM} coordinates (central differences, '
           f'h=1e-6); max |eta| on the probe {probe_eta:.3f} < ETA_CLIP {W.ETA_CLIP:.0f}, so the '
           f'identity is in scope')

    # Also check the solver reaches a stationary point.
    theta_hat, hess = W._newton(phi, y, w, offset, lam)
    gnorm = float(np.linalg.norm(W.gradient(theta_hat, phi, y, w, offset, lam)))
    eig = float(np.min(np.linalg.eigvalsh(hess)))
    record('stationary_mode', gnorm < 1e-7 and eig > 0,
           f'gradient norm at mode {gnorm:.2e}, min Hessian eigenvalue {eig:.3e} (positive definite)')

    # ---- ETA_CLIP consistency ---------------------------------------------
    # `objective` clips eta, so its derivative through a saturated row is zero;
    # `gradient` (and `_newton`) use the unclipped-likelihood formula. The two are
    # therefore gradients of DIFFERENT functions once |eta| >= ETA_CLIP. Measure that
    # divergence, then assert the property that makes it inert on these inputs.
    clip_theta = np.zeros(W.DIM)
    clip_theta[0] = 2.0 * W.ETA_CLIP
    g_clip = W.gradient(clip_theta, phi, y, w, np.zeros(len(y)), lam)
    n_clip = np.empty(W.DIM)
    for i in range(W.DIM):
        up, dn = clip_theta.copy(), clip_theta.copy()
        up[i] += 1e-4; dn[i] -= 1e-4
        n_clip[i] = (W.objective(up, phi, y, w, np.zeros(len(y)), lam)
                     - W.objective(dn, phi, y, w, np.zeros(len(y)), lam)) / 2e-4
    clip_gap = float(np.max(abs(g_clip - n_clip)))
    clip_margins = []

    # ---- opposite machine responses ---------------------------------------
    # Two machines of the same type with opposite lag-1 responses. The pooled fit sees
    # both and must cancel the effect; only the local channel can separate them.
    plus = np.zeros(W.DIM); plus[W.MAIN[0]] = 2.0
    minus = np.zeros(W.DIM); minus[W.MAIN[0]] = -2.0
    mats = {}
    for label, coef, seed in (('plus', plus, 11), ('minus', minus, 12)):
        hist_m, cut_m = synth(coef, n=400, seed=seed)
        mats[label] = (hist_m, cut_m, W.design(hist_m, cut_m, half_life=None))
    phi_all = np.vstack([mats[k][2][0] for k in ('plus', 'minus')])
    y_all = np.concatenate([mats[k][2][1] for k in ('plus', 'minus')])
    w_all = np.concatenate([mats[k][2][2] for k in ('plus', 'minus')])
    beta_pool, _ = W._newton(phi_all, y_all, w_all, np.zeros(len(y_all)), lam)
    pooled_lag1 = float(beta_pool[W.MAIN[0]])
    local_lag1 = {}
    for label in ('plus', 'minus'):
        p_m, y_m, w_m, _ = mats[label][2]
        u_m, _ = W._newton(p_m, y_m, w_m, p_m @ beta_pool, np.array(lam))
        local_lag1[label] = float(u_m[W.MAIN[0]])
        clip_margins.append(float(np.max(abs(p_m @ (beta_pool + u_m)))))

    # The claim is about each machine's lag-1 RESPONSE, so vary lag 1 with lags 2..8
    # held fixed, over many conditioning patterns, instead of reading one query row.
    prng = np.random.default_rng(5)
    patterns = [list(prng.integers(0, 2, 7)) for _ in range(16)]
    response = {'plus': [], 'minus': []}
    for label in ('plus', 'minus'):
        hist_m, cut_m, _ = mats[label]
        for pattern in patterns:
            base_hist = force(hist_m, cut_m, pattern, first_lag=2)
            got = []
            for lag1 in (1, 0):
                h2 = force(base_hist, cut_m, [lag1], first_lag=1)
                p2, y2, w2, _ = W.design(h2, cut_m, half_life=None)
                got.append(W.predict_q(beta_pool, p2, y2, w2, W.query_row(h2, cut_m),
                                       lam, adapt='full')[0])
            response[label].append(got[0] - got[1])
    gap = np.array(response['plus']) - np.array(response['minus'])
    wrong_sign = (int(np.sum(np.asarray(response['plus']) <= 0))
                  + int(np.sum(np.asarray(response['minus']) >= 0)))

    # For contrast, the weaker thing the single-query-row formulation measures: hold the
    # whole 8-lag conditioning row identical across the two machines and read one q.
    single = []
    for pattern in patterns[:12]:
        qs = []
        for label in ('plus', 'minus'):
            hist_m, cut_m, _ = mats[label]
            h2 = force(hist_m, cut_m, [1] + list(pattern[:7]), first_lag=1)
            p2, y2, w2, _ = W.design(h2, cut_m, half_life=None)
            star2 = W.query_row(h2, cut_m)
            qs.append((W.predict_q(beta_pool, p2, y2, w2, star2, lam, adapt='full')[0], star2))
        assert np.array_equal(qs[0][1], qs[1][1]), 'conditioning rows must be identical'
        single.append(qs[0][0] - qs[1][0])
    single = np.array(single)
    record('opposite',
           abs(pooled_lag1) < 0.10 and local_lag1['plus'] > 1.0 and local_lag1['minus'] < -1.0
           and bool((gap > 0).all()) and gap.mean() > 0.30,
           f'true lag-1 effects {plus[W.MAIN[0]]:+.1f} and {minus[W.MAIN[0]]:+.1f}: the pooled fit '
           f'over both machines recovers {pooled_lag1:+.6f} (the two cancel, so pooling cannot '
           f'represent either), while the local deviations recover {local_lag1["plus"]:+.4f} and '
           f'{local_lag1["minus"]:+.4f} (shrunk by lam_main=4); over {len(patterns)} held '
           f'conditioning patterns the lag-1 probability response gap is mean {gap.mean():+.4f}, '
           f'min {gap.min():+.4f}, positive in all {len(gap)}. LIMITATION, measured: the '
           f'individual per-machine response had the wrong sign in {wrong_sign} of '
           f'{2 * len(patterns)} cases, and reading a SINGLE identical query row instead of a '
           f'response separates by only mean {single.mean():+.4f} (min {single.min():+.4f}, '
           f'sign inverted in {int(np.sum(single <= 0))} of {len(single)}) because 28 interaction '
           f'coefficients inject noise into one conditioning row')

    # ---- pairwise XOR ------------------------------------------------------
    xor = np.zeros(W.DIM)
    xor[W.INTER[0]] = 2.5           # s1*s2 term: the (1,2) pair is first in PAIRS
    hist, cutoff = synth(xor, n=500, seed=21)
    q_full, q_main, inter_frozen, inter_fitted = {}, {}, 0.0, []
    for name, (l1, l2) in (('agree', (1, 1)), ('disagree', (1, 0))):
        h2 = force(hist, cutoff, [l1, l2], first_lag=1)
        star = W.query_row(h2, cutoff)
        p2, y2, w2, _ = W.design(h2, cutoff, half_life=None)
        q_full[name] = W.predict_q(beta_zero, p2, y2, w2, star, lam, adapt='full')[0]
        q_main[name] = W.predict_q(beta_zero, p2, y2, w2, star, lam, adapt='main')[0]
        # The structural reason the main-effects arm must miss it: its interaction
        # block is pinned at zero, so its logit is additive in s1 and s2.
        lam_main_only = np.array(lam); lam_main_only[W.INTER] = 1e12
        u_main, _ = W._newton(p2, y2, w2, p2 @ beta_zero, lam_main_only)
        u_full, _ = W._newton(p2, y2, w2, p2 @ beta_zero, np.array(lam))
        inter_frozen = max(inter_frozen, float(np.max(abs(u_main[W.INTER]))))
        inter_fitted.append(float(u_full[W.INTER[0]]))
        clip_margins.append(float(np.max(abs(p2 @ u_full))))
    gap_full = q_full['agree'] - q_full['disagree']
    gap_main = q_main['agree'] - q_main['disagree']
    record('xor', gap_full > 0.30 and abs(gap_main) < 0.10 and inter_frozen < 1e-6,
           f'with a true s1*s2 effect of {xor[W.INTER[0]]:+.1f} the interaction model recovers '
           f'{inter_fitted[0]:+.4f} and {inter_fitted[1]:+.4f} on the two conditioning histories '
           f'and separates agreeing from disagreeing lags by {gap_full:+.4f}, '
           f'while the main-effects-only model separates them by {gap_main:+.4f}; its interaction '
           f'block is pinned at max |u| {inter_frozen:.2e}, so its logit is additive in s1 and s2 '
           f'and cannot represent XOR (the {gap_main:+.4f} residue is main-effect noise, not an '
           f'interaction)')

    # ---- short history: exact population limit, and Laplace widening ------
    # NOTE what is and is not asserted here. The local logit deviation does NOT shrink
    # monotonically as the history shortens (measured below). What happens is that the
    # Laplace logit variance widens and the probit scaling pulls the probability toward
    # 0.5. Those coincide with "toward the population prediction" only when 0.5 lies
    # between the two, so that wording is not asserted.
    beta_pop = np.zeros(W.DIM); beta_pop[0] = -1.0      # population logit -1 => q ~ 0.269
    strong = np.zeros(W.DIM); strong[0] = 3.0
    hist, cutoff = synth(strong, n=400, seed=31)
    full_phi, full_y, full_w, full_used = W.design(hist, cutoff, half_life=None)
    star = W.query_row(hist, cutoff)
    order = np.argsort(full_used)
    q_pop = W._sigmoid(float(beta_pop @ star))
    q_long, d_long = W.predict_q(beta_pop, full_phi, full_y, full_w, star, lam, adapt='full')
    keep = order[-14:]
    q_short, d_short = W.predict_q(beta_pop, full_phi[keep], full_y[keep], full_w[keep],
                                   star, lam, adapt='full')
    # (i) the exact limit: no local evidence means exactly the population prediction
    empty = order[:0]
    q_empty, d_empty = W.predict_q(beta_pop, full_phi[empty], full_y[empty], full_w[empty],
                                   star, lam, adapt='full')
    exact_limit = q_empty == q_pop and d_empty['logit_var'] == 0.0
    # (ii) the widening is what moves the probability, and it moves it toward 0.5
    pulled = abs(q_short - 0.5) < abs(W._sigmoid(d_short['logit']) - 0.5)
    # (iii) the widening is robust across seeds, not an artefact of seed 31
    wider, ndev_full, ndev_short = 0, [], []
    for s in range(20):
        h2, c2 = synth(strong, n=400, seed=400 + s)
        p2, y2, w2, u2 = W.design(h2, c2, half_life=None)
        st2 = W.query_row(h2, c2); o2 = np.argsort(u2); base = float(beta_pop @ st2)
        _, da = W.predict_q(beta_pop, p2, y2, w2, st2, lam, adapt='full')
        k2 = o2[-14:]
        _, db = W.predict_q(beta_pop, p2[k2], y2[k2], w2[k2], st2, lam, adapt='full')
        wider += db['logit_var'] > da['logit_var']
        ndev_full.append(abs(da['logit'] - base)); ndev_short.append(abs(db['logit'] - base))
    record('shrinkage', exact_limit and pulled and d_short['logit_var'] > d_long['logit_var']
           and wider == 20,
           f'population q={q_pop:.4f}; an EMPTY local history returns q={q_empty:.6f}, bitwise '
           f'equal to the population, with logit variance 0. {len(full_y)} transitions give '
           f'q={q_long:.4f} (logit var {d_long["logit_var"]:.3f}); 14 transitions give '
           f'q={q_short:.4f} (logit var {d_short["logit_var"]:.3f}), and the variance is the '
           f'larger one in {wider}/20 seeds. The widening pulls the probability toward 0.5: '
           f'|q-0.5|={abs(q_short - 0.5):.4f} against {abs(W._sigmoid(d_short["logit"]) - 0.5):.4f} '
           f'for the unwidened sigmoid of the same logit. NOT asserted, because measured false: '
           f'the local logit deviation does not shrink with a shorter history (mean '
           f'|deviation| {np.mean(ndev_full):.3f} at full history against '
           f'{np.mean(ndev_short):.3f} at 14 transitions over 20 seeds), so this is a pull toward '
           f'0.5, not a shrink toward the population logit')

    # ---- win-head-level post-cutoff guard ---------------------------------
    # Bypasses causal_index so this tests win_head's own boundary, not the loader's.
    hist, cutoff = synth(plus, n=400, seed=11)
    raw = dict(hist)
    for off in range(8):
        raw[cutoff + off] = 1
    clean_d = W.design(hist, cutoff, half_life=None)
    dirty_d = W.design(raw, cutoff, half_life=None)
    boundary = (np.array_equal(clean_d[0], dirty_d[0]) and np.array_equal(clean_d[1], dirty_d[1])
                and np.array_equal(W.query_row(hist, cutoff), W.query_row(raw, cutoff))
                and not any(t >= cutoff for t in dirty_d[3]))

    # ---- real-data checks --------------------------------------------------
    if args.data:
        vq, vh = load_inputs(args.data / 'validation')
        index = causal_index(vq, vh)
        payload = json.loads((ROOT / 'artifacts/win_head/population.json').read_text())
        chosen = payload['selected']
        beta = {t: np.asarray(v) for t, v in payload['beta'].items()}
        half_life = chosen['half_life']
        rlam = W.penalty(chosen['lam_intercept'], chosen['lam_main'], chosen['lam_inter'])
        real_eta = [0.0]

        def predict_all(idx, track=False):
            out = {}
            for q in vq:
                ticks = idx[(q.episode_id, q.machine_id)]
                phi, y, _, used = W.design(ticks, q.cutoff, half_life=None)
                star = W.query_row(ticks, q.cutoff)
                if star is None:
                    continue
                wts = (np.ones(len(y)) if not half_life else
                       2.0 ** (-(q.cutoff - 1 - np.asarray(used, dtype=np.float64)) / half_life))
                b = beta.get(q.machine_type, np.zeros(W.DIM))
                out[q.query_id] = W.predict_q(b, phi, y, wts, star, rlam, adapt='full')[0]
                if track and len(y):
                    u, _ = W._newton(phi, y, wts, phi @ b, np.array(rlam))
                    real_eta[0] = max(real_eta[0], float(np.max(abs(phi @ (b + u)))))
            return out

        baseline = predict_all(index, track=True)
        cutoffs = {(q.episode_id, q.machine_id): q.cutoff for q in vq}
        extra = []
        for (e, m), cut in sorted(cutoffs.items()):
            observed = index[(e, m)]
            for off in range(8):
                if cut + off not in observed:
                    extra.append(Play(e, m, cut + off, (0, 1, 2, 5, 20)[(observed.get(cut - 1, 0) + 3) % 5]))
        mutated = predict_all(causal_index(vq, tuple(vh) + tuple(extra)))
        same = all(baseline[k] == mutated[k] for k in baseline)
        record('post_cutoff', same and len(extra) > 0 and boundary,
               f'{len(extra)} adversarial plays appended at ticks cutoff..cutoff+7 across '
               f'{len(cutoffs)} machines; {sum(1 for k in baseline if baseline[k] != mutated[k])} '
               f'of {len(baseline)} win estimates changed. At the win-head boundary, feeding '
               f'`design`/`query_row` an UNFILTERED tick map containing 8 post-cutoff ticks '
               f'(bypassing causal_index) leaves the design matrix, the labels and the query row '
               f'bitwise equal and uses no tick >= cutoff: {boundary}')

        # missing lag never becomes a loss
        probe = next(iter(vq))
        ticks = dict(index[(probe.episode_id, probe.machine_id)])
        before = W.design(ticks, probe.cutoff, half_life=None)[0].shape[0]
        victim = probe.cutoff - 12
        had_victim = victim in ticks
        ticks.pop(victim, None)
        after_phi, after_y, _, after_used = W.design(ticks, probe.cutoff, half_life=None)
        dropped = before - after_phi.shape[0]
        as_lag = [t for t in after_used if any(t - l == victim for l in range(1, W.HORIZON + 1))]
        # Exactly HORIZON+1: the victim as a target, plus the 8 transitions needing it as a lag.
        record('missing', had_victim and dropped == W.HORIZON + 1 and victim not in after_used
               and not as_lag,
               f'removing tick {victim} dropped exactly {dropped} transitions (expected '
               f'{W.HORIZON + 1}: the victim as a target plus the {W.HORIZON} transitions needing '
               f'it as a lag) rather than imputing a loss; the removed tick appears neither as an '
               f'observed outcome nor as a lag in any of the {len(after_used)} retained rows')

        from slot_model.official_score import scoring_inputs
        scoring = scoring_inputs(args.data)
        targets, groups, reference = scoring
        ids = sorted(reference)
        ep = np.array([{t.query_id: t.episode_id for t in targets}[q] for q in ids])
        truth = np.array([{t.query_id: t.probabilities for t in targets}[q] for q in ids])
        ref_arr = np.array([reference[q] for q in ids])
        base = read_submission(ROOT / 'artifacts/audit/oof_candidate_raw.csv')
        base = np.array([base[q] for q in ids])
        q_exist = 1 - base[:, 0]
        props = base[:, 1:] / np.maximum(q_exist, 1e-12)[:, None]
        missing_ids = [q for q in ids if q not in baseline]
        q_new = np.array([baseline.get(q, q_exist[i]) for i, q in enumerate(ids)])
        episodes = set(ep.tolist())
        record('coverage', not missing_ids and len(ids) == len(baseline),
               f'{len(baseline)} of {len(ids)} scored queries have a win-head estimate, so '
               f'{len(missing_ids)} fall back to the reference q; a fallback would silently make '
               f'the simplex and blend checks partly about the reference instead of the win head')

        # clip consistency, now that the real fits have been measured
        clip_margins.append(real_eta[0])
        worst = max(clip_margins)
        record('clip_consistency', worst < 0.5 * W.ETA_CLIP,
               f'`objective` differentiates through the ETA_CLIP saturation and `gradient` does '
               f'not, so at |eta|=2*ETA_CLIP the analytic gradient and central differences '
               f'disagree by {clip_gap:.3f} - they are gradients of different functions there, '
               f'and `objective`\'s docstring claim to be "the exact MAP objective the Newton '
               f'solver minimizes" holds only for |eta| < ETA_CLIP={W.ETA_CLIP:.0f}. That is '
               f'inert here: the largest |eta| at any fitted mode is {worst:.4f} across every '
               f'synthetic fit above and all {len(baseline)} real validation fits, a factor '
               f'{W.ETA_CLIP / max(worst, 1e-9):.1f} inside the clip')

        # outer-fold target invariance
        folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

        def row_weights(es, tr):
            w = np.zeros(len(ids))
            for condition in sorted(set(groups.values())):
                ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == condition}
                b = np.mean([np.mean(np.sum((ref_arr[ep == e] - tr[ep == e]) ** 2, axis=1)) for e in ces])
                for e in ces:
                    w[ep == e] = 1 / (4 * len(ces) * sum(ep == e) * max(b, 1e-9))
            return w

        def alpha(es, tr):
            """Returns (raw ratio, clipped) - the same closed form as win_head_evaluate."""
            w = row_weights(es, tr)
            A = 1 + np.sum(props ** 2, axis=1)
            B = 1 - tr[:, 0] + np.sum(props * tr[:, 1:], axis=1)
            d = q_new - q_exist
            den = float(np.sum(w * d * d * A))
            if den <= 0:
                return 0.0, 0.0
            raw = float(-np.sum(w * d * (q_exist * A - B)) / den)
            return raw, float(np.clip(raw, 0, 1))

        rng2 = np.random.default_rng(99)
        changed_raw, changed_clipped, saturated, raws = 0, 0, 0, []
        for fold in range(4):
            held = {e for e in episodes if folds[e] == fold}
            fit = episodes - held
            r0, c0 = alpha(fit, truth)
            tampered = truth.copy()
            mask = np.array([e in held for e in ep])
            noise = rng2.random((int(mask.sum()), 5)) + 0.05
            tampered[mask] = noise / noise.sum(axis=1, keepdims=True)
            r1, c1 = alpha(fit, tampered)
            changed_raw += r0 != r1
            changed_clipped += c0 != c1
            saturated += c0 in (0.0, 1.0)
            raws.append(r0)
        record('outer_fold', changed_raw == 0,
               f'held-fold target vectors replaced with random simplex points in all 4 folds; the '
               f'UNCLIPPED blend coefficient changed in {changed_raw} of 4 folds (raw ratios '
               f'{", ".join("%.9f" % r for r in raws)}). This is compared before the [0,1] clip on '
               f'purpose: the clipped alpha is saturated at a bound in {saturated} of 4 folds, so '
               f'comparing the clipped value alone would pass even if tampering moved the ratio '
               f'(it changed in {changed_clipped} of 4 clipped). The fitting weights and both '
               f'closed-form sums are zero on held rows by construction')

        # simplex validity on the real candidate
        vec = np.array([W.compose(q_new[i], props[i]) for i in range(len(ids))])
        ok = (np.isfinite(vec).all() and vec.min() >= 0
              and float(np.max(abs(vec.sum(1) - 1))) < 1e-12 and vec.shape[1] == 5)
        record('simplex', ok,
               f'{vec.shape[0]} five-vectors: min {vec.min():.3e}, max sum error '
               f'{float(np.max(abs(vec.sum(1) - 1))):.2e}, order [0,1,2,5,20] preserved')

    # ---- no signal field anywhere in the win head -------------------------
    source = (ROOT / 'slot_model/win_head.py').read_text()
    hits = [f'{i}: {line.strip()}' for i, line in enumerate(source.splitlines(), 1)
            if re.search(r'signal', line, re.IGNORECASE)]
    record('signal', not hits,
           f'scanned {len(source.splitlines())} lines of slot_model/win_head.py for /signal/i: '
           f'{len(hits)} matches{"" if not hits else " -> " + "; ".join(hits)}. The win head reads '
           f'no signal field, so no historical signal value is invented and an unknown historical '
           f'signal is never represented as an observed zero')

    passed = all(r['passed'] for r in RESULTS)
    report = {'checks': RESULTS, 'passed': passed,
              'failures': [r for r in RESULTS if not r['passed']],
              'scope': 'implementation capability on synthetic and real inputs; not a competition score'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"\n{sum(r['passed'] for r in RESULTS)}/{len(RESULTS)} checks passed")
    print('VERIFY', 'PASSED' if passed else 'FAILED')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
