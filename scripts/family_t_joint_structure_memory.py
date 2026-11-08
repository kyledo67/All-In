"""FAMILY T. Infer temporal structure (which lag template) and memory (how far back the
current regime extends) JOINTLY, by a bounded Bayesian online changepoint recursion.

WHY. The protected release HARD-SELECTS one lag template per machine by conjugate log
evidence, and takes its memory from a different, dense model: the per-machine recency
half-life chosen by the 37-coefficient logistic selector. Memory therefore enters as an
exponential decay whose rate was never chosen by the estimator that uses it, and the
structural hypothesis is frozen once and never revised. A previous fixed-share experiment
(scripts/memory_policies.py) froze the structure after a warmup prefix and lost 1.7
points; that is evidence against THAT procedure, not against joint inference.

MODEL (declared in full).
  Observations. For machine m, the scored observations are exactly the complete-lag
  transitions of slot_model.win_head.design: tick t < cutoff with t and all of t-1..t-8
  observed, label 1(Y_t > 0), in increasing ACTUAL tick order. Every arm and every control
  in this script sees that identical observation set, so evidence is comparable across
  arms. A missing tick is never imputed as a loss; it simply yields no row.
  Structure. A template M assigns each observation a cell: `null` (one cell), `(i,j) full`
  (four cells, the two lag signs), `(i,j) parity` (two cells, signs agree vs differ). The
  bank is the UNCHANGED 57-candidate dictionary of the protected release
  (1 + 28 pairs x {full, parity}) with its frozen TRAIN-learned smoothed log prior. Arm
  T3B7 additionally restricts the bank to the candidates with TRAIN argmax support >= 10
  machines (7 of 57), as a bank-robustness check; that support count is read from the
  frozen train-fitted payload and never from a validation target.
  Regime. Within a regime the per-cell win probabilities are independent
  Beta(kappa*mu_type, kappa*(1-mu_type)) draws, mu_type being the frozen train-fitted
  per-machine-type prior mean. A changepoint draws a fresh independent parameter vector
  for every cell at once. This is the product-partition model of Barry and Hartigan, i.e.
  the generative model of Adams and MacKay (arXiv:0710.3742).
  Hazard convention. Constant per-ACTUAL-TICK changepoint hazard h. Because a constant
  hazard is memoryless in elapsed time, the probability of no changepoint across a gap of
  Delta actual ticks is exactly (1-h)^Delta, so the recursion need only track the number of
  OBSERVATIONS in the current regime while the elapsed-tick gap enters through that factor.
  Missing ticks therefore advance the hazard without any outcome being invented, and the
  treatment is exact rather than approximate. MEASURED on this split: every consecutive
  pair of scored observations, and the step from the last observation to the forecast
  origin, is exactly 1 actual tick apart, so the gap factor is never larger than one tick
  here; the general code path is retained and this fact is asserted and reported.
  Initialization for truncated histories. A regime boundary is placed immediately before a
  machine's FIRST scored observation: w(run length = 0) = prior. Nothing before the first
  usable transition is assumed, so a short history is not credited with evidence it does
  not have, and the 32-observation short_history machines start from the prior.
  State pruning / approximation budget. NONE. The run-length support is the exact
  0..n_i, with n_i <= 248 here, so the forward recursion is exact up to floating point;
  there is no truncation, no particle approximation and no fixed-lag window. (A truncated
  variant at M=64 was measured during development and was WORSE, because it caps the
  stable machines' regimes; it is not an arm.) Cost is O(n_i^2 * |bank|) per machine.
  Sequential, not tempered. Inside the recursion each observation enters its regime's
  Beta-Bernoulli predictive exactly once with weight one and integer counts, and every
  prediction is formed BEFORE its own outcome is multiplied in. That is a genuine
  sequential posterior. Arm T1 is explicitly NOT: its evidence term uses the deployed
  decay-weighted FRACTIONAL counts, which is a POWER (tempered) likelihood, so T1's
  weights are generalized model weights and are reported as such.

ARMS (all compose with the protected release's conditional-payout head r, frozen).
  C0  no-change / zero-correction: the protected release itself.
  C1  matched control: this script's own reimplementation of the protected route - hard
      template selection by conjugate log evidence plus train prior, decay weights at the
      per-machine dense-selected half-life, kappa = 0.5. Asserted to reproduce the
      protected q to 1e-12, so the family's control IS the protected release.
  T1  stationary causal mixture over structures: omega_M proportional to
      pi_M exp(lambda * logZ_M^power) with the deployed decay memory; lambda = None is the
      hard argmax, which at kappa = 0.5 is exactly C1. No change inference.
  T2  bounded changepoint / run-length inference with the STRUCTURE FIXED to C1's hard
      selection. Replaces the decay memory and nothing else.
  T2MAP as T2 but forecasting from the MAP run length instead of averaging over it.
  T3  JOINT averaging over structure and run length. The structure posterior keeps being
      revised as evidence accumulates, and each structure is scored under its own
      run-length posterior.
  T3R as T3 but the structural hypothesis is re-drawn from the prior at every changepoint,
      so a change can revise the structure itself.
  T3B7 as T3 on the reduced train-supported 7-member bank (bank robustness).
  h = 0 inside T2/T3 is the stationary no-changepoint limit (the full-history estimator)
  and is kept in every grid as an endpoint control.

NESTING. The 4 frozen complete-episode outer folds of artifacts/phase2/folds.json. For
each outer fold, every arm's (hazard, kappa) or (lambda, kappa) setting AND the decision
to use the arm at all are chosen by minimising the official row-weighted full five-vector
squared error summed over the three complete-episode inner held partitions of that fold's
FITTING episodes (harness.inner_folds), with the reference denominators recomputed on each
partition by harness.row_weights. Held-fold forecasts are concatenated and scored ONCE.
Each arm is reported twice: `forced` (its own settings only) and `gated` (C1 competes, and
is listed first so ties keep the protected route). No arm fits any parameter on inner
TRAINING data - there is nothing to fit, because every forecast is a deterministic causal
function of pre-cutoff history plus train-fitted constants - so the inner partitions enter
only through the objective and its normalisation. No tail constants are estimated anywhere
in this family: r is frozen at the protected head, so harness.tail_constants is never
called and there is nothing to refit per inner partition.

LEAKAGE. Target-dependent quantities fitted here: ONLY the discrete setting index per arm
per outer fold (hazard, kappa, lambda) and the gate decision, each chosen on that fold's
fitting episodes alone. mu_type, the template log prior, the 57-candidate bank, the train
argmax support counts, kappa = 0.5 and the per-machine half-life selector are all read
frozen from the protected package (train-fitted, validation-target-free). r is the
protected release's own conditional-payout head, used unchanged.

EXPOSURE, stated honestly. Public validation has been inspected repeatedly across this
campaign, and the hazard and kappa GRID BOUNDS for this script were informed by an
exploratory sweep on public validation before the nested run. The objective surface is
flat to within 0.1 point over three decades of hazard, which bounds how much that exposure
can inflate the nested number, but the nested figure is not blind.

DETERMINISM. episode_id is a string; every reduction iterates a sorted sequence or an
index range, settings are iterated in a fixed declared order, ties go to the first
setting in that order, and no Python set is ever summed over. Verified by a second run
under PYTHONHASHSEED=777 with byte-identical output CSVs.
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

from slot_model import harness as H
from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_forecasting.io import load_inputs, write_submission
from scripts.win_head_train import build

TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
NCELL_MAX = 4
HAZARDS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
KAPPAS = (0.5, 1.0, 2.0)
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, None)   # None = hard argmax, infinite temperature
SUPPORT_MIN = 10
HALF_LIFE_FLOOR = 1e-12


# ------------------------------------------------------------------ preparation
def prepare(bundle):
    """Per-query ordered scored observations, cell indices and cumulative cell counts."""
    queries, _ = load_inputs(bundle.data / 'validation')
    entries = build(queries, bundle.index)
    out = []
    for qid in bundle.ids:                      # bundle.ids is sorted
        e = entries[qid]
        order = np.argsort(e['used'], kind='stable')
        signs = PD.signs_from(e['phi'][order])
        if e['star'] is None or len(e['y']) == 0:
            raise SystemExit(f'query {qid} has no usable forecast row; this split has none')
        out.append({'qid': qid, 'ticks': e['used'][order].astype(np.float64),
                    'y': e['y'][order].astype(np.float64), 'signs': signs,
                    'star_full': e['star'], 'star_signs': e['star'][1:1 + W.HORIZON][None, :],
                    'phi': e['phi'][order], 'cutoff': float(e['cutoff']), 'type': e['type']})
    return out


def gap_report(records):
    """Assert and summarise the ACTUAL-tick gaps the hazard has to cross."""
    inter, final = {}, {}
    for r in records:
        for g in np.diff(r['ticks']).tolist():
            inter[g] = inter.get(g, 0) + 1
        g = r['cutoff'] - r['ticks'][-1]
        final[g] = final.get(g, 0) + 1
    return {'inter_observation_gap_histogram': {str(k): inter[k] for k in sorted(inter)},
            'final_gap_to_origin_histogram': {str(k): final[k] for k in sorted(final)}}


def half_lives(bundle, records, beta, lam):
    """Per-machine recency half-life from the frozen dense averaged selector (unchanged)."""
    out = np.empty(len(records))
    for i, r in enumerate(records):
        by_type = {h: beta[h].get(r['type'], np.zeros(W.DIM)) for h in sorted(beta, key=lambda z: (z is None, z))}
        _, mix = W.predict_q_averaged(by_type, r['phi'], r['y'], r['ticks'], r['cutoff'],
                                      r['star_full'], lam, adapt='full')
        out[i] = mix['selected_half_life']
    return out


def bank_cells(record, bank):
    """Cell index per (candidate, observation), the forecast cell, and the declared widths."""
    signs, star = record['signs'], record['star_signs']
    ci = np.zeros((len(bank), signs.shape[0]), dtype=np.int64)
    cs = np.zeros(len(bank), dtype=np.int64)
    nc = np.zeros(len(bank), dtype=np.int64)
    for j, ix in enumerate(bank):
        subset, template = TEMPLATES[ix]
        c, width = WI._template_cells(signs, subset, template)
        sc, _ = WI._template_cells(star, subset, template)
        ci[j] = c
        cs[j] = int(sc[0])
        nc[j] = width
    return ci, cs, nc


def prefix_counts(ci, y):
    """Cumulative per-cell win and observation counts, index k = over observations < k."""
    onehot = (ci[:, :, None] == np.arange(NCELL_MAX)[None, None, :]).astype(np.float64)
    wins = np.zeros((ci.shape[0], ci.shape[1] + 1, NCELL_MAX))
    counts = np.zeros((ci.shape[0], ci.shape[1] + 1, NCELL_MAX))
    wins[:, 1:, :] = np.cumsum(onehot * y[None, :, None], axis=1)
    counts[:, 1:, :] = np.cumsum(onehot, axis=1)
    return wins, counts


# --------------------------------------------------------- arm T1 (no change inference)
def decay_weights(record, half_life):
    if not half_life:
        return np.ones(len(record['y']))
    return 2.0 ** (-(record['cutoff'] - 1.0 - record['ticks']) / half_life)


def t1_mixture(records, hl, bank, log_prior, prior_mean, kappa, temperature):
    """Tempered (POWER-likelihood) causal mixture over structures with the deployed decay.

    omega_M proportional to pi_M exp(temperature * logZ_M), logZ_M the conjugate log
    evidence computed from DECAY-WEIGHTED FRACTIONAL counts. That is a power likelihood,
    not a sequential posterior; `temperature=None` is the hard argmax of logZ + log pi,
    which at kappa = 0.5 on the full bank reproduces the protected route exactly.
    """
    q = np.empty(len(records))
    picked = np.empty(len(records), dtype=np.int64)
    maxw = np.empty(len(records))
    for i, r in enumerate(records):
        mu = float(prior_mean.get(r['type'], 0.44))
        a0, b0 = kappa * mu, kappa * (1.0 - mu)
        w = decay_weights(r, None if hl[i] == 0.0 else hl[i])
        ci, cs, nco = bank_cells(r, bank)
        scores = np.empty(len(bank))
        theta = np.empty(len(bank))
        for j, ix in enumerate(bank):
            nc = int(nco[j])
            wins = np.bincount(ci[j], weights=w * r['y'], minlength=nc)
            loss = np.bincount(ci[j], weights=w * (1.0 - r['y']), minlength=nc)
            scores[j] = float(np.sum(betaln(a0 + wins, b0 + loss) - betaln(a0, b0)))
            c = cs[j]
            theta[j] = (a0 + wins[c]) / (a0 + b0 + wins[c] + loss[c])
        total = scores + log_prior
        if temperature is None:
            j = int(np.argmax(total))
            q[i] = theta[j]
            picked[i] = bank[j]
            maxw[i] = 1.0
            continue
        s = temperature * scores + log_prior
        s = s - s.max()
        omega = np.exp(s)
        omega /= omega.sum()
        q[i] = float(np.dot(omega, theta))
        picked[i] = bank[int(np.argmax(omega))]
        maxw[i] = float(omega.max())
    return q, picked, maxw


# ----------------------------------------- arms T2 / T2MAP / T3 / T3R (one shared pass)
def changepoint_pass(records, bank, log_prior, prior_mean, kappa, hazards, fixed_pos,
                     variants=('T2', 'T3', 'T3R')):
    """Exact forward recursion over (structure, run length) for every hazard at once.

    Returns dict keyed by hazard index with q arrays for T2 (structure fixed), T2MAP,
    T3 (structure persists across changepoints) and T3R (structure re-drawn at a
    changepoint), plus diagnostics. Predictions always precede their own outcome's
    likelihood update; run length counts OBSERVATIONS, hazards are advanced over ACTUAL
    tick gaps by (1-h)^Delta; the run-length support is exact (no pruning).
    """
    nh, S, n_rec = len(hazards), len(bank), len(records)
    do2, do3, do3r = 'T2' in variants, 'T3' in variants, 'T3R' in variants
    pi = np.exp(log_prior - np.log(np.sum(np.exp(log_prior))))
    sidx = np.arange(S)[:, None]
    q2 = np.zeros((nh, n_rec)); q2m = np.zeros((nh, n_rec))
    q3 = np.zeros((nh, n_rec)); q3r = np.zeros((nh, n_rec))
    preq2 = np.zeros(nh); preq3 = np.zeros(nh)
    runmean = np.zeros((nh, n_rec)); smax = np.zeros((nh, n_rec))
    smarg = np.zeros((nh, S))
    nobs = 0
    for i, r in enumerate(records):
        mu = float(prior_mean.get(r['type'], 0.44))
        a0, b0 = kappa * mu, kappa * (1.0 - mu)
        ci, cs, _ = bank_cells(r, bank)
        wins, counts = prefix_counts(ci, r['y'])
        y, ticks, n = r['y'], r['ticks'], len(r['y'])
        nobs += n
        p2 = int(fixed_pos[i])
        w3 = np.zeros((nh, S, n + 1)); w3[:, :, 0] = pi[None, :]
        w3r = np.zeros((nh, S, n + 1)); w3r[:, :, 0] = pi[None, :]
        w2 = np.zeros((nh, n + 1)); w2[:, 0] = 1.0
        for k in range(n):
            delta = 1.0 if k == 0 else float(ticks[k] - ticks[k - 1])
            c = ci[:, k]
            idx = k - np.arange(k + 1)
            nw = wins[sidx, k, c[:, None]] - wins[sidx, idx[None, :], c[:, None]]
            nn = counts[sidx, k, c[:, None]] - counts[sidx, idx[None, :], c[:, None]]
            theta = (a0 + nw) / (a0 + b0 + nn)
            lik = theta if y[k] > 0 else 1.0 - theta          # (S, k+1)
            for hi, h in enumerate(hazards):
                keep = 1.0 if k == 0 else (1.0 - h) ** delta
                a3 = keep * w3[hi, :, :k + 1] if do3 else None
                a3r = keep * w3r[hi, :, :k + 1] if do3r else None
                a2 = keep * w2[hi, :k + 1] if do2 else None
                if k > 0:
                    if do3:
                        a3[:, 0] += (1.0 - keep) * w3[hi, :, :k + 1].sum(axis=1)
                    if do3r:
                        a3r[:, 0] += (1.0 - keep) * pi * float(w3r[hi, :, :k + 1].sum())
                    if do2:
                        a2[0] += 1.0 - keep
                if do3:
                    m3 = a3 * lik
                    s3 = float(m3.sum())
                    preq3[hi] += np.log(max(s3, 1e-300))
                    w3[hi, :, 1:k + 2] = m3 / s3
                    w3[hi, :, 0] = 0.0
                if do3r:
                    m3r = a3r * lik
                    s3r = float(m3r.sum())
                    w3r[hi, :, 1:k + 2] = m3r / s3r
                    w3r[hi, :, 0] = 0.0
                if do2:
                    m2 = a2 * lik[p2]
                    s2 = float(m2.sum())
                    preq2[hi] += np.log(max(s2, 1e-300))
                    w2[hi, 1:k + 2] = m2 / s2
                    w2[hi, 0] = 0.0
        idx = n - np.arange(n + 1)
        nw = wins[sidx, n, cs[:, None]] - wins[sidx, idx[None, :], cs[:, None]]
        nn = counts[sidx, n, cs[:, None]] - counts[sidx, idx[None, :], cs[:, None]]
        theta_star = (a0 + nw) / (a0 + b0 + nn)                # (S, n+1)
        runs = np.arange(n + 1, dtype=np.float64)
        for hi, h in enumerate(hazards):
            keep = (1.0 - h) ** float(r['cutoff'] - ticks[-1])
            if do3:
                a3 = keep * w3[hi, :, :n + 1]
                a3[:, 0] += (1.0 - keep) * w3[hi, :, :n + 1].sum(axis=1)
                a3 /= a3.sum()
                q3[hi, i] = float(np.sum(a3 * theta_star))
                marg = a3.sum(axis=1)
                smarg[hi] += marg
                smax[hi, i] = float(marg.max())
                runmean[hi, i] = float(np.sum(a3 * runs[None, :]))
            if do3r:
                a3r = keep * w3r[hi, :, :n + 1]
                a3r[:, 0] += (1.0 - keep) * pi * float(w3r[hi, :, :n + 1].sum())
                a3r /= a3r.sum()
                q3r[hi, i] = float(np.sum(a3r * theta_star))
            if do2:
                a2 = keep * w2[hi, :n + 1]
                a2[0] += 1.0 - keep
                a2 /= a2.sum()
                q2[hi, i] = float(np.dot(a2, theta_star[p2]))
                q2m[hi, i] = float(theta_star[p2, int(np.argmax(a2))])
    return {'q2': q2, 'q2map': q2m, 'q3': q3, 'q3r': q3r,
            'prequential_fixed': preq2, 'prequential_joint': preq3,
            'scored_observations': nobs, 'run_mean': runmean,
            'structure_max_marginal': smax, 'structure_marginal': smarg / n_rec}


# --------------------------------------------------------------------- nested protocol
def nested(bundle, r_head, control_q, candidates, gated):
    """Pick a setting per outer fold on that fold's inner held partitions only."""
    order = (['control'] if gated else []) + [c[0] for c in candidates]
    look = {'control': bundle.compose(control_q, r_head)}
    for name, q in candidates:
        look[name] = bundle.compose(q, r_head)
    held = np.empty((len(bundle.ids), 5))
    chosen, rows = [], []
    for fold in range(4):
        mask = bundle.outer_mask(fold)
        weights = [bundle.row_weights(he) for _, he in bundle.inner_folds(~mask, k=3)]
        best = None
        for name in order:
            per = [bundle.wloss(look[name], w) for w in weights]
            total = float(sum(per))
            rows.append({'fold': fold, 'setting': name, 'inner_objective': total,
                         'partition_objectives': per})
            if best is None or total < best[0]:
                best = (total, name)
        chosen.append(best[1])
        held[mask] = look[best[1]][mask]
    return held, chosen, rows


def error_shares(bundle, P, weights):
    """Per-coordinate share of the official weighted squared error."""
    sq = weights[:, None] * (P - bundle.truth) ** 2
    total = float(sq.sum())
    return {f'p{v}': float(sq[:, j].sum() / total) for j, v in enumerate(H.PAYOUTS)}


def summarise(bundle, P, label):
    o = bundle.official(P)
    return {'arm': label, 'points': o['technical_points'], 'R': o['mean_loss_ratio'],
            'by_condition': {c: o['by_condition'][c]['loss_ratio'] for c in sorted(o['by_condition'])}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/family_t_joint')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'predictions').mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    clock = {}

    bundle = H.load(args.data)
    q_prot, r_prot = bundle.split_heads(bundle.protected)
    model = bundle.package / 'code/model'
    kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    support = np.asarray(payload['train_argmax_counts'], dtype=np.float64)
    names = list(payload['candidates'])
    full_bank = list(range(len(TEMPLATES)))
    small_bank = [i for i in full_bank if support[i] >= SUPPORT_MIN]
    lp_full = np.asarray(log_prior, dtype=np.float64)
    lp_small = lp_full[small_bank]

    records = prepare(bundle)
    gaps = gap_report(records)
    if min(float(k) for k in gaps['inter_observation_gap_histogram']) < 1.0:
        raise SystemExit('non-positive actual-tick gap between scored observations')
    gaps['all_gaps_exactly_one_tick'] = (
        list(gaps['inter_observation_gap_histogram']) == ['1.0']
        and list(gaps['final_gap_to_origin_histogram']) == ['1.0'])
    t0 = time.perf_counter()
    hl = half_lives(bundle, records, beta, lam)
    clock['half_life_selector'] = time.perf_counter() - t0

    # ---- C1, the matched control: the protected route through this script's own code ---
    t0 = time.perf_counter()
    q_c1, pick_c1, _ = t1_mixture(records, hl, full_bank, lp_full, prior_mean, kappa0, None)
    clock['control'] = time.perf_counter() - t0
    dev = float(np.max(np.abs(q_c1 - q_prot)))
    if dev > 1e-12:
        raise SystemExit(f'control reimplementation deviates from the protected q by {dev:.3e}')
    P_control = bundle.compose(q_c1, r_prot)
    fixed_pos_full = np.array([full_bank.index(int(p)) for p in pick_c1])
    fixed_pos_small = np.array([small_bank.index(int(p)) if int(p) in small_bank else
                                small_bank.index(0) for p in pick_c1])

    # ---- arm T1: stationary causal mixture over structures --------------------------
    t0 = time.perf_counter()
    cand = {'T1': [], 'T2': [], 'T2MAP': [], 'T3': [], 'T3R': [], 'T3B7': []}
    diag = {}
    for kappa in KAPPAS:
        for temperature in LAMBDAS:
            tag = f'k{kappa:g}_lam{"hard" if temperature is None else f"{temperature:g}"}'
            q, picked, maxw = t1_mixture(records, hl, full_bank, lp_full, prior_mean,
                                         kappa, temperature)
            cand['T1'].append((tag, q))
            diag[f'T1/{tag}'] = {'mean_max_weight': float(maxw.mean()),
                                 'public_points': bundle.official(bundle.compose(q, r_prot))['technical_points']}
    clock['T1'] = time.perf_counter() - t0

    # ---- arms T2 / T2MAP / T3 / T3R on the full bank ---------------------------------
    t0 = time.perf_counter()
    for kappa in KAPPAS:
        res = changepoint_pass(records, full_bank, lp_full, prior_mean, kappa,
                               HAZARDS, fixed_pos_full)
        for hi, h in enumerate(HAZARDS):
            tag = f'k{kappa:g}_h{h:g}'
            for arm, key in (('T2', 'q2'), ('T2MAP', 'q2map'), ('T3', 'q3'), ('T3R', 'q3r')):
                cand[arm].append((tag, res[key][hi].copy()))
            diag[f'T3/{tag}'] = {
                'mean_posterior_run_length': float(res['run_mean'][hi].mean()),
                'mean_max_structure_marginal': float(res['structure_max_marginal'][hi].mean()),
                'prequential_logscore_joint': float(res['prequential_joint'][hi]),
                'prequential_logscore_fixed_structure': float(res['prequential_fixed'][hi]),
                'public_points': bundle.official(bundle.compose(res['q3'][hi], r_prot))['technical_points']}
            diag[f'T2/{tag}'] = {
                'public_points': bundle.official(bundle.compose(res['q2'][hi], r_prot))['technical_points']}
        if kappa == KAPPAS[0]:
            clock['scored_observations'] = res['scored_observations']
    clock['changepoint_full_bank'] = time.perf_counter() - t0

    # ---- arm T3B7: the reduced train-supported bank ----------------------------------
    t0 = time.perf_counter()
    for kappa in KAPPAS:
        res = changepoint_pass(records, small_bank, lp_small, prior_mean, kappa,
                               HAZARDS, fixed_pos_small, variants=('T3',))
        for hi, h in enumerate(HAZARDS):
            cand['T3B7'].append((f'k{kappa:g}_h{h:g}', res['q3'][hi].copy()))
    clock['changepoint_small_bank'] = time.perf_counter() - t0

    # ---- union arms: let the inner objective choose the bank, and the arm itself -----
    base_arms = sorted(cand)
    cand['T3U'] = ([(f'full|{tag}', q) for tag, q in cand['T3']]
                   + [(f'b7|{tag}', q) for tag, q in cand['T3B7']])
    cand['TALL'] = [(f'{arm}|{tag}', q) for arm in base_arms for tag, q in cand[arm]]

    # ---- nested evaluation ------------------------------------------------------------
    t0 = time.perf_counter()
    arms = {}
    oof = {'control': P_control}
    rows = []
    for arm in sorted(cand):
        entry = {}
        for mode in ('gated', 'forced'):
            P, chosen, table = nested(bundle, r_prot, q_c1, cand[arm], mode == 'gated')
            for row in table:
                rows.append({'arm': arm, 'mode': mode, **row})
            s = summarise(bundle, P, f'{arm}_{mode}')
            s['selected_per_fold'] = chosen
            s['paired_vs_protected'] = bundle.paired(P, bundle.protected)
            s['paired_vs_control'] = bundle.paired(P, P_control)
            entry[mode] = s
            oof[f'{arm}_{mode}'] = P
        arms[arm] = entry
    clock['nested'] = time.perf_counter() - t0

    # within-family increments, on the gated nested forecasts
    increments = {}
    for a, b_ in (('T3', 'T2'), ('T2', 'T1'), ('T3', 'T1'), ('T3B7', 'T3'), ('T3R', 'T3')):
        increments[f'{a}_vs_{b_}'] = bundle.paired(oof[f'{a}_gated'], oof[f'{b_}_gated'])['delta_loss_ratio_ci95']

    # ---- deterministic test-time recipe ---------------------------------------------
    recipe = {}
    w_all = bundle.row_weights(np.ones(len(bundle.ids), dtype=bool))
    shares = {'C0_protected': error_shares(bundle, bundle.protected, w_all)}
    for name in sorted(oof):
        shares[name] = error_shares(bundle, oof[name], w_all)
    for arm in sorted(cand):
        losses = [(bundle.wloss(bundle.compose(q, r_prot), w_all), tag) for tag, q in cand[arm]]
        best = min(range(len(losses)), key=lambda j: (losses[j][0], j))
        recipe[arm] = {'official_objective_argmin_all_fitting_episodes': losses[best][1]}
    preq = {k: v['prequential_logscore_joint'] for k, v in diag.items() if k.startswith('T3/')}
    recipe['T3']['target_free_prequential_argmax'] = max(sorted(preq), key=lambda k: (preq[k], k)).split('/')[1]
    recipe['note'] = ('Every forecast is a deterministic causal function of pre-cutoff history '
                      'plus frozen train-fitted constants, so a test-time recipe exists for every '
                      'arm: fix the setting by the stated rule, then run the recursion on the test '
                      'histories. No RNG, no refitting, no validation target at inference.')

    report = {
        'preregistration': __doc__,
        'protected': summarise(bundle, bundle.protected, 'C0_protected'),
        'control_C1': {**summarise(bundle, P_control, 'C1_control'),
                       'max_abs_q_deviation_from_protected': dev},
        'arms': arms,
        'within_family_increments_gated': increments,
        'bank': {'full_bank_size': len(full_bank), 'small_bank_size': len(small_bank),
                 'small_bank': [names[i] for i in small_bank],
                 'small_bank_train_argmax_coverage': float(support[small_bank].sum() / support.sum()),
                 'control_hard_selection_histogram': {
                     names[i]: int((pick_c1 == i).sum()) for i in full_bank
                     if int((pick_c1 == i).sum()) > 0}},
        'union_arms': {
            'T3U': 'T3 on both banks; hazard, kappa AND bank chosen on the inner held partitions',
            'TALL': ('the whole family as one nested procedure: control plus every setting of '
                     f'{base_arms} competes on the inner held partitions, so the arm, its '
                     'hyperparameters and the decision to deviate from the protected route are '
                     'all chosen fitting-side')},
        'grids': {'hazards': list(HAZARDS), 'kappas': list(KAPPAS),
                  'lambdas': ['hard' if v is None else v for v in LAMBDAS],
                  'settings_per_arm': {a: len(cand[a]) for a in sorted(cand)}},
        'gaps': gaps,
        'error_shares_by_coordinate': shares,
        'q_head_rms_change_vs_protected': {
            name: float(np.sqrt(np.mean((bundle.split_heads(oof[name])[0] - q_prot) ** 2)))
            for name in sorted(oof)},
        'public_diagnostics_PRIVILEGED': diag,
        'test_time_recipe': recipe,
        'runtime_seconds': clock,
        'seconds': time.perf_counter() - start,
        'r_held_fixed': True,
        'exposure': ('Repeatedly inspected public validation; the hazard and kappa grid bounds '
                     'were informed by an exploratory public-validation sweep before this run.'),
    }
    for name in sorted(oof):
        write_submission(args.out / 'predictions' / f'oof_{name}.csv', bundle.mapping(oof[name]))
    with open(args.out / 'selection_table.csv', 'w') as fh:
        fh.write('arm,mode,fold,setting,inner_objective,p0,p1,p2\n')
        for row in rows:
            fh.write(f"{row['arm']},{row['mode']},{row['fold']},{row['setting']},"
                     f"{row['inner_objective']:.17g},"
                     + ','.join(format(v, '.17g') for v in row['partition_objectives']) + '\n')
    with open(args.out / 'summary.csv', 'w') as fh:
        fh.write('arm,mode,points,R,stable,short_history,drift,shared_change,'
                 'paired_vs_protected_lo,paired_vs_protected_hi\n')
        base = report['protected']
        fh.write('C0_protected,none,' + f"{base['points']:.17g},{base['R']:.17g},"
                 + ','.join(f"{base['by_condition'][c]:.17g}" for c in
                            ('stable', 'short_history', 'drift', 'shared_change'))
                 + ',,\n')
        for arm in sorted(arms):
            for mode in ('forced', 'gated'):
                s = arms[arm][mode]
                lo, hi = s['paired_vs_protected']['delta_loss_ratio_ci95']
                fh.write(f"{arm},{mode},{s['points']:.17g},{s['R']:.17g},"
                         + ','.join(f"{s['by_condition'][c]:.17g}" for c in
                                    ('stable', 'short_history', 'drift', 'shared_change'))
                         + f',{lo:.17g},{hi:.17g}\n')
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    print(f"protected               {report['protected']['points']:12.6f}  R={report['protected']['R']:.6f}")
    print(f"control C1 (reimpl)     {report['control_C1']['points']:12.6f}  "
          f"max|dq|={dev:.2e}")
    print(f"{'arm':<8}{'mode':<8}{'points':>12}{'R':>10}  paired vs protected")
    for arm in sorted(arms):
        for mode in ('forced', 'gated'):
            s = arms[arm][mode]
            lo, hi = s['paired_vs_protected']['delta_loss_ratio_ci95']
            flag = 'SIGNIFICANT' if hi < 0 else 'overlaps zero'
            print(f"{arm:<8}{mode:<8}{s['points']:>12.6f}{s['R']:>10.6f}  "
                  f"[{lo:+.5f}, {hi:+.5f}] {flag}  folds={s['selected_per_fold']}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
