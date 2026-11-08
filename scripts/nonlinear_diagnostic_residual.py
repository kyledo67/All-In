"""FAMILY N: a small nonlinear correction driven by model DIAGNOSTICS, not by the signal.

QUESTION. The protected release leaves 45.6% of its remaining error on p0 and 54.4% on the
conditional-payout side (p1 28.0%, p2 20.2%, p5 5.7%, p20 0.5%). A previous attempt corrected
the whole five-vector with a LINEAR log-mass residual on six/eight inputs and was not
significant (+0.103787 points, paired dR 95% [-0.00612, +0.00179]). That input set was built
out of one query feature (the lag-1 signal, now a closed question) plus two parity-count
summaries. This family asks a different question: does the baseline carry an exploitable
error signature in its OWN DIAGNOSTICS -- how much evidence stands behind the queried cell,
how uncertain the win logit is, how much its memory experts and its structure experts
disagree, how badly a comparable local predictor has been surprised RECENTLY versus EARLIER
on the same machine, and how far the audited candidate pool spreads around it?

CORRECTION FORM, declared before any number below was read.
    delta  = additive five-vector residual, predicted from features X
    p      = Pi( protected + delta ),  Pi = Euclidean projection onto the 5-simplex
Rows of delta are forced to sum to zero before the projection (for the closed-form weighted
ridge this is automatic, because both truth and protected rows sum to one). The additive form
is used because the official loss is squared error in exactly this space, so the fitting
objective IS the objective: minimising sum_i w_i ||protected_i + delta_i - t_i||^2 over a
linear delta is weighted least squares on the residual, with w from harness.row_weights.
EXACTLY ZERO CORRECTION is a grid member of every arm ('zero' for the ridge arms, shrinkage
s = 0 for the boosted arm), so "no change" can win on the inner folds.

ARMS, all sharing one pipeline, one transform, one grid machinery, so that the differences
between them isolate one thing each:
  A0 zero                 protected release unchanged. The floor.
  C1 linear_old8_log      LITERAL existing control: the published OOF forecast of
                          scripts/residual_fallback.py (log-mass softmax, 6/8 inputs), read
                          from its artifact, scored and paired here. Not re-run, not re-fitted.
  C2 ridge_old8_additive  the SAME eight inputs as C1, in THIS family's additive/ridge/
                          projection pipeline. Matched control that absorbs the change of
                          functional form, so A1 - C2 is attributable to the new inputs.
  C3 ridge_recal_only     ridge on the protected vector itself plus machine type only (six
                          columns, no diagnostics, no history). A pure RECALIBRATION control:
                          whatever a monotone-ish reshaping of the baseline's own numbers can
                          buy. If A1 ~ C3 the "gain" is recalibration, not information.
  A1 ridge_enriched       ridge on the enriched diagnostic feature set (core or all, chosen on
                          inner folds).
  A2 boost_enriched       depth-2/3 CatBoost MultiRMSE on the SAME enriched features, same
                          targets, same weights, same transform. A2 - A1 is attributable to
                          nonlinear interaction, nothing else.

ENRICHED FEATURES (28 columns, every one computable at test time from pre-cutoff data, the
query row and frozen package constants):
  current forecast          logit(1-p0), p1/(1-p0), (p5+p20)/(1-p0), entropy of the 5-vector
  query                     machine type (2 dummies, steady is the reference), signal coded as
                            two indicators so that an ABSENT tick cutoff-1 is its own state
  support / uncertainty     log1p(complete-lag transitions), log1p(decayed observation mass in
                            the QUERIED cell of the selected template), Beta posterior sd of
                            that cell, Laplace logit variance of the win head
  expert disagreement       half-life mixture entropy, sd of the per-half-life win
                            probabilities, best-minus-second template log evidence margin,
                            entropy of the 57-template posterior, (template cell mean - q),
                            sd across the 15 audited RAW pool members' q, mean per-component sd
                            across the pool, (q - pool mean q)
  prequential surprise      Brier and signed bias of a decay-weighted conjugate one-step
                            predictor evaluated on the 8 most recent OBSERVED ticks and on the
                            next 32 older observed ticks, plus the recent-minus-older
                            difference. Every prediction is formed from ticks strictly before
                            the tick it is scored on, so this is prequential, not in-sample.
  missingness / length      count of absent ticks among cutoff-1..cutoff-8, log history
                            length, signed current run over ACTUAL ticks capped at 8
'core' is a predeclared 13-column subset (see CORE); the ridge arms choose core vs all on the
inner folds. The boosted arm always uses all columns.

CAUSALITY. Every history feature reads obs.get(cutoff - k) by ACTUAL tick arithmetic; a
missing tick is a state of its own and never a zero outcome. Older sampled ticks are never
treated as adjacent: decay weights use true tick differences and the recent/older prequential
blocks are defined by OBSERVED-tick rank, not row offset. No other machine's outcome at the
target tick is read, and the hidden condition label is never used.

NESTING. The four frozen complete-episode outer folds. Within each outer fold everything
target-dependent -- ridge strength, feature subset, CatBoost depth, boosted shrinkage, the
feature centring/scaling statistics, the regression coefficients, and the decision to apply
any correction at all -- is chosen on three complete-episode inner folds carved from that
fold's fitting episodes, and is REFITTED inside each inner training partition. No tail
constants are fitted from truth anywhere: the tail is never reshaped by hand, the additive
residual carries it. Held-fold forecasts are concatenated and scored ONCE.

DETERMINISM. Every reduction iterates a sorted sequence; CatBoost runs single-threaded with
bootstrap_type='No' and a fixed seed. The script is verified twice, once with
PYTHONHASHSEED=777, for byte-identical prediction CSVs.

Conditional exploratory development: these 96 episodes have been inspected repeatedly across
the campaign, so nested fitting here creates no independent holdout.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
np.seterr(divide='ignore', over='ignore', invalid='ignore')
from scipy.special import betaln

from slot_forecasting.io import load_inputs, read_submission, write_submission
from slot_model import harness as H
from slot_model import pair_dict as PD
from slot_model import win_head as W
from slot_model import win_head_infer as WI
from scripts.win_head_train import build

TEMPLATES = [(None, 'null')] + [(p, t) for p in PD.PAIRS for t in ('full', 'parity')]
HORIZON = 8
PREQ_HALF_LIFE = 48.0          # predeclared, matches the shipped mid half-life
PREQ_RECENT = 8                # most recent observed ticks
PREQ_OLDER = 32                # the next 32 observed ticks below those
LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEPTHS = (2, 3)
SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)
CB_ITERATIONS = 300
CB_LEARNING_RATE = 0.05
CB_L2 = 6.0

NAMES = [
    'type_balanced', 'type_jackpot', 'signal_pos', 'signal_neg',
    'logit_q', 'r1', 'rtail', 'entropy',
    'log1p_transitions', 'log1p_cell_support', 'cell_post_sd', 'log1p_logit_var',
    'mix_entropy', 'q_halflife_sd', 'template_margin', 'template_entropy', 'tmpl2_q_gap',
    'pool_q_sd', 'pool_vec_sd', 'pool_q_gap',
    'preq_err_recent', 'preq_err_older', 'preq_err_diff', 'preq_bias_recent',
    'preq_bias_older', 'other_recent', 'log_history', 'run_signed', 'other_change',
]
BLOCKS = {
    'forecast': ('logit_q', 'r1', 'rtail', 'entropy'),
    'query': ('type_balanced', 'type_jackpot', 'signal_pos', 'signal_neg'),
    'support': ('log1p_transitions', 'log1p_cell_support', 'cell_post_sd', 'log1p_logit_var'),
    'disagree': ('mix_entropy', 'q_halflife_sd', 'template_margin', 'template_entropy',
                 'tmpl2_q_gap'),
    'pool': ('pool_q_sd', 'pool_vec_sd', 'pool_q_gap'),
    'preq': ('preq_err_recent', 'preq_err_older', 'preq_err_diff', 'preq_bias_recent',
             'preq_bias_older'),
    'neighbour': ('other_recent', 'other_change'),
    'history': ('log_history', 'run_signed'),
}
CORE = ('type_balanced', 'type_jackpot', 'signal_pos', 'signal_neg', 'logit_q', 'r1',
        'log1p_cell_support', 'cell_post_sd', 'template_margin', 'tmpl2_q_gap',
        'pool_q_sd', 'preq_bias_recent', 'preq_err_diff', 'other_change')
OLD8 = ['old_intercept_dummy', 'type_balanced', 'type_jackpot', 'old_signal', 'old_qfrac',
        'old_omean', 'run_signed', 'old_miss']
RECAL = ('type_balanced', 'type_jackpot', 'logit_q', 'r1', 'rtail', 'entropy')


# --------------------------------------------------------------------------- features
def prequential(obs, cutoff, a0, b0):
    """Decay-weighted conjugate one-step surprise on observed ticks below the cutoff.

    For each scored tick t the predictor uses ONLY ticks strictly below t, with weight
    2**(-(t-1-s)/half_life) by actual tick difference. Returns the mean Brier and mean
    signed error over the most recent PREQ_RECENT observed ticks and over the next
    PREQ_OLDER observed ticks below them.
    """
    ticks = np.array(sorted(t for t in obs if t < cutoff), dtype=np.float64)
    if len(ticks) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    y = np.array([1.0 if obs[int(t)] > 0 else 0.0 for t in ticks])
    n = len(ticks)
    lo = max(0, n - (PREQ_RECENT + PREQ_OLDER))
    errs, bias = {}, {}
    for i in range(max(1, lo), n):
        wt = 2.0 ** (-(ticks[i] - 1.0 - ticks[:i]) / PREQ_HALF_LIFE)
        p = (a0 + float(np.sum(wt * y[:i]))) / (a0 + b0 + float(np.sum(wt)))
        errs[i] = (y[i] - p) ** 2
        bias[i] = y[i] - p
    rec = sorted(errs)[-PREQ_RECENT:]
    old = sorted(errs)[:-PREQ_RECENT] if len(errs) > PREQ_RECENT else []
    er = float(np.mean([errs[i] for i in rec])) if rec else 0.0
    eo = float(np.mean([errs[i] for i in old])) if old else er
    br = float(np.mean([bias[i] for i in rec])) if rec else 0.0
    bo = float(np.mean([bias[i] for i in old])) if old else br
    return er, eo, er - eo, br, bo


def build_features(b, package, verbose=True):
    """Per-query causal diagnostic features. No target probability is read anywhere here."""
    model = package / 'code/model'
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta_by_h, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    vq, _ = load_inputs(b.data / 'validation')
    by_id = {q.query_id: q for q in vq}
    entries = build(vq, b.index)
    by_episode = {}
    for (episode, machine), hist in sorted(b.index.items()):
        by_episode.setdefault(episode, {})[machine] = hist

    pool_names, pool_P, pool_Q, _, pool_info = H.raw_pool(b)
    pool_q_sd = pool_Q.std(axis=0)
    pool_vec_sd = pool_P.std(axis=0).mean(axis=1)
    pool_q_mean = pool_Q.mean(axis=0)

    prot = b.protected
    q_prot = 1.0 - prot[:, 0]
    logit_q = np.log(np.clip(q_prot, 1e-9, 1 - 1e-9) / np.clip(1 - q_prot, 1e-9, 1 - 1e-9))
    r1 = prot[:, 1] / np.maximum(q_prot, 1e-12)
    rtail = (prot[:, 3] + prot[:, 4]) / np.maximum(q_prot, 1e-12)
    entropy = -np.sum(prot * np.log(np.clip(prot, 1e-12, None)), axis=1)

    rows, old_rows, routes = [], [], []
    for i, qid in enumerate(b.ids):
        q = by_id[qid]
        e = entries[qid]
        mu = float(prior_mean.get(q.machine_type, 0.44))
        a0, b0 = kappa * mu, kappa * (1.0 - mu)
        obs = {t: v for t, v in b.index[(q.episode_id, q.machine_id)].items() if t < q.cutoff}

        # --- win-head diagnostics (memory-expert mixture, Laplace variance, support)
        if e['star'] is not None and len(e['y']):
            _, mix = W.predict_q_averaged(
                {h: beta_by_h[h].get(q.machine_type, np.zeros(W.DIM)) for h in sorted(
                    beta_by_h, key=lambda h: (h is None, h))},
                e['phi'], e['y'], e['used'], q.cutoff, e['star'], lam, adapt='full')
            n_trans = int(mix['transitions'])
            mix_entropy = float(mix['mix_entropy'])
            q_hl_sd = float(np.std(mix['per_half_life_q']))
            logit_var = float(mix['logit_var'])
            picked = mix['selected_half_life']
            half_life = None if picked == 0.0 else picked
            wts = (np.ones(len(e['y'])) if not half_life
                   else 2.0 ** (-(q.cutoff - 1 - e['used']) / half_life))
            signs = PD.signs_from(e['phi'])
            star_signs = e['star'][1:1 + W.HORIZON][None, :]
            scores = np.empty(len(TEMPLATES))
            cells = []
            for ix, (subset, template) in enumerate(TEMPLATES):
                ci, nc = WI._template_cells(signs, subset, template)
                si, _ = WI._template_cells(star_signs, subset, template)
                wn = np.bincount(ci, weights=wts * e['y'], minlength=nc)
                ls = np.bincount(ci, weights=wts * (1.0 - e['y']), minlength=nc)
                ev = float(np.sum(betaln(a0 + wn, b0 + ls) - betaln(a0, b0)))
                scores[ix] = ev + float(log_prior[ix])
                cells.append((int(si[0]), wn, ls))
            rank = np.argsort(-scores, kind='stable')
            best, second = int(rank[0]), int(rank[1])
            margin = float(np.clip(scores[best] - scores[second], 0.0, 20.0))
            sm = np.exp(scores - scores.max())
            sm /= sm.sum()
            tmpl_entropy = float(-np.sum(sm * np.log(np.maximum(sm, 1e-15))))
            c, wn, ls = cells[best]
            support = float(wn[c] + ls[c])
            post = float((a0 + wn[c]) / (a0 + b0 + wn[c] + ls[c]))
            post_sd = float(np.sqrt(max(post * (1 - post), 0.0)
                                    / (a0 + b0 + support + 1.0)))
            c2, wn2, ls2 = cells[second]
            post2 = float((a0 + wn2[c2]) / (a0 + b0 + wn2[c2] + ls2[c2]))
            subset, template = TEMPLATES[best]
            routes.append(template)
            if template == 'parity':
                tot = float(wn.sum() + ls.sum())
                own = support
                old_qfrac = own / tot if tot > 0 else 1.0
                oc = 1 - c
                old_omean = float((a0 + wn[oc]) / (a0 + b0 + wn[oc] + ls[oc]))
            else:
                old_qfrac, old_omean = 1.0, mu
        else:
            n_trans, mix_entropy, q_hl_sd, logit_var = 0, 0.0, 0.0, 0.0
            margin, tmpl_entropy = 0.0, float(np.log(len(TEMPLATES)))
            support, post = 0.0, mu
            post_sd = float(np.sqrt(mu * (1 - mu) / (a0 + b0 + 1.0)))
            post2 = mu
            routes.append('fallback')
            old_qfrac, old_omean = 1.0, mu

        # --- signal, run, missingness, history (actual tick arithmetic)
        last = obs.get(q.cutoff - 1)
        sig = 0.0 if last is None else (1.0 if last > 0 else -1.0)
        run = 0
        for k in range(1, HORIZON + 1):
            v = obs.get(q.cutoff - k)
            if v is None:
                break
            s = 1.0 if v > 0 else -1.0
            if k == 1:
                run = s
            elif s == np.sign(run):
                run += s
            else:
                break
        miss8 = sum(1 for k in range(1, HORIZON + 1) if obs.get(q.cutoff - k) is None)
        run = float(np.clip(run, -HORIZON, HORIZON))

        er, eo, ed, br, bo = prequential(obs, q.cutoff, a0, b0)

        # OTHER machines of the same episode, ticks strictly below the shared cutoff only
        rec_p = rec_n = old_p = old_n = 0
        for m in sorted(by_episode[q.episode_id]):
            if m == q.machine_id:
                continue
            seq = by_episode[q.episode_id][m]
            for t in sorted(seq):
                if t >= q.cutoff:
                    continue
                if t >= q.cutoff - 16:
                    rec_n += 1
                    rec_p += 1 if seq[t] > 0 else 0
                elif t >= q.cutoff - 80:
                    old_n += 1
                    old_p += 1 if seq[t] > 0 else 0
        other_recent = (rec_p + 2.0) / (rec_n + 4.0)
        other_older = (old_p + 2.0) / (old_n + 4.0)

        rows.append([
            float(q.machine_type == 'balanced'), float(q.machine_type == 'jackpot'),
            float(sig > 0), float(sig < 0),
            logit_q[i], r1[i], rtail[i], entropy[i],
            np.log1p(n_trans), np.log1p(support), post_sd, np.log1p(logit_var),
            mix_entropy, q_hl_sd, margin, tmpl_entropy, post2 - q_prot[i],
            pool_q_sd[i], pool_vec_sd[i], q_prot[i] - pool_q_mean[i],
            er, eo, ed, br, bo, other_recent, np.log1p(len(obs)), run,
            other_recent - other_older,
        ])
        # the published control's own eight inputs, reconstructed from the same primitives
        old_rows.append([1.0, float(q.machine_type == 'balanced'),
                         float(q.machine_type == 'jackpot'), sig, old_qfrac, old_omean,
                         run, 1.0 if miss8 > 0 else 0.0])

    X = np.asarray(rows, dtype=np.float64)
    XO = np.asarray(old_rows, dtype=np.float64)
    assert X.shape[1] == len(NAMES), (X.shape, len(NAMES))
    assert np.isfinite(X).all() and np.isfinite(XO).all(), 'non-finite feature'
    if verbose:
        print(f'  features {X.shape}, pool members {len(pool_names)} '
              f'(q heads {pool_info["distinct_q_heads"]}, r heads {pool_info["distinct_r_heads"]})',
              flush=True)
        print('  template routes: ' + ', '.join(
            f'{t}={routes.count(t)}' for t in sorted(set(routes))), flush=True)
    return X, XO, routes


# --------------------------------------------------------------------------- fitting
def project_simplex(V):
    """Euclidean projection of each row onto the probability simplex (Duchi et al.)."""
    V = np.asarray(V, dtype=np.float64)
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, V.shape[1] + 1)
    cond = U - css / ind > 0
    rho = V.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(V.shape[0]), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


def ridge_prepare(Xf, Yf, wf):
    """Weighted centring/scaling plus the Gram pieces, from FITTING rows only.

    numpy's vectorised matmul raises spurious divide/overflow FP flags on this platform
    even for finite well-scaled inputs, so the flags are silenced and the outputs are
    asserted finite instead.
    """
    assert np.isfinite(Xf).all() and np.isfinite(Yf).all() and np.isfinite(wf).all()
    s = wf.sum()
    tw = wf / s
    xbar = tw @ Xf
    var = tw @ (Xf - xbar) ** 2
    sd = np.sqrt(np.maximum(var, 0.0))
    sd[sd < 1e-12] = 1.0
    Z = (Xf - xbar) / sd
    ybar = tw @ Yf
    Yc = Yf - ybar
    G = Z.T @ (tw[:, None] * Z)
    Hm = Z.T @ (tw[:, None] * Yc)
    assert np.isfinite(G).all() and np.isfinite(Hm).all() and np.isfinite(ybar).all()
    return dict(xbar=xbar, sd=sd, ybar=ybar, G=G, Hm=Hm)


def ridge_delta(prep, Xall, lam):
    p = prep['G'].shape[0]
    C = np.linalg.solve(prep['G'] + lam * np.eye(p), prep['Hm'])
    Z = (Xall - prep['xbar']) / prep['sd']
    D = prep['ybar'] + Z @ C
    assert np.isfinite(D).all()
    return D, C


def boost_delta(Xf, Yf, wf, Xall, depth, want_importance=False):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function='MultiRMSE', depth=depth, iterations=CB_ITERATIONS,
                          learning_rate=CB_LEARNING_RATE, l2_leaf_reg=CB_L2,
                          bootstrap_type='No', rsm=1.0, random_seed=0, thread_count=1,
                          boosting_type='Plain', allow_writing_files=False, verbose=0)
    m.fit(Xf, Yf, sample_weight=wf)
    D = np.asarray(m.predict(Xall), dtype=np.float64)
    D = D - D.mean(axis=1, keepdims=True)
    assert np.isfinite(D).all()
    if want_importance:
        return D, [float(v) for v in m.get_feature_importance()]
    return D


def apply_delta(prot, delta, scale=1.0):
    return project_simplex(prot + scale * delta)


def boost_scalar(Xf, yf, wf, Xall, depth):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function='RMSE', depth=depth, iterations=CB_ITERATIONS,
                          learning_rate=CB_LEARNING_RATE, l2_leaf_reg=CB_L2,
                          bootstrap_type='No', rsm=1.0, random_seed=0, thread_count=1,
                          boosting_type='Plain', allow_writing_files=False, verbose=0)
    m.fit(Xf, yf, sample_weight=wf)
    d = np.asarray(m.predict(Xall), dtype=np.float64)
    assert np.isfinite(d).all()
    return d, [float(v) for v in m.get_feature_importance()]


# --------------------------------------------------------------------------- arms
def run_ridge_arm(b, X, cols_options, label, verbose=True):
    """Nested ridge arm. cols_options: {subset_name: column index array}."""
    held = b.protected.copy()
    prov = []
    for fold in range(4):
        outer = b.outer_mask(fold)
        fit = ~outer
        parts = [(tr, he, b.row_weights(he)) for tr, he in b.inner_folds(fit, k=3)]
        baseline = sum(b.wloss(b.protected, wh) for _, _, wh in parts)
        scores = {'zero': baseline}
        for sub in sorted(cols_options):
            cols = cols_options[sub]
            preps = []
            for tr, he, wh in parts:
                wtr = b.row_weights(tr)
                prep = ridge_prepare(X[tr][:, cols], (b.truth - b.protected)[tr], wtr[tr])
                preps.append((prep, he, wh))
            tot = 0.0
            for prep, he, wh in preps:
                d = np.repeat(prep['ybar'][None, :], len(b.ids), axis=0)
                tot += b.wloss(apply_delta(b.protected, d), wh)
            scores[f'{sub}|intercept'] = tot
            for lam in LAMBDAS:
                tot = 0.0
                for prep, he, wh in preps:
                    d, _ = ridge_delta(prep, X[:, cols], lam)
                    tot += b.wloss(apply_delta(b.protected, d), wh)
                scores[f'{sub}|{lam:g}'] = tot
        key = min(sorted(scores), key=lambda k: (scores[k], k))
        adopted = bool(key != 'zero' and scores[key] < baseline - 1e-15)
        if adopted:
            sub, ls = key.split('|')
            cols = cols_options[sub]
            wf = b.row_weights(fit)
            prep = ridge_prepare(X[fit][:, cols], (b.truth - b.protected)[fit], wf[fit])
            if ls == 'intercept':
                d = np.repeat(prep['ybar'][None, :], len(b.ids), axis=0)
                coef = None
            else:
                d, C = ridge_delta(prep, X[:, cols], float(ls))
                coef = C.tolist()
            held[outer] = apply_delta(b.protected, d)[outer]
        else:
            coef = None
        prov.append({'fold': fold, 'selected': key, 'adopted': adopted,
                     'inner_zero': baseline, 'inner_selected': scores[key],
                     'inner_scores': {k: scores[k] for k in sorted(scores)},
                     'coefficients': coef})
        if verbose:
            print(f'  [{label}] fold {fold}: pick {key:<18} adopted={adopted} '
                  f'inner {scores[key]:.6f} vs zero {baseline:.6f}', flush=True)
    return held, prov


def run_boost_arm(b, X, label, cols=None, names=None, cols_options=None, verbose=True):
    """Nested boosted arm. depth, shrinkage and (optionally) the feature subset are all
    chosen on the inner complete-episode folds of each outer fold's fitting episodes."""
    if cols_options is None:
        cols_options = {'fixed': (np.arange(X.shape[1]) if cols is None else cols,
                                  names or [f'x{i}' for i in range(X.shape[1])])}
    held = b.protected.copy()
    prov = []
    Y = b.truth - b.protected
    for fold in range(4):
        outer = b.outer_mask(fold)
        fit = ~outer
        parts = [(tr, he, b.row_weights(he)) for tr, he in b.inner_folds(fit, k=3)]
        baseline = sum(b.wloss(b.protected, wh) for _, _, wh in parts)
        scores = {'zero': baseline}
        cache = {}
        for sub in sorted(cols_options):
            XS = X[:, cols_options[sub][0]]
            for depth in DEPTHS:
                for ip, (tr, he, wh) in enumerate(parts):
                    wtr = b.row_weights(tr)
                    cache[(sub, depth, ip)] = boost_delta(XS[tr], Y[tr], wtr[tr], XS, depth)
                for s in SHRINKS:
                    if s == 0.0:
                        continue
                    tot = 0.0
                    for ip, (tr, he, wh) in enumerate(parts):
                        tot += b.wloss(apply_delta(b.protected, cache[(sub, depth, ip)], s), wh)
                    scores[f'{sub}|d{depth}|s{s:g}'] = tot
        key = min(sorted(scores), key=lambda k: (scores[k], k))
        adopted = bool(key != 'zero' and scores[key] < baseline - 1e-15)
        if adopted:
            sub, ds, ss = key.split('|')
            depth, s = int(ds[1:]), float(ss[1:])
            cols_sel, names_sel = cols_options[sub]
            XS = X[:, cols_sel]
            wf = b.row_weights(fit)
            D, raw = boost_delta(XS[fit], Y[fit], wf[fit], XS, depth, want_importance=True)
            imp = dict(zip(names_sel, raw))
            held[outer] = apply_delta(b.protected, D, s)[outer]
        else:
            imp = None
        prov.append({'fold': fold, 'selected': key, 'adopted': adopted,
                     'inner_zero': baseline, 'inner_selected': scores[key],
                     'inner_scores': {k: scores[k] for k in sorted(scores)},
                     'importance': imp})
        if verbose:
            print(f'  [{label}] fold {fold}: pick {key:<18} adopted={adopted} '
                  f'inner {scores[key]:.6f} vs zero {baseline:.6f}', flush=True)
    return held, prov


# --------------------------------------------------------------------------- probe
def probe(b, X, XO):
    """Unnested in-sample diagnostics. NEVER a reported result; sizing information only."""
    w = b.row_weights(np.ones(len(b.ids), dtype=bool))
    Y = b.truth - b.protected
    print(f'protected R {b.official(b.protected)["mean_loss_ratio"]:.9f} '
          f'wloss {b.wloss(b.protected, w):.9f}')
    for nm, cols in (('all', np.arange(len(NAMES))),
                     ('core', np.array([NAMES.index(c) for c in CORE]))):
        prep = ridge_prepare(X[:, cols], Y, w)
        for lam in (0.0, 0.03, 0.3, 3.0, 30.0):
            d, _ = ridge_delta(prep, X[:, cols], max(lam, 1e-12))
            print(f'  in-sample ridge[{nm}] lam={lam:<6g} R={b.wloss(apply_delta(b.protected, d), w):.9f}')
    prep = ridge_prepare(XO, Y, w)
    for lam in (0.03, 0.3, 3.0):
        d, _ = ridge_delta(prep, XO, lam)
        print(f'  in-sample ridge[old8] lam={lam:<6g} R={b.wloss(apply_delta(b.protected, d), w):.9f}')
    for depth in DEPTHS:
        D = boost_delta(X, Y, w, X, depth)
        for s in (0.5, 1.0):
            print(f'  in-sample boost d{depth} s={s} '
                  f'R={b.wloss(apply_delta(b.protected, D, s), w):.9f}')
    # univariate weighted correlation of each feature with the weighted residual norm
    tw = w / w.sum()
    print('  feature | weighted corr with residual on p0 / p1 / p2')
    for j, nm in enumerate(NAMES):
        x = X[:, j]
        xc = x - tw @ x
        sx = np.sqrt(tw @ xc ** 2) or 1.0
        out = []
        for k in (0, 1, 2):
            y = Y[:, k]
            yc = y - tw @ y
            sy = np.sqrt(tw @ yc ** 2) or 1.0
            out.append(float((tw @ (xc * yc)) / (sx * sy)))
        print(f'    {nm:<20} ' + '  '.join(f'{v:+.3f}' for v in out))


def run_q_arm(b, X, label, kind, cols=None, names=None, verbose=True):
    """Correct the WIN PROBABILITY only, at the protected release's own frozen r.

    Motivation, disclosed: the five-output arms moved essentially all of their gain onto
    p0, so a one-output correction should carry the same signal with a fifth of the
    variance. At frozen r the official loss is exactly
        L_i(q) = w_i (A_i q^2 - 2 B_i q) + const,
    so a weighted squared-error regression onto the per-query exact optimum B_i/A_i with
    weights w_i A_i IS the official objective, not a surrogate. Scoring still uses the full
    five-vector loss. The target B/A is target-dependent and is therefore only ever formed
    on fitting / inner-training episodes.
    """
    X = X if cols is None else X[:, cols]
    names = names or [f'x{i}' for i in range(X.shape[1])]
    q0, r0 = b.split_heads(b.protected)
    A, Bc = b.q_coeffs(r0)
    resid = Bc / A - q0
    held = b.protected.copy()
    prov = []

    def compose_q(d, s):
        return b.compose(np.clip(q0 + s * d, 1e-6, 1.0 - 1e-6), r0)

    for fold in range(4):
        outer = b.outer_mask(fold)
        fit = ~outer
        parts = [(tr, he, b.row_weights(he)) for tr, he in b.inner_folds(fit, k=3)]
        baseline = sum(b.wloss(b.protected, wh) for _, _, wh in parts)
        scores = {'zero': baseline}
        cache = {}
        if kind == 'ridge':
            for ip, (tr, he, wh) in enumerate(parts):
                wtr = b.row_weights(tr)
                cache[ip] = ridge_prepare(X[tr], resid[tr, None], (wtr * A)[tr])
            tot = 0.0
            for ip, (tr, he, wh) in enumerate(parts):
                d = np.repeat(cache[ip]['ybar'], len(b.ids))
                tot += b.wloss(compose_q(d, 1.0), wh)
            scores['intercept'] = tot
            for lam in LAMBDAS:
                tot = 0.0
                for ip, (tr, he, wh) in enumerate(parts):
                    d, _ = ridge_delta(cache[ip], X, lam)
                    tot += b.wloss(compose_q(d[:, 0], 1.0), wh)
                scores[f'{lam:g}'] = tot
        else:
            for depth in DEPTHS:
                for ip, (tr, he, wh) in enumerate(parts):
                    wtr = b.row_weights(tr)
                    cache[(depth, ip)] = boost_scalar(X[tr], resid[tr], (wtr * A)[tr],
                                                      X, depth)[0]
                for sc in SHRINKS:
                    if sc == 0.0:
                        continue
                    tot = 0.0
                    for ip, (tr, he, wh) in enumerate(parts):
                        tot += b.wloss(compose_q(cache[(depth, ip)], sc), wh)
                    scores[f'd{depth}|s{sc:g}'] = tot
        key = min(sorted(scores), key=lambda k: (scores[k], k))
        adopted = bool(key != 'zero' and scores[key] < baseline - 1e-15)
        imp = None
        if adopted:
            wf = b.row_weights(fit)
            if kind == 'ridge':
                prep = ridge_prepare(X[fit], resid[fit, None], (wf * A)[fit])
                if key == 'intercept':
                    d, sc = np.repeat(prep['ybar'], len(b.ids)), 1.0
                else:
                    dd, _ = ridge_delta(prep, X, float(key))
                    d, sc = dd[:, 0], 1.0
            else:
                ds, ss = key.split('|')
                sc = float(ss[1:])
                d, raw = boost_scalar(X[fit], resid[fit], (wf * A)[fit], X, int(ds[1:]))
                imp = dict(zip(names, raw))
            held[outer] = compose_q(d, sc)[outer]
        prov.append({'fold': fold, 'selected': key, 'adopted': adopted,
                     'inner_zero': baseline, 'inner_selected': scores[key],
                     'inner_scores': {k: scores[k] for k in sorted(scores)},
                     'importance': imp})
        if verbose:
            print(f'  [{label}] fold {fold}: pick {key:<18} adopted={adopted} '
                  f'inner {scores[key]:.6f} vs zero {baseline:.6f}', flush=True)
    return held, prov


# ------------------------------------------------------------------- supplementary test
def supplementary(b, out):
    """Extra quantiles and the one-sided mass of the SAME stratified episode bootstrap.

    The resampling construction below is copied line for line from
    slot_model.official_score.paired_bootstrap, so at repeats=2000 and seed 314159 it must
    reproduce the official 95% interval exactly; that equality is asserted. The only
    additions are more repeats, a 90% interval and P(delta R < 0). This is reported as
    SUPPLEMENTARY colour on an interval that is already published at 95%, never as a
    replacement decision rule.
    """
    from slot_forecasting.schema import CONDITIONS
    targets, groups, reference = b.scoring
    rows_dir = out / 'predictions'
    arms = {f.stem[4:]: f for f in sorted(rows_dir.glob('oof_*.csv'))}
    base = b.mapping(b.protected)
    report = {}
    for name, f in arms.items():
        if name == 'zero_protected':
            continue
        d = read_submission(f)
        a = {k: list(v) for k, v in d.items()}
        rows = {}
        for condition in CONDITIONS:
            es = sorted({t.episode_id for t in targets if groups[t.query_id] == condition})
            values = []
            for episode in es:
                ts = [t for t in targets if t.episode_id == episode]
                values.append([np.mean([np.sum((np.asarray(p[t.query_id]) - t.probabilities) ** 2)
                                        for t in ts]) for p in (a, base, reference)])
            rows[condition] = np.asarray(values)
        for repeats, tag in ((2000, 'check'), (20000, 'fine')):
            rng = np.random.default_rng(314159)
            totals = np.zeros(repeats)
            for condition, values in rows.items():
                draws = rng.integers(0, len(values), (repeats, len(values)))
                means = values[draws].mean(1)
                totals += ((means[:, 0] - means[:, 1]) / np.maximum(means[:, 2], 1e-9)) / 4
            if tag == 'check':
                official = b.paired(np.array([d[i] for i in b.ids]), b.protected)
                mine = np.quantile(totals, [.025, .975]).tolist()
                assert np.allclose(mine, official['delta_loss_ratio_ci95'], atol=1e-12), \
                    (name, mine, official['delta_loss_ratio_ci95'])
            else:
                report[name] = {'ci95': np.quantile(totals, [.025, .975]).tolist(),
                                'ci90': np.quantile(totals, [.05, .95]).tolist(),
                                'p_delta_negative': float(np.mean(totals < 0)),
                                'median_delta': float(np.median(totals)), 'repeats': repeats}
        print(f"  {name:<28} P(dR<0)={report[name]['p_delta_negative']:.4f}  "
              f"90% [{report[name]['ci90'][0]:+.5f},{report[name]['ci90'][1]:+.5f}]  "
              f"20k 95% [{report[name]['ci95'][0]:+.5f},{report[name]['ci95'][1]:+.5f}]", flush=True)
    (out / 'supplementary_bootstrap.json').write_text(
        json.dumps({'note': supplementary.__doc__, 'arms_vs_protected': report},
                   indent=2, sort_keys=True) + '\n')


def leaktest(b, X, XO):
    """Empirical nesting check: corrupt the TRUTH of outer fold 0's episodes and confirm
    that fold 0's held forecasts do not move by a single bit. Anything fitted, selected or
    gated on fold-0 targets would change the forecast; nothing may."""
    import copy
    out = {}
    for name, fn in (('boost_enriched', lambda bb: run_boost_arm(bb, X, 'leak', names=NAMES,
                                                                 verbose=False)),
                     ('boost_q_only_enriched', lambda bb: run_q_arm(bb, X, 'leak', 'boost',
                                                                    names=NAMES, verbose=False)),
                     ('ridge_enriched', lambda bb: run_ridge_arm(
                         bb, X, {'all': np.arange(len(NAMES)),
                                 'core': np.array([NAMES.index(c) for c in CORE])},
                         'leak', verbose=False))):
        clean, _ = fn(b)
        b2 = copy.copy(b)
        b2.truth = b.truth.copy()
        b2._wcache = {}
        mask0 = b.outer_mask(0)
        rng = np.random.default_rng(99)
        bad = rng.random((int(mask0.sum()), 5)) ** 3
        b2.truth[mask0] = bad / bad.sum(axis=1, keepdims=True)
        dirty, _ = fn(b2)
        same_held = bool(np.array_equal(clean[mask0], dirty[mask0]))
        other_moved = bool(not np.array_equal(clean[~mask0], dirty[~mask0]))
        out[name] = {'fold0_held_identical_under_corrupted_fold0_truth': same_held,
                     'other_folds_did_move': other_moved,
                     'max_abs_fold0_difference': float(np.max(np.abs(clean[mask0] - dirty[mask0])))}
        print(f'  {name:<24} fold0 held identical={same_held}  '
              f'(other folds moved={other_moved})', flush=True)
        assert same_held, f'{name}: fold-0 held forecast depends on fold-0 truth'
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    ap.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_exact')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/nonlinear_diag_residual')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--supplementary', action='store_true')
    ap.add_argument('--leaktest', action='store_true')
    args = ap.parse_args()
    start = time.perf_counter()
    (args.out / 'predictions').mkdir(parents=True, exist_ok=True)

    b = H.load(args.data, args.package)
    print(f'loaded {len(b.ids)} queries / {len(b.episodes)} episodes', flush=True)
    if args.supplementary:
        supplementary(b, args.out)
        return
    t0 = time.perf_counter()
    X, XO, routes = build_features(b, args.package)
    feat_seconds = time.perf_counter() - t0
    print(f'features in {feat_seconds:.1f}s', flush=True)
    if args.probe:
        probe(b, X, XO)
        return
    if args.leaktest:
        res = leaktest(b, X, XO)
        (args.out / 'leaktest.json').write_text(json.dumps(res, indent=2, sort_keys=True) + '\n')
        print(f"wrote {args.out / 'leaktest.json'}")
        return

    cols_all = np.arange(len(NAMES))
    cols_core = np.array([NAMES.index(c) for c in CORE])
    cols_recal = np.array([NAMES.index(c) for c in RECAL])

    arms, timings = {}, {}
    t = time.perf_counter()
    arms['ridge_old8_additive'] = run_ridge_arm(b, XO, {'old8': np.arange(XO.shape[1])},
                                                'C2 old8')
    timings['ridge_old8_additive'] = time.perf_counter() - t
    t = time.perf_counter()
    arms['ridge_recal_only'] = run_ridge_arm(b, X, {'recal': cols_recal}, 'C3 recal')
    timings['ridge_recal_only'] = time.perf_counter() - t
    t = time.perf_counter()
    arms['ridge_enriched'] = run_ridge_arm(b, X, {'all': cols_all, 'core': cols_core},
                                           'A1 enriched')
    timings['ridge_enriched'] = time.perf_counter() - t
    t = time.perf_counter()
    arms['boost_enriched'] = run_boost_arm(b, X, 'A2 boost', names=NAMES)
    timings['boost_enriched'] = time.perf_counter() - t
    # matched nonlinear controls: same budget and grid, narrower information
    t = time.perf_counter()
    arms['boost_old8'] = run_boost_arm(b, XO, 'C4 boost-old8', names=OLD8)
    timings['boost_old8'] = time.perf_counter() - t
    t = time.perf_counter()
    arms['boost_recal_only'] = run_boost_arm(b, X, 'C5 boost-recal', cols=cols_recal,
                                             names=list(RECAL))
    timings['boost_recal_only'] = time.perf_counter() - t
    # block ablations, and the purest test of added information: diagnostics with no
    # access to the protected vector and no access to the candidate pool at all
    subsets = {'all': list(NAMES), 'core': list(CORE)}
    for tag, drop in (('nopreq', BLOCKS['preq']), ('nopool', BLOCKS['pool']),
                      ('noother', BLOCKS['neighbour'])):
        subsets[tag] = [n for n in NAMES if n not in drop]
    subsets['diag_only'] = [n for n in NAMES
                            if n not in BLOCKS['forecast'] and n not in BLOCKS['pool']]
    for tag in ('nopreq', 'nopool', 'noother', 'diag_only'):
        keep = subsets[tag]
        t = time.perf_counter()
        arms[f'boost_enriched_{tag}' if tag != 'diag_only' else 'boost_diag_only'] = (
            run_boost_arm(b, X, f'AB {tag}',
                          cols=np.array([NAMES.index(n) for n in keep]), names=keep))
        timings[f'boost_enriched_{tag}' if tag != 'diag_only' else 'boost_diag_only'] = (
            time.perf_counter() - t)
    # A3: the honest version of "choose the feature subset" -- the SAME five candidate
    # subsets, but chosen per outer fold on that fold's inner folds, never on the outer
    # table. Disclosure: the candidate list was fixed after the four fixed-subset arms
    # above had been scored, so the LIST is exposed even though the CHOICE is nested.
    # A4/A5: one-output q-side correction at frozen r, same features and same grids.
    t = time.perf_counter()
    arms['ridge_q_only_enriched'] = run_q_arm(b, X, 'A4 ridge-q', 'ridge', names=NAMES)
    timings['ridge_q_only_enriched'] = time.perf_counter() - t
    t = time.perf_counter()
    arms['boost_q_only_enriched'] = run_q_arm(b, X, 'A5 boost-q', 'boost', names=NAMES)
    timings['boost_q_only_enriched'] = time.perf_counter() - t
    t = time.perf_counter()
    arms['boost_q_only_old8'] = run_q_arm(b, XO, 'C6 boost-q-old8', 'boost', names=OLD8)
    timings['boost_q_only_old8'] = time.perf_counter() - t
    t = time.perf_counter()
    arms['boost_subset_nested'] = run_boost_arm(
        b, X, 'A3 nested-subset',
        cols_options={k: (np.array([NAMES.index(n) for n in v]), v)
                      for k, v in sorted(subsets.items()) if k != 'diag_only'})
    timings['boost_subset_nested'] = time.perf_counter() - t

    # the literal published linear control, read not re-fitted
    c1 = read_submission(ROOT / 'artifacts/residual_fallback/predictions/oof_residual_fallback.csv')
    c1 = np.array([c1[i] for i in b.ids])
    c1 = c1 / c1.sum(axis=1, keepdims=True)

    forecasts = {'zero_protected': b.protected, 'linear_old8_log_published': c1}
    for k, (held, _) in arms.items():
        forecasts[k] = held / held.sum(axis=1, keepdims=True)

    controls = {'zero_protected': None,
                'linear_old8_log_published': 'zero_protected',
                'ridge_old8_additive': 'zero_protected',
                'ridge_recal_only': 'zero_protected',
                'ridge_enriched': 'ridge_old8_additive',
                'boost_enriched': 'ridge_enriched',
                'boost_old8': 'zero_protected',
                'boost_recal_only': 'zero_protected',
                'boost_enriched_nopreq': 'boost_enriched',
                'boost_enriched_nopool': 'boost_enriched',
                'boost_enriched_noother': 'boost_enriched',
                'boost_diag_only': 'zero_protected',
                'boost_subset_nested': 'boost_enriched',
                'ridge_q_only_enriched': 'ridge_enriched',
                'boost_q_only_enriched': 'boost_enriched',
                'boost_q_only_old8': 'boost_old8'}
    # attribution comparisons, each one isolating a single difference
    pairs = [('ridge_enriched', 'ridge_old8_additive'),
             ('boost_enriched', 'ridge_enriched'),
             ('boost_enriched', 'boost_old8'),
             ('boost_enriched', 'boost_recal_only'),
             ('boost_enriched', 'boost_enriched_nopreq'),
             ('boost_enriched', 'boost_enriched_nopool'),
             ('boost_enriched', 'boost_enriched_noother'),
             ('boost_old8', 'linear_old8_log_published'),
             ('boost_enriched', 'linear_old8_log_published'),
             ('boost_diag_only', 'zero_protected'),
             ('boost_subset_nested', 'boost_enriched'),
             ('boost_subset_nested', 'boost_old8'),
             ('boost_enriched_nopreq', 'boost_subset_nested'),
             ('boost_q_only_enriched', 'boost_enriched'),
             ('boost_q_only_enriched', 'boost_q_only_old8'),
             ('boost_q_only_enriched', 'boost_recal_only'),
             ('ridge_q_only_enriched', 'ridge_enriched')]

    wall = b.row_weights(np.ones(len(b.ids), dtype=bool))
    results = {}
    for name in sorted(forecasts):
        P = forecasts[name]
        res = b.official(P)
        per_cat = [float(np.sum(wall * (P[:, k] - b.truth[:, k]) ** 2)) for k in range(5)]
        tot_cat = sum(per_cat)
        row = {'points': res['technical_points'], 'R': res['mean_loss_ratio'],
               'by_condition': {c: v['loss_ratio'] for c, v in res['by_condition'].items()},
               'max_abs_change_vs_protected': float(np.max(np.abs(P - b.protected))),
               'mean_abs_change_vs_protected': float(np.mean(np.abs(P - b.protected))),
               'control': controls[name],
               'weighted_sse_by_category': per_cat,
               'error_share_by_category': [v / tot_cat for v in per_cat],
               'seconds': timings.get(name)}
        if name != 'zero_protected':
            row['paired_vs_protected_ci95'] = b.paired(P, b.protected)['delta_loss_ratio_ci95']
            cn = controls[name]
            if cn and cn != 'zero_protected':
                row['paired_vs_control_ci95'] = b.paired(P, forecasts[cn])['delta_loss_ratio_ci95']
            else:
                row['paired_vs_control_ci95'] = row['paired_vs_protected_ci95']
        if name in arms:
            row['folds'] = arms[name][1]
            row['folds_adopted'] = sum(1 for p in arms[name][1] if p['adopted'])
        results[name] = row
        write_submission(args.out / 'predictions' / f'oof_{name}.csv', b.mapping(P))

    comparisons = {f'{a}__vs__{c}': b.paired(forecasts[a], forecasts[c])['delta_loss_ratio_ci95']
                   for a, c in pairs}
    importance = {}
    for name, (_, prov) in arms.items():
        acc = {}
        for pr in prov:
            for k, v in (pr.get('importance') or {}).items():
                acc[k] = acc.get(k, 0.0) + v / 4.0
        if acc:
            importance[name] = dict(sorted(acc.items(), key=lambda kv: -kv[1]))

    rep = {'protocol': __doc__, 'results': results,
           'attribution_comparisons_ci95': comparisons,
           'feature_blocks': {k: list(v) for k, v in BLOCKS.items()},
           'candidate_subsets': subsets,
           'mean_feature_importance': importance,
           'feature_names': NAMES, 'core_subset': list(CORE), 'old8_inputs': OLD8,
           'recal_subset': list(RECAL),
           'grid': {'ridge_lambda': list(LAMBDAS), 'ridge_extra': ['zero', 'intercept'],
                    'boost_depth': list(DEPTHS), 'boost_shrink': list(SHRINKS),
                    'catboost': {'iterations': CB_ITERATIONS, 'learning_rate': CB_LEARNING_RATE,
                                 'l2_leaf_reg': CB_L2, 'loss_function': 'MultiRMSE',
                                 'bootstrap_type': 'No', 'thread_count': 1, 'random_seed': 0}},
           'template_route_counts': {t: routes.count(t) for t in sorted(set(routes))},
           'transformation': 'p = Euclidean projection of (protected + s*delta) onto the simplex',
           'leakage': 'Fitted on fitting-side episodes only: ridge coefficients and intercept, '
                      'feature centring/scaling, ridge lambda, feature subset, CatBoost depth, '
                      'boosted shrinkage, and the per-fold adoption gate. No tail constants '
                      'were fitted from truth. All features are functions of the protected '
                      'release, the query row, pre-cutoff history, frozen package constants '
                      'and the audited RAW pool.',
           'feature_seconds': feat_seconds,
           'seconds': time.perf_counter() - start}
    (args.out / 'evaluation.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')

    conds = sorted(results['zero_protected']['by_condition'])
    print(f"\n{'arm':<28}{'points':>11}{'R':>11}  " + '  '.join(f'{c[:9]:>9}' for c in conds)
          + '   dR vs protected        dR vs control')
    for name in ('zero_protected', 'linear_old8_log_published', 'ridge_old8_additive',
                 'ridge_recal_only', 'ridge_enriched', 'boost_recal_only', 'boost_old8',
                 'boost_diag_only', 'boost_enriched_nopreq', 'boost_enriched_nopool',
                 'boost_enriched_noother', 'boost_enriched', 'boost_subset_nested',
                 'ridge_q_only_enriched', 'boost_q_only_old8', 'boost_q_only_enriched'):
        v = results[name]
        line = (f"{name:<28}{v['points']:>11.6f}{v['R']:>11.6f}  "
                + '  '.join(f"{v['by_condition'][c]:>9.4f}" for c in conds))
        if 'paired_vs_protected_ci95' in v:
            a = v['paired_vs_protected_ci95']
            c = v['paired_vs_control_ci95']
            line += (f"   [{a[0]:+.5f},{a[1]:+.5f}]   [{c[0]:+.5f},{c[1]:+.5f}]"
                     + ('  SIG' if a[1] < 0 else ''))
        print(line)
    print('\nattribution (negative favours the FIRST arm)')
    for k, v in comparisons.items():
        print(f'  {k:<52} [{v[0]:+.5f}, {v[1]:+.5f}]'
              + ('  SIG' if v[1] < 0 else ('  SIG(worse)' if v[0] > 0 else '')))
    top = importance.get('boost_enriched', {})
    print('\ntop enriched importances: ' + ', '.join(
        f'{k}={v:.1f}' for k, v in list(top.items())[:10]))
    print(f"\nwrote {args.out / 'evaluation.json'} in {rep['seconds']:.1f}s")


if __name__ == '__main__':
    main()
