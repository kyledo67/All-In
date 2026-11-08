"""The one offline inference path for the win-head candidate packages.

This module is used unchanged by the evaluation replay (``--split validation``) and by
the final prediction (``--split test``). It imports no training script, reads no data
path at import time, runs no fitting of blend weights or hyperparameters, and performs
no network access. Everything it needs is read from a frozen model directory.

What it computes
----------------
The five-vector is  ``[1 - q, q*r_1, q*r_2, q*r_5, q*r_20]``:

* ``q`` is the hierarchical logistic autoregressive win probability from
  ``slot_model.win_head``: pooled per-machine-type effects on the predeclared 37-dim
  sign/pair basis, plus regularized machine-specific deviations fitted from that
  machine's own pre-cutoff outcomes, averaged over recency half-lives by each
  machine's own held-out recent block, with the Laplace predictive logit scaling.
  MAP (penalized maximum likelihood) plus a one-dimensional Laplace integral; this is
  not posterior inference.
* ``r`` are the positive-payout proportions of a reference predictor, taken from
  ``deployment.json``:

  - ``three_component``: a fixed convex combination of the raw two-seed TabPFN(hmm)
    average, PairMTD(lag-7) over the saved changepoint MTD, and the two-head CatBoost.
  - ``two_head``: the two-head CatBoost alone (no torch, no TabPFN, no checkpoint).

``alpha`` in ``deployment.json`` is the frozen blend between the reference predictor's
own win mass and ``q``; the measured candidate used ``alpha = 1`` (q only). It is kept
as an explicit frozen constant rather than being re-derived here.

Determinism
-----------
No RNG is drawn anywhere in this path. CatBoost inference, the MTD EM recursion, the
HMM forward pass and the Newton/IRLS solves are all deterministic given their inputs.
TabPFN's forward pass is seeded from the saved per-component seed and runs on CPU with
a fixed thread count.
"""
import json
import time
from pathlib import Path

import numpy as np
from slot_model import pair_dict as PD

from . import win_head as W
from .features import causal_index
from .two_head_infer import SEEDS as TWO_HEAD_SEEDS, feature_table, predict_proba as two_head_predict
from slot_forecasting.io import load_inputs, write_submission

PAYOUT_ORDER = (0, 1, 2, 5, 20)
PROPORTION_FLOOR = 1e-12


# --------------------------------------------------------------------- win head
def load_win_head(path):
    """Read the frozen half-life-averaged win head and assert its declared shape."""
    payload = json.loads(Path(path).read_text())
    selected = payload['selected']
    if list(payload['basis']) != list(W.basis_names()):
        raise ValueError('Saved win-head basis does not match slot_model.win_head')
    if int(payload['dim']) != W.DIM:
        raise ValueError(f"Saved win-head dim {payload['dim']} != {W.DIM}")
    lam = W.penalty(selected['lam_intercept'], selected['lam_main'], selected['lam_inter'])
    beta = {}
    for key, table in payload['beta_by_half_life'].items():
        half_life = float(key)
        coefficients = {t: np.asarray(v, dtype=np.float64) for t, v in table.items()}
        for value in coefficients.values():
            if value.shape != (W.DIM,):
                raise ValueError('Saved win-head coefficient block has the wrong width')
        beta[None if half_life == 0.0 else half_life] = coefficients
    if sorted(h if h else 0.0 for h in beta) != sorted(payload['half_lives']):
        raise ValueError('Saved half-life set does not match beta_by_half_life')
    return beta, lam, selected


def win_probabilities(queries, index, beta_by_half_life, lam, fallback):
    """Per-query win probability in the order of `queries`.

    `fallback[i]` is used only when the prediction tick's own basis row cannot be built
    (a lag within 1..8 is unobserved) or the machine offers no complete transition. A
    missing lag is never imputed as an observed loss; the row is simply unavailable, so
    the reference predictor's win mass is used for that query instead.
    """
    values = np.empty(len(queries), dtype=np.float64)
    details = []
    for i, query in enumerate(queries):
        ticks = index[query.episode_id, query.machine_id]
        phi, labels, _, used = W.design(ticks, query.cutoff, half_life=None)
        star = W.query_row(ticks, query.cutoff)
        if star is None or len(labels) == 0:
            values[i] = fallback[i]
            details.append({'transitions': int(len(labels)), 'adapted': False, 'fallback': True})
            continue
        by_type = {h: table.get(query.machine_type, np.zeros(W.DIM))
                   for h, table in beta_by_half_life.items()}
        value, detail = W.predict_q_averaged(by_type, phi, labels, used, query.cutoff,
                                             star, lam, adapt='full')
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            value = fallback[i]
            detail['fallback'] = True
        details.append(detail)
        values[i] = value
    return values, details


def load_pair_dict(path):
    """Read the frozen compact pair-dictionary settings and assert their shape."""
    payload = json.loads(Path(path).read_text())
    names = list(payload['candidates'])
    if names != list(PD.candidate_names()):
        raise ValueError('Saved pair-dictionary candidate set does not match slot_model.pair_dict')
    log_prior = np.asarray(payload['log_prior'], dtype=np.float64)
    if log_prior.shape != (len(PD.CANDIDATES),):
        raise ValueError('Saved pair-identity log prior has the wrong length')
    return float(payload['lam_scale']), float(payload['lam_intercept_scale']), log_prior, payload


def pair_win_probabilities(queries, index, beta_by_half_life, lam, fallback,
                           lam_scale, lam_intercept_scale, log_prior, select='hard'):
    """Compact pair-model win probability per query, in the order of `queries`.

    The per-machine half-life comes from the SAME verified averaging procedure used by
    the dense model, so the only thing that differs here is the local model: four
    coefficients for one lag pair instead of 37 coefficients for all of them.

    `select='hard'` takes the single highest-evidence candidate, which is what scored
    best on both the train surrogate and validation. `select='average'` averages
    PROBABILITIES across all 29 candidates.
    """
    values = np.empty(len(queries), dtype=np.float64)
    details = []
    for i, query in enumerate(queries):
        ticks = index[query.episode_id, query.machine_id]
        phi, labels, _, used = W.design(ticks, query.cutoff, half_life=None)
        star = W.query_row(ticks, query.cutoff)
        if star is None or len(labels) == 0:
            values[i] = fallback[i]
            details.append({'transitions': int(len(labels)), 'fallback': True})
            continue
        by_type = {h: table.get(query.machine_type, np.zeros(W.DIM))
                   for h, table in beta_by_half_life.items()}
        _, mix = W.predict_q_averaged(by_type, phi, labels, used, query.cutoff, star,
                                      lam, adapt='full')
        picked = mix['selected_half_life']
        half_life = None if picked == 0.0 else picked
        weights = (np.ones(len(labels)) if not half_life else
                   2.0 ** (-(query.cutoff - 1 - np.asarray(used, dtype=np.float64)) / half_life))
        beta = by_type[half_life]
        population = (phi @ beta, float(beta @ star))
        omega, qs, _ = PD.machine_posterior(phi, labels, weights, star, population,
                                            lam_scale * lam_intercept_scale, lam_scale,
                                            log_prior=log_prior)
        index_best = int(np.argmax(omega))
        value = float(qs[index_best]) if select == 'hard' else PD.average(omega, qs)
        detail = {'transitions': int(len(labels)), 'half_life': picked,
                  'candidate': PD.candidate_names()[index_best],
                  'max_weight': float(omega.max()), 'fallback': False}
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            value = fallback[i]
            detail['fallback'] = True
        values[i] = value
        details.append(detail)
    return values, details


def proportions(vectors):
    """Positive-payout proportions of a reference five-vector block.

    Deliberately NOT renormalized here: `win_head.compose` normalizes the assembled
    five-vector, which is the composition the measured candidate used.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    mass = np.maximum(1.0 - vectors[:, 0], PROPORTION_FLOOR)
    return vectors[:, 1:] / mass[:, None]


# ------------------------------------------------------------- reference blocks
def two_head_component(rows, x, model):
    return two_head_predict(x, Path(model) / 'two_head')


def tabpfn_component(rows, x, model, threads=2):
    """Raw two-seed TabPFN average on the 182 observable + 7 HMM summary columns."""
    from .phase2_predict import hmm_summary
    from .tabpfn_model import TabPFNEnsemble
    population = json.loads((Path(model) / 'population.json').read_text())
    _, summary = hmm_summary(population, rows)
    values = np.column_stack((x, summary))
    ensemble = TabPFNEnsemble.load(Path(model) / 'tabpfn',
                                   checkpoints=Path(model) / 'tabpfn_checkpoints')
    if ensemble.variant != 'hmm':
        raise ValueError(f'Saved TabPFN variant is {ensemble.variant!r}, expected hmm')
    out = ensemble.predict_proba(values)
    return out, {'seeds': ensemble.seeds, 'variant': ensemble.variant,
                 'hmm_summary_columns': int(summary.shape[1])}


def mtd_component(rows, queries, history, model):
    """PairMTD(lag-7) over the saved changepoint MTD. Predict-only; nothing is refitted."""
    from .mtd import MTDModel
    from .mtd_pairs import PairMTDModel, LAG7_PAIRS
    base = MTDModel.load(Path(model) / 'mtd' / 'change.json')
    if not base.changepoint:
        raise ValueError('Saved MTD base is not the changepoint variant')
    predictions = PairMTDModel(pairs=LAG7_PAIRS, prior=40.0, base=base).predict(queries, history)
    return np.array([predictions[e['q'].query_id] for e in rows], dtype=np.float64)


def reference_vectors(variant, weights, rows, x, queries, history, model, threads=2):
    """The reference five-vectors whose positive-payout proportions become r."""
    notes = {}
    if variant == 'two_head':
        block = two_head_component(rows, x, model)
        notes['two_head'] = {'seeds': list(TWO_HEAD_SEEDS)}
        return block, notes
    if variant != 'three_component':
        raise ValueError(f'Unknown reference variant {variant!r}')
    # ORDER MATTERS, measured. TabPFN must run before CatBoost touches anything.
    # With CatBoost's native runtime initialized first, the TabPFN forward pass deviated
    # from the measured component CSVs by 3.975e-04 (max abs, 768 validation rows);
    # running TabPFN first on the same saved context and the same bitwise-identical
    # 189-column table reproduces them to 1.110e-16. CatBoost loads its own OpenMP
    # runtime, which changes how torch partitions its intra-op work and therefore the
    # reduction order inside the transformer. The components themselves are unchanged.
    components = {}
    components['tabpfn_raw'], notes['tabpfn'] = tabpfn_component(rows, x, model, threads=threads)
    components['two_head'] = two_head_component(rows, x, model)
    components['mtd_lag7'] = mtd_component(rows, queries, history, model)
    if sorted(weights) != sorted(components):
        raise ValueError(f'Deployment weights {sorted(weights)} do not match '
                         f'the available components {sorted(components)}')
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9 or any(w < 0 for w in weights.values()):
        raise ValueError('Deployment weights are not a convex combination')
    block = np.zeros_like(components['two_head'])
    for name in sorted(components):
        part = components[name]
        if not np.isfinite(part).all() or part.min() < 0 or np.max(abs(part.sum(1) - 1)) > 1e-9:
            raise ValueError(f'Component {name} is not a valid simplex block')
        block = block + weights[name] * part
    block = block / block.sum(1, keepdims=True)
    notes['weights'] = {name: float(weights[name]) for name in sorted(weights)}
    return block, notes


# ------------------------------------------------------------------- entry point
def predict(data, out, model, *, split='test', threads=2, diagnostics=None):
    """Write the candidate submission CSV for `split` and return a report dict."""
    start = time.perf_counter()
    model = Path(model)
    deployment = json.loads((model / 'deployment.json').read_text())
    variant = deployment['reference_variant']
    alpha = float(deployment['alpha'])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be in [0, 1]')

    queries, history = load_inputs(Path(data) / split)
    index = causal_index(queries, history)
    prior = json.loads((model / 'population_prior.json').read_text())
    schema = json.loads((model / 'feature_schema.json').read_text())['observable']
    rows, x, names = feature_table(queries, index, prior, schema=schema)
    ordered = [e['q'] for e in rows]
    ids = [q.query_id for q in ordered]
    if len(set(ids)) != len(ids) or set(ids) != {q.query_id for q in queries}:
        raise ValueError('Query alignment changed')
    feature_seconds = time.perf_counter() - start

    reference, notes = reference_vectors(variant, deployment.get('weights', {}), rows, x,
                                         queries, history, model, threads=threads)
    reference_seconds = time.perf_counter() - start - feature_seconds

    beta, lam, selected = load_win_head(model / 'win_head' / deployment['win_head'])
    r = proportions(reference)
    reference_q = 1.0 - reference[:, 0]
    pair_name = deployment.get('pair_dict')
    if pair_name:
        scale, intercept_scale, log_prior, pair_payload = load_pair_dict(
            model / 'win_head' / pair_name)
        q, details = pair_win_probabilities(ordered, index, beta, lam, reference_q,
                                            scale, intercept_scale, log_prior,
                                            select=deployment.get('pair_select', 'hard'))
        win_model = {'local_model': 'compact_pair_dictionary',
                     'select': deployment.get('pair_select', 'hard'),
                     'lam_scale': scale, 'candidates': len(log_prior)}
    else:
        q, details = win_probabilities(ordered, index, beta, lam, reference_q)
        win_model = {'local_model': 'dense_37_coefficient'}
    blended = (1.0 - alpha) * reference_q + alpha * q
    vectors = np.array([W.compose(blended[i], r[i]) for i in range(len(ids))])
    if not np.isfinite(vectors).all() or vectors.min() < 0 or vectors.max() > 1:
        raise ValueError('Composed probabilities left the unit interval')
    if np.max(abs(vectors.sum(1) - 1.0)) > 1e-12:
        raise ValueError('Composed probabilities do not sum to one')
    write_submission(out, dict(zip(ids, vectors.tolist())))

    fallbacks = [ids[i] for i, d in enumerate(details) if d.get('fallback')]
    report = {'queries': len(ids), 'split': split, 'reference_variant': variant,
              'alpha': alpha, 'payout_order': list(PAYOUT_ORDER),
              'win_head': {k: selected[k] for k in sorted(selected)},
              'win_model': win_model,
              'reference_notes': notes,
              'win_head_fallback_queries': fallbacks,
              'win_head_fallbacks': len(fallbacks),
              'max_abs_sum_error': float(np.max(abs(vectors.sum(1) - 1.0))),
              'min_probability': float(vectors.min()),
              'feature_seconds': feature_seconds,
              'reference_seconds': reference_seconds,
              'seconds': time.perf_counter() - start}
    if diagnostics is not None:
        diagnostics = Path(diagnostics)
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        write_submission(diagnostics.with_name(diagnostics.stem + '_reference.csv'),
                         dict(zip(ids, reference.tolist())))
        diagnostics.write_text(json.dumps(
            {**report, 'q': dict(zip(ids, q.tolist())),
             'blended_q': dict(zip(ids, blended.tolist()))}, indent=2, sort_keys=True) + '\n')
    return report
