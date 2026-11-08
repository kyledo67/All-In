"""Explicit candidate manifest with provenance classes, replacing directory discovery.

Phase 0 of the qualification. Every candidate is classified by WHERE its numbers come
from, not by its directory name:

  RAW          deterministic given released data plus TRAIN-only fitting. Safe in any
               outer fold, because no validation target entered its construction.
  FOLD_FITTED  fitted against validation TARGETS using the same 4-fold map. Its value on
               an episode in fold j came from a fit on folds != j, which for any OTHER
               outer held fold H includes H. So its FITTING-SIDE values leak H's targets
               into the stacking weights even though each row is individually out-of-fold.
               A per-row OOF label is therefore not sufficient.
  DEPLOY_FIT   fitted on ALL validation targets (a deployment replay). Leaks every fold.
  PRIVILEGED   reads evaluation targets as the forecast itself. Never a candidate.

The RAW set is the only pool that is properly nested for a new stacking layer without
regenerating inner out-of-fold forecasts.

Duplicate detection is by rounded array hash on the five-vector AND separately on each
head, so two columns that differ only in an unused head are not counted as independent
evidence.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Producing script -> provenance class. Classification was read off the sources:
# a script that calls row_weights() fits against validation truth.
RAW_SOURCES = {
    # primitive component forecasts: train-fitted models predicting validation
    'artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv': 'TabPFN hmm variant, seed 20261108, trained on public/train',
    'artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv': 'TabPFN hmm variant, seed 20261109, trained on public/train',
    'artifacts/tabpfn/predictions/tabpfn_observable_20261108.csv': 'TabPFN observable variant, train-fitted',
    'artifacts/tabpfn/predictions/tabpfn_observable_20261109.csv': 'TabPFN observable variant, train-fitted',
    'artifacts/two_head_01/validation.csv': 'two-head CatBoost, trained on public/train',
    'artifacts/dense_catboost_01/validation_dense.csv': 'dense CatBoost, trained on public/train',
    'artifacts/dense_catboost_01/validation_sparse.csv': 'sparse CatBoost, trained on public/train',
    'artifacts/mtd_pairs_01/validation_additive.csv': 'additive MTD, train-fitted, pair family frozen on train OOF',
    'artifacts/mtd_pairs_01/validation_lag7.csv': 'lag-7 pair MTD, train-fitted',
    'artifacts/mtd_pairs_01/validation_control.csv': 'control pair MTD, train-fitted',
    'artifacts/final/validation_predictions.csv': 'Dynamax shared HMM, train-fitted',
    # conjugate / parity family: hyperparameters selected on the TRAIN surrogate only
    'artifacts/beta_cells/predictions/oof_beta_hard.csv': 'four-cell Beta, kappa from train surrogate',
    'artifacts/beta_cells/predictions/oof_beta_average.csv': 'four-cell Beta averaged, train surrogate',
    'artifacts/beta_templates/predictions/oof_parity_templates.csv': 'null/full/parity dictionary, train surrogate',
    'artifacts/beta_subsets/predictions/oof_parity_templates.csv': 'subset-parity dictionary, train surrogate',
    'artifacts/beta_memory/predictions/oof_beta_memory.csv': 'Beta-specific causal memory selection, per-machine history only',
    'artifacts/hier_beta/predictions/oof_pooled_endpoint.csv': 'hierarchical Beta, pooled endpoint, train surrogate',
    'artifacts/hier_beta/predictions/oof_tau32.csv': 'hierarchical Beta tau=32, train surrogate',
    'artifacts/hier_beta/predictions/oof_tau8.csv': 'hierarchical Beta tau=8, train surrogate',
    'artifacts/hier_beta/predictions/oof_tau2.csv': 'hierarchical Beta tau=2, train surrogate',
    'artifacts/hier_beta/predictions/oof_independent_4cell.csv': 'independent four-cell control, train surrogate',
    'artifacts/endpoint_family/predictions/oof_tail_train.csv': 'tail rescale with TRAIN-estimated constants',
    'artifacts/endpoint_family/predictions/oof_tail_train_smoothed.csv': 'tail rescale, train constants, add-1 smoothed',
    'artifacts/projection/predictions/oof_tail_projection_train.csv': 'exact tail projection, TRAIN constants',
    'artifacts/endpoint_predictor/predictions/oof_beta_hard.csv': 'conjugate endpoint predictor arm, train EM',
    'artifacts/dirichlet_r/predictions/oof_r_dirichlet.csv': 'conjugate Dirichlet r, tau from train surrogate',
    'artifacts/win_head/predictions/oof_population_alpha1.csv': 'dense win head, population only',
    'artifacts/win_head/predictions/oof_local_main_alpha1.csv': 'dense win head, local main effects',
    'artifacts/win_head/predictions/oof_local_interactions_alpha1.csv': 'dense win head, local interactions',
    'artifacts/pair_dict/predictions/oof_compact_hard.csv': 'compact logistic pair dictionary, hard selection',
    'artifacts/pair_dict/predictions/oof_compact_average.csv': 'compact logistic pair dictionary, averaged',
}

FOLD_FITTED_PREFIXES = (
    'artifacts/memory_policies/', 'artifacts/context_correction/', 'artifacts/finalist/',
    'artifacts/finalist_parity/', 'artifacts/finalist_beta/', 'artifacts/parity_prior/',
    'artifacts/phase2/predictions/oof_', 'artifacts/stacking/', 'artifacts/separable/',
    'artifacts/memory_envelope/',
)
DEPLOY_FIT_FILES = (
    'artifacts/win_head_release_exact/validation_submission.csv',
    'artifacts/win_head_release_tail/validation_submission.csv',
    'artifacts/win_head_release_dirichlet/validation_submission.csv',
    'artifacts/win_head_release_parity/validation_submission.csv',
    'artifacts/win_head_release_nofx/validation_submission.csv',
    'artifacts/endpoint_family/predictions/oof_tail_crossfit.csv',
    'artifacts/projection/predictions/oof_tail_projection_crossfit.csv',
    'artifacts/projection/predictions/oof_endpoint_projection_crossfit.csv',
    'artifacts/dirichlet_r/predictions/oof_r_blend.csv',
)
PRIVILEGED_MARKERS = ('PRIVILEGED', 'privileged_envelope', 'true_r')


def classify(path):
    p = str(path)
    if any(m in p for m in PRIVILEGED_MARKERS):
        return 'PRIVILEGED'
    if p in DEPLOY_FIT_FILES:
        return 'DEPLOY_FIT'
    if any(p.startswith(pre) for pre in FOLD_FITTED_PREFIXES):
        return 'FOLD_FITTED'
    if p in RAW_SOURCES:
        return 'RAW'
    return 'UNCLASSIFIED'


def manifest(ids, read_submission, numpy):
    """Build the manifest: only RAW entries are returned as a nested-safe pool."""
    import numpy as np
    rows = []
    seen_full, seen_q, seen_r = {}, {}, {}
    for path, recipe in sorted(RAW_SOURCES.items()):
        f = ROOT / path
        if not f.exists():
            rows.append({'path': path, 'class': 'RAW', 'status': 'missing'})
            continue
        d = read_submission(f)
        if set(d) != set(ids):
            rows.append({'path': path, 'class': 'RAW', 'status': 'misaligned'})
            continue
        a = np.array([d[i] for i in ids])
        a = a / a.sum(axis=1, keepdims=True)
        full = hashlib.sha256(a.round(12).tobytes()).hexdigest()[:16]
        q = 1 - a[:, 0]
        r = a[:, 1:] / np.maximum(q, 1e-12)[:, None]
        r = r / r.sum(axis=1, keepdims=True)
        hq = hashlib.sha256(q.round(12).tobytes()).hexdigest()[:16]
        hr = hashlib.sha256(r.round(12).tobytes()).hexdigest()[:16]
        rows.append({'path': path, 'class': 'RAW', 'recipe': recipe, 'sha_full': full,
                     'sha_q': hq, 'sha_r': hr,
                     'dup_full_of': seen_full.get(full), 'dup_q_of': seen_q.get(hq),
                     'dup_r_of': seen_r.get(hr), 'status': 'ok'})
        seen_full.setdefault(full, path)
        seen_q.setdefault(hq, path)
        seen_r.setdefault(hr, path)
    return rows
