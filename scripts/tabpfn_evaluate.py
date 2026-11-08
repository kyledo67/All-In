"""Does TabPFN earn a slot next to CatBoost and the Dynamax HMM?

Follows scripts/phase2_evaluate.py exactly: four stratified validation-episode folds,
configurations and convex blend weights chosen on the other three folds, held-fold
predictions concatenated, then handed to the organizer's own scorer. All public
validation has been examined before, so every number here is exploratory development
evidence, not an untouched holdout.

Reports, each with per-condition loss ratios:
  (a) TabPFN alone
  (b) the current CatBoost + Dynamax + local blend (the 31.70 bar)
  (c) the three-model blend at fixed TabPFN weights 0.15 / 0.25 / 0.35
  (d) a free five-component simplex, to see what weight TabPFN is actually given
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.optimize import minimize

from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import read_submission, write_submission

PHASE2 = ROOT / 'artifacts/phase2'
BASE = ('hmm', 'local', 'cat_observable', 'cat_hmm')
NAMES = BASE + ('tabpfn',)
FIXED_WEIGHTS = (0.15, 0.25, 0.35)
VARIANTS = ('observable', 'hmm')


def checker_points(out, variants, data):
    """Run organizer/check_submission.py on each written OOF CSV and parse its report."""
    import re
    import subprocess
    results = {}
    for name in variants:
        path = out / 'predictions' / f'oof_{name}.csv'
        process = subprocess.run([sys.executable, str(ROOT / 'organizer/check_submission.py'), str(path.resolve()),
                                  '--data', str(Path(data).resolve()), '--split', 'validation'],
                                 capture_output=True, text=True, cwd=str(ROOT / 'organizer'))
        text = process.stdout
        points = re.search(r'Validation points: ([0-9.]+)', text)
        ratios = dict((m.group(1), float(m.group(2)))
                      for m in re.finditer(r'^\s+(\w+): loss ratio ([0-9.]+)', text, re.M))
        results[name] = {'returncode': process.returncode, 'valid_format': 'VALID FORMAT' in text,
                         'points': float(points.group(1)) if points else None,
                         'by_condition_loss_ratio': ratios,
                         'stderr': process.stderr.strip()[:400] or None}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--tabpfn', type=Path, default=ROOT / 'artifacts/tabpfn')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/tabpfn')
    parser.add_argument('--weights', nargs='+', type=float, default=list(FIXED_WEIGHTS))
    args = parser.parse_args()
    start = time.perf_counter()

    protocol = json.loads((PHASE2 / 'protocol.json').read_text())
    folds = json.loads((PHASE2 / 'folds.json').read_text())['validation']
    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    episode_of = {t.query_id: t.episode_id for t in targets}
    ep = np.array([episode_of[q] for q in ids])
    truth = np.array([next(t.probabilities for t in targets if t.query_id == q) for q in ids])
    ref = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())

    def load(path):
        d = read_submission(path)
        return np.array([d[q] for q in ids])

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def official(a, es=episodes):
        return score(mapping(a), scoring, es)

    def row_weights(es):
        """Reproduce the official per-row weight so the optimizer fits the real score."""
        w = np.zeros(len(ids))
        for condition in sorted(set(groups.values())):
            ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == condition}
            b = np.mean([np.mean(np.sum((ref[ep == e] - truth[ep == e]) ** 2, axis=1)) for e in ces])
            for e in ces:
                w[ep == e] = 1 / (4 * len(ces) * sum(ep == e) * max(b, 1e-9))
        return w

    # ---- components ---------------------------------------------------------
    hmm = load(ROOT / 'artifacts/final/validation_predictions.csv')
    local = [load(PHASE2 / 'predictions' / f'local_{i}.csv') for i in range(len(protocol['local']))]
    cat = {v: {i: {s: load(PHASE2 / 'predictions' / f'cb_{v}_{i}_{s}.csv') for s in protocol['seeds']}
               for i in range(len(protocol['catboost']))} for v in VARIANTS}
    cat_avg = {v: [np.mean(list(cat[v][i].values()), axis=0) for i in cat[v]] for v in cat}

    runs = json.loads((args.tabpfn / 'training_runs.json').read_text())
    tab_seeds = runs['seeds']
    tab_variants = [v for v in VARIANTS if v in runs['variants']]
    tab = {v: {s: load(args.tabpfn / 'predictions' / f'tabpfn_{v}_{s}.csv') for s in tab_seeds}
           for v in tab_variants}
    tab_avg = {v: np.mean(list(tab[v].values()), axis=0) for v in tab}

    for name, array in [('hmm', hmm)] + [(f'tabpfn_{v}', a) for v, a in tab_avg.items()]:
        if not np.isfinite(array).all() or array.min() < 0 or np.max(abs(array.sum(1) - 1)) > 1e-6:
            raise ValueError(f'{name} predictions are not a valid simplex')

    # ---- fold-held selection and blending -----------------------------------
    def select(es):
        chosen = {'local': min(range(len(local)), key=lambda i: official(local[i], es)['mean_loss_ratio'])}
        for v in cat:
            chosen[v] = min(range(len(cat_avg[v])), key=lambda i: official(cat_avg[v][i], es)['mean_loss_ratio'])
        chosen['tabpfn'] = min(tab_variants, key=lambda v: official(tab_avg[v], es)['mean_loss_ratio'])
        return chosen

    def blend_weights(arrays, es):
        """Same complementarity test and SLSQP simplex fit as phase2_evaluate."""
        p = np.array(arrays)
        w = row_weights(es)
        residual = p - truth
        flat = (residual * np.sqrt(w)[None, :, None]).reshape(len(p), -1)
        gram = flat @ flat.T
        norm = np.sqrt(np.diag(gram))
        cos = gram / np.maximum(norm[:, None] * norm[None, :], 1e-15)
        if not bool(np.min(cos) < .98):
            return np.eye(len(p))[np.argmin(np.diag(gram))], cos, False
        res = minimize(lambda x: float(x @ gram @ x), np.ones(len(p)) / len(p), jac=lambda x: 2 * gram @ x,
                       method='SLSQP', bounds=[(0, 1)] * len(p),
                       constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1, 'jac': lambda x: np.ones_like(x)},
                       options={'ftol': 1e-12, 'maxiter': 200})
        assert res.success, res.message
        return res.x / res.x.sum(), cos, True

    variants_out = ['tabpfn', 'base_blend', 'free_blend'] + [f'blend_w{w:g}' for w in args.weights]
    oof = {n: np.empty_like(hmm) for n in list(NAMES) + variants_out}
    fold_reports = []
    for fold in range(4):
        held = {e for e in episodes if folds[e] == fold}
        fit = episodes - held
        chosen = select(fit)
        base_arrays = [hmm, local[chosen['local']], cat_avg['observable'][chosen['observable']],
                       cat_avg['hmm'][chosen['hmm']]]
        tabpfn = tab_avg[chosen['tabpfn']]
        mask = np.array([e in held for e in ep])

        base_w, base_cos, base_complementary = blend_weights(base_arrays, fit)
        base_blend = np.einsum('m,mnc->nc', base_w, np.array(base_arrays))
        free_w, free_cos, free_complementary = blend_weights(base_arrays + [tabpfn], fit)
        free_blend = np.einsum('m,mnc->nc', free_w, np.array(base_arrays + [tabpfn]))

        for name, array in zip(NAMES, base_arrays + [tabpfn]):
            oof[name][mask] = array[mask]
        oof['tabpfn'][mask] = tabpfn[mask]
        oof['base_blend'][mask] = base_blend[mask]
        oof['free_blend'][mask] = free_blend[mask]
        for w in args.weights:
            oof[f'blend_w{w:g}'][mask] = ((1 - w) * base_blend + w * tabpfn)[mask]

        # The optimizer's objective must equal the official score on the fitting folds.
        for array in base_arrays + [tabpfn]:
            assert abs(np.sum(row_weights(fit) * np.sum((array - truth) ** 2, axis=1))
                       - official(array, fit)['mean_loss_ratio']) < 1e-12
        fold_reports.append({
            'fold': fold, 'selection': chosen, 'held_episodes': sorted(held),
            'base_blend_weights': dict(zip(BASE, base_w.tolist())),
            'free_blend_weights': dict(zip(NAMES, free_w.tolist())),
            'free_tabpfn_weight': float(free_w[-1]),
            'base_complementary': base_complementary, 'free_complementary': free_complementary,
            'free_residual_cosine': free_cos.tolist(),
            'score': {n: official(a, held) for n, a in
                      [('tabpfn', tabpfn), ('base_blend', base_blend), ('free_blend', free_blend)]
                      + [(f'blend_w{w:g}', (1 - w) * base_blend + w * tabpfn) for w in args.weights]},
        })
        print(f'fold {fold}: free tabpfn weight {free_w[-1]:.4f} '
              f"selection {chosen}", flush=True)

    scores = {n: official(a) for n, a in oof.items()}
    comparisons = {n: paired_bootstrap(mapping(oof[n]), mapping(oof['base_blend']), scoring, episodes)
                   for n in variants_out if n != 'base_blend'}
    free_weights = [r['free_tabpfn_weight'] for r in fold_reports]
    earns_slot = bool(np.mean(free_weights) >= 0.15)

    report = {
        'exposure': 'Exploratory: all public validation previously examined. No untouched evaluation set.',
        'aggregation': 'Official score on concatenated held-fold predictions, not mean fold points.',
        'bar': {'current_blend_points': scores['base_blend']['technical_points'],
                'recorded_phase2_points': 31.7023},
        'scores': scores, 'folds': fold_reports, 'paired_vs_current_blend': comparisons,
        'tabpfn': {'variants': tab_variants, 'seeds': tab_seeds,
                   'n_estimators': runs['n_estimators'],
                   'context_rows': runs['provenance']['train_rows'],
                   'free_blend_weight_per_fold': free_weights,
                   'free_blend_weight_mean': float(np.mean(free_weights)),
                   'earns_slot_at_0.15': earns_slot},
        'seconds': time.perf_counter() - start,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, array in oof.items():
        write_submission(out / 'predictions' / f'oof_{name}.csv', mapping(array))

    # Independent confirmation through the organizer's own entry point, so the headline
    # numbers do not rest on our scoring call alone.
    report['kit_checker'] = checker_points(out, variants_out, args.data)
    for name, checked in report['kit_checker'].items():
        ours = scores[name]['technical_points']
        checked['our_points'] = ours
        checked['agrees'] = (checked.get('points') is not None and abs(checked['points'] - ours) <= 1e-6)
    disagreements = sorted(n for n, c in report['kit_checker'].items() if not c['agrees'])
    report['kit_checker_agrees'] = not disagreements
    if disagreements:
        print(f'WARNING: kit checker disagrees with our scorer for {disagreements}; '
              'treat the checker output as authoritative and investigate before reporting.')
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    print('\n--- official development score (held-fold concatenation) ---')
    print(f"{'variant':<16}{'points':>9}  " + '  '.join(f'{c:>14}' for c in sorted(scores['base_blend']['by_condition'])))
    for name in ['tabpfn', 'base_blend'] + [f'blend_w{w:g}' for w in args.weights] + ['free_blend']:
        row = scores[name]
        ratios = '  '.join(f"{row['by_condition'][c]['loss_ratio']:>14.4f}" for c in sorted(row['by_condition']))
        print(f"{name:<16}{row['technical_points']:>9.4f}  {ratios}")
    print(f"\nfree-simplex TabPFN weight per fold: {[round(w, 4) for w in free_weights]}")
    print(f"mean {np.mean(free_weights):.4f} -> earns a slot at 0.15: {earns_slot}")
    print(f"\nwrote {out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
