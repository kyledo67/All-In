"""One predeclared scalar sharpness calibration of the frozen fold recipes."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from scipy.optimize import minimize_scalar
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_rows, read_submission, read_targets, write_submission
from slot_model.frozen_blend import predict

DATA = Path('/Users/kyledo/Downloads/main-participants-60-40/public')
OUT = ROOT / 'artifacts/calibration_cycle_01'
MODEL = ROOT / 'artifacts/blend_release/model'
FOLDS = json.loads((MODEL / 'validation_fold_map.json').read_text())
assert not OUT.exists(), 'Use the saved run; do not silently repeat the experiment.'
OUT.mkdir(parents=True)
start = time.perf_counter()
targets = read_targets(DATA / 'validation/targets.csv')
groups = {r['query_id']: r['condition'] for r in read_rows(DATA / 'validation/conditions.csv', ('query_id', 'condition'))}
reference = read_submission(DATA / 'validation/reference.csv')
ids = [t.query_id for t in targets]
episodes = np.array([t.episode_id for t in targets])
folds = np.array([FOLDS[e] for e in episodes])
truth = np.array([t.probabilities for t in targets])
recipes = []
for fold in range(4):
    path = OUT / f'recipe_{fold}_validation.csv'
    predict(DATA, path, MODEL, split='validation', mode=f'fold{fold}')
    p = read_submission(path)
    assert set(p) == set(ids)
    recipes.append(np.array([p[q] for q in ids]))

def transform(a, gamma):
    x = np.exp(gamma * np.log(np.maximum(a, 1e-12)))
    return x / x.sum(axis=1, keepdims=True)

def official(a, mask=None):
    keep = np.ones(len(ids), dtype=bool) if mask is None else mask
    p = {ids[i]: a[i].tolist() for i in np.flatnonzero(keep)}
    ts = [targets[i] for i in np.flatnonzero(keep)]
    return contest_score(p, ts, {q: groups[q] for q in p}, {q: reference[q] for q in p}, max_points=60)

base = np.empty_like(recipes[0]);candidate = np.empty_like(base)
fold_reports = []
for fold in range(4):
    held = folds == fold
    fit = ~held
    recipe = recipes[fold]
    # Strictly use the recipe trained without the held fold's probability targets.
    result = minimize_scalar(lambda g: official(transform(recipe, g), fit)['mean_loss_ratio'],
                             bounds=(0.8, 1.2), method='bounded', options={'xatol': 1e-5})
    gamma = float(result.x)
    base[held] = recipe[held]
    candidate[held] = transform(recipe[held], gamma)
    fold_reports.append({'fold': fold, 'gamma': gamma, 'episodes': int(len(set(episodes[held]))),
                         'base': official(recipe, held), 'candidate': official(transform(recipe, gamma), held)})

write_submission(OUT / 'base_oof.csv', dict(zip(ids, base.tolist())))
write_submission(OUT / 'candidate_oof.csv', dict(zip(ids, candidate.tolist())))
base_score = official(base); candidate_score = official(candidate)
assert abs(base_score['technical_points'] - 31.70233166770079) < 1e-9
delta = candidate_score['mean_loss_ratio'] - base_score['mean_loss_ratio']
per_episode = []
for e in sorted(set(episodes)):
    mask = episodes == e
    per_episode.append({'episode': e, 'condition': groups[ids[np.flatnonzero(mask)[0]]],
                        'base_error': float(np.mean(np.sum((base[mask]-truth[mask])**2, axis=1))),
                        'candidate_error': float(np.mean(np.sum((candidate[mask]-truth[mask])**2, axis=1)))})
by_condition = {}
for c in sorted(set(groups.values())):
    subset = [r for r in per_episode if r['condition'] == c]
    by_condition[c] = {'base_error': float(np.mean([r['base_error'] for r in subset])),
                       'candidate_error': float(np.mean([r['candidate_error'] for r in subset]))}

# Paired, whole-episode exploratory interval; frozen reference denominator is recomputed per draw.
rng = np.random.default_rng(314159)
boot = np.zeros(2000)
for c in sorted(by_condition):
    rows = [r for r in per_episode if r['condition'] == c]
    ref_errors = []
    for r in rows:
        m = episodes == r['episode']
        ref_errors.append(float(np.mean(np.sum((np.array([reference[ids[i]] for i in np.flatnonzero(m)]) - truth[m])**2, axis=1))))
    sample = rng.integers(0, len(rows), (2000, len(rows)))
    diffs = np.array([r['candidate_error'] - r['base_error'] for r in rows])[sample].mean(axis=1)
    refs = np.array(ref_errors)[sample].mean(axis=1)
    boot += diffs / np.maximum(refs, 1e-9) / 4

diagnostics = {'category_normalized_contribution': {}, 'category_bias': {},
               'confidence_quartiles': []}
for c in sorted(by_condition):
    cm = np.array([groups[q] == c for q in ids])
    denom = base_score['by_condition'][c]['reference_error']
    diagnostics['category_normalized_contribution'][c] = (np.mean((base[cm]-truth[cm])**2, axis=0)/denom/4).tolist()
    diagnostics['category_bias'][c] = np.mean(base[cm]-truth[cm], axis=0).tolist()
conf = np.max(base, axis=1)
for idx in np.array_split(np.argsort(conf), 4):
    diagnostics['confidence_quartiles'].append({'mean_max_probability': float(conf[idx].mean()),
        'mean_error': float(np.mean(np.sum((base[idx]-truth[idx])**2, axis=1)))})
report = {'status': 'exploratory development evidence; public validation previously inspected',
          'hypothesis': 'A systematic sharpness bias may be corrected by one regularized scalar power transform.',
          'intervention': 'For each held fold, optimize gamma in [0.8,1.2] on the other three folds using that fold recipe and the exact organizer score.',
          'base': base_score, 'candidate': candidate_score, 'delta_normalized_error': delta,
          'relative_error_reduction': -delta/base_score['mean_loss_ratio'], 'folds': fold_reports,
          'paired_delta_ratio_ci95': np.quantile(boot, [.025,.975]).tolist(),
          'diagnostics': diagnostics, 'by_condition_errors': by_condition,
          'seconds': time.perf_counter()-start,
          'input_hashes': {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'validation/targets.csv',DATA/'validation/conditions.csv',DATA/'validation/reference.csv')},
          'source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / 'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
(OUT / 'per_episode.json').write_text(json.dumps(per_episode, indent=2)+'\n')
print(json.dumps({'base_points': base_score['technical_points'], 'candidate_points': candidate_score['technical_points'],
                  'delta_ratio': delta, 'gamma': [r['gamma'] for r in fold_reports], 'seconds': report['seconds']}, indent=2))
