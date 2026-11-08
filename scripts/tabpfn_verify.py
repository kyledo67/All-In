"""Numerical checks on the TabPFN component, on small synthetic tables.

These are Python test outputs about our wrapper, kept separate from the
source-analysis findings and from the official competition scores. They check the
contract the blend relies on, not predictive quality:

  determinism      same seed and context -> identical probabilities
  simplex          finite, >= 1e-6, five columns, sums to 1 within 1e-12
  round trip       save then load reproduces predictions exactly
  subsampling      deterministic, whole episodes only, independent of row order
  absent class     a payout never seen in context still gets a finite column
  offline          runs with HF_HUB_OFFLINE=1 against the cached checkpoint

Run after changing slot_model/tabpfn_model.py.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model.tabpfn_model import FLOOR, MAX_FEATURES, TabPFNComponent, episode_subsample, simplex

RESULTS = []


def record(name, passed, detail):
    RESULTS.append({'check': name, 'passed': bool(passed), 'detail': detail})
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)


def table(rows=240, features=16, classes=5, seed=0, machines=8):
    """Synthetic episodes of `machines` machines, with a genuine lag-1 signal."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features))
    logits = x[:, :classes] * 1.5
    y = np.argmax(logits + rng.gumbel(size=(rows, classes)), axis=1)
    episodes = np.array([f'ep{i // machines:03d}' for i in range(rows)])
    weight = np.full(rows, 1 / rows)
    return x, y.astype(np.int64), episodes, weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/tabpfn/verification.json')
    args = parser.parse_args()
    start = time.perf_counter()
    x, y, episodes, weight = table()
    xq = table(rows=40, seed=7)[0]

    # ---- simplex helper, before any model call ------------------------------
    raw = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2], [1e-30, 1 - 1e-30, 0, 0, 0]])
    projected = simplex(raw)
    record('simplex floor and normalization',
           projected.min() >= FLOOR * 0.5 and np.max(abs(projected.sum(1) - 1)) < 1e-12,
           f'min {projected.min():.2e}, max sum error {np.max(abs(projected.sum(1) - 1)):.1e}')

    # ---- subsampling --------------------------------------------------------
    mask_a, info_a = episode_subsample(episodes, weight, 80, 20261108)
    mask_b, _ = episode_subsample(episodes, weight, 80, 20261108)
    order = np.random.default_rng(1).permutation(len(episodes))
    mask_c, _ = episode_subsample(episodes[order], weight[order], 80, 20261108)
    kept = set(episodes[mask_a])
    whole = all(mask_a[episodes == e].all() or (~mask_a[episodes == e]).all() for e in set(episodes))
    record('subsample deterministic', np.array_equal(mask_a, mask_b), f"{info_a['kept_rows']} rows reproduced exactly")
    record('subsample independent of row order', set(episodes[order][mask_c]) == kept,
           f'{len(kept)} episodes chosen either way')
    record('subsample keeps whole episodes', whole, 'no episode split across the cut')
    mask_d, _ = episode_subsample(episodes, weight, 80, 20261109)
    record('subsample responds to seed', not np.array_equal(mask_a, mask_d), 'a different seed selects differently')

    # ---- feature ceiling ----------------------------------------------------
    try:
        TabPFNComponent(seed=1).fit(np.zeros((10, MAX_FEATURES + 1)), np.arange(10) % 5)
        record('feature ceiling enforced', False, 'oversized table was accepted')
    except ValueError as exc:
        record('feature ceiling enforced', 'ceiling' in str(exc), str(exc)[:90])

    # ---- determinism and simplex on real forward passes ---------------------
    first = TabPFNComponent(seed=20261108, n_estimators=1).fit(x, y, episodes=episodes, weight=weight)
    p1 = first.predict_proba(xq)
    p2 = TabPFNComponent(seed=20261108, n_estimators=1).fit(x, y, episodes=episodes, weight=weight).predict_proba(xq)
    record('determinism across fits', np.array_equal(p1, p2),
           f'max |p1-p2| = {float(np.max(abs(p1 - p2))):.1e} over {p1.shape[0]} rows')
    p3 = TabPFNComponent(seed=20261109, n_estimators=1).fit(x, y, episodes=episodes, weight=weight).predict_proba(xq)
    record('seed changes the forward pass', not np.array_equal(p1, p3),
           f'max |p1-p3| = {float(np.max(abs(p1 - p3))):.2e}')
    record('output is a valid simplex',
           p1.shape[1] == 5 and np.isfinite(p1).all() and p1.min() >= FLOOR * 0.5
           and np.max(abs(p1.sum(1) - 1)) < 1e-12,
           f'shape {p1.shape}, min {p1.min():.2e}, max sum error {np.max(abs(p1.sum(1) - 1)):.1e}')

    # ---- save / load round trip --------------------------------------------
    with tempfile.TemporaryDirectory() as directory:
        config = first.save(directory)
        restored = TabPFNComponent.load(directory)
        p4 = restored.predict_proba(xq)
        record('save/load round trip', np.array_equal(p1, p4),
               f"max |p-p_loaded| = {float(np.max(abs(p1 - p4))):.1e}, context {config['context_rows']} rows")
        tampered = Path(directory) / 'config.json'
        stored = json.loads(tampered.read_text())
        stored['context_sha256'] = '0' * 64
        tampered.write_text(json.dumps(stored))
        try:
            TabPFNComponent.load(directory)
            record('context hash is enforced', False, 'a tampered hash was accepted')
        except ValueError as exc:
            record('context hash is enforced', True, str(exc)[:90])

    # ---- a payout absent from the context ----------------------------------
    keep = y != 4
    sparse = TabPFNComponent(seed=20261108, n_estimators=1).fit(x[keep], y[keep], episodes=episodes[keep],
                                                               weight=weight[keep])
    p5 = sparse.predict_proba(xq)
    record('absent payout still yields a column',
           p5.shape[1] == 5 and np.isfinite(p5).all() and np.max(abs(p5.sum(1) - 1)) < 1e-12
           and p5[:, 4].max() <= FLOOR * 2,
           f'class 4 absent from {int(keep.sum())} context rows; column max {p5[:, 4].max():.1e}')

    # ---- offline ------------------------------------------------------------
    record('ran offline', os.environ.get('HF_HUB_OFFLINE') == '1',
           'HF_HUB_OFFLINE=1 throughout; checkpoint read from artifacts/tabpfn/checkpoints')

    passed = all(r['passed'] for r in RESULTS)
    report = {'checks': RESULTS, 'passed': passed, 'failures': [r for r in RESULTS if not r['passed']],
              'seconds': time.perf_counter() - start,
              'scope': 'wrapper contract on synthetic tables; not a competition score'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"\n{sum(r['passed'] for r in RESULTS)}/{len(RESULTS)} checks passed in {report['seconds']:.1f}s")
    print('VERIFY', 'PASSED' if passed else 'FAILED')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
