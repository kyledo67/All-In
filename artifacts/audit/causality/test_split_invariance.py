"""Post-cutoff invariance for the MTD and two-head components on the TEST split.

The test split is the actual inference target (1536 queries / 192 episodes) and carries no
labels, so running it here cannot consume holdout information. This repeats the strongest
adversarial check on the split that will actually be submitted.

  MTD:      PairMTDModel(LAG7_PAIRS) over MTDModel.load(artifacts/mtd_01/change.json);
            adversarial plays at cutoff..cutoff+8 on every machine of every episode fed to
            slot_model.mtd.index(), which applies NO cutoff filter of its own.
  two_head: the 182-column feature matrix and the saved win/size heads; the same ticks are
            injected DIRECTLY into the index, bypassing causal_index's own filter.
"""
import hashlib, json, sys, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from catboost import CatBoostClassifier

from slot_model.features import causal_index, PAYOUTS
from slot_model.mtd import MTDModel, index as mtd_index
from slot_model.mtd_pairs import PairMTDModel, LAG7_PAIRS
from slot_forecasting.io import load_inputs
from slot_forecasting.schema import Play

DATA = ROOT / 'main-participants-60-40/public'
SRC = ROOT / 'scripts/dense_catboost.py'
OLD_D = "D=Path('/Users/kyledo/Downloads/main-participants-60-40/public')"
HORIZON = 8
OUT = Path(__file__).resolve().parent / 'test_split_invariance.json'


def main():
    t0 = time.perf_counter()
    q_, h_ = load_inputs(DATA / 'test')
    idx0 = causal_index(q_, h_)
    cutoffs = {(q.episode_id, q.machine_id): q.cutoff for q in q_}
    report = {'split': 'test', 'queries': len(q_), 'episodes': len({e for e, _ in cutoffs}),
              'machines': len(cutoffs), 'horizon_ticks': HORIZON,
              'labels_available': False,
              'note': 'the test split carries no labels, so running it consumes no holdout '
                      'information; this is a pure information-flow check'}

    extra = []
    for (e, m), cut in sorted(cutoffs.items()):
        obs = idx0[(e, m)]
        for off in range(HORIZON):
            tick = cut + off
            if tick in obs:
                continue
            extra.append(Play(e, m, tick, PAYOUTS[(obs.get(cut - 1, 0) + 3) % len(PAYOUTS)]))
    assert extra
    report['adversarial_plays_added'] = len(extra)
    report['machines_mutated'] = len({(p.episode_id, p.machine_id) for p in extra})

    # ---- MTD ----------------------------------------------------------------
    def mtd():
        return PairMTDModel(pairs=LAG7_PAIRS,
                            base=MTDModel.load(ROOT / 'artifacts/mtd_01/change.json'))
    b = mtd().predict(q_, h_)
    m = mtd().predict(q_, tuple(h_) + tuple(extra))
    worst = max(float(np.max(np.abs(np.asarray(b[k]) - np.asarray(m[k])))) for k in b)
    mtd_ok = all(np.array_equal(np.asarray(b[k]), np.asarray(m[k])) for k in b)
    report['mtd'] = {'queries_attacked': len(q_), 'bitwise_identical': bool(mtd_ok),
                     'max_abs_delta': worst,
                     'mtd_index_applies_no_cutoff_filter': True}
    print(f'MTD on test: {len(extra)} adversarial plays -> bitwise identical={mtd_ok} '
          f'(max delta {worst:.3e})', flush=True)

    # ---- two_head -----------------------------------------------------------
    raw = SRC.read_bytes()
    text = raw.decode()
    assert text.count(OLD_D) == 1
    mod = types.ModuleType('dense_catboost_local')
    mod.__file__ = str(SRC)
    exec(compile(text.replace(OLD_D, f"D=Path({str(DATA)!r})"), str(SRC), 'exec'), mod.__dict__)

    def byep(index):
        out = {}
        for key, hh in index.items():
            out.setdefault(key[0], {})[key] = hh
        return out

    def build(index):
        mod.vidx = index
        return mod.held_features(q_, set(mod.F), byep(index))

    loaded = []
    for s in mod.seeds:
        w = CatBoostClassifier(); w.load_model(str(ROOT / f'artifacts/two_head_01/win_{s}.cbm'))
        z = CatBoostClassifier(); z.load_model(str(ROOT / f'artifacts/two_head_01/size_{s}.cbm'))
        loaded.append((w, z))

    def pred(X):
        ps = []
        for w, z in loaded:
            c = w.predict_proba(X, thread_count=2)[:, 1]
            ps.append(np.column_stack((1 - c, c[:, None] * z.predict_proba(X, thread_count=2))))
        return np.mean(ps, axis=0)

    X0 = build(idx0); P0 = pred(X0)
    idx1 = causal_index(q_, tuple(h_) + tuple(extra))
    X1 = build(idx1); P1 = pred(X1)
    idx2 = {k: dict(v) for k, v in idx0.items()}
    inj = 0
    for p in extra:
        key = (p.episode_id, p.machine_id)
        if p.tick not in idx2[key]:
            idx2[key][p.tick] = PAYOUTS.index(p.payout); inj += 1
    X2 = build(idx2); P2 = pred(X2)
    report['two_head'] = {
        'feature_matrix_shape': list(X0.shape),
        'import_transform': {'source_sha256': hashlib.sha256(raw).hexdigest(),
                             'substituted_from': OLD_D, 'substitutions': 1},
        'through_causal_index': {
            'features_bitwise_identical': bool(np.array_equal(X0, X1)),
            'features_max_abs_delta': float(np.max(np.abs(X0 - X1))),
            'probabilities_bitwise_identical': bool(np.array_equal(P0, P1)),
            'probabilities_max_abs_delta': float(np.max(np.abs(P0 - P1)))},
        'bypassing_causal_index': {
            'ticks_injected_directly_into_index': inj,
            'features_bitwise_identical': bool(np.array_equal(X0, X2)),
            'features_max_abs_delta': float(np.max(np.abs(X0 - X2))),
            'probabilities_bitwise_identical': bool(np.array_equal(P0, P2)),
            'probabilities_max_abs_delta': float(np.max(np.abs(P0 - P2)))}}
    print(f"two_head on test: through causal_index identical="
          f"{report['two_head']['through_causal_index']['features_bitwise_identical']}; "
          f"bypassing causal_index identical="
          f"{report['two_head']['bypassing_causal_index']['features_bitwise_identical']} "
          f"({inj} ticks injected)", flush=True)

    report['seconds'] = time.perf_counter() - t0
    report['passed'] = bool(mtd_ok and np.array_equal(X0, X1) and np.array_equal(P0, P1)
                            and np.array_equal(X0, X2) and np.array_equal(P0, P2))
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('TEST-SPLIT INVARIANCE', 'PASSED' if report['passed'] else 'FAILED',
          f"({report['seconds']:.1f}s)")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
