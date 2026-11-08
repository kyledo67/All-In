"""Adversarial causality mutation test for the two-head CatBoost component.

Target under test: the exact validation inference path of scripts/two_head_catboost.py,
  X   = held_features(vq, set(F), vbyepisode)          # scripts/dense_catboost.py
  p   = [1-chance, chance * conditional]               # win head + 4-class size head
  out = mean over the two saved seeds
using the SAVED heads artifacts/two_head_01/{win,size}_{seed}.cbm. Nothing is re-fitted.

IMPORTING THE REAL FEATURE BUILDER: scripts/dense_catboost.py hardcodes
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public'), which does not exist on
this machine, so a plain import dies at module scope. This test execs the file's original
source with ONLY that one assignment rewritten to the local data root, and records the
sha256 of the original bytes plus the exact substitution, so the transform is auditable.
make_examples / train_features / held_features / matrix_examples are otherwise untouched.

NOTE ON HOLDOUT STATUS: public validation, inspected many times across this project.
Not an independent holdout. These are causality (information-flow) checks only.

Checks
  R   reproducibility: recomputed probabilities vs saved artifacts/two_head_01/validation.csv
  A1  post-cutoff invariance THROUGH causal_index: append plays at ticks cutoff..cutoff+8
      for every machine (target AND all 7 episode neighbours), rebuild the index with
      causal_index, rebuild the 182-column feature matrix -> bitwise identical, and the
      final head probabilities -> bitwise identical
  A2  post-cutoff invariance BYPASSING causal_index: inject the same post-cutoff ticks
      DIRECTLY into the index dicts, so causal_index's own `tick >= cutoff` drop cannot
      help. Only the `t < q.cutoff` guards inside sample_history/observable can save it.
      This is the defence-in-depth check A1 cannot make.
  P   power/sensitivity control: inject ONE play at a MISSING tick strictly below cutoff
      -> the feature matrix MUST change (otherwise A1/A2 are vacuous)
  N   neighbour columns: confirm the other_recent_* columns are the neighbour-derived ones
      and that they are non-degenerate, so A1/A2 actually exercise a neighbour path
"""
import copy, hashlib, json, sys, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from catboost import CatBoostClassifier

from slot_model.features import causal_index, PAYOUTS
from slot_forecasting.io import read_submission
from slot_forecasting.schema import Play

DATA = ROOT / 'main-participants-60-40/public'
SRC = ROOT / 'scripts/dense_catboost.py'
OLD_D = "D=Path('/Users/kyledo/Downloads/main-participants-60-40/public')"
HORIZON = 8
OUT = Path(__file__).resolve().parent / 'two_head_mutation.json'


def load_builder():
    """Exec scripts/dense_catboost.py with only the data-root assignment rewritten."""
    raw = SRC.read_bytes()
    text = raw.decode()
    assert text.count(OLD_D) == 1, 'data-root assignment not found exactly once'
    new_d = f"D=Path({str(DATA)!r})"
    mod = types.ModuleType('dense_catboost_local')
    mod.__file__ = str(SRC)
    exec(compile(text.replace(OLD_D, new_d), str(SRC), 'exec'), mod.__dict__)
    return mod, {'source_sha256': hashlib.sha256(raw).hexdigest(),
                 'substituted_from': OLD_D, 'substituted_to': new_d,
                 'substitutions': 1}


def heads(seeds):
    out = []
    for s in seeds:
        w = CatBoostClassifier()
        w.load_model(str(ROOT / f'artifacts/two_head_01/win_{s}.cbm'))
        z = CatBoostClassifier()
        z.load_model(str(ROOT / f'artifacts/two_head_01/size_{s}.cbm'))
        out.append((s, w, z))
    return out


def predict(X, loaded):
    ps = []
    for _, w, z in loaded:
        chance = w.predict_proba(X, thread_count=2)[:, 1]
        conditional = z.predict_proba(X, thread_count=2)
        p = np.column_stack((1 - chance, chance[:, None] * conditional))
        assert np.isfinite(p).all() and abs(p.sum(1) - 1).max() < 1e-12 and p.min() >= 0
        ps.append(p)
    return np.mean(ps, axis=0)


def main():
    t0 = time.perf_counter()
    mod, transform = load_builder()
    vq, vh, F = mod.vq, mod.vh, mod.F
    report = {'holdout_status': 'public validation, inspected many times; NOT an independent '
                                'holdout. Causality checks only, no generalisation claim.',
              'import_transform': transform, 'queries': len(vq),
              'feature_columns': len(mod.schema), 'seeds': mod.seeds}

    def build(vidx, vbyepisode):
        mod.vidx, mod.vbyepisode = vidx, vbyepisode
        return mod.held_features(vq, set(F), vbyepisode)

    def byep(vidx):
        out = {}
        for key, h in vidx.items():
            out.setdefault(key[0], {})[key] = h
        return out

    vidx0 = causal_index(vq, vh)
    X0 = build(vidx0, byep(vidx0))
    loaded = heads(mod.seeds)
    P0 = predict(X0, loaded)
    report['feature_matrix_shape'] = list(X0.shape)

    # ---- R reproducibility --------------------------------------------------
    got = {q.query_id: P0[i].tolist() for i, q in enumerate(vq)}
    saved = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    assert set(got) == set(saved)
    deltas = [float(np.max(np.abs(np.asarray(got[k]) - np.asarray(saved[k])))) for k in got]
    report['reproducibility'] = {
        'saved_bitwise_identical': all(d == 0 for d in deltas),
        'saved_max_abs_delta': max(deltas), 'saved_queries_differing': sum(1 for d in deltas if d > 0),
        'saved_within_1e_12': max(deltas) < 1e-12,
        'note': 'recomputed from the saved .cbm heads and a freshly built feature matrix'}
    print(f"R reproducibility vs saved validation.csv: bitwise="
          f"{report['reproducibility']['saved_bitwise_identical']} max_delta={max(deltas):.3e}",
          flush=True)

    # ---- adversarial post-cutoff plays -------------------------------------
    cutoffs = {(q.episode_id, q.machine_id): q.cutoff for q in vq}
    extra = []
    for (e, m), cut in sorted(cutoffs.items()):
        obs = vidx0[(e, m)]
        for off in range(HORIZON):
            tick = cut + off
            if tick in obs:
                continue
            recent = obs.get(cut - 1, 0)
            extra.append(Play(e, m, tick, PAYOUTS[(recent + 3) % len(PAYOUTS)]))
    assert extra

    # ---- A1 through causal_index -------------------------------------------
    vidx1 = causal_index(vq, tuple(vh) + tuple(extra))
    X1 = build(vidx1, byep(vidx1))
    P1 = predict(X1, loaded)
    a1_x = bool(np.array_equal(X0, X1))
    a1_p = bool(np.array_equal(P0, P1))
    report['post_cutoff_invariance_through_causal_index'] = {
        'adversarial_plays_added': len(extra),
        'machines_mutated': len({(p.episode_id, p.machine_id) for p in extra}),
        'episodes_mutated': len({p.episode_id for p in extra}),
        'features_bitwise_identical': a1_x,
        'features_max_abs_delta': float(np.max(np.abs(X0 - X1))),
        'probabilities_bitwise_identical': a1_p,
        'probabilities_max_abs_delta': float(np.max(np.abs(P0 - P1)))}
    print(f'A1 through causal_index: {len(extra)} plays -> features identical={a1_x}, '
          f'probabilities identical={a1_p}', flush=True)

    # ---- A2 bypassing causal_index -----------------------------------------
    vidx2 = {k: dict(v) for k, v in vidx0.items()}
    injected = 0
    for p in extra:
        key = (p.episode_id, p.machine_id)
        if p.tick not in vidx2[key]:
            vidx2[key][p.tick] = PAYOUTS.index(p.payout)
            injected += 1
    X2 = build(vidx2, byep(vidx2))
    P2 = predict(X2, loaded)
    a2_x = bool(np.array_equal(X0, X2))
    a2_p = bool(np.array_equal(P0, P2))
    report['post_cutoff_invariance_bypassing_causal_index'] = {
        'ticks_injected_directly_into_index': injected,
        'features_bitwise_identical': a2_x,
        'features_max_abs_delta': float(np.max(np.abs(X0 - X2))),
        'probabilities_bitwise_identical': a2_p,
        'probabilities_max_abs_delta': float(np.max(np.abs(P0 - P2))),
        'why': 'causal_index is bypassed, so only the t < q.cutoff guards inside '
               'sample_history() and observable() can produce invariance here'}
    print(f'A2 bypassing causal_index: {injected} ticks injected -> features identical={a2_x}, '
          f'probabilities identical={a2_p}', flush=True)

    # ---- P power control ----------------------------------------------------
    probes = []
    for q in vq:
        obs = vidx0[(q.episode_id, q.machine_id)]
        gap = next((t for t in range(q.cutoff - 128, q.cutoff - 40) if t >= 0 and t not in obs), None)
        if gap is not None:
            probes.append((q, gap))
        if len(probes) >= 4:
            break
    detail = []
    for q, gap in probes:
        vidx3 = {k: dict(v) for k, v in vidx0.items()}
        vidx3[(q.episode_id, q.machine_id)][gap] = PAYOUTS.index(2)
        X3 = build(vidx3, byep(vidx3))
        d = float(np.max(np.abs(X0 - X3)))
        row = list(vq).index(q)
        detail.append({'query_id': q.query_id, 'injected_tick': gap, 'cutoff': q.cutoff,
                       'feature_max_abs_delta': d,
                       'own_row_changed': bool(not np.array_equal(X0[row], X3[row]))})
    report['power_control_pre_cutoff'] = {
        'probes': len(probes), 'probes_that_changed': sum(1 for d in detail if d['own_row_changed']),
        'test_has_power': bool(detail) and all(d['own_row_changed'] for d in detail),
        'detail': detail}
    print(f"P power control: {report['power_control_pre_cutoff']['probes_that_changed']}"
          f"/{len(probes)} pre-cutoff injections changed the feature row "
          f"(test has power = {report['power_control_pre_cutoff']['test_has_power']})", flush=True)

    # ---- N neighbour columns -----------------------------------------------
    nbr = [i for i, n in enumerate(mod.schema) if n.startswith('other_recent')]
    report['neighbour_columns'] = {
        'names': [mod.schema[i] for i in nbr], 'count': len(nbr),
        'nonconstant': int(sum(1 for i in nbr if X0[:, i].std() > 0)),
        'other_recent_count_range': [float(X0[:, mod.schema.index('other_recent_count')].min()),
                                     float(X0[:, mod.schema.index('other_recent_count')].max())],
        'source': "observable() reads index.items() filtered to episode==q.episode_id and "
                  "machine!=q.machine_id, with q.cutoff-16 <= t < q.cutoff (strict)"}
    print(f"N neighbour columns: {len(nbr)} columns, "
          f"{report['neighbour_columns']['nonconstant']} non-constant, other_recent_count range "
          f"{report['neighbour_columns']['other_recent_count_range']}", flush=True)

    report['seconds'] = time.perf_counter() - t0
    report['passed'] = bool(a1_x and a1_p and a2_x and a2_p and
                            report['reproducibility']['saved_within_1e_12'] and
                            report['power_control_pre_cutoff']['test_has_power'])
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('TWO-HEAD MUTATION AUDIT', 'PASSED' if report['passed'] else 'FAILED',
          f"({report['seconds']:.1f}s)")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
