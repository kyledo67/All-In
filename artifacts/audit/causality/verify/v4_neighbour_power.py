"""NEIGHBOUR-PATH power control, which the prior audit never ran.

mtd_mutation.py:119-123, two_head_mutation.py:183-199 and tabpfn_bypass.py:92-97 all
build their power control by injecting a pre-cutoff tick into
  idx[(q.episode_id, q.machine_id)]
-- the TARGET's OWN machine. None of them perturbs a NEIGHBOUR observation. So the
neighbour half of each post-cutoff invariance result (the half the whole 'neighbour
future' question turns on) is not shown to be non-vacuous: a pipeline that ignored
neighbours entirely would pass A/A1/A2/S1 identically.

Here: perturb neighbour observations STRICTLY BELOW the cutoff and require the output to
move. All neighbour payouts in the read window are driven to all-win and then to all-loss.
"""
import json, sys, time
from pathlib import Path
ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path[:0] = [str(ROOT), str(ROOT / 'organizer')]
import numpy as np
from slot_model.features import PAYOUTS, causal_index
from slot_model.phase2_features import examples, fit_prior, matrix, observable, sample_history
from slot_model.phase2_predict import hmm_summary
from slot_model.phase2_tables import PHASE2, load_population
from slot_model.mtd import MTDModel, index as mtd_index, inputs as mtd_inputs
from slot_model.mtd_pairs import PairMTDModel, LAG7_PAIRS
from slot_forecasting.io import load_inputs
from slot_forecasting.schema import Play

DATA = ROOT / 'main-participants-60-40/public'
t0 = time.perf_counter()
vq, vh = load_inputs(DATA / 'validation')
folds = json.loads((PHASE2 / 'folds.json').read_text())
tq, th = load_inputs(DATA / 'train')
tidx = causal_index(tq, th)
prior = fit_prior(tq, tidx, set(folds['train']))
population, _ = load_population(set(folds['train']))
vidx = causal_index(vq, vh)
midx = mtd_index(vh)
report = {}

def nbr_overwrite(q, lo, hi, payout):
    """Plays overwriting EVERY neighbour observation of q's episode in [lo, hi)."""
    out = []
    for (e, m), h in midx.items():
        if e != q.episode_id or m == q.machine_id:
            continue
        for t in sorted(h):
            if lo <= t < hi:
                out.append(Play(e, m, t, payout))
    return out

# ---------- 1. MTD neighbour power (neighbour lags 1..3 at the cutoff) ----------
model = lambda: PairMTDModel(pairs=LAG7_PAIRS, base=MTDModel.load(ROOT / 'artifacts/mtd_01/change.json'))
probe = []
for q in vq[:6]:
    base = model().predict([q], vh)[q.query_id]
    row = {'query_id': q.query_id, 'cutoff': q.cutoff,
           'baseline_nbr_symbols_at_cutoff': mtd_inputs(q, q.cutoff, midx)[8:]}
    for tag, payout in (('all_win_20x', 20), ('all_loss_0x', 0)):
        plays = nbr_overwrite(q, q.cutoff - 3, q.cutoff, payout)
        h2 = tuple(vh) + tuple(plays)
        got = model().predict([q], h2)[q.query_id]
        row[tag] = {'neighbour_plays_overwritten': len(plays),
                    'nbr_symbols_at_cutoff': mtd_inputs(q, q.cutoff, mtd_index(h2))[8:],
                    'max_abs_delta': float(np.max(np.abs(np.asarray(got) - np.asarray(base)))),
                    'changed': bool(np.max(np.abs(np.asarray(got) - np.asarray(base))) > 0)}
    probe.append(row)
report['mtd_neighbour_power'] = {
    'probes': len(probe),
    'all_win_changed': sum(1 for r in probe if r['all_win_20x']['changed']),
    'all_loss_changed': sum(1 for r in probe if r['all_loss_0x']['changed']),
    'has_power': all(r['all_win_20x']['changed'] and r['all_loss_0x']['changed'] for r in probe),
    'detail': probe}
print(f"MTD neighbour power: win {report['mtd_neighbour_power']['all_win_changed']}/{len(probe)}, "
      f"loss {report['mtd_neighbour_power']['all_loss_changed']}/{len(probe)}, "
      f"has_power={report['mtd_neighbour_power']['has_power']}", flush=True)

# ---------- 2. observable() neighbour power (two_head / TabPFN share this) ------
es = examples(vq, vidx)
priors = {q.episode_id: prior for q in vq}
X0, names = matrix(es, vidx, priors)
_, H0 = hmm_summary(population, es)
nbr_cols = [i for i, n in enumerate(names) if n.startswith('other_recent')]
det = []
for tag, code in (('all_win_20x', PAYOUTS.index(20)), ('all_loss_0x', PAYOUTS.index(0))):
    for q in vq[:4]:
        idx2 = {k: dict(v) for k, v in vidx.items()}
        n = 0
        for (e, m) in list(idx2):
            if e != q.episode_id or m == q.machine_id:
                continue
            for t in list(idx2[(e, m)]):
                if q.cutoff - 16 <= t < q.cutoff:
                    idx2[(e, m)][t] = code
                    n += 1
        es2 = examples(vq, idx2)
        X2, _ = matrix(es2, idx2, priors)
        _, H2 = hmm_summary(population, es2)
        rows = [i for i, e in enumerate(es) if e['q'].query_id == q.query_id]
        det.append({'tag': tag, 'query_id': q.query_id, 'neighbour_ticks_overwritten': n,
                    'own_row_other_recent_changed':
                        bool(not np.array_equal(X0[np.ix_(rows, nbr_cols)], X2[np.ix_(rows, nbr_cols)])),
                    'own_row_other_recent_max_delta':
                        float(np.max(np.abs(X0[np.ix_(rows, nbr_cols)] - X2[np.ix_(rows, nbr_cols)]))),
                    'own_row_hmm_changed': bool(not np.array_equal(H0[rows], H2[rows])),
                    'table_max_delta': float(np.max(np.abs(X0 - X2)))})
report['observable_neighbour_power'] = {
    'neighbour_columns': [names[i] for i in nbr_cols],
    'probes': len(det),
    'probes_that_changed_own_neighbour_columns': sum(1 for d in det if d['own_row_other_recent_changed']),
    'has_power': all(d['own_row_other_recent_changed'] for d in det),
    'hmm_summary_responds_to_neighbours': any(d['own_row_hmm_changed'] for d in det),
    'detail': det}
print(f"observable() neighbour power: "
      f"{report['observable_neighbour_power']['probes_that_changed_own_neighbour_columns']}/{len(det)} "
      f"changed, has_power={report['observable_neighbour_power']['has_power']}; "
      f"HMM summary responds to neighbours={report['observable_neighbour_power']['hmm_summary_responds_to_neighbours']}",
      flush=True)
report['seconds'] = time.perf_counter() - t0
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print(f"done {report['seconds']:.1f}s")
