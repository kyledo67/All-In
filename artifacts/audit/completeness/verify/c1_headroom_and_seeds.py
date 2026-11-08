"""Completeness-critic probes: (1) where the remaining error lives once the win head's
win mass is in place, (2) the seed spread of the raw TabPFN component.

Every number is public-validation over the same 96 repeatedly-inspected episodes.
Nothing here is confirmed or independent.
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from slot_forecasting.io import read_submission
from slot_model.official_score import score, scoring_inputs, paired_bootstrap

DATA = ROOT / 'main-participants-60-40/public'
scoring = scoring_inputs(DATA)
targets, groups, reference = scoring
ids = sorted(reference)
episodes = sorted({t.episode_id for t in targets})
T = {t.query_id: np.asarray(t.probabilities) for t in targets}
truth = np.array([T[q] for q in ids])

def load(p):
    s = read_submission(ROOT / p)
    return np.array([s[q] for q in ids])

def pts(arr):
    return score({q: arr[i] for i, q in enumerate(ids)}, scoring, episodes)['technical_points']

cand = load('artifacts/audit/oof_candidate_raw.csv')
win  = load('artifacts/win_head/predictions/oof_win_head.csv')

def split(v):
    q = 1.0 - v[:, 0]
    tail = v[:, 1:].sum(1)
    r = np.where(tail[:, None] > 0, v[:, 1:] / np.maximum(tail, 1e-300)[:, None], 0.25)
    return q, r

def compose(q, r):
    out = np.empty((len(q), 5)); out[:, 0] = 1.0 - q; out[:, 1:] = q[:, None] * r
    return out / out.sum(1, keepdims=True)

q_cand, r_cand = split(cand)
q_win,  r_win  = split(win)
q_true, r_true = split(truth)
# r is undefined where the true win mass is zero: keep the model's own r there.
zero = truth[:, 1:].sum(1) <= 0
r_true_safe = np.where(zero[:, None], r_cand, r_true)

out = {
 'A_candidate_three_component': pts(cand),
 'B_win_head_headline': pts(win),
 'C_win_head_q_plus_ORACLE_r': pts(compose(q_win, r_true_safe)),
 'D_candidate_q_plus_ORACLE_r': pts(compose(q_cand, r_true_safe)),
 'E_ORACLE_q_plus_win_head_r': pts(compose(q_true, r_win)),
 'F_ORACLE_q_plus_candidate_r': pts(compose(q_true, r_cand)),
 'G_perfect': pts(truth),
 'queries_with_zero_true_win_mass': int(zero.sum()),
 'r_identical_win_vs_candidate_maxabs': float(np.max(np.abs(r_win - r_cand))),
}
out['remaining_q_headroom_from_win_head'] = out['E_ORACLE_q_plus_win_head_r'] - out['B_win_head_headline']
out['remaining_r_headroom_from_win_head'] = out['C_win_head_q_plus_ORACLE_r'] - out['B_win_head_headline']

# ---- TabPFN seed spread -------------------------------------------------
seeds = {}
for s in ('20261108', '20261109'):
    v = load(f'artifacts/tabpfn/predictions/tabpfn_hmm_{s}.csv')
    seeds[s] = v
    out[f'tabpfn_seed_{s}_points'] = pts(v)
avg = (seeds['20261108'] + seeds['20261109']) / 2.0
out['tabpfn_seed_average_points'] = pts(avg)
out['tabpfn_oof_tabpfn_points'] = pts(load('artifacts/tabpfn/predictions/oof_tabpfn.csv'))
out['tabpfn_seed_spread_points'] = abs(out['tabpfn_seed_20261108_points'] - out['tabpfn_seed_20261109_points'])
out['tabpfn_seed_maxabs_prob_diff'] = float(np.max(np.abs(seeds['20261108'] - seeds['20261109'])))
out['tabpfn_seed_rms_prob_diff'] = float(np.sqrt(np.mean((seeds['20261108'] - seeds['20261109']) ** 2)))
# What the candidate scores if one TabPFN seed is swapped for the other, weights unchanged.
# per-condition comparison candidate vs frozen blend
blend = load('artifacts/phase2/predictions/oof_ensemble.csv')
out['A_frozen_blend_points'] = pts(blend)
per = {}
ref_arr = np.array([reference[q] for q in ids])
ep = np.array([next(t.episode_id for t in targets if t.query_id == q) for q in ids])
for cond in sorted(set(groups.values())):
    m = np.array([groups[q] == cond for q in ids])
    es = sorted(set(ep[m]))
    def cerr(v):
        return float(np.mean([np.mean(np.sum((v[ep == e] - truth[ep == e]) ** 2, axis=1)) for e in es]))
    b = cerr(ref_arr)
    per[cond] = {'B_c': b, 'cand_ratio': cerr(cand) / b, 'blend_ratio': cerr(blend) / b,
                 'win_ratio': cerr(win) / b}
out['per_condition'] = per
out['candidate_beats_blend_in_all_four_conditions'] = all(v['cand_ratio'] < v['blend_ratio'] for v in per.values())
out['win_beats_candidate_in_all_four_conditions'] = all(v['win_ratio'] < v['cand_ratio'] for v in per.values())

# paired bootstrap win head vs candidate, organizer helper, same seed as the repo uses
a = {q: win[i] for i, q in enumerate(ids)}
b = {q: cand[i] for i, q in enumerate(ids)}
out['paired_bootstrap_win_vs_candidate'] = paired_bootstrap(a, b, scoring, episodes)
c = {q: cand[i] for i, q in enumerate(ids)}
d = {q: blend[i] for i, q in enumerate(ids)}
out['paired_bootstrap_candidate_vs_blend'] = paired_bootstrap(c, d, scoring, episodes)

dest = ROOT / 'artifacts/audit/completeness/verify/c1_headroom_and_seeds.json'
dest.write_text(json.dumps(out, indent=2, sort_keys=True))
print(json.dumps(out, indent=2, sort_keys=True))
