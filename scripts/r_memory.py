"""DECISION RECORD and the single intervention: give r its own memory.

STRONGEST SUPPORTED EXPLANATION OF THE ~53/60 PLATEAU
The official R decomposes exactly (residual 1.39e-17) into
    A = 0.048108  38.8%  error that cannot be removed by changing q at all, given this r
    B = 0.041958  33.8%  the seven memory experts do not span the q the loss wants
    C = 0.034015  27.4%  error recoverable by choosing better among the current experts
Term C is precisely the capacity the privileged envelope advertised, and the causal
fixed-share experiment recovered none of it (51.71 against 53.38, monotonically worse in
the share rate). So the plateau is NOT mainly a weighting problem. The largest single
share, A, is unreachable through q by construction and moves only if r improves, and the
refreshed oracle gives true r a +2.3366 headroom on the current candidate.

MEASUREMENTS THAT SELECT THE INTERVENTION
  - r was verified frozen across every memory policy (max |dr| = 2.22e-16), so the
    earlier q attribution was sound and no q/r factorial is needed.
  - r nevertheless INHERITS q's selected half-life inside dirichlet_proportions. q's
    selector overwhelmingly picks very short memory (H=8 for most machines, and a global
    H=16 beats per-machine selection), so r is being estimated from a decayed positive
    mass whose median is only 8.8 - while a positive outcome is roughly 40% of ticks, so
    most of the machine's positive history is discarded by a half-life chosen for q.
  - short_history carries both the largest A (0.0568) and the largest B (0.0606).

COMPETING EXPLANATIONS STILL CONSISTENT WITH THE DATA
  - Term A may be dominated by genuine hidden-state uncertainty rather than by r's
    estimation error. The decomposition cannot separate those: A is defined relative to
    THIS r, not to the best possible r. If A is mostly hidden state, this intervention
    will produce little.
  - Term B could reflect a missing q representation rather than missing memory diversity.
  - The within-parity residual question (++ vs -- and +- vs -+) was not measured, so an
    omitted-structure explanation is not excluded.

THE SINGLE INTERVENTION
Give r an independently selected memory, holding q, the Dirichlet prior (tau=32, the
per-type population centre) and the blend form fixed. Candidates for the r half-life:
the existing grid 8, 16, 32, 64, 128, 256, and no decay. The deployed policy (inherit
q's half-life) is the control, and no-decay is the hypothesis-driven extreme.

FITTING PROTOCOL
The r half-life and the blend weight are selected on each outer fold's FITTING episodes
by the exact official weighted five-vector objective, then applied to the held fold;
held-fold forecasts are concatenated and scored once. q is byte-identical throughout.

ADVANCEMENT RULE, fixed before seeing any result below: the policy replaces the deployed
r only if held-fold points improve AND the paired whole-episode interval against the
deployed forecast lies below zero. This is the same rule that withheld global H=16, and
it is not relaxed here.

STOPPING RULE: one batch. If it fails, the run ends with a quantified negative diagnosis
and no new candidate.

Exposure: 96 repeatedly inspected validation episodes; exploratory research round.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import win_head as W
from slot_model import win_head_infer as WI
from slot_model.features import causal_index
from slot_model.official_score import paired_bootstrap, score, scoring_inputs
from slot_forecasting.io import load_inputs, read_submission, write_submission
from scripts.win_head_train import build

R_HALF_LIVES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, None)
LABELS = ['inf' if h is None else str(int(h)) for h in R_HALF_LIVES]
BLEND_GRID = np.linspace(0.0, 1.0, 51)


def positive_counts(ticks, cutoff, half_life):
    counts = np.zeros(4)
    for tick, value in ticks.items():
        if tick >= cutoff or value <= 0:
            continue
        w = 1.0 if not half_life else 2.0 ** (-(cutoff - 1 - tick) / half_life)
        counts[value - 1] += w
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT / 'artifacts/win_head_release_dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/r_memory')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    model = args.package / 'code/model'
    tau, prior, w_dep, dir_payload = WI.load_dirichlet_r(model / 'win_head/dirichlet_r_deploy.json')
    kappa, prior_mean, log_prior, _ = WI.load_beta_templates(
        model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    # q is held byte-identical to the deployed candidate.
    dep = read_submission(args.package / 'validation_submission.csv')
    dep = np.array([dep[q] for q in ids])
    q_fixed = 1.0 - dep[:, 0]

    # pooled two-head r, the component the Dirichlet estimate is blended into
    th = read_submission(ROOT / 'artifacts/two_head_01/validation.csv')
    th = np.array([th[q] for q in ids])
    r_pool = th[:, 1:] / np.maximum(1.0 - th[:, 0], 1e-12)[:, None]
    r_pool /= r_pool.sum(axis=1, keepdims=True)

    vq, vh = load_inputs(args.data / 'validation')
    vindex = causal_index(vq, vh)
    vent = build(vq, vindex)
    vmeta = {q.query_id: (q.episode_id, q.machine_id, q.cutoff, q.machine_type) for q in vq}

    # Dirichlet r under each candidate memory, plus the deployed inherited-memory version.
    conj = {lab: np.empty((len(ids), 4)) for lab in LABELS}
    conj['inherited'] = np.empty((len(ids), 4))
    support = {lab: np.empty(len(ids)) for lab in list(LABELS) + ['inherited']}
    for i, qid in enumerate(ids):
        e = vent[qid]
        ep_, mid, cut, typ = vmeta[qid]
        ticks = vindex[(ep_, mid)]
        m = np.asarray(prior.get(typ, np.full(4, 0.25)))
        inherited_h = None
        if e['star'] is not None and len(e['y']):
            _, mix = W.predict_q_averaged(
                {h: beta[h].get(typ, np.zeros(W.DIM)) for h in beta},
                e['phi'], e['y'], e['used'], cut, e['star'], lam, adapt='full')
            p = mix['selected_half_life']
            inherited_h = None if p == 0.0 else p
        for lab, h in list(zip(LABELS, R_HALF_LIVES)) + [('inherited', inherited_h)]:
            c = positive_counts(ticks, cut, h)
            support[lab][i] = c.sum()
            conj[lab][i] = (tau * m + c) / (tau + c.sum())
    for lab in conj:
        conj[lab] /= conj[lab].sum(axis=1, keepdims=True)

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def compose(r):
        out = np.column_stack((1.0 - q_fixed, q_fixed[:, None] * r))
        return out / out.sum(axis=1, keepdims=True)

    def official(a, es=episodes):
        return score(mapping(a), scoring, es)

    def row_weights(es):
        w = np.zeros(len(ids))
        for c in sorted(set(groups.values())):
            ces = {x for i, x in enumerate(ep) if x in es and groups[ids[i]] == c}
            b = np.mean([np.mean(np.sum((ref_arr[ep == x] - truth[ep == x]) ** 2, axis=1)) for x in ces])
            for x in ces:
                w[ep == x] = 1 / (4 * len(ces) * sum(ep == x) * max(b, 1e-9))
        return w

    def blend(r_conj, g):
        r = (1 - g) * r_pool + g * r_conj
        return r / r.sum(axis=1, keepdims=True)

    def wloss(r, w):
        return float(np.sum(w * np.sum((compose(r) - truth) ** 2, axis=1)))

    # ---- control: deployed inherited memory at the frozen deployment weight ----
    control = official(compose(blend(conj['inherited'], w_dep)))

    # ---- intervention: r half-life AND blend weight selected fitting-side ----
    held = np.empty((len(ids), 4))
    picks = []
    for fold in range(4):
        m = np.array([folds[x] == fold for x in ep])
        fit = {x for x in episodes if folds[x] != fold}
        w = row_weights(fit)
        best = None
        for lab in LABELS:
            for g in BLEND_GRID:
                v = wloss(blend(conj[lab], g), w)
                if best is None or v < best[0]:
                    best = (v, lab, float(g))
        picks.append({'fold': fold, 'r_half_life': best[1], 'blend_weight': best[2]})
        held[m] = blend(conj[best[1]], best[2])[m]
    intervention = official(compose(held))

    # per-memory diagnostics at the frozen deployment weight, for interpretation only
    per_memory = {lab: official(compose(blend(conj[lab], w_dep)))['technical_points']
                  for lab in LABELS}
    paired = paired_bootstrap(mapping(compose(held)),
                              mapping(compose(blend(conj['inherited'], w_dep))), scoring, episodes)
    gain = intervention['technical_points'] - control['technical_points']
    lo, hi = paired['delta_loss_ratio_ci95']
    advanced = bool(gain > 0 and hi < 0)

    report = {
        'decision_record': __doc__,
        'control_deployed_inherited_memory': {
            'points': control['technical_points'], 'R': control['mean_loss_ratio'],
            'by_condition': {c: v['loss_ratio'] for c, v in control['by_condition'].items()}},
        'intervention_independent_r_memory': {
            'points': intervention['technical_points'], 'R': intervention['mean_loss_ratio'],
            'by_condition': {c: v['loss_ratio'] for c, v in intervention['by_condition'].items()}},
        'gain': gain, 'selected_per_fold': picks,
        'per_memory_at_frozen_weight': per_memory,
        'positive_support_mass': {lab: {'median': float(np.median(support[lab])),
                                        'p10': float(np.percentile(support[lab], 10))}
                                  for lab in list(LABELS) + ['inherited']},
        'paired_vs_control': paired,
        'advancement_rule': 'held-fold points improve AND paired interval upper bound < 0',
        'advanced': advanced,
        'q_byte_identical_to_deployed': True,
        'exposure': 'Repeatedly inspected public validation; exploratory research round.',
        'seconds': time.perf_counter() - start}
    write_submission(args.out / 'predictions' / 'oof_r_memory.csv', mapping(compose(held)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(control['by_condition'])
    print(f"{'candidate':<38}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for nm, v in (('control: r inherits q memory', control),
                  ('intervention: independent r memory', intervention)):
        print(f"{nm:<38}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print('\nr memory at the frozen deployment weight (interpretation only):')
    for lab in LABELS:
        print(f"   H={lab:<5} {per_memory[lab]:12.6f}   positive mass median "
              f"{np.median(support[lab]):7.2f}")
    print(f"   inherited          positive mass median {np.median(support['inherited']):7.2f}")
    print(f"\nselected per fold: {[(p['r_half_life'], round(p['blend_weight'],2)) for p in picks]}")
    print(f"gain {gain:+.6f}; paired 95% [{lo:+.5f}, {hi:+.5f}]")
    print(f"ADVANCEMENT: {'GRANTED' if advanced else 'WITHHELD'} under the preregistered rule")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
