"""Full-vector stacking and head recombination over archived forecasts only.

PREREGISTERED PROTOCOL, fixed before any held-fold number below was read.

No new features, no new model family, no new external signal. Every input is an archived
five-probability validation forecast already produced in this project. The two PRIVILEGED
arrays (true-r substitution and the memory envelope) are EXCLUDED from every candidate
pool, because they read the evaluation targets as forecasts and are diagnostics only.

Three arms, each with all selection and weight fitting done inside the outer folds:

  A  best_single        the archived forecast with the lowest loss on each fold's FITTING
                        episodes, applied to its held fold. This is the honest "best
                        single model" comparator: picking the best by its held-fold score
                        would be selection on the evaluation data.
  B  head_recombination q and r are separable heads of p = (1-q, q*r). For each outer fold
                        the (q-source, r-source) pair minimising the official weighted loss
                        on the FITTING episodes is chosen and applied to the held fold.
                        Exploits that the best q and the best r were produced by different
                        archived candidates.
  C  convex_stacking    convex weights over a pool of archived forecasts, fitted on the
                        fitting episodes against the exact official weighted squared error
                        (a proper rule, matched to the evaluation loss, not a surrogate).
                        The pool size K is chosen by inner complete-episode folds from a
                        predeclared grid, because unconstrained stacking over ~140
                        candidates on 72 fitting episodes would overfit the weights.

Comparator: the protected candidate at 53.532918. Advancement requires held-fold points to
improve AND the paired whole-episode interval against the protected forecast to lie below
zero, the same rule used throughout; it is not relaxed after seeing results.

Blum and Hardt's caution applies with full force here: these 96 episodes have been reused
for the whole project, and stacking is exactly the kind of step that can fit the reuse
rather than the signal. Fold-internal selection limits but does not remove that.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import glob
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

PRIVILEGED = ('PRIVILEGED', 'privileged_envelope', 'true_r')
# This script's OWN outputs must never enter its pool. Re-stacking on previous held-fold
# outputs is a self-feedback loop that inflates the score on these same 96 episodes by
# construction - the Ladder effect, generated internally rather than by a leaderboard.
SELF_OUTPUT = 'artifacts/stacking/'
KGRID = (1, 2, 3, 5, 8, 12)
TOPQ, TOPR = 5, 5
PATTERNS = ('artifacts/*/validation_submission.csv', 'artifacts/*/predictions/*.csv',
            'artifacts/win_head/predictions/*.csv', 'artifacts/phase2/predictions/oof_*.csv',
            'artifacts/final/validation_predictions.csv', 'artifacts/two_head_01/validation.csv',
            'artifacts/mtd_pairs_01/validation_*.csv', 'artifacts/dense_catboost_01/validation_*.csv',
            'artifacts/shared_state_01/validation.csv', 'artifacts/tabpfn/predictions/*.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protected', type=Path,
                        default=ROOT / 'artifacts/win_head_release_exact/validation_submission.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/stacking')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    scoring = scoring_inputs(args.data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    want = set(ids)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref_arr = np.array([reference[q] for q in ids])
    episodes = set(ep.tolist())
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']

    # ---- load the archived pool, excluding privileged arrays and duplicates ----
    seen, names, arrs = {}, [], []
    excluded = []
    for path in sorted({f for pat in PATTERNS for f in glob.glob(pat)}):
        if any(tag in path for tag in PRIVILEGED) or path.startswith(SELF_OUTPUT):
            excluded.append(path)
            continue
        if 'reference' in path or 'diagnostics' in path:
            continue
        try:
            d = read_submission(path)
        except Exception:
            continue
        if set(d) != want:
            continue
        a = np.array([d[i] for i in ids])
        if not (np.isfinite(a).all() and a.min() >= -1e-12 and abs(a.sum(1) - 1).max() < 1e-6):
            continue
        key = hash(a.round(12).tobytes())
        if key in seen:
            continue
        seen[key] = path
        names.append(path)
        arrs.append(a / a.sum(axis=1, keepdims=True))
    P = np.stack(arrs)                                   # (m, n, 5)
    print(f'pool: {len(names)} distinct archived forecasts; excluded {len(excluded)} privileged',
          flush=True)

    prot = read_submission(args.protected)
    prot = np.array([prot[q] for q in ids])
    prot = prot / prot.sum(axis=1, keepdims=True)

    def row_weights(mask):
        w = np.zeros(len(ids))
        es = {x for i, x in enumerate(ep) if mask[i]}
        for c in sorted(set(groups.values())):
            ces = {e for i, e in enumerate(ep) if e in es and groups[ids[i]] == c}
            if not ces:
                continue
            b = np.mean([np.mean(np.sum((ref_arr[ep == e] - truth[ep == e]) ** 2, axis=1))
                         for e in sorted(ces)])
            for e in ces:
                w[ep == e] = 1 / (4 * len(ces) * sum(ep == e) * max(b, 1e-9))
        return w

    def wloss(p, w):
        return float(np.sum(w * np.sum((p - truth) ** 2, axis=1)))

    def mapping(a):
        return dict(zip(ids, a.tolist()))

    def official(a):
        a = a / a.sum(axis=1, keepdims=True)
        return score(mapping(a), scoring, episodes)

    # separable heads of every pool member
    Q = 1.0 - P[:, :, 0]                                 # (m, n)
    R = P[:, :, 1:] / np.maximum(Q, 1e-12)[:, :, None]
    R = R / R.sum(axis=2, keepdims=True)

    held = {'best_single': np.empty_like(prot), 'head_recombination': np.empty_like(prot),
            'convex_stacking': np.empty_like(prot), 'augmented_stacking': np.empty_like(prot)}
    prov = []
    for fold in range(4):
        outer = np.array([folds[x] == fold for x in ep])
        fit = ~outer
        w = row_weights(fit)

        # A: best single by FITTING loss
        losses = np.array([wloss(P[i], w) for i in range(len(names))])
        ia = int(np.argmin(losses))
        held['best_single'][outer] = P[ia][outer]

        # B: best (q, r) pair by FITTING loss, using the exact quadratic form
        A_ = 1.0 + np.sum(R ** 2, axis=2)                # (m, n) per r-source
        Bc = (1.0 - truth[:, 0])[None, :] + np.einsum('mnc,nc->mn', R, truth[:, 1:])
        # loss(q_i, r_j) = sum_n w_n [ A_j q_i^2 - 2 B_j q_i ] + const(w)
        q2 = (Q ** 2)                                     # (m, n)
        M = (q2 * w) @ A_.T - 2.0 * ((Q * w) @ Bc.T)      # (m_q, m_r)
        iq, ir = np.unravel_index(int(np.argmin(M)), M.shape)
        rec = np.column_stack((1 - Q[iq], Q[iq][:, None] * R[ir]))
        rec = rec / rec.sum(axis=1, keepdims=True)
        held['head_recombination'][outer] = rec[outer]

        # D: augment the pool with head recombinations, formed ONLY from fitting-fold
        # rankings, then stack over the augmented pool. Separable heads mean the archived
        # pool does not contain these products, so this adds genuinely new members without
        # any new model family or feature.
        qbest = np.argsort(M.min(axis=1))[:TOPQ]
        rbest = np.argsort(M.min(axis=0))[:TOPR]
        extra, extra_names = [], []
        for i in qbest:
            for j in rbest:
                v = np.column_stack((1 - Q[i], Q[i][:, None] * R[j]))
                extra.append(v / v.sum(axis=1, keepdims=True))
                extra_names.append(f'recomb[q={Path(names[i]).stem}|r={Path(names[j]).stem}]')
        PA = np.concatenate((P, np.stack(extra)))
        NA = names + extra_names

        # C: convex stacking, pool size K chosen by inner complete-episode folds
        order = np.argsort(losses)
        fit_eps = sorted({x for i, x in enumerate(ep) if fit[i]})
        inner = {e: i % 3 for i, e in enumerate(fit_eps)}

        def stack_weights(mask, idx, pool=None):
            ww = row_weights(mask)
            sub = (P if pool is None else pool)[idx]
            G = np.einsum('n,ind,jnd->ij', ww, sub - truth, sub - truth)
            res = minimize(lambda x: float(x @ G @ x), np.ones(len(idx)) / len(idx),
                           jac=lambda x: 2 * G @ x, method='SLSQP', bounds=[(0, 1)] * len(idx),
                           constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1},
                           options={'ftol': 1e-14, 'maxiter': 600})
            x = np.clip(res.x, 0, None)
            return x / max(x.sum(), 1e-12)

        kscore = {}
        for K in KGRID:
            tot = 0.0
            for ib in range(3):
                itr = np.array([fit[i] and inner.get(ep[i], -1) != ib for i in range(len(ids))])
                ihe = np.array([fit[i] and inner.get(ep[i], -1) == ib for i in range(len(ids))])
                iloss = np.array([wloss(P[i], row_weights(itr)) for i in range(len(names))])
                idx = np.argsort(iloss)[:K]
                x = stack_weights(itr, idx)
                tot += wloss(np.einsum('i,ind->nd', x, P[idx]), row_weights(ihe))
            kscore[K] = tot
        K = min(kscore, key=lambda k: kscore[k])
        idx = order[:K]
        x = stack_weights(fit, idx)
        st = np.einsum('i,ind->nd', x, P[idx])
        held['convex_stacking'][outer] = (st / st.sum(axis=1, keepdims=True))[outer]

        # E: stacking over the AUGMENTED pool, K again chosen by inner folds
        ascore = {}
        for K2 in KGRID:
            tot = 0.0
            for ib in range(3):
                itr = np.array([fit[i] and inner.get(ep[i], -1) != ib for i in range(len(ids))])
                ihe = np.array([fit[i] and inner.get(ep[i], -1) == ib for i in range(len(ids))])
                wi = row_weights(itr)
                il = np.array([wloss(PA[i], wi) for i in range(len(NA))])
                idx2 = np.argsort(il)[:K2]
                x2 = stack_weights(itr, idx2, PA)
                tot += wloss(np.einsum('i,ind->nd', x2, PA[idx2]), row_weights(ihe))
            ascore[K2] = tot
        K2 = min(ascore, key=lambda k: ascore[k])
        aloss = np.array([wloss(PA[i], w) for i in range(len(NA))])
        idx2 = np.argsort(aloss)[:K2]
        x2 = stack_weights(fit, idx2, PA)
        sa = np.einsum('i,ind->nd', x2, PA[idx2])
        held['augmented_stacking'][outer] = (sa / sa.sum(axis=1, keepdims=True))[outer]

        prov.append({'fold': fold, 'best_single': names[ia],
                     'recombination_q': names[iq], 'recombination_r': names[ir],
                     'stack_K': K, 'stack_inner_scores': {str(k): v for k, v in kscore.items()},
                     'stack_members': [names[i] for i in idx],
                     'stack_weights': x.tolist(),
                     'augmented_K': K2, 'augmented_members': [NA[i] for i in idx2],
                     'augmented_weights': x2.tolist(),
                     'augmented_inner_scores': {str(k): v for k, v in ascore.items()}})
        print(f"  fold {fold}: single={Path(names[ia]).parent.name}/{Path(names[ia]).stem} | "
              f"q={Path(names[iq]).stem} r={Path(names[ir]).stem} | K={K}", flush=True)

    res = {'protected': official(prot), **{k: official(v) for k, v in held.items()}}
    pairs = {k: paired_bootstrap(mapping(v / v.sum(axis=1, keepdims=True)), mapping(prot),
                                 scoring, episodes) for k, v in held.items()}
    qualified = {k: res[k]['technical_points'] - res['protected']['technical_points']
                 for k in held
                 if res[k]['technical_points'] > res['protected']['technical_points']
                 and pairs[k]['delta_loss_ratio_ci95'][1] < 0}
    best = (max(qualified, key=lambda k: qualified[k]) if qualified
            else max(held, key=lambda k: res[k]['technical_points']))
    gain = res[best]['technical_points'] - res['protected']['technical_points']
    lo, hi = pairs[best]['delta_loss_ratio_ci95']

    report = {'protocol': __doc__, 'pool_size': len(names), 'excluded_privileged': excluded,
              'results': {k: {'points': v['technical_points'], 'R': v['mean_loss_ratio'],
                              'by_condition': {c: x['loss_ratio'] for c, x in v['by_condition'].items()}}
                          for k, v in res.items()},
              'gains_over_protected': {k: res[k]['technical_points'] - res['protected']['technical_points']
                                       for k in held},
              'paired_vs_protected': {k: {'ci95_delta_R': v['delta_loss_ratio_ci95'],
                                          'units': 'change in normalized error R; negative '
                                                   'favours the candidate'}
                                      for k, v in pairs.items()},
              'fold_provenance': prov, 'best_arm': best,
              'qualified_arms': {k: v for k, v in qualified.items()},
              'advancement_rule': 'applied PER ARM: points improve AND paired interval upper '
                                  'bound < 0',
              'advanced': bool(gain > 0 and hi < 0),
              'exposure': 'Conditional exploratory development; these 96 episodes underpin the '
                          'entire project and stacking can fit that reuse.',
              'seconds': time.perf_counter() - start}
    for k, v in held.items():
        write_submission(args.out / 'predictions' / f'oof_{k}.csv',
                         mapping(v / v.sum(axis=1, keepdims=True)))
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    conds = sorted(res['protected']['by_condition'])
    print(f"\n{'arm':<24}{'points':>12}{'R':>10}  " + '  '.join(f'{c[:9]:>9}' for c in conds))
    for k in ('protected', 'best_single', 'head_recombination', 'convex_stacking',
              'augmented_stacking'):
        v = res[k]
        print(f"{k:<24}{v['technical_points']:>12.6f}{v['mean_loss_ratio']:>10.6f}  "
              + '  '.join(f"{v['by_condition'][c]['loss_ratio']:>9.4f}" for c in conds))
    print()
    for k, v in report['paired_vs_protected'].items():
        ci = v['ci95_delta_R']
        print(f"   {k:<22} dR 95% [{ci[0]:+.5f}, {ci[1]:+.5f}]"
              + ('  significant' if ci[1] < 0 else '  overlaps zero'))
    print(f"\nbest arm {best}: gain {gain:+.6f}")
    print(f"ADVANCEMENT: {'GRANTED' if (gain > 0 and hi < 0) else 'WITHHELD'}")
    print(f"wrote {args.out / 'evaluation.json'} in {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
