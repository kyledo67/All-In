"""Primary experiment: is the transferred Round-1 template prior stale for Round 2?

Arms, all sharing one cached evidence matrix so only the prior changes:
  A  the actual packaged prior (exact control; reproduces the staged forecast to 1e-16)
  B  family masses preserved, pair mass spread UNIFORMLY over the 28 lag pairs
  C  family masses preserved, pair mass from ROUND-2 TRAIN histories via the project's own
     estimator (argmax counts under a neutral pair prior, add-one smoothed, normalised),
     interpolated with the source pair preference as pi = (1-l)*pi_src + l*pi_adapt,
     l in {0, 0.5, 1} selected on the fitting side only.

Frozen across every arm, verified from arrays not filenames: the 57-template dictionary, the
per-query selected half-life, kappa and the type prior means, eligible ticks and decay
weights, hard selection, the full r path, and the composition operator.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.r2_common import (R2, CONDITIONS, targets, queries, conditions, score,
                               reference_episode_errors, episode_errors, paired_bootstrap)
from scripts.r2_prior_cache import select, TEMPLATES

EXP = ROOT / 'artifacts/r2_prior_experiment'
PAIRS = tuple((a, b) for a in range(1, 9) for b in range(a + 1, 9))
FULL_I = [1 + 2 * j for j in range(28)]
PAR_I = [2 + 2 * j for j in range(28)]


def load(split):
    z = np.load(EXP / f'cache_{split}.npz', allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_prior(m_null, m_full, m_par, pi):
    p = np.empty(57)
    p[0] = m_null
    p[FULL_I] = m_full * pi
    p[PAR_I] = m_par * pi
    assert abs(p.sum() - 1) < 1e-12 and (p > 0).all()
    return np.log(p)


def pair_of(index):
    """template index -> pair index, or -1 for null."""
    return -1 if index <= 0 else (index - 1) // 2


def estimate_pi(cache, log_prior, smoothing=1.0):
    """The project's own estimator: argmax counts, balanced per episode/machine, smoothed."""
    _, pick = select(cache, log_prior)
    eps = cache['episode']
    uniq = sorted(set(eps.tolist()))
    w = np.zeros(len(pick))
    for e in uniq:
        s = eps == e
        w[s] = 1.0 / (len(uniq) * s.sum())
    counts = np.zeros(28)
    for i, t in enumerate(pick):
        j = pair_of(int(t))
        if j >= 0:
            counts[j] += w[i] * len(pick)        # scale back to machine units
    pi = (counts + smoothing) / (counts.sum() + smoothing * 28)
    return pi, counts, pick


def inner_blocks(fit_eps, cond_of, k=3, seed=4242):
    rng = np.random.default_rng(seed)
    assign = {}
    for c in CONDITIONS:
        e = sorted(x for x in fit_eps if cond_of[x] == c)
        for pos, j in enumerate(rng.permutation(len(e))):
            assign[e[j]] = pos % k
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=EXP / 'arms.json')
    args = ap.parse_args()

    st = json.loads((EXP / 'prior_structure.json').read_text())
    m_null, m_full, m_par = st['m_null'], st['m_full'], st['m_parity']
    pi_src = np.load(EXP / 'pi_source.npy')
    cv, ct = load('validation'), load('train')
    cand = [str(x) for x in cv['candidates']]
    lp_A = cv['log_prior_saved']
    lp_B = build_prior(m_null, m_full, m_par, np.full(28, 1 / 28))

    # ---- the Round-2 training diagnostic, reproduced with the current loaders ----
    print('=== Round-2 TRAIN template evidence (768 machines) ===')
    flat = np.zeros(57)
    pi_flat, cnt_flat, pick_flat = estimate_pi(ct, flat)
    pi_adapt, cnt_neu, pick_neu = estimate_pi(ct, lp_B)
    _, pick_srcprior = select(ct, lp_A)
    for tag, pick in (('flat prior over all 57', pick_flat),
                      ('neutral pair prior (arm B)', pick_neu),
                      ('SOURCE Round-1 prior (arm A)', pick_srcprior)):
        pc = np.zeros(28); nnull = 0
        for t in pick:
            j = pair_of(int(t))
            if j < 0: nnull += 1
            else: pc[j] += 1
        top = np.argsort(-pc)[:4]
        print(f'  {tag:30s} null={nnull:4d}  ' +
              '  '.join(f'({PAIRS[k][0]},{PAIRS[k][1]})={int(pc[k])}' for k in top))
    chg = int((pick_neu != pick_srcprior).sum())
    print(f'  templates changed by reintroducing the source prior: {chg} of {len(pick_neu)}')
    print(f'  pi_adapt top: ' + '  '.join(
        f'({PAIRS[k][0]},{PAIRS[k][1]})={pi_adapt[k]:.4f}' for k in np.argsort(-pi_adapt)[:5]))
    print(f'  effective pairs: source {1/np.sum(pi_src**2):.2f}  adaptation '
          f'{1/np.sum(pi_adapt**2):.2f}  (of 28)')

    # ---- evaluation scaffolding ----
    tg = targets(R2); qm = queries(R2, 'validation'); cond = conditions(R2, 'validation')
    ids = [str(x) for x in cv['ids']]
    assert set(ids) == set(tg)
    T = np.array([tg[i] for i in ids])
    refep = reference_episode_errors(R2)
    cond_of_q = {i: cond[i] for i in ids}
    ep_of = cv['episode']
    cond_of_ep = {}
    for i, e in zip(ids, ep_of):
        cond_of_ep[e] = cond_of_q[i]
    eps_sorted = sorted(set(ep_of.tolist()))
    rng = np.random.default_rng(777)
    fold_of_ep = {}
    for c in CONDITIONS:
        e = sorted(x for x in eps_sorted if cond_of_ep[x] == c)
        for pos, j in enumerate(rng.permutation(len(e))):
            fold_of_ep[e[j]] = pos % 4
    fold = np.array([fold_of_ep[e] for e in ep_of])
    fh = hashlib.sha256(json.dumps({k: int(v) for k, v in sorted(fold_of_ep.items())},
                                   sort_keys=True).encode()).hexdigest()[:16]
    print(f'\nouter folds: 4 x 24 complete episodes, 6 per condition per fold, hash {fh}')

    r = cv['r']; refq = cv['ref_q']; nobs = cv['nobs']
    ALPHAS = {'current_rule': lambda n: np.where(n < 128, 0.83, 0.94),
              'alpha_1': lambda n: np.ones(len(n))}

    def compose(q, alpha_fn):
        a = alpha_fn(nobs)
        b = (1 - a) * refq + a * q
        return np.column_stack([1 - b, b[:, None] * r])

    def R_of(P, mask=None):
        sel = np.ones(len(ids), bool) if mask is None else mask
        sub = [ids[k] for k in range(len(ids)) if sel[k]]
        return score({i: P[ids.index(i)] for i in sub}, {i: tg[i] for i in sub}, qm,
                     {i: cond[i] for i in sub}, refep)

    def loss(P, mask):
        """Official normalisation on a subset: mean over conditions of E[c]/B[c]."""
        return R_of(P, mask)['R']

    LAMS = [0.0, 0.5, 1.0]
    priors = {'A_source': lp_A, 'B_neutral': lp_B}
    for l in LAMS:
        priors[f'C_lam{l}'] = build_prior(m_null, m_full, m_par,
                                          (1 - l) * pi_src + l * pi_adapt)
    qs = {k: select(cv, v)[0] for k, v in priors.items()}
    picks = {k: select(cv, v)[1] for k, v in priors.items()}

    results = {}
    print('\n=== fixed-prior arms, fixed deployment alpha rule (REPLAY, not a procedure) ===')
    for k in ('A_source', 'B_neutral', 'C_lam0.0', 'C_lam0.5', 'C_lam1.0'):
        for an in ('current_rule', 'alpha_1'):
            s = R_of(compose(qs[k], ALPHAS[an]))
            tag = f'{k} + {an}'
            results[tag] = {'R': s['R'], 'points_40': 40 / (1 + s['R']),
                            'points_20': 20 / (1 + s['R']),
                            'ratios': {c: s['by_condition'][c]['loss_ratio'] for c in CONDITIONS},
                            'eval': 'fixed-prior replay'}
            print(f'  {tag:26s} R={s["R"]:.5f}  {40/(1+s["R"]):.4f}/40  ratios=' +
                  ' '.join(f'{s["by_condition"][c]["loss_ratio"]:.3f}' for c in CONDITIONS))

    # ---- nested procedure: choose lambda and alpha inside the fitting folds ----
    print('\n=== NESTED procedure: lambda and alpha chosen on the fitting side only ===')
    for name, cands in (('lambda only (alpha fixed to current rule)',
                         [(f'C_lam{l}', 'current_rule') for l in LAMS]),
                        ('lambda x alpha (predeclared 2-option alpha)',
                         [(f'C_lam{l}', a) for l in LAMS for a in ALPHAS]),
                        ('lambda x alpha, with A and B in the candidate set',
                         [(k, a) for k in ('A_source', 'B_neutral', 'C_lam0.0', 'C_lam0.5',
                                           'C_lam1.0') for a in ALPHAS])):
        P = np.zeros_like(T)
        chosen = []
        for f in range(4):
            te = fold == f
            fit_eps = [e for e in eps_sorted if fold_of_ep[e] != f]
            ib = inner_blocks(fit_eps, cond_of_ep)
            tot = {c: 0.0 for c in cands}
            for blk in range(3):
                m = np.array([fold_of_ep[e] != f and ib[e] == blk for e in ep_of])
                if m.sum() == 0:
                    continue
                for c in cands:
                    tot[c] += loss(compose(qs[c[0]], ALPHAS[c[1]]), m)
            # prefer the simpler configuration on effective ties
            best = min(tot, key=lambda c: (round(tot[c], 9), cands.index(c)))
            chosen.append(best)
            P[te] = compose(qs[best[0]], ALPHAS[best[1]])[te]
        s = R_of(P)
        results[f'NESTED: {name}'] = {
            'R': s['R'], 'points_40': 40 / (1 + s['R']), 'points_20': 20 / (1 + s['R']),
            'ratios': {c: s['by_condition'][c]['loss_ratio'] for c in CONDITIONS},
            'chosen_per_fold': [list(c) for c in chosen], 'eval': 'out-of-fold procedure'}
        print(f'  {name}')
        print(f'    R={s["R"]:.5f}  {40/(1+s["R"]):.4f}/40  ratios=' +
              ' '.join(f'{s["by_condition"][c]["loss_ratio"]:.3f}' for c in CONDITIONS))
        print(f'    chosen per fold: {chosen}')
        np.save(EXP / f'oof_{name.split()[0]}.npy', P)

    # ---- how often does the prior change the selected template, on validation ----
    print('\n=== selected-template change on the 768 validation queries ===')
    for k in ('B_neutral', 'C_lam0.5', 'C_lam1.0'):
        ch = int((picks[k] != picks['A_source']).sum())
        print(f'  A -> {k:10s} changed {ch:4d} of 768 ({100*ch/768:.1f}%)  '
              f'mean |dq| {np.abs(qs[k]-qs["A_source"]).mean():.5f}  '
              f'max {np.abs(qs[k]-qs["A_source"]).max():.5f}')
    print('\n  selected-pair frequency on validation, before -> after:')
    for k in ('A_source', 'C_lam1.0'):
        pc = np.zeros(28); nn = 0
        for t in picks[k]:
            j = pair_of(int(t))
            if j < 0: nn += 1
            else: pc[j] += 1
        print(f'    {k:10s} null={nn:3d}  ' + '  '.join(
            f'({PAIRS[q][0]},{PAIRS[q][1]})={int(pc[q])}' for q in np.argsort(-pc)[:5]))

    json.dump({'fold_hash': fh, 'pi_source': pi_src.tolist(),
               'pi_adapt': pi_adapt.tolist(), 'family_masses': st,
               'train_neutral_pair_counts': cnt_neu.tolist(),
               'templates_changed_train': chg, 'results': results},
              open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
