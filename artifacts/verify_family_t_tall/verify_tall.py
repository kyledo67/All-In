"""Independent verification of Family-T arms T3B7 / T3U / TALL (claimed 54.919878102996826).

Writes artifacts/verify_family_t_tall/*.json. Does not touch any other artifacts dir.
"""
import os, sys, json, time, hashlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = '/Users/ayan/Downloads/All-In'
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/organizer')
import numpy as np
from pathlib import Path

from slot_model import harness as H
from slot_model import win_head_infer as WI
import scripts.family_t_joint_structure_memory as FT

OUT = Path(ROOT) / 'artifacts/verify_family_t_tall'
CACHE = OUT / 'cand_cache.npz'
DATA = Path(ROOT) / 'main-participants-60-40/public'
CLAIM = 54.919878102996826

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]

def build_candidates():
    bundle = H.load(DATA)
    q_prot, r_prot = bundle.split_heads(bundle.protected)
    model = bundle.package / 'code/model'
    kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(model / 'win_head/beta_templates_deploy.json')
    beta, lam, _ = WI.load_win_head(model / 'win_head/population_averaged.json')
    support = np.asarray(payload['train_argmax_counts'], dtype=np.float64)
    full_bank = list(range(len(FT.TEMPLATES)))
    small_bank = [i for i in full_bank if support[i] >= FT.SUPPORT_MIN]
    lp_full = np.asarray(log_prior, dtype=np.float64)
    lp_small = lp_full[small_bank]
    records = FT.prepare(bundle)
    hl = FT.half_lives(bundle, records, beta, lam)
    q_c1, pick_c1, _ = FT.t1_mixture(records, hl, full_bank, lp_full, prior_mean, kappa0, None)
    dev = float(np.max(np.abs(q_c1 - q_prot)))
    fixed_full = np.array([full_bank.index(int(p)) for p in pick_c1])
    fixed_small = np.array([small_bank.index(int(p)) if int(p) in small_bank else small_bank.index(0) for p in pick_c1])
    cand = {'T1': [], 'T2': [], 'T2MAP': [], 'T3': [], 'T3R': [], 'T3B7': []}
    for kappa in FT.KAPPAS:
        for temperature in FT.LAMBDAS:
            tag = f'k{kappa:g}_lam{"hard" if temperature is None else f"{temperature:g}"}'
            q, _, _ = FT.t1_mixture(records, hl, full_bank, lp_full, prior_mean, kappa, temperature)
            cand['T1'].append((tag, q))
    for kappa in FT.KAPPAS:
        res = FT.changepoint_pass(records, full_bank, lp_full, prior_mean, kappa, FT.HAZARDS, fixed_full)
        for hi, h in enumerate(FT.HAZARDS):
            tag = f'k{kappa:g}_h{h:g}'
            for arm, key in (('T2', 'q2'), ('T2MAP', 'q2map'), ('T3', 'q3'), ('T3R', 'q3r')):
                cand[arm].append((tag, res[key][hi].copy()))
    for kappa in FT.KAPPAS:
        res = FT.changepoint_pass(records, small_bank, lp_small, prior_mean, kappa, FT.HAZARDS,
                                  fixed_small, variants=('T3',))
        for hi, h in enumerate(FT.HAZARDS):
            cand['T3B7'].append((f'k{kappa:g}_h{h:g}', res['q3'][hi].copy()))
    base_arms = sorted(cand)
    cand['T3U'] = ([(f'full|{t}', q) for t, q in cand['T3']] + [(f'b7|{t}', q) for t, q in cand['T3B7']])
    cand['TALL'] = [(f'{a}|{t}', q) for a in base_arms for t, q in cand[a]]
    return bundle, q_c1, r_prot, cand, small_bank, support, dev, hl, records, pick_c1

if __name__ == '__main__':
    t0 = time.perf_counter()
    bundle, q_c1, r_prot, cand, small_bank, support, dev, hl, records, pick_c1 = build_candidates()
    report = {'control_max_abs_q_dev_vs_protected': dev,
              'small_bank': small_bank,
              'support_at_small_bank': support[small_bank].tolist(),
              'n_settings': {a: len(cand[a]) for a in sorted(cand)}}
    # ---------- CHECK 1 / 6: reproduce + scored once -------------------------
    res = {}
    for arm in ('T3', 'T3B7', 'T3U', 'TALL'):
        for mode in ('forced', 'gated'):
            P, chosen, table = FT.nested(bundle, r_prot, q_c1, cand[arm], mode == 'gated')
            o = bundle.official(P)
            res[f'{arm}_{mode}'] = {'points': o['technical_points'], 'R': o['mean_loss_ratio'],
                                    'chosen': chosen, 'sha_P': sha(P),
                                    'by_condition': {c: o['by_condition'][c]['loss_ratio']
                                                     for c in sorted(o['by_condition'])}}
            np.save(OUT / f'P_{arm}_{mode}.npy', P)
    report['check1_reproduce'] = res
    report['check1_abs_err_vs_claim'] = {k: abs(v['points'] - CLAIM) for k, v in res.items()}
    np.savez_compressed(CACHE, **{f'{a}|{t}': q for a in sorted(cand) for t, q in cand[a]},
                        q_c1=q_c1, r_prot=r_prot, hl=hl, pick_c1=pick_c1)
    report['seconds'] = time.perf_counter() - t0
    (OUT / 'check1.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
