"""Adversarial verification of Family E's E2r_calibrator_direct claim.

Independent re-run + causal leakage probe: perturb the OUTER HELD fold's targets and
check whether that fold's held forecast changes at all. If any target-dependent quantity
were fitted or selected on the held episodes, the held forecast would move.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))

from slot_model import harness as H
import scripts.endpoint_supervised as E

OUT = ROOT / 'artifacts/verify_family_e'
OUT.mkdir(parents=True, exist_ok=True)


def build(data=ROOT / 'main-participants-60-40/public'):
    b = H.load(data)
    qh, r_prot = b.split_heads(b.protected)
    raw = E.gather(b)
    ctx = {'bundle': b, 'raw': raw, 'n': len(b.ids), 'q': qh, 'u_prot': r_prot[:, 0],
           'train_mixture': None}
    return b, ctx


def main():
    rep = {}
    b, ctx = build()
    n = ctx['n']

    arms = {'E2r_calibrator_direct': E.Calibrator(regression=True),
            'E2_calibrator': E.Calibrator(),
            'E0a_zero_correction': E.Zero()}
    base = {}
    for name, arm in arms.items():
        t0 = time.perf_counter()
        base[name] = E.run_arm(b, arm, ctx, name)
        o = b.official(base[name]['P'])
        pv = b.paired(base[name]['P'], b.protected)
        rep[name] = {'points': o['technical_points'], 'R': o['mean_loss_ratio'],
                     'by_condition': {c: v['loss_ratio'] for c, v in o['by_condition'].items()},
                     'paired_vs_protected_ci95': pv['delta_loss_ratio_ci95'],
                     'seconds': time.perf_counter() - t0,
                     'chosen': base[name]['chosen']}
        print(name, o['technical_points'], o['mean_loss_ratio'],
              pv['delta_loss_ratio_ci95'], flush=True)

    prot = b.official(b.protected)
    rep['protected'] = {'points': prot['technical_points'], 'R': prot['mean_loss_ratio']}
    rep['paired_E2r_vs_E0a'] = b.paired(base['E2r_calibrator_direct']['P'],
                                        base['E0a_zero_correction']['P'])['delta_loss_ratio_ci95']
    rep['E0a_equals_protected_maxabs'] = float(
        np.max(np.abs(base['E0a_zero_correction']['P'] - b.protected)))

    # ---- leakage probe: perturb the held fold's TRUTH, refit, compare held forecast ----
    rng = np.random.default_rng(20261108)
    probe = {}
    for name in ('E2r_calibrator_direct', 'E2_calibrator'):
        arm = arms[name]
        rows = []
        for fold in range(4):
            held = b.outer_mask(fold)
            b2 = copy.copy(b)
            b2._wcache = {}
            t2 = b.truth.copy()
            # replace the held fold's targets with a random simplex: nothing legitimate
            # may depend on them when forecasting that same fold
            draw = rng.dirichlet(np.ones(5), size=int(held.sum()))
            t2[held] = draw
            b2.truth = t2
            ctx2 = dict(ctx); ctx2['bundle'] = b2
            res2 = E.run_arm(b2, arm, ctx2, name + '_probe')
            du = float(np.max(np.abs(res2['u'][held] - base[name]['u'][held])))
            dP = float(np.max(np.abs(res2['P'][held] - base[name]['P'][held])))
            same_cfg = (res2['chosen'][fold]['config'] == base[name]['chosen'][fold]['config']
                        and res2['chosen'][fold]['blend'] == base[name]['chosen'][fold]['blend'])
            # control: the probe MUST change the OTHER folds (truth there is fitting data)
            other = ~held
            dP_other = float(np.max(np.abs(res2['P'][other] - base[name]['P'][other])))
            rows.append({'fold': fold, 'max_abs_delta_u_on_held': du,
                         'max_abs_delta_P_on_held': dP,
                         'selected_config_unchanged': bool(same_cfg),
                         'max_abs_delta_P_on_other_folds_sanity': dP_other})
            print(name, 'fold', fold, 'd_u_held', du, 'd_P_held', dP,
                  'cfg_same', same_cfg, 'd_P_other', dP_other, flush=True)
        probe[name] = rows
    rep['held_target_perturbation_probe'] = probe

    # ---- what inputs are touched: no archived candidate files are read by E2r ----
    (OUT / 'verification.json').write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                      default=float) + '\n')
    print('wrote', OUT / 'verification.json')


if __name__ == '__main__':
    main()
