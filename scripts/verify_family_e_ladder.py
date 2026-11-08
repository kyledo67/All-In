"""Does E2r's gain depend on the hand-designed 9-rung ladder?

Family E discloses that the 9-rung feature ladder was DESIGNED by inspecting fold-0
fitting episodes, which are HELD episodes for folds 1-3. The per-fold choice among the
rungs is nested, but the menu itself is not. This re-runs E2r with the ladder collapsed
to ONE rung at a time, so no rung selection happens at all; only the ridge strength and
the blend weight are chosen on the inner partitions. A gain that survives a single
pre-specifiable rung does not depend on the menu.
"""
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
FULL = list(E.LADDER)


def main():
    b = H.load(ROOT / 'main-participants-60-40/public')
    qh, r_prot = b.split_heads(b.protected)
    raw = E.gather(b)
    ctx = {'bundle': b, 'raw': raw, 'n': len(b.ids), 'q': qh, 'u_prot': r_prot[:, 0]}
    prot = b.official(b.protected)
    rows = {}
    for name, cols in FULL:
        E.LADDER = [(name, cols)]
        t0 = time.perf_counter()
        res = E.run_arm(b, E.Calibrator(regression=True), ctx, 'E2r_' + name)
        o = b.official(res['P'])
        pv = b.paired(res['P'], b.protected)['delta_loss_ratio_ci95']
        rows[name] = {'points': o['technical_points'], 'R': o['mean_loss_ratio'],
                      'gain': o['technical_points'] - prot['technical_points'],
                      'ci95_vs_protected': pv, 'sig': bool(pv[1] < 0),
                      'blends': [c['blend'] for c in res['chosen']],
                      'seconds': time.perf_counter() - t0}
        print(f"{name:<18}{o['technical_points']:>12.6f}  "
              f"gain {rows[name]['gain']:+.6f}  [{pv[0]:+.6f}, {pv[1]:+.6f}]"
              f"{'  SIG' if pv[1] < 0 else ''}  blends={rows[name]['blends']}", flush=True)
    E.LADDER = FULL
    (OUT / 'ladder_robustness.json').write_text(
        json.dumps({'protected_points': prot['technical_points'], 'single_rung': rows},
                   indent=2, sort_keys=True, default=float) + '\n')
    print('wrote', OUT / 'ladder_robustness.json')


if __name__ == '__main__':
    main()
