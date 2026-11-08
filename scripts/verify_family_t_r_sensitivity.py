"""Does Family T's q-side gain survive a NON-target-fitted conditional-payout head?

The whole family composes its q with r taken from artifacts/win_head_release_exact/
validation_submission.csv. stack_manifest.classify() calls that file DEPLOY_FIT, and the
release's per-type tail constants are bit-equal to harness.tail_constants(all) computed
from VALIDATION TRUTH. The official loss at frozen r is A q^2 - 2 B q + const with
A = 1 + |r|^2 and B = q_true (1 + r . r_true), so the q-optimum is
q* = q_true (1 + r . r_true)/(1 + |r|^2): it equals q_true only when r is right. A
truth-fitted r therefore AMPLIFIES how much the objective rewards an accurate q.

So recompose BOTH q heads (the nested T3 q and the protected q) with r heads that are
admissible under the manifest, and re-measure the paired delta. Nothing here is proposed
as a forecast; it bounds how much of the measured +1.373 is the q head.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import harness as H
from slot_forecasting.io import read_submission

b = H.load(ROOT / 'main-participants-60-40/public')
q_prot, r_prot = b.split_heads(b.protected)
P3 = np.array([read_submission(ROOT / 'artifacts/verify_family_t_joint/predictions/oof_T3_gated.csv')[i]
               for i in b.ids])
q3, _ = b.split_heads(P3)
names, _, _, Rraw, info = H.raw_pool(b)
out = {'raw_pool_members': len(names), 'distinct_r_heads': info['distinct_r_heads'],
       'max_abs_q3_minus_q_from_r_prot_composition': float(np.max(np.abs(
           b.compose(q3, r_prot) - P3))),
       'arms': {}}
rs = {'release_r_DEPLOY_FIT': r_prot,
      'raw_pool_mean_r': Rraw.mean(axis=0) / Rraw.mean(axis=0).sum(axis=1, keepdims=True)}
for j, n in enumerate(names):
    if 'tabpfn_observable_20261108' in n or 'two_head_01' in n:
        rs[f'raw:{Path(n).parent.name}/{Path(n).stem}'] = Rraw[j]
for tag, r in sorted(rs.items()):
    A = b.compose(q3, r)
    B = b.compose(q_prot, r)
    oa, ob = b.official(A), b.official(B)
    pr = b.paired(A, B)
    out['arms'][tag] = {
        'points_T3_q': oa['technical_points'], 'points_protected_q': ob['technical_points'],
        'point_gain_from_q_head': oa['technical_points'] - ob['technical_points'],
        'R_T3_q': oa['mean_loss_ratio'], 'R_protected_q': ob['mean_loss_ratio'],
        'paired_ci95_T3q_vs_protq': pr['delta_loss_ratio_ci95'],
        'significant': bool(pr['delta_loss_ratio_ci95'][1] < 0)}
(ROOT / 'artifacts/verify_family_t_joint/r_sensitivity.json').write_text(
    json.dumps(out, indent=2, sort_keys=True, default=float) + '\n')
print(json.dumps(out, indent=2, sort_keys=True, default=float))
