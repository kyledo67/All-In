"""Target-free setting rule for T2/T2MAP: does the gain survive without the official target?

The family's own report records, per (kappa, hazard), the PREQUENTIAL log score of the
fixed-structure run-length model (the sequential marginal likelihood of the data under that
setting).  That quantity needs no validation probability target at all, so it is the only
setting rule in this family that is genuinely target-free and would transfer to test.  Here
the T2MAP forecast is scored at the setting that criterion selects, and compared with the
setting the official row-weighted objective selects on the inner held partitions.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np
from slot_model import harness as H

OUT = ROOT / 'artifacts/verify_family_t_adv_opus'
b = H.load(ROOT / 'main-participants-60-40/public')
q_prot, r_prot = b.split_heads(b.protected)
Q = np.load(OUT / 'T2MAP_q_candidates.npy')
order = [tuple(x) for x in json.loads((OUT / 'T2MAP_q_candidate_order.json').read_text())]
tag = {f'k{k:g}_h{h:g}': i for i, (k, h) in enumerate(order)}

diag = json.loads((ROOT / 'artifacts/family_t_joint/evaluation.json').read_text())['public_diagnostics_PRIVILEGED']
preq = {k.split('/')[1]: v['prequential_logscore_fixed_structure']
        for k, v in sorted(diag.items()) if k.startswith('T3/')}
best_free = max(sorted(preq), key=lambda k: (preq[k], k))

rows = {}
for name in sorted(tag):
    P = b.compose(Q[tag[name]], r_prot)
    o = b.official(P)
    pr = b.paired(P, b.protected)
    rows[name] = {'prequential_logscore_fixed_structure': preq.get(name),
                  'points_all_episodes': o['technical_points'],
                  'paired_vs_protected_ci95': pr['delta_loss_ratio_ci95'],
                  'significant': bool(pr['delta_loss_ratio_ci95'][1] < 0)}
rep = {'preregistration': __doc__,
       'protected_points': b.official(b.protected)['technical_points'],
       'target_free_prequential_argmax_setting': best_free,
       'target_free_choice': rows[best_free],
       'nested_official_objective_choice_per_fold': ['k1_h0.0001', 'k1_h0.0001',
                                                     'k0.5_h0.0001', 'k1_h0.0001'],
       'every_setting': rows}
(OUT / 'target_free_rule.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
print('protected                        %.6f' % rep['protected_points'])
print('target-free prequential argmax   %-14s %.6f  ci95=[%+.6f,%+.6f] sig=%s'
      % (best_free, rows[best_free]['points_all_episodes'],
         *rows[best_free]['paired_vs_protected_ci95'], rows[best_free]['significant']))
print()
print('%-14s %-16s %-12s %s' % ('setting', 'prequential', 'points', 'paired vs protected'))
for name in sorted(rows, key=lambda n: -(rows[n]['prequential_logscore_fixed_structure'] or -1e18)):
    r = rows[name]
    print('%-14s %-16.1f %-12.6f [%+.6f,%+.6f] %s'
          % (name, r['prequential_logscore_fixed_structure'], r['points_all_episodes'],
             *r['paired_vs_protected_ci95'], 'SIG' if r['significant'] else ''))
