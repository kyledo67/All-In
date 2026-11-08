import os, sys, copy, json
os.environ.setdefault('MPLCONFIGDIR','/tmp/all-in-mpl')
ROOT='/Users/ayan/Downloads/All-In'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT+'/organizer')
import numpy as np
from slot_model import harness as H
from slot_model import win_head_infer as WI
from slot_forecasting.io import load_inputs
from scripts.family_t_joint_structure_memory import TEMPLATES, prepare, half_lives, t1_mixture
from scripts.verify_family_t2_independent import t2_q

b = H.load(ROOT + '/main-participants-60-40/public')
model = b.package / 'code/model'
kappa0, prior_mean, log_prior, payload = WI.load_beta_templates(model/'win_head/beta_templates_deploy.json')
beta, lam, _ = WI.load_win_head(model/'win_head/population_averaged.json')
full = list(range(len(TEMPLATES)))
recs = prepare(b)
hl = half_lives(b, recs, beta, lam)
_, pick, _ = t1_mixture(recs, hl, full, np.asarray(log_prior), prior_mean, kappa0, None)
tpl = [TEMPLATES[int(p)] for p in pick]
queries,_ = load_inputs(b.data/'validation')
byq = {q.query_id: q for q in queries}

def qvec(records, k=1.0, h=1e-4):
    return np.array([t2_q(records[i], tpl[i][0], tpl[i][1], k,
                          float(prior_mean.get(records[i]['type'],0.44)), h)
                     for i in range(len(records))])

base = qvec(recs)
out = {}
# post-cutoff injection: fabricate outcomes AT the cutoff and after it
inj = copy.deepcopy(b.index)
rng = np.random.default_rng(7)
for qid in b.ids:
    q = byq[qid]; t = inj[(q.episode_id, q.machine_id)]
    for d in range(0, 12):
        t[q.cutoff + d] = int(rng.integers(0,5))
sv = b.index; b.index = inj
try: r2 = prepare(b)
finally: b.index = sv
out['post_cutoff_injection_max_abs_dq'] = float(np.max(np.abs(base - qvec(r2))))
# delete a mid-history tick -> must change q (missing tick is a state, not a zero)
dele = copy.deepcopy(b.index)
for qid in b.ids:
    q = byq[qid]; t = dele[(q.episode_id, q.machine_id)]
    ks = sorted(t)
    victim = ks[len(ks)//2]
    t.pop(victim)
b.index = dele
try: r3 = prepare(b)
finally: b.index = sv
q3 = qvec(r3)
out['mid_history_tick_deletion_max_abs_dq'] = float(np.max(np.abs(base - q3)))
out['mid_history_tick_deletion_n_obs_change'] = sorted({len(r3[i]['y'])-len(recs[i]['y']) for i in range(len(recs))})
# flipping a tick 40 ticks before the cutoff: must change q for long histories only a little
print(json.dumps(out, indent=2))
