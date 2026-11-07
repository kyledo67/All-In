"""Targeted causality and fitting-provenance tests for the added pipeline."""
import sys,json
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,'organizer')
import numpy as np
from slot_forecasting.io import load_inputs,read_labels
from slot_model.features import causal_index
from slot_model.phase2_features import observable,sample_history,fit_prior,examples,local_probability
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=Path('artifacts/phase2')
qs,hs=load_inputs(D/'train');idx=causal_index(qs,hs);q=qs[0];eps={x.episode_id for x in qs};allowed=eps-{q.episode_id}
p=fit_prior(qs,idx,allowed);h=sample_history(q,idx);f=observable(q,h,idx,p)
changed={k:dict(v) for k,v in idx.items()}
for k,v in changed.items():
 if k[0]==q.episode_id:
  for t in range(q.cutoff,q.cutoff+8):v[t]=4
h2=dict(h);h2.update({t:4 for t in range(q.cutoff,q.cutoff+8)})
assert f==observable(q,h2,changed,p)
assert np.array_equal(local_probability(q,h,p),local_probability(q,h2,p))
# All of this episode's observations can change without changing its learned prior.
for k,v in changed.items():
 if k[0]==q.episode_id:
  for t in v:v[t]=4
assert fit_prior(qs,changed,allowed)==p
# Lag 2 addresses an actual tick; unknown does not equal payout class zero.
gap=dict(h);gap.pop(q.cutoff-2);g=observable(q,gap,idx,p)
assert g['lag2_missing']==1 and sum(g[f'lag2_c{c}'] for c in range(5))==0
zero=dict(h);zero[q.cutoff-2]=0;z=observable(q,zero,idx,p)
assert z['lag2_missing']==0 and z['lag2_c0']==1
ex=examples(qs,idx,read_labels(D/'train/labels.csv',qs),True);weights=defaultdict(float)
for e in ex:
 qq=e['q'];weights[qq.episode_id]+=e['weight'];assert all(t<qq.cutoff for t in e['h'])
 assert 192<=qq.cutoff<=256
 if not e['signal_known']:assert qq.signal==0
assert max(abs(v-1) for v in weights.values())<1e-12
# Historical current outcome never affects its feature row.
e=next(e for e in ex if not e['signal_known']);qq=e['q'];original=observable(qq,e['h'],idx,p,False)
change={k:dict(v) for k,v in idx.items()};change[qq.episode_id,qq.machine_id][qq.cutoff]=4
assert original==observable(qq,e['h'],change,p,False)
report={'passed':True,'post_cutoff_self_and_peer_invariance':True,'prior_excludes_own_episode':True,'missing_lag_not_zero':True,'historical_target_excluded':True,'historical_signal_missing_flag':True,'equal_total_episode_weight':True,'rows':len(ex),'max_episode_weight_error':max(abs(v-1) for v in weights.values())}
(O/'causality_tests.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
