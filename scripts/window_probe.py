"""Quantify discarded-history effects in a known persistent-regime process.

No competition targets, tuning, or simulator-internal claims are involved.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slot_model import runtime
import numpy as np
from scripts.verify import reference_filter

rng=np.random.default_rng(617)
trans=np.array([[.99,.01],[.01,.99]])
emissions=np.array([[.65,.16,.10,.07,.02],[.45,.25,.16,.10,.04]])
errors={8:[],16:[],32:[]}
for _ in range(256):
    state=int(rng.integers(2)); y=[]
    for t in range(128):
        y.append(int(rng.choice(5,p=emissions[state])))
        state=int(rng.choice(2,p=trans[state]))
    likelihood=emissions[:,y].T
    full,_=reference_filter([.5,.5],trans,likelihood)
    truth=(full[-1]@trans)@emissions
    for window in errors:
        f,_=reference_filter([.5,.5],trans,likelihood[-window:])
        forecast=(f[-1]@trans)@emissions
        errors[window].append(float(np.sum((forecast-truth)**2)))
result={'seed':617,'machines':256,'full_history_ticks':128,
        'transition':trans.tolist(),'emission':emissions.tolist(),
        'mean_squared_difference_from_full_history':{w:float(np.mean(v)) for w,v in errors.items()},
        'scope':'Synthetic sensitivity only; the competition window stays frozen at 32.',
        'conclusion':'Restarting the state prior discards information. Missing AR inputs are avoided, not marginalized.'}
Path('artifacts/verification/window_approximation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
