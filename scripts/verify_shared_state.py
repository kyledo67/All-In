import sys,json
from dataclasses import replace
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_forecasting.io import load_inputs
from slot_model.shared_state import neighbor_state,adjust
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');q,h=load_inputs(D/'validation');q=q[0]
a=neighbor_state([q],h,.35)[q.query_id]
modified=[replace(row,payout=20) if row.episode_id==q.episode_id and row.tick>=q.cutoff else row for row in h]
assert neighbor_state([q],modified,.35)[q.query_id]==a
p=adjust([.6,.2,.1,.08,.02],a,.04)
assert np.isfinite(p).all() and min(p)>=0 and abs(sum(p)-1)<1e-12
print('shared-state cutoff and normalization checks passed')
