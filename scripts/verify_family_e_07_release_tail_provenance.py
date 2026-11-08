import sys, json
sys.path.insert(0,'/Users/ayan/Downloads/All-In'); sys.path.insert(0,'/Users/ayan/Downloads/All-In/organizer')
import numpy as np
from pathlib import Path
from slot_model import harness as H
from slot_forecasting.io import load_inputs, read_targets

b = H.load(Path('main-participants-60-40/public'))
pk = json.load(open('artifacts/win_head_release_exact/packaging.json'))
val_tail = pk['csv']['validation']['report']['r_model']['tail_by_type']
print('release packaging tail_by_type (validation report):')
for t in sorted(val_tail): print('  ', t, np.array(val_tail[t]))
print('deployment_blend_weight', pk['csv']['validation']['report']['r_model']['deployment_blend_weight'],
      'tail_projection', pk['csv']['validation']['report']['r_model']['tail_projection'])
tc = b.tail_constants(np.ones(len(b.ids), bool))
print('validation TRUTH tail constants:')
for t in sorted(tc): print('  ', t, tc[t])
for t in sorted(tc):
    print('  max|release-truth|', t, float(np.abs(np.array(val_tail[t])-tc[t]).max()))
# train truth tails
tq, th = load_inputs(Path('main-participants-60-40/public/train'))
lab = read_targets(Path('main-participants-60-40/public/train/labels.csv')) if Path('main-participants-60-40/public/train/labels.csv').exists() else None
print('train labels available:', lab is not None)
if lab is not None:
    tt = {q.query_id: q.machine_type for q in tq}
    byt = {}
    for r in lab:
        p = np.asarray(r.probabilities)
        tail = p[2:]/max(p[2:].sum(),1e-15)
        byt.setdefault(tt[r.query_id], []).append(tail)
    for t in sorted(byt):
        a=np.array(byt[t]); print('  train', t, 'n',len(a), 'mean', a.mean(0), 'spread', (a.max(0)-a.min(0)))
# distinct u in train
if lab is not None:
    for t in sorted(byt):
        pass
    byu={}
    for r in lab:
        p=np.asarray(r.probabilities); q=1-p[0]
        byu.setdefault(tt[r.query_id],set()).add(round(float(p[1]/max(q,1e-15)),9))
    for t in sorted(byu): print('  train distinct u', t, sorted(byu[t]))
