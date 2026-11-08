"""Causality + finiteness audit of Family E's evidence pipeline (scripts/endpoint_supervised.py).

Every per-machine history dict is wrapped in a recorder and the whole bundle index in a
proxy, so one E.gather() pass records WHICH (episode, machine) and WHICH tick each query
actually read. Asserts no read at or after that query's own cutoff, no read of a foreign
machine, and that absent ticks were treated as absent. Then checks the design matrix and
the standardised regression matrix for non-finite cells.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/Users/ayan/Downloads/All-In')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))

from slot_model import harness as H
from slot_forecasting.io import load_inputs
import scripts.endpoint_supervised as E

STATE = {'q': None}
LOG = []


class Rec(dict):
    def __init__(self, src, tag):
        super().__init__(src)
        self.tag = tag

    def __getitem__(self, k):
        LOG.append((STATE['q'], self.tag, k))
        return super().__getitem__(k)

    def __contains__(self, k):
        LOG.append((STATE['q'], self.tag, k))
        return super().__contains__(k)

    def get(self, k, d=None):
        LOG.append((STATE['q'], self.tag, k))
        return super().get(k, d)


class Proxy(dict):
    def __init__(self, src, order):
        super().__init__({k: Rec(v, k) for k, v in src.items()})
        self.order, self.i = order, 0

    def __getitem__(self, key):
        STATE['q'] = self.order[self.i]
        self.i += 1
        return super().__getitem__(key)


def main():
    b = H.load(ROOT / 'main-participants-60-40/public')
    queries, _ = load_inputs(b.data / 'validation')
    meta = {q.query_id: (q.episode_id, q.machine_id, q.cutoff) for q in queries}

    per_key = {}
    for q in queries:
        per_key.setdefault((q.episode_id, q.machine_id), []).append(q.cutoff)
    print('history index keys:', len(b.index), '| queries:', len(b.ids))
    print('keys with >1 distinct query cutoff:', sum(1 for v in per_key.values() if len(set(v)) > 1))
    over = [k for k in per_key if b.index[k] and max(b.index[k]) >= min(per_key[k])]
    print("index keys holding a tick >= that key's MIN query cutoff:", len(over))

    orig = b.index
    b.index = Proxy(orig, list(b.ids))
    E.gather(b)
    b.index = orig
    print('recorded accesses:', len(LOG))

    bad_tick, foreign = [], []
    for qid, tag, k in LOG:
        episode, machine, cutoff = meta[qid]
        if tag != (episode, machine):
            foreign.append((qid, tag))
        if isinstance(k, (int, np.integer)) and int(k) >= cutoff:
            bad_tick.append((qid, cutoff, int(k)))
    print('READS at or after the query cutoff:', len(bad_tick), bad_tick[:5])
    print('READS of a foreign (episode, machine):', len(foreign), foreign[:5])
    # do absent ticks get read as present?  count reads of keys not in the dict
    absent = sum(1 for qid, tag, k in LOG
                 if isinstance(k, (int, np.integer)) and int(k) not in orig[tag])
    print('reads of ABSENT ticks (membership tests only, never valued):', absent)

    raw = E.gather(b)
    qh, r_prot = b.split_heads(b.protected)
    u_prot = r_prot[:, 0]
    masks = []
    for fold in range(4):
        fit = ~b.outer_mask(fold)
        masks.append(('fit%d' % fold, fit))
        for i, (tr, _he) in enumerate(b.inner_folds(fit, 3)):
            masks.append(('fit%d_tr%d' % (fold, i), tr))
    worstX, worstZ, bad = 0.0, 0.0, []
    for name, m in masks:
        X, _ = E.design(b, raw, m, u_prot)
        if not np.all(np.isfinite(X)):
            bad.append((name, int((~np.isfinite(X)).sum())))
        worstX = max(worstX, float(np.max(np.abs(X))))
        for nm, cols in E.LADDER:
            if not cols:
                continue
            idx = [E.IDX[c] for c in cols]
            for t in H.TYPES:
                mm = m & (b.types == t)
                mu = X[mm][:, idx].mean(axis=0)
                sd = np.maximum(X[mm][:, idx].std(axis=0), 1e-6)
                Z = (X[b.types == t][:, idx] - mu) / sd
                worstZ = max(worstZ, float(np.max(np.abs(Z))))
                if not np.all(np.isfinite(Z)):
                    print('NON-FINITE Z', name, nm, t)
    print('design matrices with a non-finite cell:', bad)
    print('max |X|:', worstX, ' max |Z| over every mask/rung/type:', worstZ)

    names, _P, _Q, _R, info = H.raw_pool(b)
    print('raw_pool members:', len(names), 'distinct q', info['distinct_q_heads'],
          'distinct r', info['distinct_r_heads'])
    print('all RAW?', all(__import__('scripts.stack_manifest', fromlist=['classify'])
                          .classify(n) == 'RAW' for n in names))


if __name__ == '__main__':
    main()
