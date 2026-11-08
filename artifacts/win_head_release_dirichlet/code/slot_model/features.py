"""Tick-addressed features; no fabricated outcomes or historical signals.

We use complete contiguous windows only, restarting the HMM at each window.
This deliberately discards older information, rather than treating sparse rows
as adjacent or pretending transition advancement integrates missing AR inputs.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
TYPES = ('steady', 'balanced', 'jackpot')
PAIRS = ((1, 2), (2, 4), (4, 8), (1, 8))
FEATURES = ([f'type_{t}' for t in TYPES] + [f'win_lag_{i}' for i in range(1, 9)]
            + [f'win_pair_{i}_{j}' for i, j in PAIRS])
# Bias and two selected coefficients: lag-1 and lag-1 x lag-2.
ADAPT_COLUMNS = (3, 11)
WINDOW = 32

def causal_index(queries, history):
    cutoffs = {(q.episode_id, q.machine_id): q.cutoff for q in queries}
    result = defaultdict(dict)
    for p in history:
        key = (p.episode_id, p.machine_id)
        if key not in cutoffs:
            raise ValueError(f'Unknown machine {key}')
        # Robust boundary for callers with a broader event stream. Organizer
        # loaders separately reject future rows in released files.
        if p.tick >= cutoffs[key]:
            continue
        if p.tick in result[key]:
            raise ValueError('Duplicate tick')
        result[key][p.tick] = PAYOUTS.index(p.payout)
    return result

def feature_at(ticks, tick, machine_type):
    if any(tick - lag not in ticks for lag in range(1, 9)):
        return None
    wins = [1.0 if ticks[tick-i] > 0 else -1.0 for i in range(1, 9)]
    return ([float(machine_type == t) for t in TYPES] + wins
            + [wins[i-1] * wins[j-1] for i, j in PAIRS])

def prepare(queries, history, *, training=False, window=WINDOW):
    """Return fixed-length complete segments and next-query features.

    Train uses nonoverlapping windows aligned backwards from each cutoff.
    Forecasting uses exactly the last window; incomplete inputs fail explicitly.
    Machine type is assumed stable within its episode; signal is never used.
    """
    indexed = causal_index(queries, history)
    xs, ys, owners, next_x = [], [], [], []
    for q in sorted(queries, key=lambda q: q.query_id):
        ticks = indexed[(q.episode_id, q.machine_id)]
        ends = range(q.cutoff, window + 7, -window) if training else [q.cutoff]
        for end in ends:
            xx = [feature_at(ticks, t, q.machine_type) for t in range(end-window, end)]
            if any(x is None for x in xx) or any(t not in ticks for t in range(end-window, end)):
                if not training:
                    raise ValueError(f'{q.query_id}: incomplete recent {window}+8 ticks; no imputation supported')
                continue
            xs.append(xx)
            ys.append([ticks[t] for t in range(end-window, end)])
            owners.append(q.query_id)
            if not training:
                nx = feature_at(ticks, q.cutoff, q.machine_type)
                if nx is None:
                    raise ValueError('Missing query lag')
                next_x.append(nx)
    return {'x': xs, 'y': ys, 'owners': owners, 'next_x': next_x}

def cached_prepare(directory, queries, history, *, training=False, window=WINDOW, cache='artifacts/cache'):
    import numpy as np
    directory = Path(directory)
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for name in ('queries.csv', 'history.csv'):
        digest.update((directory/name).read_bytes())
    digest.update(json.dumps([training, window]).encode())
    key = digest.hexdigest()
    path = Path(cache)/f'{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}, key
    data = prepare(queries, history, training=training, window=window)
    data = {k: np.asarray(v) for k, v in data.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data, key
