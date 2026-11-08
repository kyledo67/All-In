"""Causal observable features for the Round-2 q head.

Every value reads only ticks strictly below the query cutoff. Missing ticks are treated
as unobserved, never as zero payouts: lag features carry an explicit missing flag and the
window statistics run over OBSERVED ticks in order, so a gap shortens the window rather
than inserting a fake loss.
"""
import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
PIDX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
TYPES = ('steady', 'balanced', 'jackpot')
WINDOWS = (5, 10, 20, 40, 80, 160)
HALFLIVES = (4, 8, 16, 32, 64, 128)


def _ew(vals, hl):
    """Exponentially weighted mean over values ordered oldest -> newest."""
    if not vals:
        return -1.0
    n = len(vals)
    w = 2.0 ** (-(np.arange(n - 1, -1, -1)) / hl)
    return float(np.dot(w, vals) / w.sum())


def machine_features(obs, cutoff, machine_type, signal, neighbours):
    """obs: {tick: payout} for this machine; neighbours: list of {tick: payout}."""
    f, names = [], []

    def add(name, value):
        names.append(name)
        f.append(float(value))

    pre = sorted((t, p) for t, p in obs.items() if t < cutoff)
    ticks = [t for t, _ in pre]
    pays = [p for _, p in pre]
    wins = [1.0 if p > 0 else 0.0 for p in pays]
    n = len(pre)

    for t in TYPES:
        add(f'type_{t}', machine_type == t)
    add('signal', signal)
    for s in (-1, 0, 1):
        add(f'signal_{s}', signal == s)
    add('cutoff', cutoff / 250.0)
    add('n_obs', n / 250.0)
    add('log_n_obs', np.log1p(n))

    # lag block: one-hot payout class plus an explicit missing flag
    seen = dict(pre)
    for lag in range(1, 9):
        y = seen.get(cutoff - lag)
        add(f'lag{lag}_missing', y is None)
        for c, v in enumerate(PAYOUTS):
            add(f'lag{lag}_c{c}', (y == v) if y is not None else 0.0)
        add(f'lag{lag}_win', (y > 0) if y is not None else 0.0)

    # pair interactions on the win indicator, the structure R1 found productive
    wl = {}
    for lag in range(1, 9):
        y = seen.get(cutoff - lag)
        wl[lag] = (1.0 if y > 0 else -1.0) if y is not None else 0.0
    for i, j in ((1, 2), (2, 4), (4, 8), (1, 8), (1, 3), (2, 3), (3, 6), (5, 7)):
        add(f'pair_{i}_{j}', wl[i] * wl[j])

    # observed-order window statistics
    for w in WINDOWS:
        sub = wins[-w:]
        add(f'rate_{w}', np.mean(sub) if sub else -1.0)
        subp = pays[-w:]
        add(f'meanpay_{w}', np.mean(subp) if subp else -1.0)
    add('rate_all', np.mean(wins) if wins else -1.0)
    add('meanpay_all', np.mean(pays) if pays else -1.0)
    nz = [p for p in pays if p > 0]
    add('nz_meanpay', np.mean(nz) if nz else -1.0)
    add('nz_count', len(nz) / 100.0)
    for hl in HALFLIVES:
        add(f'ew_rate_{hl}', _ew(wins, hl))
        add(f'ew_pay_{hl}', _ew([p / 20.0 for p in pays], hl))

    # payout class mix, recent and overall
    for tagw, sub in (('40', pays[-40:]), ('all', pays)):
        tot = max(len(sub), 1)
        for c, v in enumerate(PAYOUTS):
            add(f'mix{tagw}_c{c}', sub.count(v) / tot if sub else -1.0)

    # drift signals
    r20 = np.mean(wins[-20:]) if len(wins) >= 1 else -1.0
    p20 = np.mean(wins[-40:-20]) if len(wins) >= 21 else r20
    r40 = np.mean(wins[-40:]) if wins else -1.0
    add('trend_20_prev20', r20 - p20)
    add('trend_40_all', r40 - (np.mean(wins) if wins else 0.0))
    add('trend_ew8_ew64', _ew(wins, 8) - _ew(wins, 64))
    add('trend_ew16_ew128', _ew(wins, 16) - _ew(wins, 128))

    # recency of events
    since_win = next((i + 1 for i, v in enumerate(reversed(wins)) if v > 0), 99)
    since_big = next((i + 1 for i, p in enumerate(reversed(pays)) if p >= 5), 99)
    since_jp = next((i + 1 for i, p in enumerate(reversed(pays)) if p == 20), 199)
    add('since_win', min(since_win, 40) / 40.0)
    add('since_big', min(since_big, 60) / 60.0)
    add('since_jackpot', min(since_jp, 199) / 199.0)
    streak = 0
    if wins:
        last = wins[-1]
        for v in reversed(wins):
            if v == last:
                streak += 1
            else:
                break
    add('streak_len', min(streak, 20) / 20.0)
    add('streak_is_win', wins[-1] if wins else -1.0)

    # gap / density structure
    if n > 1:
        gaps = np.diff(ticks)
        add('gap_mean', gaps.mean())
        add('gap_frac_gt1', float((gaps > 1).mean()))
        add('gap_recent', float((np.diff(ticks[-40:]) > 1).mean()) if n > 40 else float((gaps > 1).mean()))
    else:
        add('gap_mean', -1.0)
        add('gap_frac_gt1', -1.0)
        add('gap_recent', -1.0)
    add('span', (ticks[-1] - ticks[0]) / 250.0 if n > 1 else -1.0)
    add('last_tick_lag', (cutoff - ticks[-1]) if n else -1.0)

    # episode neighbours, strictly pre-cutoff
    nb_rates, nb_ew8, nb_ew32, nb_last = [], [], [], []
    for o in neighbours:
        v = [1.0 if p > 0 else 0.0 for t, p in sorted(o.items()) if t < cutoff]
        if v:
            nb_rates.append(np.mean(v[-20:]))
            nb_ew8.append(_ew(v, 8))
            nb_ew32.append(_ew(v, 32))
            nb_last.append(v[-1])
    add('nb_count', len(nb_rates) / 8.0)
    add('nb_rate20', np.mean(nb_rates) if nb_rates else -1.0)
    add('nb_ew8', np.mean(nb_ew8) if nb_ew8 else -1.0)
    add('nb_ew32', np.mean(nb_ew32) if nb_ew32 else -1.0)
    add('nb_last_win', np.mean(nb_last) if nb_last else -1.0)
    add('nb_sd_rate20', np.std(nb_rates) if len(nb_rates) > 1 else -1.0)
    add('nb_minus_self', (np.mean(nb_rates) - r20) if nb_rates else 0.0)
    add('nb_ew8_minus_ew32', (np.mean(nb_ew8) - np.mean(nb_ew32)) if nb_ew8 else 0.0)
    return np.array(f), names


def build(queries_map, hist, ids=None):
    ids = sorted(queries_map) if ids is None else list(ids)
    rows, names = [], None
    for qid in ids:
        m = queries_map[qid]
        ep = hist[m['episode_id']]
        obs = ep.get(m['machine_id'], {})
        nb = [o for mid, o in sorted(ep.items()) if mid != m['machine_id']]
        v, names = machine_features(obs, m['cutoff'], m['machine_type'], m['signal'], nb)
        rows.append(v)
    return ids, np.array(rows), names
