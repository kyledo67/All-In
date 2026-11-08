"""One shared experiment harness for the Round 1 campaign.

Every family imports this instead of re-deriving aligned arrays, official row weights, fold
structure, or the admissible candidate pool. The point is that a cross-family comparison is
only meaningful if all arms share these exactly.

DETERMINISM. episode_id is a string, so iterating a Python set of episode ids has an order
that depends on CPython's per-process string hash randomisation. Summing a float list in
that order makes results vary run to run, which previously flipped discrete grid choices.
Every reduction here iterates a SORTED sequence. Do not replace a sorted() with a set.

OFFICIAL OBJECTIVE. query loss is the full five-vector squared error; it is averaged within
episode, then within condition, then each condition is divided by the frozen reference error
for that condition (floor 1e-9), and R is the mean of the four ratios. points = 60/(1+R).
`row_weights` returns per-query weights w such that sum_i w_i * ||p_i - t_i||^2 equals R
restricted to the episodes in the mask, with the reference denominators recomputed from
exactly those episodes. That is the FITTING-side normalisation: it is what a fitting
objective must use, and it is not the same as scoring a held fold.

LEAKAGE. `raw_pool` returns only candidates whose provenance class is RAW in
scripts/stack_manifest.py, i.e. no validation target entered their construction through
either head. An independent audit found that 15 of a previous 30-member "RAW" pool were
target-fitted, mostly because they inherited a truth-derived conditional payout head rather
than estimating one. Do not widen the pool without re-auditing.
"""
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'organizer') not in sys.path:
    sys.path.insert(0, str(ROOT / 'organizer'))

from slot_forecasting.io import load_inputs, read_submission, write_submission  # noqa: E402
from slot_model.features import causal_index  # noqa: E402
from slot_model.official_score import paired_bootstrap, score, scoring_inputs  # noqa: E402

PAYOUTS = (0, 1, 2, 5, 20)
TYPES = ('balanced', 'jackpot', 'steady')


@dataclass
class Bundle:
    data: Path
    package: Path
    ids: list
    ep: np.ndarray
    truth: np.ndarray
    ref: np.ndarray
    groups: dict
    folds: dict
    types: np.ndarray
    signals: np.ndarray
    cutoffs: np.ndarray
    index: dict
    protected: np.ndarray
    scoring: tuple
    episodes: set
    _wcache: dict = field(default_factory=dict)

    # ---- official objective -------------------------------------------------
    def row_weights(self, mask):
        """Per-query weights whose weighted full-vector SSE equals R on `mask`'s episodes."""
        key = hashlib.sha1(np.asarray(mask, dtype=bool).tobytes()).hexdigest()
        if key in self._wcache:
            return self._wcache[key]
        mask = np.asarray(mask, dtype=bool)
        w = np.zeros(len(self.ids))
        inmask = sorted({e for i, e in enumerate(self.ep) if mask[i]})
        for c in sorted(set(self.groups.values())):
            ces = sorted({e for i, e in enumerate(self.ep)
                          if e in set(inmask) and self.groups[self.ids[i]] == c})
            if not ces:
                continue
            b = float(np.mean([np.mean(np.sum((self.ref[self.ep == e] - self.truth[self.ep == e]) ** 2,
                                              axis=1)) for e in ces]))
            for e in ces:
                n = int((self.ep == e).sum())
                w[self.ep == e] = 1.0 / (4 * len(ces) * n * max(b, 1e-9))
        self._wcache[key] = w
        return w

    def wloss(self, P, w):
        """Official weighted full five-vector squared error. The objective, not a surrogate."""
        return float(np.sum(w * np.sum((P - self.truth) ** 2, axis=1)))

    def mapping(self, P):
        P = np.asarray(P, dtype=float)
        P = P / P.sum(axis=1, keepdims=True)
        return dict(zip(self.ids, P.tolist()))

    def official(self, P, episodes=None):
        return score(self.mapping(P), self.scoring,
                     self.episodes if episodes is None else episodes)

    def paired(self, P, Q):
        return paired_bootstrap(self.mapping(P), self.mapping(Q), self.scoring, self.episodes)

    # ---- folds --------------------------------------------------------------
    def outer_mask(self, fold):
        return np.array([self.folds[e] == fold for e in self.ep])

    def inner_folds(self, fit_mask, k=3):
        """Complete-episode inner partitions of the FITTING episodes only.

        Returns a list of (train_mask, held_mask). Every query of an episode lands in the
        same partition; episodes are assigned in sorted order so the split is deterministic.
        """
        fit_mask = np.asarray(fit_mask, dtype=bool)
        eps = sorted({e for i, e in enumerate(self.ep) if fit_mask[i]})
        assign = {e: i % k for i, e in enumerate(eps)}
        out = []
        for b in range(k):
            tr = np.array([fit_mask[i] and assign.get(e, -1) != b for i, e in enumerate(self.ep)])
            he = np.array([fit_mask[i] and assign.get(e, -1) == b for i, e in enumerate(self.ep)])
            out.append((tr, he))
        return out

    # ---- decomposition helpers ---------------------------------------------
    @staticmethod
    def split_heads(P):
        """P -> (q, r) with q = P(positive) and r the conditional distribution on 1,2,5,20."""
        P = np.asarray(P, dtype=float)
        P = P / P.sum(axis=1, keepdims=True)
        q = 1.0 - P[:, 0]
        r = P[:, 1:] / np.maximum(q, 1e-12)[:, None]
        return q, r / r.sum(axis=1, keepdims=True)

    @staticmethod
    def compose(q, r):
        q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        out = np.column_stack((1.0 - q, q[:, None] * np.asarray(r, dtype=float)))
        return out / out.sum(axis=1, keepdims=True)

    @staticmethod
    def compose_qav(q, a, v):
        """p(q,a,v) = [1-q, q a, q(1-a) v2, q(1-a) v5, q(1-a) v20]. v rows sum to 1."""
        q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        a = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
        v = np.asarray(v, dtype=float)
        v = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-15)
        out = np.column_stack((1.0 - q, q * a, (q * (1.0 - a))[:, None] * v))
        return out / out.sum(axis=1, keepdims=True)

    def q_coeffs(self, r):
        """A, B with L(q) = sum_i w_i (A_i q_i^2 - 2 B_i q_i) + const at frozen r."""
        r = np.asarray(r, dtype=float)
        A = 1.0 + np.sum(r ** 2, axis=1)
        B = (1.0 - self.truth[:, 0]) + np.sum(r * self.truth[:, 1:], axis=1)
        return A, B

    def tail_constants(self, mask):
        """Per-type mean tail vector over {2,5,20} from the masked episodes' TRUTH."""
        mask = np.asarray(mask, dtype=bool)
        rt = self.truth[:, 1:] / np.maximum(self.truth[:, 1:].sum(axis=1, keepdims=True), 1e-15)
        out = {}
        for t in TYPES:
            m = mask & (self.types == t)
            tl = rt[m][:, 1:]
            tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1e-15)
            out[t] = tl.mean(axis=0) if m.any() else np.ones(3) / 3
        return out


def load(data, package=ROOT / 'artifacts/win_head_release_exact'):
    data, package = Path(data), Path(package)
    scoring = scoring_inputs(data)
    targets, groups, reference = scoring
    ids = sorted(reference)
    T = {t.query_id: t for t in targets}
    ep = np.array([T[q].episode_id for q in ids])
    truth = np.array([T[q].probabilities for q in ids])
    ref = np.array([reference[q] for q in ids])
    folds = json.loads((ROOT / 'artifacts/phase2/folds.json').read_text())['validation']
    vq, vh = load_inputs(data / 'validation')
    by = {q.query_id: q for q in vq}
    prot = read_submission(package / 'validation_submission.csv')
    prot = np.array([prot[q] for q in ids])
    return Bundle(data=data, package=package, ids=ids, ep=ep, truth=truth, ref=ref,
                  groups=groups, folds=folds,
                  types=np.array([by[i].machine_type for i in ids]),
                  signals=np.array([by[i].signal for i in ids]),
                  cutoffs=np.array([by[i].cutoff for i in ids]),
                  index=causal_index(vq, vh),
                  protected=prot / prot.sum(axis=1, keepdims=True),
                  scoring=scoring, episodes=set(ep.tolist()))


def raw_pool(bundle):
    """The audited RAW candidate pool: (names, P[m,n,5], Q[m,n], R[m,n,4], dup info)."""
    from scripts.stack_manifest import RAW_SOURCES, classify
    want = set(bundle.ids)
    names, arrs, seen, skipped = [], [], {}, []
    for path in sorted(RAW_SOURCES):
        if classify(path) != 'RAW':
            skipped.append([path, f'provenance {classify(path)}'])
            continue
        f = ROOT / path
        if not f.exists():
            skipped.append([path, 'missing'])
            continue
        d = read_submission(f)
        if set(d) != want:
            skipped.append([path, 'misaligned'])
            continue
        a = np.array([d[i] for i in bundle.ids])
        if not (np.isfinite(a).all() and a.min() >= -1e-12
                and abs(a.sum(1) - 1).max() < 1e-6):
            skipped.append([path, 'not on simplex'])
            continue
        a = a / a.sum(axis=1, keepdims=True)
        h = hashlib.sha256(a.round(12).tobytes()).hexdigest()
        if h in seen:
            skipped.append([path, f'duplicate of {seen[h]}'])
            continue
        seen[h] = path
        names.append(path)
        arrs.append(a)
    P = np.stack(arrs)
    Q, R = [], []
    for i in range(len(P)):
        q, r = Bundle.split_heads(P[i])
        Q.append(q); R.append(r)
    Q, R = np.stack(Q), np.stack(R)
    dq, dr = {}, {}
    for i, n in enumerate(names):
        dq.setdefault(hashlib.sha256(Q[i].round(12).tobytes()).hexdigest(), []).append(n)
        dr.setdefault(hashlib.sha256(R[i].round(12).tobytes()).hexdigest(), []).append(n)
    info = {'skipped': skipped,
            'distinct_q_heads': len(dq), 'distinct_r_heads': len(dr),
            'duplicate_q_groups': [v for v in dq.values() if len(v) > 1],
            'duplicate_r_groups': [v for v in dr.values() if len(v) > 1]}
    return names, P, Q, R, info
