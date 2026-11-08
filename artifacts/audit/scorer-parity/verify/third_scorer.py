#!/usr/bin/env python
"""Third, deliberately different implementation of the Round-1 contract.

No organizer import, no numpy. Uses Fraction (exact rational) arithmetic for the
whole pipeline so the result is independent of float accumulation order, then
reports the exact value and its float64 image. Also reports a NO-RENORMALIZATION
variant to size the renormalization sensitivity.
"""
import csv
import json
import sys
from fractions import Fraction as F

CATS = ("p_0", "p_1", "p_2", "p_5", "p_20")
CONDS = ("stable", "short_history", "drift", "shared_change")
TOL = F(1, 10**6)


def rows(path, cols):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        r = csv.DictReader(fh)
        assert set(r.fieldnames) == set(cols), (path, r.fieldnames)
        return list(r)


def vec(row, renorm=True):
    # Fraction(str) is the EXACT decimal; float(str) then Fraction(float) is the
    # exact binary double. The organizer parses to double, so do that.
    v = [F(float(row[c])) for c in CATS]
    for x in v:
        assert 0 <= x <= 1, row
    t = sum(v)
    assert abs(t - 1) <= TOL, (row, float(t))
    return [x / t for x in v] if renorm else v


def score(sub_path, data, renorm=True):
    val = f"{data}/validation"
    meta = json.load(open(f"{data}/metadata.json", encoding="utf-8"))
    assert meta["round"] in ("main", "adaptation")
    assert meta["max_points"] == {"main": 60, "adaptation": 40}[meta["round"]]
    mp = F(meta["max_points"])
    tgt = rows(f"{val}/targets.csv", ("query_id", "episode_id", *CATS, "outcome"))
    truth = {r["query_id"]: vec(r, renorm) for r in tgt}
    ep = {r["query_id"]: r["episode_id"] for r in tgt}
    cond = {r["query_id"]: r["condition"] for r in rows(f"{val}/conditions.csv", ("query_id", "condition"))}
    sub = {r["query_id"]: vec(r, renorm) for r in rows(sub_path, ("query_id", *CATS))}
    ref = {r["query_id"]: vec(r, renorm) for r in rows(f"{val}/reference.csv", ("query_id", *CATS))}
    assert set(sub) == set(truth) == set(cond) == set(ref)
    assert set(cond.values()) == set(CONDS)

    def E(pred, qids):
        buckets = {}
        for q in qids:
            buckets.setdefault(ep[q], []).append(q)
        per = []
        for name in sorted(buckets):
            ids = buckets[name]
            per.append(sum(sum((a - b) ** 2 for a, b in zip(pred[q], truth[q])) for q in ids) / len(ids))
        return sum(per) / len(per), len(buckets)

    out, ratios = {}, []
    for c in CONDS:
        ids = sorted(q for q in truth if cond[q] == c)
        e, nep = E(sub, ids)
        b, _ = E(ref, ids)
        denom = b if b > F(1, 10**9) else F(1, 10**9)
        ratios.append(e / denom)
        out[c] = {"E": float(e), "B": float(b), "ratio": float(e / denom), "episodes": nep, "queries": len(ids)}
    R = sum(ratios) / len(CONDS)
    return {"R_exact_float": float(R), "points": float(mp / (1 + R)), "by_condition": out,
            "max_points": int(mp)}


if __name__ == "__main__":
    data = "main-participants-60-40/public"
    for p in sys.argv[1:]:
        for renorm in (True, False):
            r = score(p, data, renorm)
            print(f"{p}  renorm={renorm}")
            print(f"  points = {r['points']!r}   R = {r['R_exact_float']!r}")
            for c in CONDS:
                row = r["by_condition"][c]
                print(f"    {c:14s} E={row['E']!r} B={row['B']!r} ratio={row['ratio']!r} ep={row['episodes']} q={row['queries']}")
