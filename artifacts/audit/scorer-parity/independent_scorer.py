#!/usr/bin/env python
"""Independent Round-1 scorer. Written from the published contract only.

Imports nothing from organizer/.  Only `csv`, `numpy`, `json`, `argparse`, `sys`.

CONTRACT (organizer/RULES.md "How points work" + public/metadata.json):

    query loss  L_i  = SUM over the 5 payout categories of (p_ik - q_ik)**2
    episode err      = MEAN of L_i over the queries in that episode
    condition err E_c= MEAN of episode err over the episodes in that condition
    B_c              = E_c recomputed with the frozen reference.csv as prediction
    R                = (1/4) * SUM over the 4 announced conditions of E_c / max(B_c, 1e-9)
    technical points = max_points / (1 + R)       with max_points read from metadata.json

Row admission (organizer/RULES.md "Submit your forecast"):
    * exactly the columns query_id,p_0,p_1,p_2,p_5,p_20
    * every query exactly once, no unknown query, no missing query
    * each probability finite and in [0, 1]
    * the five probabilities sum to 1 within 1e-6
    * accepted rows are then divided by their own sum ("after normalization")

Usage:
    independent_scorer.py SUBMISSION.csv [--data DIR] [--round main] [--json]
"""

import argparse
import csv
import json
import sys

import numpy as np

CATEGORIES = ("p_0", "p_1", "p_2", "p_5", "p_20")
SUBMISSION_HEADER = ("query_id",) + CATEGORIES
SUM_TOL = 1e-6
DENOM_FLOOR = 1e-9
FOUR_CONDITIONS = ("stable", "short_history", "drift", "shared_change")
ROUND_MAX = {"main": 60, "adaptation": 40}


class Rejected(Exception):
    """A submission or dataset the contract does not admit."""


def exact_sum(values):
    """Neumaier compensated summation.

    The 1e-6 admission gate is defined on the TRUE sum of the five printed
    decimals, so a naive float64 accumulation can straddle the gate by one ULP
    and admit a row the contract excludes.  This keeps the running error term.
    """
    total = np.float64(0.0)
    compensation = np.float64(0.0)
    for value in values:
        value = np.float64(value)
        candidate = total + value
        if abs(total) >= abs(value):
            compensation += (total - candidate) + value
        else:
            compensation += (value - candidate) + total
        total = candidate
    return np.float64(total + compensation)


# ---------------------------------------------------------------- row admission

def _vector(raw, where):
    """Validate five probability strings and return them normalized to sum 1."""
    out = np.empty(5, dtype=np.float64)
    for i, text in enumerate(raw):
        try:
            value = np.float64(text)
        except (TypeError, ValueError):
            raise Rejected(f"{where}: {CATEGORIES[i]}={text!r} is not a number")
        if not np.isfinite(value):
            raise Rejected(f"{where}: {CATEGORIES[i]}={text!r} is not finite")
        if value < 0.0 or value > 1.0:
            raise Rejected(f"{where}: {CATEGORIES[i]}={text!r} is outside [0,1]")
        out[i] = value
    total = exact_sum(out)
    if abs(total - 1.0) > SUM_TOL:
        raise Rejected(f"{where}: the five probabilities sum to {total!r}, "
                       f"which is further than {SUM_TOL} from 1")
    return out / total


def _table(path, header):
    """Read a CSV whose header must be exactly `header`, in that order."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise Rejected(f"{path}: file is empty")
    if tuple(rows[0]) != tuple(header):
        raise Rejected(f"{path}: header is {tuple(rows[0])}, contract wants {tuple(header)}")
    body = rows[1:]
    if not body:
        raise Rejected(f"{path}: no data rows")
    for number, row in enumerate(body, start=2):
        if len(row) != len(header):
            raise Rejected(f"{path} line {number}: {len(row)} fields, wanted {len(header)}")
    return [dict(zip(header, row)) for row in body]


def load_probability_file(path, expected_ids=None):
    """query_id -> normalized 5-vector, with duplicate/missing/extra id checks."""
    table = _table(path, SUBMISSION_HEADER)
    out = {}
    for row in table:
        key = row["query_id"]
        if key in out:
            raise Rejected(f"{path}: query_id {key} appears more than once")
        out[key] = _vector([row[c] for c in CATEGORIES], f"{path} row {key}")
    if expected_ids is not None:
        want, got = set(expected_ids), set(out)
        if want - got:
            raise Rejected(f"{path}: missing query_id(s) {sorted(want - got)[:5]}")
        if got - want:
            raise Rejected(f"{path}: unknown query_id(s) {sorted(got - want)[:5]}")
    return out


def load_truth(path):
    """targets.csv -> (query_id -> normalized truth vector, query_id -> episode_id)."""
    header = ("query_id", "episode_id") + CATEGORIES + ("outcome",)
    truth, episode = {}, {}
    for row in _table(path, header):
        key = row["query_id"]
        if key in truth:
            raise Rejected(f"{path}: duplicate target query_id {key}")
        truth[key] = _vector([row[c] for c in CATEGORIES], f"{path} target {key}")
        episode[key] = row["episode_id"]
    return truth, episode


def load_conditions(path):
    condition = {}
    for row in _table(path, ("query_id", "condition")):
        key = row["query_id"]
        if key in condition:
            raise Rejected(f"{path}: duplicate condition query_id {key}")
        condition[key] = row["condition"]
    return condition


def load_max_points(data_dir, expect_round=None):
    with open(f"{data_dir}/metadata.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    name = meta.get("round")
    if name not in ROUND_MAX:
        raise Rejected(f"metadata.json: round={name!r} is not one of {tuple(ROUND_MAX)}")
    if meta.get("max_points") != ROUND_MAX[name]:
        raise Rejected(f"metadata.json: round {name} must carry "
                       f"max_points={ROUND_MAX[name]}, found {meta.get('max_points')!r}")
    if expect_round is not None and name != expect_round:
        raise Rejected(f"metadata.json: this kit is round {name!r}, not {expect_round!r}")
    return name, int(meta["max_points"]), meta


# ------------------------------------------------------------------ the formula

def condition_error(predictions, truth, episode_of, query_ids):
    """MEAN over episodes of (MEAN over that episode's queries of SUM of squares)."""
    buckets = {}
    for qid in query_ids:
        buckets.setdefault(episode_of[qid], []).append(qid)
    per_episode = np.empty(len(buckets), dtype=np.float64)
    for slot, name in enumerate(sorted(buckets)):
        ids = buckets[name]
        losses = np.empty(len(ids), dtype=np.float64)
        for i, qid in enumerate(ids):
            gap = predictions[qid] - truth[qid]
            losses[i] = exact_sum(gap * gap)       # SUM over 5 categories, not a mean
        per_episode[slot] = exact_sum(losses) / np.float64(len(losses))
    return float(exact_sum(per_episode) / np.float64(len(per_episode))), len(buckets)


def score(submission_path, data_dir, expect_round=None):
    split = f"{data_dir}/validation"
    round_name, max_points, _ = load_max_points(data_dir, expect_round)
    truth, episode_of = load_truth(f"{split}/targets.csv")
    condition_of = load_conditions(f"{split}/conditions.csv")
    ids = set(truth)
    if set(condition_of) != ids:
        raise Rejected("conditions.csv and targets.csv cover different query_ids")
    seen = set(condition_of.values())
    if seen != set(FOUR_CONDITIONS):
        raise Rejected(f"conditions present {sorted(seen)}, contract wants {sorted(FOUR_CONDITIONS)}")
    span = {}                                      # one episode -> one condition
    for qid in ids:
        span.setdefault(episode_of[qid], set()).add(condition_of[qid])
    bad = [e for e, c in span.items() if len(c) != 1]
    if bad:
        raise Rejected(f"episode(s) spanning several conditions: {sorted(bad)[:5]}")

    mine = load_probability_file(submission_path, ids)
    reference = load_probability_file(f"{split}/reference.csv", ids)

    rows, ratios = {}, []
    for name in FOUR_CONDITIONS:
        subset = sorted(q for q in ids if condition_of[q] == name)
        e_c, n_ep = condition_error(mine, truth, episode_of, subset)
        b_c, _ = condition_error(reference, truth, episode_of, subset)
        ratio = e_c / max(DENOM_FLOOR, b_c)
        ratios.append(ratio)
        rows[name] = {"error": e_c, "reference_error": b_c, "loss_ratio": ratio,
                      "episodes": n_ep, "queries": len(subset)}
    r = float(exact_sum(ratios) / np.float64(len(FOUR_CONDITIONS)))
    return {"round": round_name, "max_points": max_points, "R": r,
            "technical_points": max_points / (1.0 + r), "by_condition": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission")
    ap.add_argument("--data", default="main-participants-60-40/public")
    ap.add_argument("--round", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        result = score(args.submission, args.data, args.round)
    except Rejected as exc:
        print(f"INDEPENDENT REJECT: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"round={result['round']} max_points={result['max_points']}")
        print(f"R                = {result['R']!r}")
        print(f"technical_points = {result['technical_points']!r}")
        for name in FOUR_CONDITIONS:
            row = result["by_condition"][name]
            print(f"  {name:14s} E={row['error']!r} B={row['reference_error']!r} "
                  f"ratio={row['loss_ratio']!r} ep={row['episodes']} q={row['queries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
