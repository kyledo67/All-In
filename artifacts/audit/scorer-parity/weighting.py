#!/usr/bin/env python
"""Weighting level checks + the exact 1e-6 tolerance boundary + diagnostic-vs-official."""
import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "organizer"))
from slot_forecasting.io import read_rows, read_submission, read_targets      # noqa: E402
from slot_forecasting.scoring import score_predictions                       # noqa: E402
from slot_forecasting.rounds import score_round                              # noqa: E402
from slot_forecasting.schema import ValidationError                          # noqa: E402

spec = importlib.util.spec_from_file_location("indep", OUT / "independent_scorer.py")
indep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indep)

DATA = ROOT / "main-participants-60-40/public"
VAL = DATA / "validation"
targets = read_targets(VAL / "targets.csv")
cond = {r["query_id"]: r["condition"] for r in read_rows(VAL / "conditions.csv", ("query_id", "condition"))}

sizes = Counter(t.episode_id for t in targets)
print("episodes =", len(sizes), " queries =", len(targets))
print("distinct queries-per-episode counts =", sorted(set(sizes.values())))
per_cond_ep = Counter(cond[t.query_id] for t in targets)
print("queries per condition =", dict(per_cond_ep))
ep_cond = {}
for t in targets:
    ep_cond.setdefault(cond[t.query_id], set()).add(t.episode_id)
print("episodes per condition =", {k: len(v) for k, v in ep_cond.items()})
print("-> every episode has the same query count, so episode-weighted and")
print("   query-weighted condition means COINCIDE on this kit; the mean-based")
print("   episode weighting is established by code reading, not by this data.")

CAND = ROOT / "artifacts/audit/oof_candidate_raw.csv"
pred = read_submission(CAND, {t.query_id for t in targets})
diag = score_predictions(pred, targets)
off = score_round(CAND, DATA)
mine = indep.score(str(CAND), str(DATA), "main")
print()
print("OFFICIAL  technical_points (rounds.score_round)      =", repr(off["technical_points"]))
print("OFFICIAL  R (mean_loss_ratio)                        =", repr(off["mean_loss_ratio"]))
print("DIAGNOSTIC scoring.score_predictions['score']        =", repr(diag["score"]))
print("DIAGNOSTIC probability_error (pooled, 96 episodes)   =", repr(diag["probability_error"]))
print("check: 100*(1-error/2)                               =",
      repr(100 * (1 - diag["probability_error"] / 2)))
four = [mine["by_condition"][c]["error"] for c in indep.FOUR_CONDITIONS]
print("mean of the four condition errors                    =", repr(float(np.mean(four))))
print("  delta vs pooled probability_error                  =",
      repr(float(np.mean(four)) - diag["probability_error"]))

# exact tolerance boundary: abs(total-1) > 1e-6 rejects, == 1e-6 accepts
with (VAL / "targets.csv").open(newline="", encoding="utf-8-sig") as fh:
    raw = {r["query_id"]: [r[c] for c in indep.CATEGORIES] for r in csv.DictReader(fh)}
ids = [t.query_id for t in targets]
victim = ids[0]
cases = OUT / "cases"
cases.mkdir(exist_ok=True)
for label, delta in (("exactly_plus_1e-6", 1e-6), ("exactly_minus_1e-6", -1e-6),
                     ("one_ulp_over_plus", 1e-6 + 1e-21)):
    rows = []
    for q in ids:
        vals = [float(x) for x in raw[q]]
        if q == victim:
            vals[0] = vals[0] + delta
            total = sum(vals)
            vals[0] += (1.0 + delta) - total          # force the sum to land on 1+delta
        rows.append([q] + [repr(v) for v in vals])
    path = cases / f"tol_{label}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(indep.SUBMISSION_HEADER))
        w.writerows(rows)
    s = sum(float(x) for x in rows[ids.index(victim)][1:])
    try:
        read_submission(path, set(ids))
        verdict = "ACCEPT"
    except ValidationError as exc:
        verdict = f"REJECT: {exc}"
    try:
        indep.load_probability_file(str(path), set(ids))
        iv = "ACCEPT"
    except indep.Rejected as exc:
        iv = f"REJECT: {exc}"
    print(f"\ntolerance {label:20s} realised row sum = {s!r}  |sum-1| = {abs(s - 1.0)!r}")
    print(f"  organizer read_submission : {verdict}")
    print(f"  independent               : {iv}")
