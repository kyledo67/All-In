#!/usr/bin/env python
"""Parity / rejection battery against organizer gatekeepers.

Gatekeepers exercised per case:
  * organizer/check_submission.py --split validation   (CLI, full official path)
  * slot_forecasting.io.read_submission                (the parser itself)
  * artifacts/audit/scorer-parity/independent_scorer.py
"""
import csv
import importlib.util
import io as _io
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
WORK = OUT / "cases"
WORK.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "organizer"))

from slot_forecasting.io import read_submission, read_targets, read_rows   # noqa: E402
from slot_forecasting.schema import ValidationError, SUM_TOLERANCE          # noqa: E402
from slot_forecasting.rounds import score_round                            # noqa: E402

spec = importlib.util.spec_from_file_location("indep", OUT / "independent_scorer.py")
indep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indep)

DATA = ROOT / "main-participants-60-40/public"
VAL = DATA / "validation"
HEADER = ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"]
CATS = HEADER[1:]

targets = read_targets(VAL / "targets.csv")
IDS = [t.query_id for t in targets]
EPISODE = {t.query_id: t.episode_id for t in targets}
COND = {r["query_id"]: r["condition"] for r in read_rows(VAL / "conditions.csv", ("query_id", "condition"))}
# Perfect rows = the literal target probability strings (same text the scorer normalizes for truth).
raw_targets = {}
with (VAL / "targets.csv").open(newline="", encoding="utf-8-sig") as fh:
    for row in csv.DictReader(fh):
        raw_targets[row["query_id"]] = [row[c] for c in CATS]
PERFECT = [[q] + list(raw_targets[q]) for q in IDS]


def write_case(name, rows, header=HEADER):
    path = WORK / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def probe(name, path):
    """Run all three gatekeepers and return a verdict record."""
    rec = {"case": name, "file": str(path.relative_to(ROOT))}
    cli = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "organizer/check_submission.py"),
                          str(path), "--data", str(DATA), "--split", "validation"],
                         capture_output=True, text=True, cwd=str(ROOT))
    rec["check_submission_exit"] = cli.returncode
    rec["check_submission_out"] = (cli.stdout + cli.stderr).strip()
    try:
        read_submission(path, set(IDS))
        rec["read_submission"] = "ACCEPT"
    except ValidationError as exc:
        rec["read_submission"] = f"REJECT: {exc}"
    except Exception as exc:                                        # noqa: BLE001
        rec["read_submission"] = f"REJECT({type(exc).__name__}): {exc}"
    try:
        rec["organizer_points"] = repr(score_round(path, DATA)["technical_points"])
        rec["organizer_R"] = repr(score_round(path, DATA)["mean_loss_ratio"])
    except Exception as exc:                                        # noqa: BLE001
        rec["organizer_points"] = f"REJECT({type(exc).__name__}): {exc}"
        rec["organizer_R"] = None
    try:
        mine = indep.score(str(path), str(DATA), "main")
        rec["independent_points"] = repr(mine["technical_points"])
        rec["independent_R"] = repr(mine["R"])
    except indep.Rejected as exc:
        rec["independent_points"] = f"REJECT: {exc}"
        rec["independent_R"] = None
    except Exception as exc:                                        # noqa: BLE001
        rec["independent_points"] = f"REJECT({type(exc).__name__}): {exc}"
        rec["independent_R"] = None
    return rec


records = []


def run(name, rows, header=HEADER):
    rec = probe(name, write_case(name, rows, header))
    records.append(rec)
    print(f"--- {name}")
    print(f"    check_submission exit={rec['check_submission_exit']}: {rec['check_submission_out'].splitlines()[0] if rec['check_submission_out'] else ''}")
    if rec["check_submission_exit"] != 0:
        for line in rec["check_submission_out"].splitlines():
            print(f"      | {line}")
    print(f"    read_submission      : {rec['read_submission']}")
    print(f"    organizer points     : {rec['organizer_points']}")
    print(f"    independent points   : {rec['independent_points']}")
    return rec


# ---------------------------------------------------------------- (a) perfect
run("a_perfect", PERFECT)

# -------------------------------------- (b) controlled epsilon per category
EPS = 1e-3
victim = IDS[0]
vcond, vepi = COND[victim], EPISODE[victim]
n_in_episode = sum(1 for q in IDS if EPISODE[q] == vepi)
n_episodes_in_cond = len({EPISODE[q] for q in IDS if COND[q] == vcond})
B = indep.score(str(VAL / "reference.csv"), str(DATA), "main")["by_condition"][vcond]["reference_error"]
print(f"\n[epsilon setup] victim={victim} condition={vcond} episode={vepi} "
      f"queries_in_episode={n_in_episode} episodes_in_condition={n_episodes_in_cond} B_c={B!r}")
eps_rows = []
for k, cat in enumerate(CATS):
    donor = 0 if k != 0 else 1           # move EPS from `donor` into category k
    rows = []
    for r in PERFECT:
        if r[0] != victim:
            rows.append(list(r))
            continue
        vals = [float(x) for x in r[1:]]
        vals[k] += EPS
        vals[donor] -= EPS
        rows.append([r[0]] + [repr(v) for v in vals])
    # Analytic prediction: one query gains 2*EPS^2; episode mean /n_in_episode;
    # condition mean /n_episodes; R contribution /4 after dividing by B_c.
    dE = 2 * EPS ** 2 / n_in_episode / n_episodes_in_cond
    dR = dE / B / 4
    predicted = 60.0 / (1.0 + dR)
    rec = run(f"b_eps_{cat}_from_{CATS[donor]}", rows)
    rec["analytic_delta_E_c"] = repr(dE)
    rec["analytic_R"] = repr(dR)
    rec["analytic_points"] = repr(predicted)
    actual = float(rec["organizer_points"]) if not rec["organizer_points"].startswith("REJECT") else math.nan
    rec["analytic_vs_organizer_delta"] = repr(actual - predicted)
    print(f"    analytic points      : {predicted!r}  (delta vs organizer {actual - predicted!r})")
    eps_rows.append(rec)

# (b2) the simplex constraint: the same +EPS with NO compensating decrease.
rows = []
for r in PERFECT:
    if r[0] != victim:
        rows.append(list(r))
        continue
    vals = [float(x) for x in r[1:]]
    vals[2] += EPS
    rows.append([r[0]] + [repr(v) for v in vals])
run("b2_uncompensated_eps_1e-3", rows)

# (b3) an uncompensated bump SMALLER than SUM_TOLERANCE: admitted, then renormalized,
# so the error lands on all five categories instead of the one that was edited.
TINY = 5e-7
rows = []
for r in PERFECT:
    if r[0] != victim:
        rows.append(list(r))
        continue
    vals = [float(x) for x in r[1:]]
    vals[2] += TINY
    rows.append([r[0]] + [repr(v) for v in vals])
run("b3_uncompensated_eps_5e-7_renormalized", rows)

# ------------------------------------------ (c) sums straddling SUM_TOLERANCE
for label, delta in (("c_sum_plus_2e-6", 2e-6), ("c_sum_minus_2e-6", -2e-6),
                     ("c_sum_plus_9e-7", 9e-7), ("c_sum_minus_9e-7", -9e-7)):
    rows = []
    for r in PERFECT:
        if r[0] != victim:
            rows.append(list(r))
            continue
        vals = [float(x) for x in r[1:]]
        vals[0] += delta
        rows.append([r[0]] + [repr(v) for v in vals])
    run(label, rows)

# ------------------------------------------------- (d) negative probability
rows = []
for r in PERFECT:
    if r[0] != victim:
        rows.append(list(r))
        continue
    vals = [float(x) for x in r[1:]]
    vals[1] = -0.05
    vals[0] = 1.0 - sum(vals[1:])
    rows.append([r[0]] + [repr(v) for v in vals])
run("d_negative_probability", rows)

# ------------------------------------------------------ (e) probability > 1
rows = []
for r in PERFECT:
    if r[0] != victim:
        rows.append(list(r))
        continue
    rows.append([r[0], "1.5", "-0.5", "0.0", "0.0", "0.0"])
run("e_probability_above_one", rows)

# ------------------------------------------------------------ (f) NaN / inf
for label, text in (("f_nan", "nan"), ("f_inf", "inf"), ("f_neg_inf", "-inf")):
    rows = []
    for r in PERFECT:
        if r[0] != victim:
            rows.append(list(r))
            continue
        rows.append([r[0], text, "0.0", "0.0", "0.0", "0.0"])
    run(label, rows)

# ----------------------------------------------------- (g) duplicate query_id
run("g_duplicate_query_id", PERFECT + [list(PERFECT[0])])

# ------------------------------------------------------- (h) missing query_id
run("h_missing_query_id", PERFECT[1:])

# --------------------------------------------------- (i) extra unknown query_id
run("i_extra_unknown_query_id", PERFECT + [["main-validation-e99999-m000-next", "1", "0", "0", "0", "0"]])

# ------------------------------------------------ (j) reversed / shuffled order
run("j_reversed_order", list(reversed(PERFECT)))
import random                                                      # noqa: E402
shuffled = list(PERFECT)
random.Random(20261107).shuffle(shuffled)
run("j_shuffled_order", shuffled)

# order independence must also hold for a NON-perfect file, where ties cannot hide it
cand = ROOT / "artifacts/audit/oof_candidate_raw.csv"
with cand.open(newline="", encoding="utf-8-sig") as fh:
    rd = csv.reader(fh)
    chead = next(rd)
    cbody = [list(r) for r in rd]
base = probe("j_candidate_as_written", cand)
records.append(base)
rev = list(reversed(cbody))
shuf = list(cbody)
random.Random(99).shuffle(shuf)
r1 = run("j_candidate_reversed", rev, chead)
r2 = run("j_candidate_shuffled", shuf, chead)
print(f"\n[ordering] as-written  = {base['organizer_points']}")
print(f"[ordering] reversed    = {r1['organizer_points']}  identical={base['organizer_points'] == r1['organizer_points']}")
print(f"[ordering] shuffled    = {r2['organizer_points']}  identical={base['organizer_points'] == r2['organizer_points']}")
print(f"[ordering] independent as-written={base['independent_points']} reversed={r1['independent_points']} "
      f"shuffled={r2['independent_points']}")

# --------------------------------- (k) header order / header names
perm = ["query_id", "p_20", "p_5", "p_2", "p_1", "p_0"]
# k1: header permuted AND values permuted to match -> same meaning
run("k1_header_permuted_values_follow", [[r[0], r[5], r[4], r[3], r[2], r[1]] for r in PERFECT], perm)
# k2: header permuted but values left in canonical positional order -> silent remap
run("k2_header_permuted_values_not_moved", [list(r) for r in PERFECT], perm)
# k3: a renamed column
run("k3_header_renamed", [list(r) for r in PERFECT],
    ["query_id", "p0", "p_1", "p_2", "p_5", "p_20"])
# k4: a duplicated column name
run("k4_header_duplicate_name", [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in PERFECT],
    ["query_id", "p_0", "p_0", "p_2", "p_5", "p_20"])
# k5: an extra trailing column
run("k5_extra_column", [list(r) + ["x"] for r in PERFECT], HEADER + ["note"])
# k6: empty value
rows = [list(r) for r in PERFECT]
rows[0] = [rows[0][0], "", "0", "0", "0", "1"]
run("k6_empty_value", rows)

(OUT / "battery.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
print(f"\nSUM_TOLERANCE = {SUM_TOLERANCE!r}")
print(f"wrote {OUT / 'battery.json'}")
