#!/usr/bin/env python
"""Full-precision side-by-side: organizer rounds.score_round vs independent_scorer.score.

Also prints the DIAGNOSTIC scoring.score_predictions['score'] key, which is a
different quantity from technical_points and is NOT contest points.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "organizer"))

from slot_forecasting.rounds import score_round, ROUND_POINTS, round_metadata   # noqa: E402
from slot_forecasting.scoring import score_predictions                          # noqa: E402
from slot_forecasting.io import read_submission, read_targets                    # noqa: E402

spec = importlib.util.spec_from_file_location("indep", Path(__file__).with_name("independent_scorer.py"))
indep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indep)

DATA = ROOT / "main-participants-60-40/public"
VAL = DATA / "validation"
FILES = [
    ("frozen_reference", VAL / "reference.csv"),
    ("oof_ensemble", ROOT / "artifacts/phase2/predictions/oof_ensemble.csv"),
    ("oof_candidate_raw", ROOT / "artifacts/audit/oof_candidate_raw.csv"),
    ("oof_tabpfn", ROOT / "artifacts/tabpfn/predictions/oof_tabpfn.csv"),
]

targets = read_targets(VAL / "targets.csv")
out = {}
for name, path in FILES:
    org = score_round(path, DATA)
    mine = indep.score(str(path), str(DATA), "main")
    diag = score_predictions(read_submission(path, {t.query_id for t in targets}), targets)
    row = {
        "path": str(path.relative_to(ROOT)),
        "organizer_technical_points": repr(org["technical_points"]),
        "independent_technical_points": repr(mine["technical_points"]),
        "points_delta": repr(org["technical_points"] - mine["technical_points"]),
        "organizer_R": repr(org["mean_loss_ratio"]),
        "independent_R": repr(mine["R"]),
        "R_delta": repr(org["mean_loss_ratio"] - mine["R"]),
        "organizer_max_points": org["max_points"],
        "independent_max_points": mine["max_points"],
        "diagnostic_score_key": repr(diag["score"]),
        "diagnostic_probability_error_pooled": repr(diag["probability_error"]),
        "per_condition": {
            c: {"E_org": repr(org["by_condition"][c]["error"]),
                "E_ind": repr(mine["by_condition"][c]["error"]),
                "E_delta": repr(org["by_condition"][c]["error"] - mine["by_condition"][c]["error"]),
                "B_org": repr(org["by_condition"][c]["reference_error"]),
                "B_ind": repr(mine["by_condition"][c]["reference_error"]),
                "ratio_org": repr(org["by_condition"][c]["loss_ratio"]),
                "ratio_ind": repr(mine["by_condition"][c]["loss_ratio"]),
                "episodes": org["by_condition"][c]["episodes"],
                "queries": org["by_condition"][c]["queries"]}
            for c in indep.FOUR_CONDITIONS},
    }
    out[name] = row
    print(f"== {name} ==")
    print(f"  organizer   technical_points = {row['organizer_technical_points']}")
    print(f"  independent technical_points = {row['independent_technical_points']}")
    print(f"  delta                        = {row['points_delta']}")
    print(f"  organizer   R = {row['organizer_R']}   independent R = {row['independent_R']}  "
          f"delta={row['R_delta']}")
    print(f"  DIAGNOSTIC scoring.score_predictions['score'] = {row['diagnostic_score_key']} "
          f"(NOT contest points)")
    for c in indep.FOUR_CONDITIONS:
        pc = row["per_condition"][c]
        print(f"    {c:14s} E_delta={pc['E_delta']:>8s}  ratio_org={pc['ratio_org']}")

print("\nROUND_POINTS =", ROUND_POINTS)
print("metadata max_points =", round_metadata(DATA)["max_points"])
Path(__file__).with_name("parity.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
