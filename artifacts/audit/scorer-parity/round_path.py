#!/usr/bin/env python
"""Round-aware path: max_points comes from public/metadata.json, nothing else.

Builds doctored metadata.json copies in a scratch kit whose validation/ is a
symlink to the real one, so the released kit is never touched.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "organizer"))
from slot_forecasting.rounds import round_metadata, score_round, ROUND_POINTS   # noqa: E402
from slot_forecasting.contest import contest_score, score_contest_files          # noqa: E402
from slot_forecasting.schema import ValidationError                              # noqa: E402

DATA = ROOT / "main-participants-60-40/public"
REAL = json.loads((DATA / "metadata.json").read_text())
SUB = DATA / "validation/reference.csv"

CASES = {
    "as_released": {},
    "round_adaptation_points_60": {"round": "adaptation"},
    "round_adaptation_points_40": {"round": "adaptation", "max_points": 40},
    "round_bogus": {"round": "bonus"},
    "round_missing": {"round": None},
    "main_with_max_points_70": {"max_points": 70},
    "main_with_max_points_80": {"max_points": 80},
    "suite_version_wrong": {"suite_version": "experimentation-suite-v9"},
    "season_id_wrong": {"season_id": "not-a-24-hex-string"},
}

scratch = Path(tempfile.mkdtemp(prefix="roundpath-"))
try:
    for name, patch in CASES.items():
        kit = scratch / name / "public"
        kit.mkdir(parents=True)
        os.symlink(DATA / "validation", kit / "validation")
        meta = dict(REAL)
        for k, v in patch.items():
            if v is None:
                meta.pop(k, None)
            else:
                meta[k] = v
        (kit / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        try:
            got = round_metadata(kit)
            md = f"ACCEPT round={got['round']!r} max_points={got['max_points']!r}"
        except ValidationError as exc:
            md = f"REJECT: {exc}"
        try:
            sr = repr(score_round(SUB, kit)["technical_points"])
        except ValidationError as exc:
            sr = f"REJECT: {exc}"
        print(f"{name:32s} round_metadata -> {md}")
        print(f"{'':32s} score_round(reference) -> {sr}")
finally:
    shutil.rmtree(scratch, ignore_errors=True)

print()
print("ROUND_POINTS (rounds.py:12)                 =", ROUND_POINTS)
print("contest_score default max_points            =",
      contest_score.__defaults__, contest_score.__kwdefaults__)
print("score_contest_files default max_points      =", score_contest_files.__kwdefaults__)
print("score_round signature forces metadata value : rounds.py:35-37 passes "
      "max_points=metadata['max_points']")
