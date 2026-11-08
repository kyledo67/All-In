"""Admission cases the 30-case battery did not cover."""
import csv, sys, subprocess
from pathlib import Path
R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R / "organizer"))
from slot_forecasting.io import read_submission, read_targets           # noqa
from slot_forecasting.rounds import score_round                        # noqa
from slot_forecasting.schema import ValidationError, probabilities      # noqa

VAL = R / "main-participants-60-40/public/validation"
OUT = Path(__file__).parent / "cases"; OUT.mkdir(exist_ok=True)
tgt = read_targets(VAL / "targets.csv")
ids = [t.query_id for t in tgt]
raw = {}
with (VAL / "targets.csv").open(newline="", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        raw[r["query_id"]] = [r[c] for c in ("p_0","p_1","p_2","p_5","p_20")]

HDR = "query_id,p_0,p_1,p_2,p_5,p_20"
def body(mut=None):
    out = []
    for q in ids:
        v = list(raw[q])
        if mut and q == ids[0]:
            v = mut(v)
        out.append(",".join([q] + v))
    return out

def write(name, text, mode="w", enc="utf-8"):
    p = OUT / name
    p.write_bytes(text.encode(enc) if isinstance(text, str) else text)
    return p

cases = {}
cases["x1_bom"] = write("x1_bom.csv", "﻿" + HDR + "\n" + "\n".join(body()) + "\n")
cases["x2_crlf"] = write("x2_crlf.csv", HDR + "\r\n" + "\r\n".join(body()) + "\r\n")
cases["x3_trailing_blank_line"] = write("x3_trailing_blank_line.csv", HDR + "\n" + "\n".join(body()) + "\n\n")
cases["x4_quoted_values"] = write("x4_quoted_values.csv",
    HDR + "\n" + "\n".join(",".join(['"%s"' % f for f in line.split(",")]) for line in body()) + "\n")
cases["x5_space_padded"] = write("x5_space_padded.csv",
    HDR + "\n" + "\n".join(body(lambda v: [" " + x + " " for x in v])) + "\n")
cases["x6_plus_sign"] = write("x6_plus_sign.csv", HDR + "\n" + "\n".join(body(lambda v: ["+" + x for x in v])) + "\n")
cases["x7_one_and_zeros"] = write("x7_one_and_zeros.csv", HDR + "\n" + "\n".join(body(lambda v: ["1.0","0.0","0.0","0.0","0.0"])) + "\n")
cases["x8_NaN_spelling"] = write("x8_NaN_spelling.csv", HDR + "\n" + "\n".join(body(lambda v: ["NaN","0.0","0.0","0.0","0.0"])) + "\n")
cases["x9_Infinity_spelling"] = write("x9_Infinity_spelling.csv", HDR + "\n" + "\n".join(body(lambda v: ["Infinity","0.0","0.0","0.0","0.0"])) + "\n")
cases["x10_overflow_1e400"] = write("x10_overflow_1e400.csv", HDR + "\n" + "\n".join(body(lambda v: ["1e400","0.0","0.0","0.0","0.0"])) + "\n")
cases["x11_header_only_space"] = write("x11_header_leading_space.csv", "query_id, p_0,p_1,p_2,p_5,p_20\n" + "\n".join(body()) + "\n")
cases["x12_latin1_bytes"] = write("x12_latin1_bytes.csv", (HDR + "\n" + "\n".join(body())).encode("utf-8") + b",\xff\n")
cases["x13_subnormal"] = write("x13_subnormal.csv", HDR + "\n" + "\n".join(body(lambda v: ["5e-324","0.0","0.0","0.0","1.0"])) + "\n")

for name, path in cases.items():
    try:
        read_submission(path, set(ids)); adm = "ACCEPT"
    except ValidationError as e: adm = f"REJECT: {e}"
    except Exception as e: adm = f"{type(e).__name__}: {e}"
    try:
        pts = repr(score_round(path, R / "main-participants-60-40/public")["technical_points"])
    except ValidationError as e: pts = f"REJECT: {e}"
    except Exception as e: pts = f"{type(e).__name__}: {e}"
    cs = subprocess.run([str(R / ".venv/bin/python"), str(R / "organizer/check_submission.py"), str(path),
                         "--data", str(R / "main-participants-60-40/public"), "--split", "validation"],
                        capture_output=True, text=True)
    first = (cs.stdout or cs.stderr).strip().splitlines()[0] if (cs.stdout or cs.stderr).strip() else ""
    print(f"--- {name}")
    print(f"    read_submission : {adm}")
    print(f"    score_round     : {pts}")
    print(f"    check_submission exit={cs.returncode}: {first}")

# Can renormalization ever push a value above 1 (second normalization raising)?
print("\n[probe] is v/fsum(v) > 1 reachable with all v in [0,1] and |sum-1|<=1e-6 ?")
import math, random
rng = random.Random(7)
worst = 0.0
for _ in range(400000):
    k = rng.randrange(5)
    v = [0.0]*5
    v[k] = 1.0 - rng.random()*1e-6
    rest = 1.0 - v[k] + rng.uniform(-1e-6, 1e-6)
    v[(k+1) % 5] = max(0.0, rest)
    t = math.fsum(v)
    if abs(t-1.0) > 1e-6: continue
    worst = max(worst, max(x/t for x in v))
print("    max observed normalized value =", repr(worst), "(>1 would break scoring.py:63)")
