"""Close the payout-weighting question: transfer eps between EVERY unordered pair,
including pairs that do not touch p_0 (the battery only tested pairs involving p_0)."""
import csv, itertools, sys
from pathlib import Path
R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R / "organizer"))
from slot_forecasting.io import read_rows, read_targets                  # noqa
from slot_forecasting.rounds import score_round                          # noqa

PUB = R / "main-participants-60-40/public"; VAL = PUB / "validation"
OUT = Path(__file__).parent / "cases"; OUT.mkdir(exist_ok=True)
CATS = ("p_0", "p_1", "p_2", "p_5", "p_20")
tgt = read_targets(VAL / "targets.csv"); ids = [t.query_id for t in tgt]
cond = {r["query_id"]: r["condition"] for r in read_rows(VAL / "conditions.csv", ("query_id", "condition"))}
raw = {}
with (VAL / "targets.csv").open(newline="", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        raw[r["query_id"]] = [float(r[c]) for c in CATS]

victim = ids[0]
EPS = 1e-3
base = score_round(VAL / "reference.csv", PUB)          # just to get B_c
B = base["by_condition"][cond[victim]]["reference_error"]
nq = sum(1 for t in tgt if t.episode_id == dict((t.query_id, t.episode_id) for t in tgt)[victim])
ep_of = {t.query_id: t.episode_id for t in tgt}
nq = sum(1 for t in tgt if t.episode_id == ep_of[victim])
nep = len({t.episode_id for t in tgt if cond[t.query_id] == cond[victim]})
analytic_dE = 2 * EPS**2 / nq / nep
analytic_R = analytic_dE / B / 4
analytic_pts = 60 / (1 + analytic_R)
print(f"victim={victim} cond={cond[victim]} nq={nq} nep={nep} B={B!r}")
print(f"analytic: dE={analytic_dE!r} dR={analytic_R!r} points={analytic_pts!r}\n")

rows_out = []
for a, b in itertools.combinations(range(5), 2):
    v = list(raw[victim])
    if v[b] < EPS:   # need room to take mass out of b
        donor_ok = False
    else:
        donor_ok = True
    if not donor_ok:
        rows_out.append((CATS[a], CATS[b], "skipped: donor < eps", ""))
        continue
    v[a] += EPS; v[b] -= EPS
    p = OUT / f"pair_{CATS[a]}_{CATS[b]}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["query_id", *CATS])
        for q in ids:
            w.writerow([q] + [repr(x) for x in (v if q == victim else raw[q])])
    pts = score_round(p, PUB)["technical_points"]
    rows_out.append((CATS[a], CATS[b], repr(pts), repr(pts - analytic_pts)))

for a, b, pts, d in rows_out:
    print(f"  {a:5s} <- {b:5s}  points={pts:22s} delta_vs_analytic={d}")
