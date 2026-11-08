import csv, json, math, sys, collections
sys.path.insert(0, "/Users/ayan/Downloads/All-In/organizer")
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_targets, read_submission, read_queries, read_rows
from slot_forecasting.schema import PAYOUTS, CONDITIONS

ROOT = "/Users/ayan/Downloads/All-In"
PUB = f"{ROOT}/main-participants-60-40/public/validation"
CAND = f"{ROOT}/artifacts/audit/oof_candidate_raw.csv"
MAXP = 60.0

queries = {q.query_id: q for q in read_queries(f"{PUB}/queries.csv")}
targets = tuple(read_targets(f"{PUB}/targets.csv"))
groups = {r["query_id"]: r["condition"] for r in read_rows(f"{PUB}/conditions.csv", ("query_id","condition"))}
ref = read_submission(f"{PUB}/reference.csv", queries)
cand = read_submission(CAND, queries)

def official(pred):
    return contest_score(pred, targets, groups, ref, max_points=MAXP)

base = official(cand)
R = base["mean_loss_ratio"]; pts = base["technical_points"]

# ---- plain arrays
tmap = {t.query_id: t for t in targets}
qids = sorted(tmap)
P = {q: list(cand[q]) for q in qids}
Q = {q: list(tmap[q].probabilities) for q in qids}
RF = {q: list(ref[q]) for q in qids}
by_cond = collections.defaultdict(list)
for q in qids: by_cond[groups[q]].append(q)
ep_of = {q: tmap[q].episode_id for q in qids}

def cond_err(pred, ids):   # all episodes have 8 queries -> episode-mean == query-mean
    return math.fsum(math.fsum((pred[q][k]-Q[q][k])**2 for k in range(5)) for q in ids)/len(ids)

# verify my shortcut equals organizer per-condition error
shortcut_ok = {}
for c in CONDITIONS:
    shortcut_ok[c] = abs(cond_err(P, by_cond[c]) - base["by_condition"][c]["error"])

# ---- per condition
conds = {}
for c in CONDITIONS:
    bc = base["by_condition"][c]
    conds[c] = {"E_c": bc["error"], "B_c": bc["reference_error"], "ratio": bc["loss_ratio"],
                "contribution_to_R": bc["loss_ratio"]/4.0, "episodes": bc["episodes"], "queries": bc["queries"]}

# ---- per-episode errors within each condition
ep_err = collections.defaultdict(list); ep_ref = collections.defaultdict(list)
for q in qids:
    ep_err[ep_of[q]].append(math.fsum((P[q][k]-Q[q][k])**2 for k in range(5)))
    ep_ref[ep_of[q]].append(math.fsum((RF[q][k]-Q[q][k])**2 for k in range(5)))
episodes = []
for e in sorted(ep_err):
    c = groups[[q for q in qids if ep_of[q]==e][0]]
    ee = math.fsum(ep_err[e])/len(ep_err[e]); re_ = math.fsum(ep_ref[e])/len(ep_ref[e])
    episodes.append({"episode_id": e, "condition": c, "episode_error": ee, "reference_error": re_,
                     "excess_over_reference": ee-re_, "ratio": ee/max(re_,1e-9)})

# ---- per-category decomposition (global and per condition)
def cat_stats(ids):
    out = []
    for k in range(5):
        mse = math.fsum((P[q][k]-Q[q][k])**2 for q in ids)/len(ids)
        bias = math.fsum(P[q][k]-Q[q][k] for q in ids)/len(ids)
        rmse_ref = math.fsum((RF[q][k]-Q[q][k])**2 for q in ids)/len(ids)
        mp = math.fsum(P[q][k] for q in ids)/len(ids); mq = math.fsum(Q[q][k] for q in ids)/len(ids)
        out.append({"payout": PAYOUTS[k], "mean_sq_err": mse, "mean_signed_bias": bias,
                    "ref_mean_sq_err": rmse_ref, "mean_pred": mp, "mean_true": mq})
    return out
cat_global = cat_stats(qids)
cat_by_cond = {c: cat_stats(by_cond[c]) for c in CONDITIONS}

# ---- headroom
ratios = {c: conds[c]["ratio"] for c in CONDITIONS}
best_ratio = min(ratios.values()); best_cond = min(ratios, key=ratios.get)
def R_of(rr): return math.fsum(rr.values())/4.0
def pts_of(r): return MAXP/(1+r)
headroom = {}
for c in CONDITIONS:
    rr = dict(ratios); rr[c] = best_ratio
    R_best = R_of(rr); rr0 = dict(ratios); rr0[c] = 0.0; R_zero = R_of(rr0)
    rr1 = dict(ratios); rr1[c] = ratios[c]*0.99; R_1pct = R_of(rr1)
    headroom[c] = {
        "ratio": ratios[c],
        "R_if_matches_best_condition": R_best, "points_if_matches_best_condition": pts_of(R_best),
        "points_gain_if_matches_best": pts_of(R_best)-pts, 
        "R_if_zero_error": R_zero, "points_if_zero_error": pts_of(R_zero),
        "points_gain_if_zero": pts_of(R_zero)-pts,
        "points_per_1pct_relative_error_reduction": pts_of(R_1pct)-pts,
    }
all_best = pts_of(R_of({c: best_ratio for c in CONDITIONS}))

# ---- simplex-aware single-parameter corrections (fit per condition, in-sample)
def renorm(v):
    s = math.fsum(v)
    return [x/s for x in v]

def apply_temp(p, T):
    e = [max(x, 1e-12)**T for x in p]
    return renorm(e)

def apply_zero_shift(p, d):
    # move d of mass into p_0 (d<0 moves out), positives scaled proportionally
    p0 = p[0] + d
    p0 = min(max(p0, 1e-9), 1-1e-9)
    pos = p[1:]; s = math.fsum(pos)
    if s <= 0: return [p0] + [(1-p0)/4]*4
    f = (1-p0)/s
    return [p0] + [x*f for x in pos]

def apply_pos_temp(p, T):
    # reallocate within positive categories only, preserving p_0 and total positive mass
    pos = p[1:]; s = math.fsum(pos)
    if s <= 0: return list(p)
    e = [max(x,1e-12)**T for x in pos]; se = math.fsum(e)
    return [p[0]] + [x/se*s for x in e]

def fit_1d(ids, fn, lo, hi, iters=80):
    # golden section on condition error
    gr = (math.sqrt(5)-1)/2
    a,b = lo,hi
    c_, d_ = b-gr*(b-a), a+gr*(b-a)
    def f(t):
        pred = {q: fn(P[q], t) for q in ids}
        return cond_err(pred, ids)
    fc, fd = f(c_), f(d_)
    for _ in range(iters):
        if fc < fd: b, d_, fd = d_, c_, fc; c_ = b-gr*(b-a); fc = f(c_)
        else: a, c_, fc = c_, d_, fd; d_ = a+gr*(b-a); fd = f(d_)
    t = (a+b)/2
    return t, f(t)

def points_if_cond_error(c, new_E):
    rr = dict(ratios); rr[c] = new_E/max(conds[c]["B_c"],1e-9)
    return pts_of(R_of(rr)), R_of(rr)

worst_cond = max(ratios, key=ratios.get)
simplex = {}
for c in CONDITIONS:
    ids = by_cond[c]; E0 = conds[c]["E_c"]
    res = {}
    for name, fn, lo, hi in (("global_temperature", apply_temp, 0.2, 3.0),
                             ("zero_vs_positive_shift", apply_zero_shift, -0.30, 0.30),
                             ("within_positive_temperature", apply_pos_temp, 0.2, 3.0)):
        t, E = fit_1d(ids, fn, lo, hi)
        p_new, R_new = points_if_cond_error(c, E)
        res[name] = {"best_param": t, "fitted_E_c": E, "delta_E_c": E-E0,
                     "rel_error_reduction": (E0-E)/E0, "R_if_applied": R_new,
                     "points_if_applied": p_new, "points_gain_upper_bound_in_sample": p_new-pts}
    simplex[c] = res

# all-condition joint (each condition its own fitted param) upper bound
joint = {}
for name in ("global_temperature","zero_vs_positive_shift","within_positive_temperature"):
    rr = {c: simplex[c][name]["fitted_E_c"]/max(conds[c]["B_c"],1e-9) for c in CONDITIONS}
    joint[name] = {"R": R_of(rr), "points": pts_of(R_of(rr)), "points_gain": pts_of(R_of(rr))-pts}

# ---- tempting wrong moves
def onehot_argmax_payout(p):
    # argmax over the probability vector is the modal payout; "argmax payout" = most likely payout cell
    k = max(range(5), key=lambda i: p[i])
    v = [1e-9]*5; v[k] = 1-4e-9
    return v
wrong = {}
pred_oh = {q: onehot_argmax_payout(P[q]) for q in qids}
r_oh = official(pred_oh)
wrong["one_hot_argmax_payout"] = {"R": r_oh["mean_loss_ratio"], "points": r_oh["technical_points"],
    "points_delta": r_oh["technical_points"]-pts,
    "by_condition_ratio": {c: r_oh["by_condition"][c]["loss_ratio"] for c in CONDITIONS}}

shift20 = {}
for d in (0.005, 0.01, 0.02, 0.05, 0.10):
    def mv(p, d=d):
        take = min(d, 0.9*math.fsum(p[:4]))
        s = math.fsum(p[:4])
        f = (s-take)/s if s > 0 else 1.0
        return [x*f for x in p[:4]] + [p[4]+take]
    pr = {q: mv(P[q]) for q in qids}
    rr = official(pr)
    shift20[f"+{d}"] = {"R": rr["mean_loss_ratio"], "points": rr["technical_points"], "points_delta": rr["technical_points"]-pts}
# also multiplicative inflation of p_20
for m in (1.25, 1.5, 2.0):
    def mu(p, m=m):
        v = list(p); v[4] = min(v[4]*m, 0.999)
        return renorm(v)
    pr = {q: mu(P[q]) for q in qids}
    rr = official(pr)
    shift20[f"x{m}"] = {"R": rr["mean_loss_ratio"], "points": rr["technical_points"], "points_delta": rr["technical_points"]-pts}
wrong["shift_mass_to_20x"] = shift20

out = {
 "official_contract": {"max_points": MAXP, "R": R, "technical_points": pts,
   "formula": "R = mean_c E_c/max(B_c,1e-9); points = 60/(1+R)",
   "reference_points_check": None},
 "candidate": {"path": "artifacts/audit/oof_candidate_raw.csv", "R": R, "technical_points": pts,
   "global_error": math.fsum(conds[c]["E_c"] for c in CONDITIONS)/4.0},
 "episode_mean_equals_query_mean_residual": shortcut_ok,
 "per_condition": conds,
 "best_condition_by_ratio": best_cond, "worst_condition_by_ratio": worst_cond,
 "per_episode": episodes,
 "per_category_global": cat_global,
 "per_category_by_condition": cat_by_cond,
 "headroom": headroom,
 "points_if_all_conditions_at_best_ratio": all_best,
 "simplex_aware_in_sample_upper_bounds": simplex,
 "simplex_aware_joint_all_conditions": joint,
 "wrong_moves": wrong,
 "caveat": "All numbers are public-validation, inspected many times. Not an independent holdout.",
}
# reference scores exactly 30/60 check
rr = official(ref)
out["official_contract"]["reference_points_check"] = rr["technical_points"]
out["official_contract"]["reference_R_check"] = rr["mean_loss_ratio"]

with open(f"{ROOT}/artifacts/audit/error-breakdown/breakdown.json","w") as f:
    json.dump(out, f, indent=2, sort_keys=False)
print(json.dumps({k:v for k,v in out.items() if k not in ("per_episode","per_category_by_condition")}, indent=2))
