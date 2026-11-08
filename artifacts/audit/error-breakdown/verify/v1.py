import json, math, sys, collections
sys.path.insert(0, "/Users/ayan/Downloads/All-In/organizer")
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_targets, read_submission, read_queries, read_rows
from slot_forecasting.schema import PAYOUTS, CONDITIONS

ROOT="/Users/ayan/Downloads/All-In"; PUB=f"{ROOT}/main-participants-60-40/public/validation"; MAXP=60.0
queries={q.query_id:q for q in read_queries(f"{PUB}/queries.csv")}
targets=tuple(read_targets(f"{PUB}/targets.csv"))
groups={r["query_id"]:r["condition"] for r in read_rows(f"{PUB}/conditions.csv",("query_id","condition"))}
ref=read_submission(f"{PUB}/reference.csv",queries)
cand=read_submission(f"{ROOT}/artifacts/audit/oof_candidate_raw.csv",queries)
tmap={t.query_id:t for t in targets}; qids=sorted(tmap)
P={q:list(cand[q]) for q in qids}; Q={q:list(tmap[q].probabilities) for q in qids}; RF={q:list(ref[q]) for q in qids}
by_cond=collections.defaultdict(list)
for q in qids: by_cond[groups[q]].append(q)
ep_of={q:tmap[q].episode_id for q in qids}

out={}
# ---- structural facts
cnt=collections.Counter()
for q in qids: cnt[ep_of[q]]+=1
out["n_queries"]=len(qids); out["n_episodes"]=len(cnt)
out["distinct_episode_query_counts"]=sorted(set(cnt.values()))
out["episodes_per_condition"]={c:len({ep_of[q] for q in by_cond[c]}) for c in CONDITIONS}
out["queries_per_condition"]={c:len(by_cond[c]) for c in CONDITIONS}

# ---- raw CSV row-sum check (before probabilities() renormalisation)
import csv
mx=0.0
with open(f"{ROOT}/artifacts/audit/oof_candidate_raw.csv") as f:
    for r in csv.DictReader(f):
        s=math.fsum(float(r[f"p_{p}"]) for p in PAYOUTS)
        mx=max(mx, abs(s-1.0))
out["candidate_csv_max_abs_rowsum_minus_1"]=mx

# ---- official scores
b=contest_score(cand,targets,groups,ref,max_points=MAXP)
rr=contest_score(ref,targets,groups,ref,max_points=MAXP)
out["candidate_points"]=b["technical_points"]; out["candidate_R"]=b["mean_loss_ratio"]
out["reference_points"]=rr["technical_points"]; out["reference_R"]=rr["mean_loss_ratio"]
pts0=b["technical_points"]
Ec={c:b["by_condition"][c]["error"] for c in CONDITIONS}
Bc={c:b["by_condition"][c]["reference_error"] for c in CONDITIONS}
ratios={c:b["by_condition"][c]["loss_ratio"] for c in CONDITIONS}
out["per_condition"]={c:{"E_c":Ec[c],"B_c":Bc[c],"ratio":ratios[c],"contrib":ratios[c]/4} for c in CONDITIONS}
out["sum_contrib"]=math.fsum(ratios[c]/4 for c in CONDITIONS)

def cerr(pred,ids): return math.fsum(math.fsum((pred[q][k]-Q[q][k])**2 for k in range(5)) for q in ids)/len(ids)
out["flat_query_mean_minus_organizer_Ec"]={c:cerr(P,by_cond[c])-Ec[c] for c in CONDITIONS}
out["global_query_mean"]=cerr(P,qids)
out["mean_of_condition_errors"]=math.fsum(Ec.values())/4

# ---- per-category
cats=[]
tot=cerr(P,qids)
for k in range(5):
    mse=math.fsum((P[q][k]-Q[q][k])**2 for q in qids)/len(qids)
    bias=math.fsum(P[q][k]-Q[q][k] for q in qids)/len(qids)
    cats.append({"payout":PAYOUTS[k],"mse":mse,"share":mse/tot,"bias":bias,"bias_sq":bias*bias,
                 "bias_sq_share_of_own_mse":bias*bias/mse})
out["per_category_global"]=cats
out["sum_category_mse"]=math.fsum(c["mse"] for c in cats)
out["bias_sq_share_total"]=math.fsum(c["bias_sq"] for c in cats)/tot
bss={}
for c in CONDITIONS:
    ids=by_cond[c]; t=cerr(P,ids)
    bs=math.fsum((math.fsum(P[q][k]-Q[q][k] for q in ids)/len(ids))**2 for k in range(5))
    bss[c]=bs/t
out["bias_sq_share_by_condition"]=bss

# ---- p0 sign
sg={}
for c in ["global"]+list(CONDITIONS):
    ids=qids if c=="global" else by_cond[c]
    sg[c]={"frac_over":sum(1 for q in ids if P[q][0]>Q[q][0])/len(ids),
           "mean_abs":math.fsum(abs(P[q][0]-Q[q][0]) for q in ids)/len(ids),
           "p90":sorted(abs(P[q][0]-Q[q][0]) for q in ids)[int(0.9*len(ids))]}
out["p0_sign"]=sg

# ---- headroom
def Rof(r): return math.fsum(r.values())/4
def pf(r): return MAXP/(1+r)
hr={}
best=min(ratios.values())
for c in CONDITIONS:
    z=dict(ratios); z[c]=0.0
    m=dict(ratios); m[c]=best
    o=dict(ratios); o[c]=ratios[c]*0.99
    hr[c]={"zero_pts":pf(Rof(z)),"zero_gain":pf(Rof(z))-pts0,
           "match_gain":pf(Rof(m))-pts0,"per_1pct":pf(Rof(o))-pts0}
out["headroom"]=hr
out["all_at_best"]=pf(Rof({c:best for c in CONDITIONS}))
out["all_at_best_gain"]=pf(Rof({c:best for c in CONDITIONS}))-pts0

# ---- zero one payout category
zc={}
for k in range(5):
    ee={c:Ec[c]-math.fsum((P[q][k]-Q[q][k])**2 for q in by_cond[c])/len(by_cond[c]) for c in CONDITIONS}
    r={c:ee[c]/max(Bc[c],1e-9) for c in CONDITIONS}
    zc[f"p_{PAYOUTS[k]}"]={"pts":pf(Rof(r)),"delta":pf(Rof(r))-pts0}
out["zero_category"]=zc

print(json.dumps(out,indent=2))
