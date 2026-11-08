import json, math, sys, collections
sys.path.insert(0,"/Users/ayan/Downloads/All-In/organizer")
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_targets, read_submission, read_queries, read_rows
from slot_forecasting.schema import PAYOUTS, CONDITIONS
ROOT="/Users/ayan/Downloads/All-In"; PUB=f"{ROOT}/main-participants-60-40/public/validation"; MAXP=60.0
SP="/Users/ayan/Downloads/All-In/artifacts/audit/error-breakdown"
queries={q.query_id:q for q in read_queries(f"{PUB}/queries.csv")}
targets=tuple(read_targets(f"{PUB}/targets.csv"))
groups={r["query_id"]:r["condition"] for r in read_rows(f"{PUB}/conditions.csv",("query_id","condition"))}
ref=read_submission(f"{PUB}/reference.csv",queries); cand=read_submission(f"{ROOT}/artifacts/audit/oof_candidate_raw.csv",queries)
tmap={t.query_id:t for t in targets}; qids=sorted(tmap)
P={q:list(cand[q]) for q in qids}; Q={q:list(tmap[q].probabilities) for q in qids}
by_cond=collections.defaultdict(list)
for q in qids: by_cond[groups[q]].append(q)
base=contest_score(cand,targets,groups,ref,max_points=MAXP); pts0=base["technical_points"]
Bc={c:base["by_condition"][c]["reference_error"] for c in CONDITIONS}
Ec={c:base["by_condition"][c]["error"] for c in CONDITIONS}
def Rp(ee): return math.fsum(ee[c]/max(Bc[c],1e-9) for c in CONDITIONS)/4.0
def pf(r): return MAXP/(1+r)
def catmse(ids,k): return math.fsum((P[q][k]-Q[q][k])**2 for q in ids)/len(ids)
def catvar(ids,k):
    m=math.fsum(Q[q][k] for q in ids)/len(ids); return math.fsum((Q[q][k]-m)**2 for q in ids)/len(ids)

# A) zero out one payout category's error everywhere (hard decomposition bound)
zero_cat={}
for k in range(5):
    ee={c:Ec[c]-catmse(by_cond[c],k) for c in CONDITIONS}
    zero_cat[f"p_{PAYOUTS[k]}"]={"share_of_global_mse":catmse(qids,k)/ (math.fsum(Ec.values())/4),
        "R":Rp(ee),"points":pf(Rp(ee)),"points_delta":pf(Rp(ee))-pts0}
# B) lift each condition's per-category R2 to the best R2 any condition achieves for that category
r2={c:[1-catmse(by_cond[c],k)/catvar(by_cond[c],k) for k in range(5)] for c in CONDITIONS}
best_r2=[max(r2[c][k] for c in CONDITIONS) for k in range(5)]
best_src=[max(CONDITIONS,key=lambda c:r2[c][k]) for k in range(5)]
lift={}
for c in CONDITIONS:
    newE=math.fsum((1-max(r2[c][k],best_r2[k]))*catvar(by_cond[c],k) for k in range(5))
    ee=dict(Ec); ee[c]=newE
    lift[c]={"current_R2_by_category":r2[c],"target_R2_by_category":best_r2,
             "target_source_condition":best_src,"E_c_if_lifted":newE,
             "rel_error_reduction":(Ec[c]-newE)/Ec[c],"R":Rp(ee),"points":pf(Rp(ee)),
             "points_delta":pf(Rp(ee))-pts0}
eeall={c:math.fsum((1-max(r2[c][k],best_r2[k]))*catvar(by_cond[c],k) for k in range(5)) for c in CONDITIONS}
lift["_all_conditions_jointly"]={"R":Rp(eeall),"points":pf(Rp(eeall)),"points_delta":pf(Rp(eeall))-pts0}
# C) candidate-minus-reference explained-variance edge on p_0
RFp={q:list(ref[q]) for q in qids}
edge={}
for c in CONDITIONS:
    ids=by_cond[c]; v=catvar(ids,0)
    mr=math.fsum((RFp[q][0]-Q[q][0])**2 for q in ids)/len(ids)
    edge[c]={"cand_R2_p0":1-catmse(ids,0)/v,"ref_R2_p0":1-mr/v,"edge":(1-catmse(ids,0)/v)-(1-mr/v)}
out={"zero_one_payout_category":zero_cat,"lift_skill_to_best_sibling_condition":lift,"p0_explained_variance_edge_over_reference":edge}
with open(f"{SP}/final.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))

# ---- MERGE everything into artifacts/audit/error-breakdown/breakdown.json
bd=json.load(open(f"{ROOT}/artifacts/audit/error-breakdown/breakdown.json"))
bd["bias_variance_split"]=json.load(open(f"{SP}/extras.json"))["bias_variance"]
e1=json.load(open(f"{SP}/extras.json")); e2=json.load(open(f"{SP}/extras2.json"))
bd["p0_sign_structure"]=e1["p0_sign_structure"]
bd["episode_concentration"]=e1["episode_concentration"]
bd["episode_drop_experiments"]=e1["episode_drop_experiments"]
bd["oracle_bounds"]=e1["oracle_bounds"]
bd["single_components_official"]=e1["single_components_official"]
bd["climatology_benchmark"]=e2["climatology_benchmark"]
bd["per_category_skill_vs_climatology"]=e2["per_category_skill_vs_climatology"]
bd["targeted_repair_floor_at_reference"]=e2["targeted_repair_floor_at_reference"]
bd["win_loss_split_by_episode"]=e2["win_loss_split_by_episode"]
bd.update(out)
bd["component_sha256_16_match_recompute_json"]=True
bd["provenance"]={
 "candidate":"artifacts/audit/oof_candidate_raw.csv (sha256[:16]=3e101db53475fc1e)",
 "truth":"main-participants-60-40/public/validation/targets.csv",
 "reference":"main-participants-60-40/public/validation/reference.csv (scores exactly 30.0/60, R=1.0)",
 "scorer":"organizer/slot_forecasting/contest.contest_score(..., max_points=60), unmodified",
 "scripts":[f"{SP}/breakdown.py",f"{SP}/extras.py",f"{SP}/extras2.py",f"{SP}/final.py"]}
bd["holdout_status"]="PUBLIC VALIDATION ONLY. All 96 episodes have been inspected repeatedly across this project. Every number here, including the in-sample fitted upper bounds, is NOT an independent holdout and must not be described as confirmed."
with open(f"{ROOT}/artifacts/audit/error-breakdown/breakdown.json","w") as f: json.dump(bd,f,indent=2)
print("\nWROTE", f"{ROOT}/artifacts/audit/error-breakdown/breakdown.json", "keys:", len(bd))
