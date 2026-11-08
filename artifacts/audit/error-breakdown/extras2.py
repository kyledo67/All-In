import json, math, sys, collections
sys.path.insert(0,"/Users/ayan/Downloads/All-In/organizer")
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_targets, read_submission, read_queries, read_rows
from slot_forecasting.schema import PAYOUTS, CONDITIONS
ROOT="/Users/ayan/Downloads/All-In"; PUB=f"{ROOT}/main-participants-60-40/public/validation"; MAXP=60.0
queries={q.query_id:q for q in read_queries(f"{PUB}/queries.csv")}
targets=tuple(read_targets(f"{PUB}/targets.csv"))
groups={r["query_id"]:r["condition"] for r in read_rows(f"{PUB}/conditions.csv",("query_id","condition"))}
ref=read_submission(f"{PUB}/reference.csv",queries); cand=read_submission(f"{ROOT}/artifacts/audit/oof_candidate_raw.csv",queries)
tmap={t.query_id:t for t in targets}; qids=sorted(tmap)
P={q:list(cand[q]) for q in qids}; Q={q:list(tmap[q].probabilities) for q in qids}; RF={q:list(ref[q]) for q in qids}
by_cond=collections.defaultdict(list)
for q in qids: by_cond[groups[q]].append(q)
ep_of={q:tmap[q].episode_id for q in qids}
base=contest_score(cand,targets,groups,ref,max_points=MAXP)
pts0=base["technical_points"]; R0=base["mean_loss_ratio"]
Bc={c:base["by_condition"][c]["reference_error"] for c in CONDITIONS}
Ec={c:base["by_condition"][c]["error"] for c in CONDITIONS}
def cerr(pred,ids): return math.fsum(math.fsum((pred[q][k]-Q[q][k])**2 for k in range(5)) for q in ids)/len(ids)
def Rp(ee): return math.fsum(ee[c]/max(Bc[c],1e-9) for c in CONDITIONS)/4.0
def pf(r): return MAXP/(1+r)

# --- climatology benchmark: constant = mean true vector (per condition, in-sample)
clim={}
for c in CONDITIONS:
    ids=by_cond[c]; m=[math.fsum(Q[q][k] for q in ids)/len(ids) for k in range(5)]
    clim[c]={"const_vector":m,"E_c":cerr({q:m for q in ids},ids)}
eecl={c:clim[c]["E_c"] for c in CONDITIONS}
clim_summary={"per_condition":clim,"R":Rp(eecl),"points":pf(Rp(eecl)),
              "ratio_by_condition":{c:eecl[c]/Bc[c] for c in CONDITIONS}}

# --- how much per-query signal: variance explained on each category
skill={}
for c in ["global"]+list(CONDITIONS):
    ids=qids if c=="global" else by_cond[c]
    row=[]
    for k in range(5):
        mq=math.fsum(Q[q][k] for q in ids)/len(ids)
        var=math.fsum((Q[q][k]-mq)**2 for q in ids)/len(ids)
        mse=math.fsum((P[q][k]-Q[q][k])**2 for q in ids)/len(ids)
        mser=math.fsum((RF[q][k]-Q[q][k])**2 for q in ids)/len(ids)
        row.append({"payout":PAYOUTS[k],"true_var":var,"cand_mse":mse,"ref_mse":mser,
                    "cand_R2_vs_climatology":1-mse/var if var>0 else None,
                    "ref_R2_vs_climatology":1-mser/var if var>0 else None,
                    "true_sd":math.sqrt(var)})
    skill[c]=row

# --- targeted repair: make the N episodes with largest excess-over-reference exactly match reference
eperr={}; erf={}
for q in qids:
    eperr.setdefault(ep_of[q],[]).append(math.fsum((P[q][k]-Q[q][k])**2 for k in range(5)))
    erf.setdefault(ep_of[q],[]).append(math.fsum((RF[q][k]-Q[q][k])**2 for k in range(5)))
eps=[]
for e in sorted(eperr):
    c=groups[next(q for q in qids if ep_of[q]==e)]
    ee=math.fsum(eperr[e])/8; rr=math.fsum(erf[e])/8
    eps.append({"episode_id":e,"condition":c,"E":ee,"B":rr,"excess":ee-rr})
repair={}
for c in CONDITIONS:
    sub=sorted([e for e in eps if e["condition"]==c],key=lambda d:-d["excess"])
    for n in (1,3,5,24):
        fixed=set(e["episode_id"] for e in sub[:n])
        tot=math.fsum((e["B"] if e["episode_id"] in fixed else e["E"]) for e in eps if e["condition"]==c)/24
        ee=dict(Ec); ee[c]=tot
        repair[f"{c}_floor_at_reference_top{n}"]={"n_episodes":n,"E_c":tot,"R":Rp(ee),"points":pf(Rp(ee)),
            "points_delta":pf(Rp(ee))-pts0,
            "episodes":[e["episode_id"] for e in sub[:n]] if n<=5 else "all 24"}

# --- how much of each condition's ratio gap is explained by episodes where we LOSE to reference
loss_split={}
for c in CONDITIONS:
    sub=[e for e in eps if e["condition"]==c]
    worse=[e for e in sub if e["excess"]>0]; better=[e for e in sub if e["excess"]<=0]
    loss_split[c]={"n_worse_than_ref":len(worse),"sum_excess_worse":math.fsum(e["excess"] for e in worse),
                   "sum_excess_better":math.fsum(e["excess"] for e in better),
                   "net_excess":math.fsum(e["excess"] for e in sub),
                   "E_c_minus_B_c":Ec[c]-Bc[c]}
out={"climatology_benchmark":clim_summary,"per_category_skill_vs_climatology":skill,
     "targeted_repair_floor_at_reference":repair,"win_loss_split_by_episode":loss_split}
with open("/Users/ayan/Downloads/All-In/artifacts/audit/error-breakdown/extras2.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
