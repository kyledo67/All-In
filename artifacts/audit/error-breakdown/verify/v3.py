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
b=contest_score(cand,targets,groups,ref,max_points=MAXP); pts0=b["technical_points"]
Ec={c:b["by_condition"][c]["error"] for c in CONDITIONS}
Bc={c:b["by_condition"][c]["reference_error"] for c in CONDITIONS}
def cerr(pred,ids): return math.fsum(math.fsum((pred[q][k]-Q[q][k])**2 for k in range(5)) for q in ids)/len(ids)
def Rof(r): return math.fsum(r.values())/4
def pf(r): return MAXP/(1+r)
def Rp(ee): return math.fsum(ee[c]/max(Bc[c],1e-9) for c in CONDITIONS)/4
out={}

# episodes
eperr=collections.defaultdict(list); erf=collections.defaultdict(list)
for q in qids:
    eperr[ep_of[q]].append(math.fsum((P[q][k]-Q[q][k])**2 for k in range(5)))
    erf[ep_of[q]].append(math.fsum((RF[q][k]-Q[q][k])**2 for k in range(5)))
eps=[]
for e in sorted(eperr):
    c=groups[next(q for q in qids if ep_of[q]==e)]
    E=math.fsum(eperr[e])/8; B=math.fsum(erf[e])/8
    eps.append({"episode_id":e,"condition":c,"E":E,"B":B,"excess":E-B,"ratio":E/max(B,1e-9)})
conc={}
for c in CONDITIONS:
    sub=sorted([e for e in eps if e["condition"]==c],key=lambda d:-d["E"]); tot=math.fsum(e["E"] for e in sub)
    byx=sorted([e for e in eps if e["condition"]==c],key=lambda d:-d["excess"])
    conc[c]={"top3_share":math.fsum(e["E"] for e in sub[:3])/tot,
             "top6_share":math.fsum(e["E"] for e in sub[:6])/tot,
             "n_worse_than_ref":sum(1 for e in sub if e["ratio"]>1.0),
             "worst_by_E":[(e["episode_id"],round(e["E"],5)) for e in sub[:3]],
             "worst_by_excess":[(e["episode_id"],round(e["excess"],5),round(e["ratio"],5)) for e in byx[:5]]}
out["concentration"]=conc
ls={}
for c in CONDITIONS:
    sub=[e for e in eps if e["condition"]==c]
    w=[e for e in sub if e["excess"]>0]
    ls[c]={"n_worse":len(w),"sum_excess_worse":math.fsum(e["excess"] for e in w),
           "net":math.fsum(e["excess"] for e in sub),"Ec_minus_Bc":Ec[c]-Bc[c]}
out["win_loss"]=ls
rep={}
for c in CONDITIONS:
    sub=sorted([e for e in eps if e["condition"]==c],key=lambda d:-d["excess"])
    for n in (1,3,5,24):
        fx=set(e["episode_id"] for e in sub[:n])
        tot=math.fsum((e["B"] if e["episode_id"] in fx else e["E"]) for e in eps if e["condition"]==c)/24
        ee=dict(Ec); ee[c]=tot
        rep[f"{c}_top{n}"]=pf(Rp(ee))-pts0
out["repair_floor_at_ref"]=rep

# climatology
clim={}
for c in CONDITIONS:
    ids=by_cond[c]; m=[math.fsum(Q[q][k] for q in ids)/len(ids) for k in range(5)]
    clim[c]=cerr({q:m for q in ids},ids)
out["climatology"]={"points":pf(Rp(clim)),"R":Rp(clim),"ratios":{c:clim[c]/Bc[c] for c in CONDITIONS}}

# per-category skill vs climatology, global
sk=[]
for k in range(5):
    mq=math.fsum(Q[q][k] for q in qids)/len(qids)
    var=math.fsum((Q[q][k]-mq)**2 for q in qids)/len(qids)
    mse=math.fsum((P[q][k]-Q[q][k])**2 for q in qids)/len(qids)
    mser=math.fsum((RF[q][k]-Q[q][k])**2 for q in qids)/len(qids)
    sk.append({"payout":PAYOUTS[k],"cand_R2":1-mse/var,"ref_R2":1-mser/var,"true_sd":math.sqrt(var)})
out["skill_global"]=sk

# p0 explained-variance edge per condition
def catvar(ids,k):
    m=math.fsum(Q[q][k] for q in ids)/len(ids); return math.fsum((Q[q][k]-m)**2 for q in ids)/len(ids)
def catmse(ids,k,PR): return math.fsum((PR[q][k]-Q[q][k])**2 for q in ids)/len(ids)
edge={}
for c in CONDITIONS:
    ids=by_cond[c]; v=catvar(ids,0)
    edge[c]={"cand_R2":1-catmse(ids,0,P)/v,"ref_R2":1-catmse(ids,0,RF)/v,
             "edge":(1-catmse(ids,0,P)/v)-(1-catmse(ids,0,RF)/v)}
out["p0_edge"]=edge

# lift to best sibling condition
r2={c:[1-catmse(by_cond[c],k,P)/catvar(by_cond[c],k) for k in range(5)] for c in CONDITIONS}
best=[max(r2[c][k] for c in CONDITIONS) for k in range(5)]
src=[max(CONDITIONS,key=lambda c:r2[c][k]) for k in range(5)]
lift={}
for c in CONDITIONS:
    newE=math.fsum((1-max(r2[c][k],best[k]))*catvar(by_cond[c],k) for k in range(5))
    ee=dict(Ec); ee[c]=newE
    lift[c]={"newE":newE,"rel_red":(Ec[c]-newE)/Ec[c],"delta":pf(Rp(ee))-pts0}
eeall={c:math.fsum((1-max(r2[c][k],best[k]))*catvar(by_cond[c],k) for k in range(5)) for c in CONDITIONS}
lift["_joint"]={"points":pf(Rp(eeall)),"delta":pf(Rp(eeall))-pts0}
out["lift_to_sibling"]=lift
out["lift_target_source"]=dict(zip([f"p_{p}" for p in PAYOUTS],src))
out["lift_r2_table"]=r2
# SANITY: does lift ever produce newE > Ec (i.e. are categories' own R2 already best)?
out["lift_newE_vs_Ec"]={c:lift[c]["newE"]-Ec[c] for c in CONDITIONS}
print(json.dumps(out,indent=2))
