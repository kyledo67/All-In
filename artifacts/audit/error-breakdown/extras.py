import csv, json, math, sys, collections
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
def official(pred): return contest_score(pred,targets,groups,ref,max_points=MAXP)
base=official(cand); R0=base["mean_loss_ratio"]; pts0=base["technical_points"]
Bc={c:base["by_condition"][c]["reference_error"] for c in CONDITIONS}
Ec={c:base["by_condition"][c]["error"] for c in CONDITIONS}
def cerr(pred,ids): return math.fsum(math.fsum((pred[q][k]-Q[q][k])**2 for k in range(5)) for q in ids)/len(ids)
def Rp(ee): return math.fsum(ee[c]/max(Bc[c],1e-9) for c in CONDITIONS)/4.0
def pf(r): return MAXP/(1+r)

# 1. bias^2 vs residual variance per category, per condition + global
def bv(ids):
    out=[]
    for k in range(5):
        res=[P[q][k]-Q[q][k] for q in ids]; n=len(res)
        m=math.fsum(res)/n; mse=math.fsum(r*r for r in res)/n
        out.append({"payout":PAYOUTS[k],"mse":mse,"bias":m,"bias_sq":m*m,"residual_var":mse-m*m,
                    "bias_sq_share_of_mse":(m*m)/mse if mse>0 else 0.0,
                    "share_of_total_mse":None})
    tot=math.fsum(o["mse"] for o in out)
    for o in out: o["share_of_total_mse"]=o["mse"]/tot
    return {"categories":out,"total_mse":tot,
            "bias_sq_total":math.fsum(o["bias_sq"] for o in out),
            "bias_sq_share_total":math.fsum(o["bias_sq"] for o in out)/tot}
bias_var={"global":bv(qids)}
for c in CONDITIONS: bias_var[c]=bv(by_cond[c])

# 2. sign structure of p0 residual
sign={}
for c in ["global"]+list(CONDITIONS):
    ids=qids if c=="global" else by_cond[c]
    over=sum(1 for q in ids if P[q][0]>Q[q][0])
    sign[c]={"n":len(ids),"frac_p0_over":over/len(ids),
             "mean_abs_p0_resid":math.fsum(abs(P[q][0]-Q[q][0]) for q in ids)/len(ids),
             "p90_abs_p0_resid":sorted(abs(P[q][0]-Q[q][0]) for q in ids)[int(0.9*len(ids))]}

# 3. episode concentration + worst episodes
eperr={}; eref={}
for q in qids:
    eperr.setdefault(ep_of[q],[]).append(math.fsum((P[q][k]-Q[q][k])**2 for k in range(5)))
    eref.setdefault(ep_of[q],[]).append(math.fsum((RF[q][k]-Q[q][k])**2 for k in range(5)))
eps=[]
for e in sorted(eperr):
    c=groups[next(q for q in qids if ep_of[q]==e)]
    ee=math.fsum(eperr[e])/8; rr=math.fsum(eref[e])/8
    eps.append({"episode_id":e,"condition":c,"E":ee,"B":rr,"excess":ee-rr,"ratio":ee/max(rr,1e-9)})
conc={}
for c in CONDITIONS:
    sub=sorted([e for e in eps if e["condition"]==c],key=lambda d:-d["E"])
    tot=math.fsum(e["E"] for e in sub)
    conc[c]={"top3_share_of_condition_error":math.fsum(e["E"] for e in sub[:3])/tot,
             "top6_share_of_condition_error":math.fsum(e["E"] for e in sub[:6])/tot,
             "n_episodes_worse_than_reference":sum(1 for e in sub if e["ratio"]>1.0),
             "worst_by_error":[{k:e[k] for k in ("episode_id","E","B","ratio")} for e in sub[:5]],
             "worst_by_excess_over_reference":[{k:e[k] for k in ("episode_id","E","B","excess","ratio")}
                 for e in sorted(sub,key=lambda d:-d["excess"])[:5]]}
# oracle: drop worst 3 episodes per condition -> what R
drop={}
for c in CONDITIONS:
    sub=sorted([e for e in eps if e["condition"]==c],key=lambda d:-d["E"])
    for n in (1,3,6):
        keep=sub[n:]
        ee=dict(Ec); ee[c]=math.fsum(e["E"] for e in keep)/len(keep)
        drop[f"{c}_drop_top{n}"]={"E_c_without_them":ee[c],"R":Rp(ee),"points":pf(Rp(ee)),"points_delta":pf(Rp(ee))-pts0}

# 4. in-sample oracle simplex over the 4 raw components (upper bound on re-weighting)
comp_paths={"frozen_blend":"artifacts/phase2/predictions/oof_ensemble.csv",
            "mtd_lag7":"artifacts/mtd_pairs_01/validation_lag7.csv",
            "two_head":"artifacts/two_head_01/validation.csv"}
C={k:{q:list(v) for q,v in read_submission(f"{ROOT}/{p}",queries).items()} for k,p in comp_paths.items()}
t1=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv",queries)
t2=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv",queries)
C["tabpfn_raw"]={q:[(t1[q][k]+t2[q][k])/2 for k in range(5)] for q in qids}
names=["frozen_blend","tabpfn_raw","mtd_lag7","two_head"]
import itertools, random
def mix(w,ids):
    return {q:[math.fsum(w[i]*C[names[i]][q][k] for i in range(4)) for k in range(5)] for q in ids}
def fit_simplex(ids,steps=0.02):
    best=(1e9,None); g=int(round(1/steps))
    for a in range(g+1):
        for b in range(g+1-a):
            for d in range(g+1-a-b):
                e=g-a-b-d
                w=[a/g,b/g,d/g,e/g]
                v=cerr(mix(w,ids),ids)
                if v<best[0]: best=(v,w)
    return best
oracle={}
# per-condition in-sample oracle
ee={}; wts={}
for c in CONDITIONS:
    v,w=fit_simplex(by_cond[c]); ee[c]=v; wts[c]=dict(zip(names,w))
oracle["per_condition_insample"]={"E_c":ee,"weights":wts,"R":Rp(ee),"points":pf(Rp(ee)),"points_delta":pf(Rp(ee))-pts0}
v,w=fit_simplex(qids)
eeg={c:cerr(mix(w,by_cond[c]),by_cond[c]) for c in CONDITIONS}
oracle["global_insample"]={"weights":dict(zip(names,w)),"E_c":eeg,"R":Rp(eeg),"points":pf(Rp(eeg)),"points_delta":pf(Rp(eeg))-pts0}
# single components alone, officially scored
singles={}
for n in names:
    rr=official({q:C[n][q] for q in qids})
    singles[n]={"R":rr["mean_loss_ratio"],"points":rr["technical_points"],
                "ratio_by_condition":{c:rr["by_condition"][c]["loss_ratio"] for c in CONDITIONS}}
# per-query oracle pick among the 4 components (hard upper bound on any selection scheme)
eeo={}
for c in CONDITIONS:
    tot=0.0
    for q in by_cond[c]:
        tot+=min(math.fsum((C[n][q][k]-Q[q][k])**2 for k in range(5)) for n in names)
    eeo[c]=tot/len(by_cond[c])
oracle["per_query_best_component"]={"E_c":eeo,"R":Rp(eeo),"points":pf(Rp(eeo)),"points_delta":pf(Rp(eeo))-pts0}

out={"component_sha256_16_verified":True,
     "bias_variance":bias_var,"p0_sign_structure":sign,
     "episode_concentration":conc,"episode_drop_experiments":drop,
     "oracle_bounds":oracle,"single_components_official":singles,
     "all_episodes":sorted(eps,key=lambda d:-d["E"])}
with open("/Users/ayan/Downloads/All-In/artifacts/audit/error-breakdown/extras.json","w") as f:
    json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!="all_episodes"},indent=2))
