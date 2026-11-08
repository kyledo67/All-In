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
def official(pred): return contest_score(pred,targets,groups,ref,max_points=MAXP)
b=official(cand); pts0=b["technical_points"]
Ec={c:b["by_condition"][c]["error"] for c in CONDITIONS}
Bc={c:b["by_condition"][c]["reference_error"] for c in CONDITIONS}
ratios={c:b["by_condition"][c]["loss_ratio"] for c in CONDITIONS}
def cerr(pred,ids): return math.fsum(math.fsum((pred[q][k]-Q[q][k])**2 for k in range(5)) for q in ids)/len(ids)
def Rof(r): return math.fsum(r.values())/4
def pf(r): return MAXP/(1+r)
out={}

# ===== 1. simplex-aware single-parameter corrections, BRUTE FORCE grid (not golden section)
def renorm(v):
    s=math.fsum(v); return [x/s for x in v]
def apply_temp(p,T): return renorm([max(x,1e-12)**T for x in p])
def apply_zero_shift(p,d):
    p0=min(max(p[0]+d,1e-9),1-1e-9); pos=p[1:]; s=math.fsum(pos)
    if s<=0: return [p0]+[(1-p0)/4]*4
    f=(1-p0)/s; return [p0]+[x*f for x in pos]
def apply_pos_temp(p,T):
    pos=p[1:]; s=math.fsum(pos)
    if s<=0: return list(p)
    e=[max(x,1e-12)**T for x in pos]; se=math.fsum(e)
    return [p[0]]+[x/se*s for x in e]
def grid_fit(ids,fn,lo,hi,n=2000):
    best=(1e9,None)
    for i in range(n+1):
        t=lo+(hi-lo)*i/n
        v=cerr({q:fn(P[q],t) for q in ids},ids)
        if v<best[0]: best=(v,t)
    return best[1],best[0]
simplex={}
for c in CONDITIONS:
    ids=by_cond[c]; res={}
    for name,fn,lo,hi in (("global_temperature",apply_temp,0.2,3.0),
                          ("zero_vs_positive_shift",apply_zero_shift,-0.30,0.30),
                          ("within_positive_temperature",apply_pos_temp,0.2,3.0)):
        t,E=grid_fit(ids,fn,lo,hi)
        r=dict(ratios); r[c]=E/max(Bc[c],1e-9)
        res[name]={"best_param":t,"E":E,"gain":pf(Rof(r))-pts0}
    simplex[c]=res
out["simplex_gridfit"]=simplex
joint={}
for name in ("global_temperature","zero_vs_positive_shift","within_positive_temperature"):
    r={c:simplex[c][name]["E"]/max(Bc[c],1e-9) for c in CONDITIONS}
    joint[name]={"points":pf(Rof(r)),"gain":pf(Rof(r))-pts0}
out["simplex_joint"]=joint

# ===== 2. wrong moves
def onehot(p):
    k=max(range(5),key=lambda i:p[i]); v=[1e-9]*5; v[k]=1-4e-9; return v
r=official({q:onehot(P[q]) for q in qids})
out["one_hot"]={"points":r["technical_points"],"delta":r["technical_points"]-pts0,"R":r["mean_loss_ratio"],
                "by_cond":{c:r["by_condition"][c]["loss_ratio"] for c in CONDITIONS}}
s20={}
for d in (0.005,0.01,0.02,0.05,0.10):
    def mv(p,d=d):
        s=math.fsum(p[:4]); take=min(d,0.9*s); f=(s-take)/s if s>0 else 1.0
        return [x*f for x in p[:4]]+[p[4]+take]
    rr=official({q:mv(P[q]) for q in qids}); s20[f"+{d}"]=rr["technical_points"]-pts0
for m in (1.25,1.5,2.0):
    def mu(p,m=m):
        v=list(p); v[4]=min(v[4]*m,0.999); return renorm(v)
    rr=official({q:mu(P[q]) for q in qids}); s20[f"x{m}"]=rr["technical_points"]-pts0
out["shift20"]=s20

# ===== 3. components, official
comp={"frozen_blend":"artifacts/phase2/predictions/oof_ensemble.csv",
      "mtd_lag7":"artifacts/mtd_pairs_01/validation_lag7.csv",
      "two_head":"artifacts/two_head_01/validation.csv"}
C={k:{q:list(v) for q,v in read_submission(f"{ROOT}/{p}",queries).items()} for k,p in comp.items()}
t1=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv",queries)
t2=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv",queries)
C["tabpfn_raw"]={q:[(t1[q][k]+t2[q][k])/2 for k in range(5)] for q in qids}
names=["frozen_blend","tabpfn_raw","mtd_lag7","two_head"]
singles={}
for n in names:
    rr=official({q:C[n][q] for q in qids})
    singles[n]={"points":rr["technical_points"],"R":rr["mean_loss_ratio"],
                "ratio":{c:rr["by_condition"][c]["loss_ratio"] for c in CONDITIONS}}
out["singles"]=singles
# is frozen_blend STRICTLY dominated by tabpfn per condition?
out["blend_vs_tabpfn_per_condition_ratio_diff"]={c:singles["frozen_blend"]["ratio"][c]-singles["tabpfn_raw"]["ratio"][c] for c in CONDITIONS}

# ===== 4. oracle simplex, in-sample
def mix(w,ids): return {q:[math.fsum(w[i]*C[names[i]][q][k] for i in range(4)) for k in range(5)] for q in ids}
def fit_simplex(ids,g=50):
    best=(1e9,None)
    for a in range(g+1):
        for bb in range(g+1-a):
            for d in range(g+1-a-bb):
                e=g-a-bb-d; w=[a/g,bb/g,d/g,e/g]
                v=cerr(mix(w,ids),ids)
                if v<best[0]: best=(v,w)
    return best
ee={}; wts={}
for c in CONDITIONS:
    v,w=fit_simplex(by_cond[c]); ee[c]=v; wts[c]=dict(zip(names,w))
r={c:ee[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["oracle_per_condition"]={"points":pf(Rof(r)),"delta":pf(Rof(r))-pts0,"weights":wts}
v,w=fit_simplex(qids)
eeg={c:cerr(mix(w,by_cond[c]),by_cond[c]) for c in CONDITIONS}
rg={c:eeg[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["oracle_global"]={"points":pf(Rof(rg)),"delta":pf(Rof(rg))-pts0,"weights":dict(zip(names,w))}
eeo={c:math.fsum(min(math.fsum((C[n][q][k]-Q[q][k])**2 for k in range(5)) for n in names) for q in by_cond[c])/len(by_cond[c]) for c in CONDITIONS}
ro={c:eeo[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["oracle_per_query"]={"points":pf(Rof(ro)),"delta":pf(Rof(ro))-pts0,"R":Rof(ro),"E_c":eeo}
print(json.dumps(out,indent=2))
