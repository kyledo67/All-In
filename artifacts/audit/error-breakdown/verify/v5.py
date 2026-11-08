import json, math, sys, collections
import numpy as np
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
b=contest_score(cand,targets,groups,ref,max_points=MAXP); pts0=b["technical_points"]
Bc={c:b["by_condition"][c]["reference_error"] for c in CONDITIONS}
Ec={c:b["by_condition"][c]["error"] for c in CONDITIONS}
ratios={c:b["by_condition"][c]["loss_ratio"] for c in CONDITIONS}
def Rof(r): return math.fsum(r.values())/4
def pf(r): return MAXP/(1+r)

comp={"frozen_blend":"artifacts/phase2/predictions/oof_ensemble.csv",
      "mtd_lag7":"artifacts/mtd_pairs_01/validation_lag7.csv",
      "two_head":"artifacts/two_head_01/validation.csv"}
Cd={k:read_submission(f"{ROOT}/{p}",queries) for k,p in comp.items()}
t1=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv",queries)
t2=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv",queries)
names=["frozen_blend","tabpfn_raw","mtd_lag7","two_head"]
Q=np.array([list(tmap[q].probabilities) for q in qids])
X=np.stack([
  np.array([list(Cd["frozen_blend"][q]) for q in qids]),
  np.array([[(t1[q][k]+t2[q][k])/2 for k in range(5)] for q in qids]),
  np.array([list(Cd["mtd_lag7"][q]) for q in qids]),
  np.array([list(Cd["two_head"][q]) for q in qids])])   # (4, N, 5)
PC=np.array([list(cand[q]) for q in qids])
cidx={c:np.array([i for i,q in enumerate(qids) if groups[q]==c]) for c in CONDITIONS}
out={}
out["max_abs_recomputed_minus_saved_candidate"]=None

def Emix(w, idx):
    M=np.tensordot(np.asarray(w), X[:,idx,:], axes=(0,0))
    return float(((M-Q[idx])**2).sum(axis=1).mean())

# exhaustive simplex grid, step 0.02 (g=50) -- vectorised
g=50
ws=[]
for a in range(g+1):
    for bb in range(g+1-a):
        for d in range(g+1-a-bb):
            ws.append((a/g,bb/g,d/g,(g-a-bb-d)/g))
W=np.array(ws)              # (K,4)
def best_simplex(idx):
    M=np.einsum('kc,cnp->knp', W, X[:,idx,:])
    E=((M-Q[idx][None,:,:])**2).sum(axis=2).mean(axis=1)
    j=int(E.argmin()); return float(E[j]), W[j].tolist()
ee={}; wts={}
for c in CONDITIONS:
    v,w=best_simplex(cidx[c]); ee[c]=v; wts[c]=dict(zip(names,w))
r={c:ee[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["oracle_per_condition"]={"points":pf(Rof(r)),"delta":pf(Rof(r))-pts0,"weights":wts}
allidx=np.arange(len(qids)); v,w=best_simplex(allidx)
eeg={c:Emix(w,cidx[c]) for c in CONDITIONS}
rg={c:eeg[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["oracle_global"]={"points":pf(Rof(rg)),"delta":pf(Rof(rg))-pts0,"weights":dict(zip(names,w))}
# per-query best component
L=((X-Q[None,:,:])**2).sum(axis=2)   # (4,N)
eeo={c:float(L[:,cidx[c]].min(axis=0).mean()) for c in CONDITIONS}
ro={c:eeo[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["oracle_per_query_best_component"]={"points":pf(Rof(ro)),"delta":pf(Rof(ro))-pts0,"R":Rof(ro),"E_c":eeo}

# ---- DEFECT 1 TEST: does the frozen blend have ZERO marginal value?
# Start from the candidate's own realised mixture (concatenated per-fold weights = the saved file)
# and add weight alpha on frozen_blend, rescaling the rest.
blend=X[0]
mix_tests={}
for alpha in (0.02,0.05,0.10,0.20):
    M=(1-alpha)*PC+alpha*blend
    e={c:float(((M[cidx[c]]-Q[cidx[c]])**2).sum(axis=1).mean()) for c in CONDITIONS}
    rr={c:e[c]/max(Bc[c],1e-9) for c in CONDITIONS}
    mix_tests[f"alpha={alpha}"]={"points":pf(Rof(rr)),"delta":pf(Rof(rr))-pts0}
out["add_frozen_blend_to_candidate"]=mix_tests
# unconstrained (sign-free) least-squares marginal: regress residual on (blend - candidate)
res=(Q-PC).ravel(); dirn=(blend-PC).ravel()
alpha_star=float(dirn@res/(dirn@dirn))
M=(1-alpha_star)*PC+alpha_star*blend
e={c:float(((M[cidx[c]]-Q[cidx[c]])**2).sum(axis=1).mean()) for c in CONDITIONS}
rr={c:e[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["best_unconstrained_alpha_on_frozen_blend"]={"alpha":alpha_star,"points":pf(Rof(rr)),"delta":pf(Rof(rr))-pts0}

# ---- DEFECT 2 TEST: is mtd_lag7 actually harmful? 3-component oracle without it, per condition,
#      and the candidate with mtd's weight removed.
sub=[0,1,3]  # frozen, tabpfn, two_head  (drop mtd)
Xs=X[sub]
ws3=[]
for a in range(g+1):
    for bb in range(g+1-a):
        ws3.append((a/g,bb/g,(g-a-bb)/g))
W3=np.array(ws3)
def best3(idx):
    M=np.einsum('kc,cnp->knp', W3, Xs[:,idx,:])
    E=((M-Q[idx][None,:,:])**2).sum(axis=2).mean(axis=1)
    j=int(E.argmin()); return float(E[j]), W3[j].tolist()
ee3={}; w3={}
for c in CONDITIONS:
    v,w=best3(cidx[c]); ee3[c]=v; w3[c]=dict(zip(["frozen_blend","tabpfn_raw","two_head"],w))
r3={c:ee3[c]/max(Bc[c],1e-9) for c in CONDITIONS}
out["oracle_per_condition_WITHOUT_mtd"]={"points":pf(Rof(r3)),"delta_vs_candidate":pf(Rof(r3))-pts0,
    "delta_vs_4comp_oracle":pf(Rof(r3))-pf(Rof(r)),"weights":w3}
# marginal value of mtd per condition at the oracle optimum
out["mtd_marginal_value_at_oracle_per_condition_points"]=pf(Rof(r))-pf(Rof(r3))
# leave-one-component-out of the 4-comp per-condition oracle
loo={}
for drop in range(4):
    keep=[i for i in range(4) if i!=drop]
    Xk=X[keep]
    wsk=[]
    for a in range(g+1):
        for bb in range(g+1-a): wsk.append((a/g,bb/g,(g-a-bb)/g))
    Wk=np.array(wsk)
    eek={}
    for c in CONDITIONS:
        M=np.einsum('kc,cnp->knp', Wk, Xk[:,cidx[c],:])
        E=((M-Q[cidx[c]][None,:,:])**2).sum(axis=2).mean(axis=1)
        eek[c]=float(E.min())
    rk={c:eek[c]/max(Bc[c],1e-9) for c in CONDITIONS}
    loo[names[drop]]={"oracle_points_without_it":pf(Rof(rk)),"loss_vs_full_oracle":pf(Rof(r))-pf(Rof(rk))}
out["leave_one_out_of_insample_oracle"]=loo
print(json.dumps(out,indent=2))
