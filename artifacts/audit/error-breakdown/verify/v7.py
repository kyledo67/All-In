import json, math, sys, collections
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/Users/ayan/Downloads/All-In/organizer")
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_targets, read_submission, read_queries, read_rows
from slot_forecasting.schema import CONDITIONS
ROOT="/Users/ayan/Downloads/All-In"; PUB=f"{ROOT}/main-participants-60-40/public/validation"; MAXP=60.0
queries={q.query_id:q for q in read_queries(f"{PUB}/queries.csv")}
targets=tuple(read_targets(f"{PUB}/targets.csv"))
groups={r["query_id"]:r["condition"] for r in read_rows(f"{PUB}/conditions.csv",("query_id","condition"))}
ref=read_submission(f"{PUB}/reference.csv",queries)
cand=read_submission(f"{ROOT}/artifacts/audit/oof_candidate_raw.csv",queries)
tmap={t.query_id:t for t in targets}; qids=sorted(tmap)
Q=np.array([list(tmap[q].probabilities) for q in qids])
PC=np.array([list(cand[q]) for q in qids])
Cd={k:read_submission(f"{ROOT}/{p}",queries) for k,p in
    {"frozen_blend":"artifacts/phase2/predictions/oof_ensemble.csv",
     "mtd_lag7":"artifacts/mtd_pairs_01/validation_lag7.csv",
     "two_head":"artifacts/two_head_01/validation.csv"}.items()}
t1=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv",queries)
t2=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv",queries)
names=["frozen_blend","tabpfn_raw","mtd_lag7","two_head"]
X=np.stack([np.array([list(Cd["frozen_blend"][q]) for q in qids]),
            np.array([[(t1[q][k]+t2[q][k])/2 for k in range(5)] for q in qids]),
            np.array([list(Cd["mtd_lag7"][q]) for q in qids]),
            np.array([list(Cd["two_head"][q]) for q in qids])])
ev=json.load(open(f"{ROOT}/artifacts/phase2/evaluation.json"))
held=[set(f["held_episodes"]) for f in ev["folds"]]
epof=[tmap[q].episode_id for q in qids]
fold_of=np.full(len(qids),-1)
for i,e in enumerate(epof):
    for f,s in enumerate(held):
        if e in s: fold_of[i]=f
out={"n_folds":len(held),"fold_sizes_episodes":[len(s) for s in held],
     "unassigned_queries":int((fold_of<0).sum()),
     "fold_query_counts":[int((fold_of==f).sum()) for f in range(len(held))]}
rc=json.load(open(f"{ROOT}/artifacts/audit/recompute.json"))
FW=[[w[n] for n in names] for w in rc["fold_weights"]]

# A) reconstruct the saved candidate from components + recompute.json fold weights
REC=np.zeros_like(PC)
for f in range(4):
    m=fold_of==f
    REC[m]=np.tensordot(np.array(FW[f]),X[:,m,:],axes=(0,0))
out["reconstruct_saved_candidate_max_abs_diff"]=float(np.abs(REC-PC).max())
rr=contest_score({q:list(REC[i]) for i,q in enumerate(qids)},targets,groups,ref,max_points=MAXP)
out["reconstructed_points"]=rr["technical_points"]

# B) INDEPENDENTLY refit the per-fold simplex (fit on other 3 folds, apply to held fold)
def fit(idx):
    def f(z):
        w=np.exp(z-z.max()); w/=w.sum()
        M=np.tensordot(w,X[:,idx,:],axes=(0,0))
        return ((M-Q[idx])**2).sum(axis=1).mean()
    best=(1e9,None)
    for s in range(12):
        z0=np.zeros(4); 
        if s: 
            rng=np.random.default_rng(s); z0=rng.normal(0,1.5,4)
        r=minimize(f,z0,method="Nelder-Mead",options={"maxiter":20000,"xatol":1e-12,"fatol":1e-15})
        if r.fun<best[0]:
            w=np.exp(r.x-r.x.max()); w/=w.sum(); best=(float(r.fun),w)
    return best[1]
MY=np.zeros_like(PC); myw=[]
for f in range(4):
    tr=np.where(fold_of!=f)[0]; te=np.where(fold_of==f)[0]
    w=fit(tr); myw.append(dict(zip(names,[float(x) for x in w])))
    MY[te]=np.tensordot(w,X[:,te,:],axes=(0,0))
mr=contest_score({q:list(MY[i]) for i,q in enumerate(qids)},targets,groups,ref,max_points=MAXP)
out["my_independent_refit"]={"points":mr["technical_points"],"R":mr["mean_loss_ratio"],
  "weights":myw,"max_abs_pred_diff_vs_saved":float(np.abs(MY-PC).max())}
out["recompute_json_fold_weights"]=rc["fold_weights"]
print(json.dumps(out,indent=2))
