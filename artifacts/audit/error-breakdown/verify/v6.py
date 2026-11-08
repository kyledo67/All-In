import json, math, sys, collections, random
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
Q=np.array([list(tmap[q].probabilities) for q in qids], dtype=np.float64)
PC=np.array([list(cand[q]) for q in qids], dtype=np.float64)
RFm=np.array([list(ref[q]) for q in qids], dtype=np.float64)
cidx={c:np.array([i for i,q in enumerate(qids) if groups[q]==c]) for c in CONDITIONS}
epof=np.array([tmap[q].episode_id for q in qids])
comp={"frozen_blend":"artifacts/phase2/predictions/oof_ensemble.csv",
      "mtd_lag7":"artifacts/mtd_pairs_01/validation_lag7.csv",
      "two_head":"artifacts/two_head_01/validation.csv"}
Cd={k:read_submission(f"{ROOT}/{p}",queries) for k,p in comp.items()}
t1=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv",queries)
t2=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv",queries)
names=["frozen_blend","tabpfn_raw","mtd_lag7","two_head"]
X=np.stack([np.array([list(Cd["frozen_blend"][q]) for q in qids]),
            np.array([[(t1[q][k]+t2[q][k])/2 for k in range(5)] for q in qids]),
            np.array([list(Cd["mtd_lag7"][q]) for q in qids]),
            np.array([list(Cd["two_head"][q]) for q in qids])]).astype(np.float64)
out={}

# 1. clean alpha* on frozen blend (plain python, no BLAS)
res=(Q-PC).ravel().tolist(); dirn=(X[0]-PC).ravel().tolist()
num=math.fsum(a*c for a,c in zip(dirn,res)); den=math.fsum(a*a for a in dirn)
al=num/den
M=(1-al)*PC+al*X[0]
e={c:float(((M[cidx[c]]-Q[cidx[c]])**2).sum(axis=1).mean()) for c in CONDITIONS}
out["frozen_blend_best_unconstrained_alpha"]={"alpha":al,"num":num,"den":den,
  "points":pf(Rof({c:e[c]/Bc[c] for c in CONDITIONS})),
  "gain":pf(Rof({c:e[c]/Bc[c] for c in CONDITIONS}))-pts0}

# 2. is "every component's weakest condition is short_history" true?
weak={}
for i,n in enumerate(names):
    rr=contest_score({q:list(X[i][j]) for j,q in enumerate(qids)},targets,groups,ref,max_points=MAXP)
    rd={c:rr["by_condition"][c]["loss_ratio"] for c in CONDITIONS}
    weak[n]={"ratios":rd,"worst":max(rd,key=rd.get)}
weak["candidate"]={"ratios":ratios,"worst":max(ratios,key=ratios.get)}
out["worst_condition_per_component"]=weak

# 3. the specific defect-2 claim: is mtd OVER-weighted in short_history by the fold weights?
g=100
W=np.array([(a/g,(g-a)/g) for a in range(g+1)])
sh=cidx["short_history"]
# hold tabpfn:two_head at the global-oracle ratio 0.48:0.32 and sweep mtd weight 0..0.5
sweep={}
for mw in [0.0,0.05,0.10,0.1096,0.15,0.20,0.2473,0.30,0.40]:
    rest=1-mw
    w=np.array([0.0, rest*0.48/0.80, mw, rest*0.32/0.80])
    Mx=np.tensordot(w,X[:,sh,:],axes=(0,0))
    E=float(((Mx-Q[sh])**2).sum(axis=1).mean())
    r=dict(ratios); r["short_history"]=E/Bc["short_history"]
    sweep[f"mtd_w={mw}"]={"E_short_history":E,"points":pf(Rof(r)),"delta_vs_candidate":pf(Rof(r))-pts0}
out["short_history_mtd_weight_sweep"]=sweep

# 4. p_0 DISPERSION, not bias: regression slope of q0 on p0 and sd comparison
disp={}
for c in ["global"]+list(CONDITIONS):
    idx=np.arange(len(qids)) if c=="global" else cidx[c]
    p0=PC[idx,0]; q0=Q[idx,0]; r0=RFm[idx,0]
    sl=float(np.cov(p0,q0,bias=True)[0,1]/np.var(p0))
    disp[c]={"sd_pred_p0":float(p0.std()),"sd_true_p0":float(q0.std()),
             "sd_ratio_pred_over_true":float(p0.std()/q0.std()),
             "ols_slope_q0_on_p0":sl,"corr":float(np.corrcoef(p0,q0)[0,1]),
             "ref_sd_pred_p0":float(r0.std()),
             "ref_ols_slope":float(np.cov(r0,q0,bias=True)[0,1]/np.var(r0))}
out["p0_dispersion"]=disp

# 5. CROSS-FIT the recalibration: fit on half the episodes, score the other half (2-fold, episode-level)
def renorm(a):
    s=a.sum(axis=1,keepdims=True); s[s<=0]=1.0; return a/s
def pava_fit(x,y):
    o=np.argsort(x,kind="mergesort"); xs=x[o]; ys=y[o]
    lev=[]; cnt=[]
    for v in ys:
        lev.append(float(v)); cnt.append(1)
        while len(lev)>1 and lev[-2]>lev[-1]:
            s=lev[-2]*cnt[-2]+lev[-1]*cnt[-1]; k=cnt[-2]+cnt[-1]
            lev.pop(); cnt.pop(); lev[-1]=s/k; cnt[-1]=k
    fit=np.repeat(np.array(lev),np.array(cnt))
    return xs,fit
def pava_apply(xs,fit,v):
    i=np.searchsorted(xs,v,side="left"); i=np.clip(i,0,len(fit)-1); return fit[i]
rng=random.Random(20261107)
eps_all=sorted(set(epof.tolist()))
folds=collections.defaultdict(list)
for c in CONDITIONS:
    ce=sorted({epof[i] for i in cidx[c]}); rng.shuffle(ce)
    for j,e in enumerate(ce): folds[j%2].append(e)
cf={}
for kind in ("affine","isotonic"):
    Eout={c:0.0 for c in CONDITIONS}; nout={c:0 for c in CONDITIONS}
    for f in (0,1):
        test=set(folds[f]); tr=np.array([i for i in range(len(qids)) if epof[i] not in test])
        te=np.array([i for i in range(len(qids)) if epof[i] in test])
        for c in CONDITIONS:
            trc=np.array([i for i in tr if groups[qids[i]]==c]); tec=np.array([i for i in te if groups[qids[i]]==c])
            V=np.zeros((len(tec),5))
            for k in range(5):
                if kind=="affine":
                    xp=PC[trc,k]; yq=Q[trc,k]
                    sl=np.cov(xp,yq,bias=True)[0,1]/np.var(xp) if np.var(xp)>0 else 0.0
                    a0=yq.mean()-sl*xp.mean()
                    V[:,k]=np.maximum(0.0,a0+sl*PC[tec,k])
                else:
                    xs,fit=pava_fit(PC[trc,k],Q[trc,k])
                    V[:,k]=np.maximum(0.0,pava_apply(xs,fit,PC[tec,k]))
            V=renorm(V)
            Eout[c]+=((V-Q[tec])**2).sum(); nout[c]+=len(tec)
    Ecf={c:Eout[c]/nout[c] for c in CONDITIONS}
    r={c:Ecf[c]/Bc[c] for c in CONDITIONS}
    cf[kind]={"E_c":Ecf,"points":pf(Rof(r)),"gain_vs_candidate":pf(Rof(r))-pts0}
out["recalibration_CROSSFIT_2fold_episode_level"]=cf
print(json.dumps(out,indent=2))
