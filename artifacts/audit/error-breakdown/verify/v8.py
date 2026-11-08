import json, math, sys, collections
import numpy as np
sys.path.insert(0,"/Users/ayan/Downloads/All-In/organizer")
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_targets, read_submission, read_queries, read_rows
from slot_forecasting.schema import CONDITIONS, PAYOUTS
ROOT="/Users/ayan/Downloads/All-In"; PUB=f"{ROOT}/main-participants-60-40/public/validation"
queries={q.query_id:q for q in read_queries(f"{PUB}/queries.csv")}
targets=tuple(read_targets(f"{PUB}/targets.csv"))
groups={r["query_id"]:r["condition"] for r in read_rows(f"{PUB}/conditions.csv",("query_id","condition"))}
ref=read_submission(f"{PUB}/reference.csv",queries); cand=read_submission(f"{ROOT}/artifacts/audit/oof_candidate_raw.csv",queries)
tmap={t.query_id:t for t in targets}; qids=sorted(tmap)
Q=np.array([list(tmap[q].probabilities) for q in qids]); P=np.array([list(cand[q]) for q in qids])
RF=np.array([list(ref[q]) for q in qids])
epof=np.array([tmap[q].episode_id for q in qids])
cidx={c:np.array([i for i,q in enumerate(qids) if groups[q]==c]) for c in CONDITIONS}
ep_of_cond={c:sorted({epof[i] for i in cidx[c]}) for c in CONDITIONS}
rows_of_ep={}
for i,e in enumerate(epof): rows_of_ep.setdefault(e,[]).append(i)

def r2(idx,k,PR):
    m=Q[idx,k].mean(); var=((Q[idx,k]-m)**2).mean()
    return 1-((PR[idx,k]-Q[idx,k])**2).mean()/var
pt={c:[r2(cidx[c],k,P) for k in range(5)] for c in CONDITIONS}
# bootstrap SD of each condition's per-category R2, resampling its 24 episodes
rng=np.random.default_rng(7)
NB=2000
sd={c:[] for c in CONDITIONS}
samples={c:np.zeros((NB,5)) for c in CONDITIONS}
for c in CONDITIONS:
    eps=ep_of_cond[c]
    for b in range(NB):
        pick=rng.integers(0,len(eps),len(eps))
        idx=np.concatenate([rows_of_ep[eps[j]] for j in pick])
        for k in range(5): samples[c][b,k]=r2(idx,k,P)
out={}
out["per_category_R2_point"]={c:[float(x) for x in pt[c]] for c in CONDITIONS}
out["per_category_R2_bootstrap_sd"]={c:[float(samples[c][:,k].std()) for k in range(5)] for c in CONDITIONS}
out["cross_condition_R2_spread_max_minus_min"]=[float(max(pt[c][k] for c in CONDITIONS)-min(pt[c][k] for c in CONDITIONS)) for k in range(5)]
out["ratio_spread_over_typical_bootstrap_sd"]=[
  float((max(pt[c][k] for c in CONDITIONS)-min(pt[c][k] for c in CONDITIONS))/
        np.mean([samples[c][:,k].std() for c in CONDITIONS])) for k in range(5)]
# Does short_history remain the condition with the largest "lift to best sibling" gain under resampling?
# Bootstrap the WHOLE lift statistic jointly (resample episodes within every condition each draw).
def Ec_Bc(idxmap):
    E={}; B={}
    for c in CONDITIONS:
        idx=idxmap[c]
        E[c]=((P[idx]-Q[idx])**2).sum(axis=1).mean(); B[c]=((RF[idx]-Q[idx])**2).sum(axis=1).mean()
    return E,B
win=collections.Counter(); lifts={c:[] for c in CONDITIONS}
edge_win=collections.Counter()
for b in range(NB):
    im={}
    for c in CONDITIONS:
        eps=ep_of_cond[c]; pick=rng.integers(0,len(eps),len(eps))
        im[c]=np.concatenate([rows_of_ep[eps[j]] for j in pick])
    E,B=Ec_Bc(im)
    r2b={c:[r2(im[c],k,P) for k in range(5)] for c in CONDITIONS}
    best=[max(r2b[c][k] for c in CONDITIONS) for k in range(5)]
    R0=sum(E[c]/max(B[c],1e-9) for c in CONDITIONS)/4; p0=60/(1+R0)
    g={}
    for c in CONDITIONS:
        var=[(((Q[im[c],k]-Q[im[c],k].mean())**2).mean()) for k in range(5)]
        newE=sum((1-max(r2b[c][k],best[k]))*var[k] for k in range(5))
        rr={cc:E[cc]/max(B[cc],1e-9) for cc in CONDITIONS}; rr[c]=newE/max(B[c],1e-9)
        g[c]=60/(1+sum(rr.values())/4)-p0
        lifts[c].append(g[c])
    win[max(g,key=g.get)]+=1
    # p0 edge: which condition has the THINNEST edge over reference on p_0
    ed={c:r2(im[c],0,P)-r2(im[c],0,RF) for c in CONDITIONS}
    edge_win[min(ed,key=ed.get)]+=1
out["bootstrap_argmax_lift_condition_freq"]={k:v/NB for k,v in win.items()}
out["bootstrap_lift_gain_mean_and_ci95"]={c:[float(np.mean(lifts[c])),float(np.percentile(lifts[c],2.5)),float(np.percentile(lifts[c],97.5))] for c in CONDITIONS}
out["bootstrap_thinnest_p0_edge_condition_freq"]={k:v/NB for k,v in edge_win.items()}
print(json.dumps(out,indent=2))
