# Things the audit did NOT do: (1) paired episode bootstrap on the claims that drive its
# conclusions, (2) richer-than-single-parameter recalibration upper bound.
import json, math, sys, collections, random
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
ratios={c:b["by_condition"][c]["loss_ratio"] for c in CONDITIONS}
def Rof(r): return math.fsum(r.values())/4
def pf(r): return MAXP/(1+r)
out={}

# ---- episode-level loss arrays per condition
ep_ids={c:sorted({ep_of[q] for q in by_cond[c]}) for c in CONDITIONS}
def epl(PR):
    d=collections.defaultdict(list)
    for q in qids: d[ep_of[q]].append(math.fsum((PR[q][k]-Q[q][k])**2 for k in range(5)))
    return {e:math.fsum(v)/len(v) for e,v in d.items()}
Le=epl(P); Lr=epl(RF)

comp={"frozen_blend":"artifacts/phase2/predictions/oof_ensemble.csv",
      "mtd_lag7":"artifacts/mtd_pairs_01/validation_lag7.csv",
      "two_head":"artifacts/two_head_01/validation.csv"}
C={k:{q:list(v) for q,v in read_submission(f"{ROOT}/{p}",queries).items()} for k,p in comp.items()}
t1=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv",queries)
t2=read_submission(f"{ROOT}/artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv",queries)
C["tabpfn_raw"]={q:[(t1[q][k]+t2[q][k])/2 for k in range(5)] for q in qids}
Lb=epl(C["frozen_blend"])

# ---- (1) PAIRED EPISODE BOOTSTRAP (organizer resamples episodes; see scoring.bootstrap_mean_interval)
def pts_from(Lp, Lq, sample):
    r={}
    for c in CONDITIONS:
        ids=sample[c]
        e=math.fsum(Lp[i] for i in ids)/len(ids); bb=math.fsum(Lq[i] for i in ids)/len(ids)
        r[c]=e/max(bb,1e-9)
    return pf(Rof(r)), r
rng=random.Random(314159)
NB=4000
d_cand_blend=[]; r_sh=[]; r_dr=[]; d_sh_dr=[]; argworst=collections.Counter(); pts_boot=[]
for _ in range(NB):
    s={c:[rng.choice(ep_ids[c]) for _ in ep_ids[c]] for c in CONDITIONS}
    pc,rc=pts_from(Le,Lr,s); pb,rb=pts_from(Lb,Lr,s)
    pts_boot.append(pc); d_cand_blend.append(pc-pb)
    r_sh.append(rc["short_history"]); r_dr.append(rc["drift"])
    d_sh_dr.append(rc["short_history"]-rc["drift"])
    argworst[max(rc,key=rc.get)]+=1
def q(v,a):
    v=sorted(v); p=(len(v)-1)*a; lo=math.floor(p); hi=math.ceil(p)
    return v[lo]+(v[hi]-v[lo])*(p-lo)
out["bootstrap_episodes_4000"]={
 "candidate_points_ci95":[q(pts_boot,0.025),q(pts_boot,0.975)],
 "candidate_minus_frozen_blend_points_mean":math.fsum(d_cand_blend)/NB,
 "candidate_minus_frozen_blend_points_ci95":[q(d_cand_blend,0.025),q(d_cand_blend,0.975)],
 "frac_candidate_beats_blend":sum(1 for x in d_cand_blend if x>0)/NB,
 "ratio_short_history_ci95":[q(r_sh,0.025),q(r_sh,0.975)],
 "ratio_drift_ci95":[q(r_dr,0.025),q(r_dr,0.975)],
 "short_history_minus_drift_ratio_mean":math.fsum(d_sh_dr)/NB,
 "short_history_minus_drift_ratio_ci95":[q(d_sh_dr,0.025),q(d_sh_dr,0.975)],
 "frac_short_history_ratio_worse_than_drift":sum(1 for x in d_sh_dr if x>0)/NB,
 "argmax_worst_condition_frequency":dict(argworst)}

# ---- (2) recalibration richer than one parameter: per-category linear map then renormalise,
#          and isotonic (PAVA) on p_0, both fitted IN-SAMPLE per condition.
def renorm(v):
    s=math.fsum(v)
    return [x/s for x in v] if s>0 else [0.2]*5
def cerr(PR,ids): return math.fsum(math.fsum((PR[q][k]-Q[q][k])**2 for k in range(5)) for q in ids)/len(ids)

def pava(x,y,w=None):
    # isotonic regression of y on sorted x; returns (sorted x, fitted level) blocks
    n=len(x); idx=sorted(range(n),key=lambda i:x[i])
    xs=[x[i] for i in idx]; ys=[y[i] for i in idx]
    lev=[]; cnt=[]
    for v in ys:
        lev.append(v); cnt.append(1)
        while len(lev)>1 and lev[-2]>lev[-1]:
            s=lev[-2]*cnt[-2]+lev[-1]*cnt[-1]; c=cnt[-2]+cnt[-1]
            lev.pop(); cnt.pop(); lev[-1]=s/c; cnt[-1]=c
    fit=[]; 
    for v,c in zip(lev,cnt): fit.extend([v]*c)
    return xs, fit
def iso_apply(xs,fit,v):
    import bisect
    i=bisect.bisect_left(xs,v)
    return fit[min(max(i,0),len(fit)-1)]

recal={}
for c in CONDITIONS:
    ids=by_cond[c]
    # (a) per-category affine a_k + b_k * p_k, least squares on q_k, then renormalise+clip
    new={}
    coef=[]
    for k in range(5):
        xs=[P[q_][k] for q_ in ids]; ys=[Q[q_][k] for q_ in ids]; n=len(xs)
        mx=math.fsum(xs)/n; my=math.fsum(ys)/n
        sxx=math.fsum((v-mx)**2 for v in xs); sxy=math.fsum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        bb=sxy/sxx if sxx>0 else 0.0; aa=my-bb*mx
        coef.append((aa,bb))
    for q_ in ids:
        v=[max(0.0,coef[k][0]+coef[k][1]*P[q_][k]) for k in range(5)]
        new[q_]=renorm(v)
    E_aff=cerr(new,ids)
    # (b) isotonic per category on p_k -> q_k, then renormalise
    isos=[]
    for k in range(5):
        xs,fit=pava([P[q_][k] for q_ in ids],[Q[q_][k] for q_ in ids]); isos.append((xs,fit))
    new2={}
    for q_ in ids:
        v=[max(0.0,iso_apply(isos[k][0],isos[k][1],P[q_][k])) for k in range(5)]
        new2[q_]=renorm(v)
    E_iso=cerr(new2,ids)
    recal[c]={"E_c":Ec[c],"affine_percat_E":E_aff,"affine_coef":coef,"isotonic_percat_E":E_iso}
for name,key in (("affine_percat","affine_percat_E"),("isotonic_percat","isotonic_percat_E")):
    r={c:recal[c][key]/max(Bc[c],1e-9) for c in CONDITIONS}
    out[f"recal_{name}_joint"]={"points":pf(Rof(r)),"gain_in_sample":pf(Rof(r))-pts0,
        "per_condition_gain":{c:pf(Rof({**ratios,c:recal[c][key]/max(Bc[c],1e-9)}))-pts0 for c in CONDITIONS}}
out["recal_detail"]={c:{"E_c":recal[c]["E_c"],"affine":recal[c]["affine_percat_E"],
                        "isotonic":recal[c]["isotonic_percat_E"],"coef":recal[c]["affine_coef"]} for c in CONDITIONS}
print(json.dumps(out,indent=2))
