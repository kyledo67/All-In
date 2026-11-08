import sys,json,time,os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
from slot_model.mtd import MTDModel,NAMES
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_submission
from slot_forecasting.contest import contest_score
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/mtd_01';O.mkdir(exist_ok=True)
tq,th=load_inputs(D/'train');vq,vh=load_inputs(D/'validation');labels=read_labels(D/'train/labels.csv',tq)
targets=read_targets(D/'validation/targets.csv');groups={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))};ref=read_submission(D/'validation/reference.csv')
base=read_submission(R/'artifacts/blend_release/evaluated_oof_predictions.csv')
score=lambda p:contest_score(p,targets,groups,ref,max_points=60)
report={'base_oof':score(base),'exposure':'All validation previously inspected. Fixed-weight comparisons are exploratory, not independent deployment scores.'}
for flag in (False,True):
 start=time.perf_counter();m=MTDModel(flag).fit(tq,th,labels);p=m.predict(vq,vh);name='change' if flag else 'plain';m.save(O/f'{name}.json');write_submission(O/f'{name}.csv',p)
 report[name]={'score':score(p),'seconds':time.perf_counter()-start,'strength':m.strength,'weights':{k:v['lambda'] for k,v in m.contexts.items()},'top_lags':{k:sorted(zip(NAMES,v['lambda']),key=lambda x:-x[1])[:4] for k,v in m.contexts.items()},'blends':{}}
 for w in (.15,.25,.35):
  blend={k:[(1-w)*a+w*b for a,b in zip(base[k],p[k])] for k in base};write_submission(O/f'{name}_blend_{w}.csv',blend);report[name]['blends'][str(w)]=score(blend)
 (O/'evaluation.json').write_text(json.dumps(report,indent=2));print(name,report[name]['score']['technical_points'],{w:s['technical_points'] for w,s in report[name]['blends'].items()},flush=True)
