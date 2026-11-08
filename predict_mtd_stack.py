"""Experimental fixed 75% frozen stack + 25% changepoint MTD release."""
import argparse,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'organizer'))
from slot_forecasting.io import load_inputs,read_submission,write_submission
from slot_model.frozen_blend import predict as predict_stack
from slot_model.mtd import MTDModel
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['test','validation'],default='test');p.add_argument('--model',type=Path,default=ROOT/'model' if (ROOT/'model').exists() else ROOT/'artifacts/blend_release/model');p.add_argument('--mtd',type=Path,default=ROOT/'mtd.json' if (ROOT/'mtd.json').exists() else ROOT/'artifacts/mtd_01/change.json');a=p.parse_args()
 start=time.perf_counter();q,h=load_inputs(a.data/a.split)
 with tempfile.TemporaryDirectory() as d:
  basefile=Path(d)/'base.csv';predict_stack(a.data,basefile,a.model,split=a.split,mode='mean');base=read_submission(basefile)
 extra=MTDModel.load(a.mtd).predict(q,h)
 if set(base)!=set(extra):raise ValueError('Query IDs differ')
 out={k:[.75*x+.25*y for x,y in zip(base[k],extra[k])] for k in base}
 write_submission(a.out,out)
 print(json.dumps({'rows':len(out),'seconds':time.perf_counter()-start,'weight':.25,'mode':'fixed experimental stack'}))
