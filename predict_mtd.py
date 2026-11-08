"""Standalone NumPy-only MTD inference with the organizer CSV interface."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'organizer'))
from slot_model.mtd import MTDModel
from slot_forecasting.io import load_inputs,write_submission
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--split',default='test',choices=['test','validation']);a=p.parse_args()
 q,h=load_inputs(a.data/a.split);write_submission(a.out,MTDModel.load(a.model).predict(q,h))
