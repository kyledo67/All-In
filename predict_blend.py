"""Offline CLI for frozen blends: no fitting, no target files, no network calls."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'organizer'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/all-in-mpl')
os.environ.setdefault('OMP_NUM_THREADS','2')
from slot_model.frozen_blend import predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--split',choices=['test','validation'],default='test');ap.add_argument('--model',type=Path,default=ROOT/'artifacts/blend_release/model')
    ap.add_argument('--mode',choices=['heldfold','mean','fold0','fold1','fold2','fold3'])
    ap.add_argument('--audit-dir',type=Path)
    ap.add_argument('--tabpfn-weight',type=float,default=0.,help='Fixed weight for the TabPFN component; 0 keeps the frozen blend unchanged.')
    ap.add_argument('--tabpfn-model',type=Path,help='Saved TabPFN context directory; defaults to <model>/tabpfn.')
    args=ap.parse_args();print(json.dumps(predict(args.data,args.out,args.model,split=args.split,mode=args.mode,audit_dir=args.audit_dir,tabpfn_weight=args.tabpfn_weight,tabpfn_model=args.tabpfn_model)))
if __name__=='__main__':main()
