"""Standalone round-one prediction interface; no network or credentials."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'organizer'))
from slot_forecasting.io import load_inputs, write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['test','validation'],default='test')
    parser.add_argument('--model',type=Path,default=ROOT/'artifacts/experiment/population.json')
    parser.add_argument('--selection',type=Path,default=ROOT/'artifacts/experiment/holdout.json')
    parser.add_argument('--engine',choices=['portable','dynamax'],default='portable')
    args=parser.parse_args()
    selected=json.loads(args.selection.read_text())['deployment']
    queries,history=load_inputs(args.data/args.split)
    if selected=='baseline':
        from slot_forecasting.baseline import FrequencyForecaster
        predictions=FrequencyForecaster.load(args.model.parent/'baseline.json').predict(queries,history)
    elif selected=='upstream' and args.engine=='portable':
        from slot_model.portable import predict_shared
        predictions=predict_shared(json.loads(args.model.read_text()),queries,history)
    else:
        from slot_model.model import deserialize,predict_batch
        from slot_model.features import prepare
        import numpy as np
        p=deserialize(json.loads(args.model.read_text()))
        precision=None if selected=='upstream' else 0. if selected=='unshrunk' else float(selected.split('_')[1])
        data={k:np.asarray(v) for k,v in prepare(queries,history).items()}
        probabilities,_,_=predict_batch(p,data,precision)
        predictions=dict(zip(data['owners'].tolist(),probabilities.tolist()))
    write_submission(args.out,predictions)
    print(f'{len(predictions)} {args.split} forecasts written to {args.out}; model={selected}; engine={args.engine}')


if __name__=='__main__':
    main()
