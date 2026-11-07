"""One frozen controlled experiment; holdout scoring is a separate command."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_model import runtime
import numpy as np
from slot_model.features import cached_prepare
from slot_model.model import CONFIG, fit_population, predict_batch, serialize, deserialize
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_targets, read_submission, write_json, write_submission
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import subset_score
from slot_forecasting.schema import CONDITIONS

def scoring_inputs(data):
    targets = read_targets(data/'validation/targets.csv')
    groups = {r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv', ('query_id','condition'))}
    reference = read_submission(data/'validation/reference.csv')
    return targets, groups, reference

def score(pred, scoring, episodes):
    return subset_score(pred, *scoring, set(episodes), 60)

def paired_bootstrap(a, b, scoring, episodes, repeats=2000):
    """Resample same whole episodes for both models AND frozen reference."""
    targets, groups, ref = scoring
    selected = [t for t in targets if t.episode_id in set(episodes)]
    rows = {}
    for c in CONDITIONS:
        es = sorted({t.episode_id for t in selected if groups[t.query_id] == c})
        vals = []
        for e in es:
            ts = [t for t in selected if t.episode_id == e]
            vals.append([np.mean([np.sum((np.asarray(p[t.query_id])-t.probabilities)**2) for t in ts]) for p in (a,b,ref)])
        rows[c] = np.asarray(vals)
    rng = np.random.default_rng(314159)
    totals = np.zeros(repeats)
    conditions = {}
    for c, v in rows.items():
        draws = rng.integers(0, len(v), (repeats,len(v)))
        means = v[draws].mean(1)
        diff = (means[:,0]-means[:,1])/np.maximum(means[:,2],1e-9)
        totals += diff/4
        conditions[c] = np.quantile(diff,[.025,.975]).tolist()
    return dict(delta_loss_ratio_ci95=np.quantile(totals,[.025,.975]).tolist(),
                by_condition_ci95=conditions, repeats=repeats, seed=314159,
                interpretation='negative favors first model', unit='whole episode, stratified by condition')

def log_mlflow(out, report):
    try:
        import mlflow
        mlflow.set_tracking_uri((Path('artifacts/mlruns').resolve()).as_uri())
        mlflow.set_experiment('round-one-controlled-dynamax')
        with mlflow.start_run(run_name=out.name) as run:
            mlflow.log_params({k:json.dumps(v) for k,v in CONFIG.items()})
            for p in ('artifacts/splits.json','artifacts/data_manifest.json','artifacts/upstream.json','artifacts/dynamax.patch'):
                mlflow.log_artifact(p, 'provenance')
            for name, result in report['development'].items():
                mlflow.log_metric(name+'_dev_points',result['technical_points'])
            for path in out.iterdir():
                if path.is_file():
                    mlflow.log_artifact(str(path))
            return {'run_id':run.info.run_id,'tracking_uri':mlflow.get_tracking_uri(), 'databricks':'not configured in supplied project; local fallback'}
    except Exception as exc:
        return {'error':repr(exc),'fallback':'all manifests and artifacts retained locally'}

def develop(data, out):
    if (out/'finalist.json').exists():
        raise RuntimeError('Already frozen; choose a new output directory for a disclosed new experiment')
    out.mkdir(parents=True,exist_ok=True)
    splits=json.loads(Path('artifacts/splits.json').read_text())
    tq,th=load_inputs(data/'train'); vq,vh=load_inputs(data/'validation')
    labels=read_labels(data/'train/labels.csv',tq)
    base=FrequencyForecaster('pooled').fit(tq,th,labels)
    base.save(out/'baseline.json')
    bp=base.predict(vq,vh)
    write_submission(out/'baseline.csv',bp)
    scoring=scoring_inputs(data)
    reference_diff=max(abs(a-b) for k in bp for a,b in zip(bp[k],scoring[2][k]))
    assert reference_diff < 1e-14
    train,tk=cached_prepare(data/'train',tq,th,training=True)
    val,vk=cached_prepare(data/'validation',vq,vh)
    write_json(out/'config.json', CONFIG)
    print('Training upstream control', train['x'].shape, flush=True)
    model, params, initial, diag=fit_population(train)
    write_json(out/'population.json',serialize(params)); write_json(out/'initial.json',serialize(initial))
    write_json(out/'population_diagnostics.json',diag)
    development={'baseline':score(bp,scoring,splits['development'])}
    diagnostics={}
    for name,precision in [('upstream',None),('unshrunk',0.0),('pooled_0.3',0.3),('pooled_3',3.0),('pooled_30',30.0)]:
        print('Forecasting',name,flush=True)
        probs,delta,dd=predict_batch(params,val,precision)
        pred=dict(zip(val['owners'].tolist(),probs.tolist()))
        write_submission(out/f'{name}.csv',pred)
        np.savez_compressed(out/f'{name}_adaptation.npz',deviations=delta,owners=val['owners'])
        write_json(out/f'{name}_diagnostics.json',dd)
        development[name]=score(read_submission(out/f'{name}.csv'),scoring,splits['development'])
        diagnostics[name]={'seconds':dd['seconds'],'deviation_rms':dd['deviation_rms'],'decreases':dd['decreases']}
        print(name, development[name]['technical_points'],flush=True)
    best_pool=min(['pooled_0.3','pooled_3','pooled_30'],key=lambda n:development[n]['mean_loss_ratio'])
    fallback=min(['baseline','upstream','unshrunk'],key=lambda n:development[n]['mean_loss_ratio'])
    finalist=min([fallback,best_pool],key=lambda n:development[n]['mean_loss_ratio'])
    report={'development':development,'diagnostics':diagnostics,'baseline_reference_max_difference':reference_diff,
            'cache_keys':{'train':tk,'validation':vk},'catboost':'No existing benchmark found in supplied project; not added',
            'splits_sha256':hashlib.sha256(Path('artifacts/splits.json').read_bytes()).hexdigest(),
            'training_labels':'Organizer baseline uses released labels; HMM uses observed history windows only',
            'best_pool':best_pool,'fallback':fallback,'finalist':finalist,'holdout_used_for_tuning':False}
    write_json(out/'development.json',report)
    write_json(out/'finalist.json',{'finalist':finalist,'best_pool':best_pool,'fallback':fallback,
        'frozen_at_unix':time.time(),'config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),
        'retention_rule':'partial pooling must beat upstream and unshrunk in paired holdout ratio CI; otherwise frozen development fallback'})
    report['mlflow']=log_mlflow(out,report)
    write_json(out/'development.json',report)

def audit(data,out):
    if (out/'holdout.json').exists():
        raise RuntimeError('Holdout already audited; do not silently reuse it')
    frozen=json.loads((out/'finalist.json').read_text())
    splits=json.loads(Path('artifacts/splits.json').read_text())
    scoring=scoring_inputs(data)
    # Only the preselected pooling setting, and required controls, are audited.
    names=['baseline','upstream','unshrunk',frozen['best_pool']]
    preds={n:read_submission(out/f'{n}.csv') for n in names}
    reports={n:score(p,scoring,splits['holdout']) for n,p in preds.items()}
    comparisons={n:paired_bootstrap(preds[frozen['best_pool']],preds[n],scoring,splits['holdout']) for n in ('upstream','unshrunk','baseline')}
    retain=all(comparisons[n]['delta_loss_ratio_ci95'][1]<0 for n in ('upstream','unshrunk'))
    chosen=frozen['finalist']
    if chosen.startswith('pooled') and not retain:
        chosen=frozen['fallback']
    report={'holdout':reports,'paired_bootstrap':comparisons,'retain_modification':retain,
            'deployment':chosen,'holdout_used_for_tuning':False,
            'decision':'Predeclared confirmatory retention gate; no retuning or refit on holdout targets'}
    write_json(out/'holdout.json',report)
    print(json.dumps(report,indent=2))
    try:
        import mlflow
        tracking=json.loads((out/'development.json').read_text())['mlflow']
        mlflow.set_tracking_uri(tracking['tracking_uri'])
        with mlflow.start_run(run_id=tracking['run_id']):
            mlflow.log_artifact(str(out/'holdout.json'))
            for n,r in reports.items(): mlflow.log_metric(n+'_holdout_points',r['technical_points'])
    except Exception as exc:
        write_json(out/'mlflow_audit_error.json',{'error':repr(exc)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['develop','audit'])
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('artifacts/experiment'))
    a=ap.parse_args(); (develop if a.phase=='develop' else audit)(a.data,a.out)
