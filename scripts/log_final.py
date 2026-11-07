"""Attach verification/deployment artifacts to the existing experiment MLflow run."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    report=json.loads((ROOT/'artifacts/experiment/development.json').read_text())
    tracking=report['mlflow']
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking['tracking_uri'])
        with mlflow.start_run(run_id=tracking['run_id']):
            upstream=json.loads((ROOT/'artifacts/upstream.json').read_text())
            provenance=json.loads((ROOT/'artifacts/source_patch_manifest.json').read_text())
            mlflow.log_params({'upstream_commit':upstream['dynamax'],
                'source_patch_sha256':provenance['artifacts/dynamax.patch'],
                'split_sha256':provenance['artifacts/splits.json'],
                'data_manifest_sha256':hashlib.sha256((ROOT/'artifacts/data_manifest.json').read_bytes()).hexdigest(),
                'deployed_model':'upstream','inference':'Python stdlib; upstream recurrence verified',
                'holdout_retuned':False})
            for phase in ('development','holdout'):
                data=json.loads((ROOT/f'artifacts/experiment/{phase}.json').read_text())
                for model,result in data[phase].items():
                    mlflow.log_metric(f'{phase}_{model}_points',result['technical_points'])
                    for condition,values in result['by_condition'].items():
                        mlflow.log_metric(f'{phase}_{model}_{condition}_loss_ratio',values['loss_ratio'])
            for directory in ('docs','slot_model','scripts','artifacts/verification','artifacts/final'):
                for path in (ROOT/directory).iterdir():
                    if path.is_file():mlflow.log_artifact(str(path),directory)
            for name in ('environment.json','source_patch_manifest.json','source_index.json','kit_verification.json'):
                mlflow.log_artifact(str(ROOT/'artifacts'/name),'provenance')
            mlflow.log_artifact(str(ROOT/'predict.py'),'source')
            mlflow.log_artifact(str(ROOT/'requirements-training.lock'),'source')
        result={'status':'logged','run_id':tracking['run_id'],'tracking_uri':tracking['tracking_uri'],
                'databricks':'No configured integration found; existing local MLflow fallback reused.'}
    except Exception as exc:
        result={'status':'local_files_retained','error':repr(exc)}
    (ROOT/'artifacts/final_tracking.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
