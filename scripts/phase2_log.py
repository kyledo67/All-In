"""Log the completed bounded investigation through the existing local MLflow store."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'artifacts/phase2'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 import mlflow
 parent=json.loads((ROOT/'artifacts/experiment/development.json').read_text())['mlflow']
 champion=json.loads((O/'champion/manifest.json').read_text());report=json.loads((O/'evaluation.json').read_text());protocol=json.loads((O/'protocol.json').read_text());manifest=json.loads((O/'manifest.json').read_text())
 mlflow.set_tracking_uri(parent['tracking_uri']);mlflow.set_experiment('round-one-feature-challengers')
 existing=json.loads((O/'mlflow.json').read_text()) if (O/'mlflow.json').exists() else None
 run_args={'run_id':existing['run_id']} if existing else {'run_name':'phase2-completed'}
 with mlflow.start_run(**run_args) as run:
  mlflow.set_tags({'parent_champion':champion['champion_id'],'parent_run_id':parent['run_id'],'evaluation':'previously exposed development; exploratory','deployed':'hmm','champion_changed':'false'})
  mlflow.log_params({'source_commit':champion['repository_head'],'source_identity':'content-hashed uncommitted files; see manifest','catboost_version':'1.2.8','catboost_commit':manifest['catboost_source_tag_commit'],'feature_hash':manifest['source_files']['slot_model/phase2_features.py'],'split_hash':manifest['split_sha256'],'data_hash':manifest['data_manifest_sha256'],'protocol_hash':manifest['protocol_sha256'],'dependency_hash':manifest['dependencies_sha256'],'seeds':json.dumps(protocol['seeds']),'tuning_budget':protocol['budget']})
  for model,score in report['scores'].items():
   mlflow.log_metric(model+'_points',score['technical_points']);mlflow.log_metric(model+'_ratio',score['mean_loss_ratio'])
   for c,v in score['by_condition'].items():
    mlflow.log_metric(model+'_'+c+'_error',v['error']);mlflow.log_metric(model+'_'+c+'_ratio',v['loss_ratio'])
  for p in O.iterdir():
   if p.is_file() and p.suffix in ('.json','.csv','.lock','.log'):mlflow.log_artifact(str(p),'reports')
  for directory in ('predictions','models','hmm_models','source'):
   for p in (O/directory).iterdir():
    if p.is_file():mlflow.log_artifact(str(p),directory)
  for p in ROOT.glob('docs/phase2-*.md'):mlflow.log_artifact(str(p),'docs')
  for p in ROOT.glob('scripts/phase2_*.py'):mlflow.log_artifact(str(p),'source/current')
  for p in ROOT.glob('slot_model/phase2_*.py'):mlflow.log_artifact(str(p),'source/current')
  mlflow.log_artifact(str(O/'champion/manifest.json'),'parent_champion')
  # Per-fit runs preserve actual parameters, checkpoints and measured times.
  for record in ([] if existing else json.loads((O/'training_runs.json').read_text())):
   with mlflow.start_run(run_name=record['name'],nested=True):
    mlflow.log_params({'variant':record['variant'],'seed':record['seed'],'trees':record['trees'],**record['config']})
    mlflow.log_metrics({'training_seconds':record['training_seconds'],'validation_inference_seconds':record['validation_inference_seconds']})
    mlflow.log_artifact(str(O/'models'/f'{record["name"]}.cbm'),'model')
    mlflow.log_artifact(str(O/'models'/f'{record["name"]}.json'),'configuration')
    mlflow.log_artifact(str(O/'predictions'/f'{record["name"]}.csv'),'predictions')
  result={'status':'logged','run_id':run.info.run_id,'tracking_uri':parent['tracking_uri'],'parent_champion':champion['champion_id'],'parent_run_id':parent['run_id'],'databricks':'Existing integration unchanged; local fallback reused.'}
 save(O/'mlflow.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
