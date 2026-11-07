"""Record verified local promotion and append artifacts to existing local MLflow."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'artifacts/blend_release'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    champ=json.loads((ROOT/'artifacts/phase2/champion/manifest.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in champ['hashes'].items())
    frozen=json.loads((O/'frozen_artifacts.json').read_text())
    assert all(sha(O/p)==v for p,v in frozen['files'].items())
    rep=json.loads((O/'reproduction.json').read_text());assert rep['byte_identical'] and rep['rows']==1536
    hashes=json.loads((O/'checksums.json').read_text());assert all(sha(O/p)==v for p,v in hashes.items())
    deployment=json.loads((O/'model/deployment.json').read_text());policy=json.loads((O/'policy_change.json').read_text())
    assert deployment['fixed_deployment_weights']==[.25]*4
    source_files=[ROOT/'predict_blend.py']+list((ROOT/'slot_model').glob('*.py'))+[ROOT/'scripts'/n for n in ['freeze_saved_blend.py','audit_frozen_blend.py','package_frozen_blend.py','finalize_frozen_blend.py']]
    manifest={'champion_id':'frozen-fold-mean-'+hashes['submission.csv'][:12],'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_identity':'Worktree implementation is uncommitted. Exact packaged source/model identity is pinned by entry.zip and its source_manifest.json.',
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source_files},
        'artifact_hashes':{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ('manifest.json','mlflow.json')},
        'deployed_average_independent_validation_score':None,'existing_combined_oof_points':deployment['cv_evidence_points'],
        'new_search_training_or_weight_fitting':False,'fallback_hashes_verified_unchanged':True}
    save(O/'manifest.json',manifest)
    active={'champion_id':manifest['champion_id'],'status':'verified local submission candidate; not uploaded',
        'deployment':deployment['deployment'],'recipe_identifiers':deployment['recipes'],'fixed_deployment_weights':[.25]*4,
        'submission':str(O/'submission.csv'),'entry':str(O/'entry.zip'),'sha256':hashes,
        'model':str(O/'model'),'reproduction':str(O/'reproduction.json'),'reproduction_seconds':rep['seconds'],
        'existing_combined_oof_points':deployment['cv_evidence_points'],'deployed_average_independent_validation_score':None,
        'policy_change':str(O/'policy_change.json'),'fallback':{'champion_id':champ['champion_id'],'submission':str(ROOT/'artifacts/final/submission.csv'),'entry':str(ROOT/'artifacts/final/entry.zip'),'hashes_unchanged':True},'recorded_unix':time.time()}
    save(ROOT/'artifacts/active_champion.json',active)
    try:
        import mlflow
        parent=json.loads((ROOT/'artifacts/phase2/mlflow.json').read_text());mlflow.set_tracking_uri(parent['tracking_uri']);mlflow.set_experiment('round-one-feature-challengers')
        existing=json.loads((O/'mlflow.json').read_text()) if (O/'mlflow.json').exists() else None
        kwargs={'run_id':existing['run_id']} if existing and existing.get('status')=='logged' else {'run_name':'frozen-fold-average-promotion'}
        with mlflow.start_run(**kwargs) as run:
            mlflow.set_tags({'parent_run_id':parent['run_id'],'parent_champion':champ['champion_id'],'promotion':'user-authorized post-evaluation policy revision','test_score':'unavailable','deployed_average_independent_score':'unavailable'})
            mlflow.log_params({'aggregation':'equal mean of four frozen probability vectors','deployment_weights':'[0.25,0.25,0.25,0.25]','recipe_ids':json.dumps([r['recipe_id'] for r in deployment['recipes']]),'new_fit_or_search':False,'entry_sha256':hashes['entry.zip'],'submission_sha256':hashes['submission.csv']})
            mlflow.log_metrics({'existing_combined_oof_points':deployment['cv_evidence_points'],'existing_oof_gain_vs_hmm':policy['gain_points'],'clean_reproduction_seconds':rep['seconds'],'max_reproduction_probability_difference':rep['max_probability_difference']})
            for p in O.rglob('*'):
                if p.is_file():mlflow.log_artifact(str(p),str(Path('release')/p.relative_to(O).parent))
            for p in source_files:mlflow.log_artifact(str(p),str(Path('source')/p.relative_to(ROOT).parent))
            mlflow.log_artifact(str(ROOT/'docs/blend-promotion.md'),'docs')
            mlflow.log_artifact(str(ROOT/'artifacts/active_champion.json'),'deployment')
            result={'status':'logged','run_id':run.info.run_id,'tracking_uri':parent['tracking_uri'],'parent_run_id':parent['run_id']}
    except Exception as exc:result={'status':'local_manifests_retained','error':repr(exc)}
    save(O/'mlflow.json',result);print(json.dumps({'active':active,'mlflow':result},indent=2))
if __name__=='__main__':main()
