"""Package a user-selected deployment of existing recipes, with exact offline replay."""
import argparse,hashlib,json,math,os,platform,resource,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'artifacts/blend_release'
sys.path.insert(0,str(ROOT/'organizer'))
from slot_forecasting.submission_contract import package_code,extract_archive
from slot_forecasting.io import read_submission

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):p.write_text(json.dumps(value,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args();args.data=args.data.resolve()
    deployment=json.loads((O/'model/deployment.json').read_text())
    if deployment['mode'] not in ('mean','fold0','fold1','fold2','fold3'):raise ValueError('Test deployment must be explicitly resolved')
    if (O/'entry.zip').exists():raise RuntimeError('Preserve existing archive; do not overwrite')
    frozen=json.loads((O/'frozen_artifacts.json').read_text())
    assert all(sha(O/name)==digest for name,digest in frozen['files'].items())
    # The final CSV comes from current source and the frozen saved models.
    command=[sys.executable,'-B',str(ROOT/'predict_blend.py'),'--data',str(args.data),'--out',str(O/'submission.csv'),'--audit-dir',str(O/'recipe_predictions')]
    result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=1800)
    save(O/'inference.json',json.loads(result.stdout))
    individual=[read_submission(O/'recipe_predictions'/f'fold{i}.csv') for i in range(4)]
    combined=read_submission(O/'submission.csv')
    assert all(set(p)==set(combined) for p in individual)
    mean_difference=max(abs(combined[q][c]-math.fsum(p[q][c] for p in individual)/4)
                        for q in combined for c in range(5))
    assert mean_difference<1e-12
    save(O/'aggregation_verification.json',{'queries':len(combined),'query_ids_aligned':True,
         'payout_order':[0,1,2,5,20],'deployment_weights':[0.25]*4,
         'max_difference_from_id_aligned_mean_of_serialized_recipes':mean_difference,
         'max_sum_error':max(abs(math.fsum(p)-1) for p in combined.values()),
         'all_finite_and_nonnegative':all(math.isfinite(v) and 0<=v<=1 for p in combined.values() for v in p)})
    for split,path in [('test',O/'submission.csv'),('validation',O/'reconstructed_oof_predictions.csv')]:
        checked=subprocess.run([sys.executable,str(ROOT/'organizer/check_submission.py'),str(path),'--data',str(args.data),'--split',split],check=True,capture_output=True,text=True)
        (O/f'{split}_check.txt').write_text(checked.stdout)
    with tempfile.TemporaryDirectory(prefix='all-in-frozen-blend-') as temp:
        temp=Path(temp);stage=temp/'source';stage.mkdir()
        def copy(src,dest):dest=stage/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        copy(ROOT/'predict_blend.py','predict.py')
        for name in ('__init__.py','features.py','portable.py','phase2_features.py','phase2_predict.py','frozen_blend.py'):
            copy(ROOT/'slot_model'/name,'slot_model/'+name)
        for p in (ROOT/'organizer/slot_forecasting').glob('*.py'):copy(p,'organizer/slot_forecasting/'+p.name)
        copy(ROOT/'organizer/check_submission.py','organizer/check_submission.py')
        for p in (O/'model').rglob('*'):
            if p.is_file():copy(p,'model/'+p.relative_to(O/'model').as_posix())
        for name in ('requirements.txt','CATBOOST_LICENSE','policy_change.json','frozen_artifacts.json','oof_verification.json','recipe_validity.json','aggregation_verification.json','oof_score_audit.json'):
            copy(O/name,name)
        copy(ROOT/'research/vendor/dynamax/LICENSE','DYNAMAX_LICENSE')
        copy(ROOT/'THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md')
        copy(ROOT/'scripts/freeze_saved_blend.py','scripts/freeze_saved_blend.py')
        copy(ROOT/'scripts/package_frozen_blend.py','scripts/package_frozen_blend.py')
        copy(ROOT/'docs/blend-promotion.md','docs/blend-promotion.md')
        save(stage/'reproduce.json',{'image':'python','command':['python','predict.py','--data','/data','--out','/output/submission.csv','--model','model']})
        metadata=json.loads((args.data/'metadata.json').read_text());mode=deployment['mode']
        (stage/'README.md').write_text(f'''# All-In frozen-blend main entry

Season: {metadata['season_id']}. Round: main. Select the registered team in the form.
Deployment mode: `{mode}`. All four recorded fold recipes and six CatBoost checkpoints
are unchanged. See policy_change.json and docs/blend-promotion.md for authorization,
the post-evaluation policy change, and the distinction between OOF evidence and test
aggregation. No new fit, weight optimization, calibration, or configuration search.

## Reproduce offline

Python {platform.python_version()} tested. Organizers install exact root-level
requirements.txt dependencies before the offline run. No JAX, REA, Databricks,
MLflow, credentials or APIs are required for prediction.

```sh
python predict.py --data /data --out /output/submission.csv --model model
```

Read-only inputs: /data/test/queries.csv and history.csv. Saved models/prior/recipes
and feature schema are under model/. The uploaded CSV is not an input. Outputs
follow 0,1,2,5,20 payout order. Only /output and /tmp need be writable.

Forecast SHA-256: {sha(O/'submission.csv')}
The score 31.7023316677 describes the preserved held-fold validation predictions;
it is not a measured hidden-test score or an independently validated score of
averaging all four recipes. Validation was already used during development.

The original HMM submission remains separately preserved by the project.
Source identity is in source_manifest.json; weights/checkpoint hashes are in
frozen_artifacts.json. Required license notices are included. This code loads saved
parameters and never trains; seeds remain 20261108 and 20261109 for the CatBoost fits.
''')
        # Test a code-identical archive first; only descriptive metadata is added afterward.
        trial=temp/'trial.zip';package_code(stage,trial);extracted=temp/'extracted';extract_archive(trial.read_bytes(),extracted)
        before={p.relative_to(extracted).as_posix():sha(p) for p in extracted.rglob('*') if p.is_file()}
        output=temp/'reproduced.csv'
        clean_data=temp/'inputs';(clean_data/'test').mkdir(parents=True)
        for name in ('queries.csv','history.csv'):shutil.copyfile(args.data/'test'/name,clean_data/'test'/name)
        runner='''import sys,os,runpy
sys.path.insert(0,os.getcwd())
def offline(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.bind'):
        raise RuntimeError('offline reproduction blocked '+event)
sys.addaudithook(offline)
sys.argv=['predict.py','--data',sys.argv[1],'--out',sys.argv[2],'--model','model']
runpy.run_path('predict.py',run_name='__main__')
'''
        env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['MPLCONFIGDIR']=str(temp/'mpl')
        # Match the organizer's read-only source/data mounts for the replay.
        for tree in (extracted,clean_data):
            for p in tree.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
            tree.chmod(0o555)
        start=time.perf_counter()
        try:
            run=subprocess.run([sys.executable,'-I','-B','-c',runner,str(clean_data),str(output)],cwd=extracted,env=env,text=True,capture_output=True,check=True,timeout=1800)
        finally:
            for tree in (extracted,clean_data):
                tree.chmod(0o755)
                for p in tree.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)
        seconds=time.perf_counter()-start
        after={p.relative_to(extracted).as_posix():sha(p) for p in extracted.rglob('*') if p.is_file()};assert before==after
        a,b=read_submission(O/'submission.csv'),read_submission(output);assert a.keys()==b.keys()
        maximum=max(abs(x-y) for q in a for x,y in zip(a[q],b[q]))
        byte_identical=output.read_bytes()==(O/'submission.csv').read_bytes()
        if not byte_identical:raise RuntimeError(f'Clean reproduction is not byte-identical; max difference={maximum}. Investigate before release.')
        rep={'mode':mode,'rows':len(a),'byte_identical':byte_identical,'max_probability_difference':maximum,'seconds':seconds,'python':platform.python_version(),'platform':platform.platform(),'peak_child_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'source_tree_unchanged':True,'network':'Python audit hook rejects socket connection/DNS/bind; tool execution sandbox restricts network. No network-dependent inference.','dependencies':'Pinned installed packages, isolated Python -I, no project PYTHONPATH','submission_sha256':sha(output),'stdout':run.stdout,'limitations':'Local macOS clean-extraction check; organizer Linux was not executed.'}
        rep.update({'read_only_code_and_data':True,'available_data_files':['test/queries.csv','test/history.csv'],
                    'training_validation_targets_or_labels_available':False,'deployment_weights':[0.25]*4})
        save(O/'reproduction.json',rep);copy(O/'reproduction.json','reproduction.json')
        with (stage/'README.md').open('a') as f:f.write(f'\nFresh offline clean-extraction replay was byte-identical in {seconds:.3f} seconds, peak child RSS {rep["peak_child_rss_bytes"]/1024**2:.1f} MiB. See reproduction.json.\n')
        save(stage/'source_manifest.json',{p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file()})
        package_code(stage,O/'entry.zip');assert (O/'entry.zip').stat().st_size<100_000_000 and (O/'submission.csv').stat().st_size<10_000_000
        rep['entry_sha256']=sha(O/'entry.zip');rep['entry_bytes']=(O/'entry.zip').stat().st_size;save(O/'reproduction.json',rep)
        # Verify final archive's executable content is exactly the tested code and checkpoints.
        final=temp/'final';extract_archive((O/'entry.zip').read_bytes(),final)
        for path,digest in before.items():
            if path not in ('README.md',):assert sha(final/path)==digest
        save(O/'checksums.json',{'submission.csv':sha(O/'submission.csv'),'entry.zip':sha(O/'entry.zip')})
        print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
