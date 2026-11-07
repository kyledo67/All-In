"""Package source/checkpoint with organizer tools and verify a fresh offline rerun."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'organizer'))
from slot_forecasting.io import read_submission
from slot_forecasting.submission_contract import package_code,extract_archive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_diff(a,b):
    pa,pb=read_submission(a),read_submission(b)
    assert set(pa)==set(pb)
    return max(abs(x-y) for k in pa for x,y in zip(pa[k],pb[k]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/final/entry.zip')
    args=ap.parse_args(); args.data=args.data.resolve(); args.out=args.out.resolve()
    if args.out.exists():raise SystemExit('Use a new output ZIP; existing entries are preserved.')
    final=ROOT/'artifacts/final'
    assert json.loads((ROOT/'artifacts/experiment/holdout.json').read_text())['deployment']=='upstream'
    with tempfile.TemporaryDirectory(prefix='all-in-package-') as temp:
        stage=Path(temp)/'source';stage.mkdir()
        def copy(path):
            src=ROOT/path;dest=stage/path
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        for directory in ('slot_model','scripts','organizer/slot_forecasting'):
            for path in (ROOT/directory).glob('*.py'):copy(path.relative_to(ROOT))
        for path in (ROOT/'research/vendor/dynamax/dynamax').rglob('*.py'):
            if '__pycache__' not in path.parts:copy(path.relative_to(ROOT))
        for directory in ('docs','research/rea_reference'):
            for path in (ROOT/directory).glob('*'):
                if path.is_file():copy(path.relative_to(ROOT))
        for directory in ('artifacts/experiment','artifacts/verification'):
            for path in (ROOT/directory).glob('*'):
                if path.suffix in ('.json','.npz','.csv'):copy(path.relative_to(ROOT))
        for path in (ROOT/'artifacts').glob('*'):
            if path.suffix in ('.json','.patch'):copy(path.relative_to(ROOT))
        for path in ('predict.py','requirements-training.lock','THIRD_PARTY_NOTICES.md',
                     'research/upstream_categorical_glm_hmm.py',
                     'research/vendor/dynamax/LICENSE','research/vendor/dynamax/CITATION.cff',
                     'research/vendor/dynamax/pyproject.toml','research/vendor/dynamax/README.md',
                     'organizer/check_submission.py','organizer/build_entry.py',
                     'organizer/run_starter.py','organizer/RULES.md','organizer/DATA.md',
                     'organizer/README.md','organizer/SUBMISSION_README.md','organizer/SHA256.json'):
            copy(path)
        shutil.copyfile(ROOT/'README.md',stage/'docs/project-readme.md')
        (stage/'requirements.txt').write_text('# Final inference uses only the Python 3.13 standard library.\n')
        (stage/'reproduce.json').write_text(json.dumps({'image':'python','command':[
            'python','predict.py','--data','/data','--out','/output/submission.csv']},indent=2)+'\n')
        (stage/'README.md').write_text('# All-In main-round entry\n\nOffline saved-model inference.\n')
        # Package and extract with the official implementation, then run THAT source.
        trial=Path(temp)/'trial.zip';package_code(stage,trial)
        extracted=Path(temp)/'extracted';extract_archive(trial.read_bytes(),extracted)
        before={str(p.relative_to(extracted)):digest(p) for p in extracted.rglob('*') if p.is_file()}
        output=Path(temp)/'output.csv'
        runner=('import socket,runpy,sys; '
                'deny=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("network disabled")); '
                'socket.socket=deny; socket.create_connection=deny; '
                'sys.argv=["predict.py","--data",sys.argv[1],"--out",sys.argv[2]]; '
                'runpy.run_path("predict.py",run_name="__main__")')
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        start=time.perf_counter()
        run=subprocess.run([sys.executable,'-B','-S','-c',runner,str(args.data),str(output)],
            cwd=extracted,env=env,text=True,capture_output=True,check=True,timeout=1800)
        seconds=time.perf_counter()-start
        after={str(p.relative_to(extracted)):digest(p) for p in extracted.rglob('*') if p.is_file()}
        assert before==after, 'Inference wrote to its source folder'
        dif=max_diff(final/'submission.csv',output);assert dif<1e-8
        test_diff=max_diff(final/'submission.csv',final/'dynamax_test.csv')
        val_diff=max_diff(final/'validation_predictions.csv',ROOT/'artifacts/experiment/upstream.csv')
        assert max(test_diff,val_diff)<1e-8
        report={'fresh_extracted_package_max_probability_difference':dif,
                'fresh_extracted_package_byte_identical':digest(output)==digest(final/'submission.csv'),
                'dynamax_test_max_probability_difference':test_diff,
                'dynamax_validation_max_probability_difference':val_diff,
                'test_queries':len(read_submission(output)),
                'seconds':seconds,'platform':platform.platform(),'python':platform.python_version(),
                'peak_child_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1 if sys.platform=='darwin' else 1024),
                'network':'socket construction disabled in fresh subprocess',
                'site_packages':'disabled with -S; no third-party inference dependencies',
                'source_tree_unchanged':True,'submission_sha256':digest(output),'stdout':run.stdout,
                'limitations':'Local macOS test, not organizer Linux container; pure stdlib final inference.',
                'budget':'30 minutes, two CPU cores, 16 GiB RAM; portable path single-threaded'}
        (final/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
        copy(Path('artifacts/final/reproduction.json'))
        for path in ('artifacts/final/test_check.txt','artifacts/final/validation_check.txt','artifacts/final/validation_score.json'):copy(Path(path))
        season=json.loads((args.data/'metadata.json').read_text())['season_id']
        readme=f'''# All-In main-round entry

Round: main. Season: {season}. Select the registered team in the organizer form.
Model: shared categorical-regression HMM trained with Dynamax; partial pooling
was tested and rejected by the frozen confirmatory gate. See docs/results.md.

## Reproduction

Python 3.13.7 tested, standard library only. No package installation is needed for
the default command. Run from the extracted ZIP root:

```sh
python predict.py --data /data --out /output/submission.csv
```

Inputs: released queries.csv and history.csv under /data/test/. Required checkpoint:
artifacts/experiment/population.json. Selection: artifacts/experiment/holdout.json.
The runner reads saved parameters; it does not retrain. No network, APIs, REA,
Databricks or credentials are required. Source can be mounted read-only. The
uploaded forecast CSV is not an input. Scratch files are not needed.

Forecast SHA-256: `{report['submission_sha256']}`.
Output order: query_id,p_0,p_1,p_2,p_5,p_20. Seed: 20261107.
Settings and data/split/source hashes: artifacts/experiment/config.json and
artifacts/*.json. Code identity: source_manifest.json. Licenses and attribution:
THIRD_PARTY_NOTICES.md and research/vendor/dynamax/LICENSE.

The official checker accepted all {report['test_queries']} test rows. A fresh
extracted package rerun with site packages and sockets disabled produced a
byte-identical CSV in {seconds:.3f} seconds on {platform.platform()},
peak child RSS {report['peak_child_rss_bytes']/1024**2:.1f} MiB, using one thread.
The measured maximum discrepancy against JAX on test was {test_diff:.3g},
below the 1e-8 limit. Organizer Linux has not been executed locally.

## Training and investigation

Final inference needs no dependencies. To reproduce training, separately install
the exact versions in requirements-training.lock into a Python 3.13 environment,
then use the commands in docs/project-readme.md with the released data path.
Use new output folders. Training and adaptation source, clean upstream control,
readable patch, saved checkpoints and numerical checks are included. Repeating
the holdout audit reproduces a result; it does not create fresh holdout evidence.
No adaptation-round data or practice-demo internals were used.
'''
        (stage/'README.md').write_text(readme)
        manifest={str(p.relative_to(stage)):digest(p) for p in stage.rglob('*') if p.is_file()}
        (stage/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        package_code(stage,args.out)
        assert args.out.stat().st_size < 100_000_000
        assert (final/'submission.csv').stat().st_size < 10_000_000
        report['entry_zip_sha256']=digest(args.out);report['entry_zip_bytes']=args.out.stat().st_size
        report['entry_zip']=str(args.out)
        (final/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
