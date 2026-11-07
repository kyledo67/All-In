"""Complete reporting and package-preservation audit; no fitting or model selection."""
import csv,hashlib,json,sys,subprocess,tempfile,time,platform,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,'organizer')
import numpy as np
from scripts.experiment import scoring_inputs,paired_bootstrap
from slot_forecasting.io import read_submission
from slot_forecasting.submission_contract import extract_archive,inspect_archive
O=Path('artifacts/phase2');D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');E=json.loads((O/'evaluation.json').read_text());DG=json.loads((O/'diagnostics.json').read_text());rows=json.loads((O/'diagnostic_rows.json').read_text());C=json.loads((O/'protocol.json').read_text());champ=json.loads((O/'champion/manifest.json').read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Verify every frozen original file, including the entire authoritative code ZIP.
assert all(sha(Path(p))==h for p,h in champ['hashes'].items())
config=inspect_archive(Path('artifacts/final/entry.zip').read_bytes())
with tempfile.TemporaryDirectory(prefix='all-in-phase2-fallback-') as tmp:
 root=Path(tmp)/'source';extract_archive(Path('artifacts/final/entry.zip').read_bytes(),root);out=Path(tmp)/'submission.csv'
 runner=('import socket,sys,runpy; deny=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("network disabled")); socket.socket=deny; socket.create_connection=deny; sys.argv=["predict.py","--data",sys.argv[1],"--out",sys.argv[2]]; runpy.run_path("predict.py",run_name="__main__")')
 start=time.perf_counter();subprocess.run([sys.executable,'-B','-S','-c',runner,str(D),str(out)],cwd=root,check=True,capture_output=True);elapsed=time.perf_counter()-start
 assert out.read_bytes()==Path('artifacts/final/submission.csv').read_bytes()
 reproduction={'champion_unchanged':True,'all_original_manifest_hashes_match':True,'fresh_original_zip_byte_identical':True,'seconds':elapsed,'network':'socket creation disabled','third_party_packages':'disabled (-S)','python':platform.python_version(),'submission_sha256':sha(out),'entry_sha256':sha(Path('artifacts/final/entry.zip')),'command':'python predict.py --data <public> --out <submission.csv>'}
 save(O/'fallback_reproduction.json',reproduction)
S=scoring_inputs(D);episodes={t.episode_id for t in S[0]}
comparison=paired_bootstrap(read_submission(O/'predictions/oof_cat_hmm.csv'),read_submission(O/'predictions/oof_cat_observable.csv'),S,episodes)
save(O/'hmm_feature_ablation_bootstrap.json',comparison)
# Category contributions respect official per-condition normalizers.
category={}
for cond,official in DG['official']['by_condition'].items():
 rr=[r for r in rows if r['condition']==cond];es={r['episode'] for r in rr}
 values=np.mean([np.mean([r['category_error'] for r in rr if r['episode']==e],axis=0) for e in es],axis=0)
 category[cond]={'error_by_category':values.tolist(),'ratio_contribution_by_category':(values/max(official['reference_error'],1e-9)).tolist()}
save(O/'category_contributions.json',category)
with (O/'per_episode.csv').open() as f:r=list(csv.DictReader(f))
names=list(E['scores'])
with (O/'paired_episode_differences.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['episode','condition']+[n+'_minus_hmm' for n in names if n!='hmm'])
 for v in r:w.writerow([v['episode'],v['condition']]+[float(v[n])-float(v['hmm']) for n in names if n!='hmm'])
source_paths=[Path(p) for p in ['slot_model/phase2_features.py','slot_model/phase2_predict.py','scripts/phase2_train.py','scripts/phase2_evaluate.py','scripts/phase2_diagnose.py','scripts/phase2_verify.py','scripts/phase2_setup.py','scripts/phase2_report.py','docs/phase2-protocol.md']]
manifest={'parent_champion':champ['champion_id'],'repository_head':champ['repository_head'],'source_status':'Uncommitted implementation; content hashes and executed source snapshots identify this run.','catboost_version':'1.2.8','catboost_source_tag_commit':'0bcf252505e3d1cf01acd925dcd7026799512fb9','catboost_tag_object':'dbecc8ddf0c0ee833def28401d54d61d6224850b','source_files':{str(p):sha(p) for p in source_paths},'executed_source':{str(p):sha(p) for p in (O/'source').glob('*.py')},'protocol_sha256':sha(O/'protocol.json'),'split_sha256':sha(O/'folds.json'),'dependencies_sha256':sha(O/'requirements.lock'),'data_manifest_sha256':sha(Path('artifacts/data_manifest.json')),'artifacts':{str(p):sha(p) for folder in ('models','predictions','hmm_models','cache') for p in (O/folder).glob('*') if p.is_file()}}
save(O/'manifest.json',manifest)
cols=['stable','short_history','drift','shared_change']
lines=['# Phase-two feature challengers','', '**The deployed champion is unchanged.** The original verified shared HMM ZIP and CSV remain the submission. '
       'The planned search is complete; partial pooling was not reopened.','',
       'All scores below use the organizer calculation on concatenated held-development-fold predictions. '
       'These are the same 96 previously examined validation episodes, regrouped for configuration and blend selection—not an untouched evaluation. '
       'No new independent labeled data was available.','',
       '| Predictor | Points / 60 | Mean ratio | Stable | Short history | Drift | Shared change |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for n,v in E['scores'].items():lines.append('| '+n+' | '+' | '.join(f'{x:.6f}' for x in [v['technical_points'],v['mean_loss_ratio']]+[v['by_condition'][c]['loss_ratio'] for c in cols])+' |')
lines+=['','The original HMM’s earlier 48-episode holdout score remains 30.759921/60; that holdout is now development evidence. '
        'The phase-two HMM score 31.163114 uses all public validation and must not be compared to 30.760 as if the difference were a model gain.','',
        '## Decision','',
        'CatBoost plus HMM features gains 0.388837 points over the champion, improves 3/4 folds and improves under both seeds. '
        'It does not meet the frozen 0.5-point replacement gate. Its 95% exploratory normalized-error difference interval includes zero. '
        'The simplex blend gains 0.539218 over the champion and 0.150381 over the best individual; it fails the frozen 0.25-point ensemble gate. '
        'The blend’s improvement over the champion has an exploratory interval below zero, but its incremental gain over the best individual does not. '
        'The predeclared gates are respected rather than relaxed after seeing these results.','',
        '| Paired comparison | 95% normalized-error difference interval |','| --- | --- |']
for n,b in E['paired_vs_champion'].items():lines.append(f'| {n} minus HMM | {b["delta_loss_ratio_ci95"]} |')
lines+=[f'| ensemble minus {E["best_single"]} | {E["ensemble_vs_best"]["delta_loss_ratio_ci95"]} |',f'| CatBoost HMM features minus observable-only | {comparison["delta_loss_ratio_ci95"]} |','',
        'Bootstrap: 2,000 paired whole-episode draws stratified by condition; reference denominators are recomputed in each draw. '
        'Intervals are exploratory and unadjusted for selection/multiple comparisons. They do not restore independence after development exposure. '
        'Per-condition intervals, all fold scores, seed comparisons and selected settings are in evaluation.json. '
        'Per-episode absolute errors and paired differences are in per_episode.csv and paired_episode_differences.csv.','',
        '## Hypothesis outcomes','',
        '1. **Local recency: partial support for complementarity, no replacement.** Local history alone scores 30.912771, below the HMM, '
        'but improves the drift condition (ratio 0.9394 versus 0.9762). Every fold selects half-life 16 and prior strength 8. '
        'It receives positive blend weight in every fold. This supports some complementary information, not a proven mechanism of HMM overreaction.','',
        '2. **Observable feature CatBoost: modest development gain, insufficient practical evidence.** It scores 31.398446; '
        'the zero-payout/type/signal diagnostics motivated it, but the improvement is small and its interval overlaps zero. '
        'Depth 4/L2=8 is selected in every fold. The aggregate test does not identify which individual feature caused the gain.','',
        '3. **HMM-derived features: small, unstable incremental gain.** The matched variant scores 31.551951 versus 31.398446. '
        'Depth choice changes across folds; it beats observable-only in three folds, while the incremental paired interval overlaps zero. '
        'Neither feature stacking nor a positive ensemble coefficient establishes independent HMM value.','',
        'No added features or ensemble components are retained in deployment. The original 15-feature, two-state shared HMM and its standard-library '
        'inference remain the champion. All challenger checkpoints and experiments are preserved for inspection.','',
        '## Champion diagnostics','',
        '| Condition | Absolute error | Reference error | Normalized error |','| --- | ---: | ---: | ---: |']
for c,v in DG['official']['by_condition'].items():lines.append(f'| {c} | {v["error"]:.6f} | {v["reference_error"]:.6f} | {v["loss_ratio"]:.6f} |')
lines+=['','| Payout | Squared error contribution | Share of total | Mean prediction minus truth |','| --- | ---: | ---: | ---: |']
for i,c in enumerate([0,1,2,5,20]):
 er=DG['overall']['category_error'][i];lines.append(f'| {c}× | {er:.6f} | {100*er/DG["overall"]["error"]:.2f}% | {DG["overall"]["bias"][i]:+.6f} |')
lines+=['','Category contributions normalized by each condition reference are in category_contributions.json. '
        'The following subgroup summaries average within each represented episode, then across represented episodes; they are descriptive subgroup errors, not standalone official points.','',
        '| Group | Queries | Episodes | Error | 0× bias |','| --- | ---: | ---: | ---: | ---: |']
for field in ['machine_type','signal','n_history','coverage_40']:
 for key,v in DG['groups'][field].items():lines.append(f'| {field}={key} | {v["queries"]} | {v["episodes"]} | {v["error"]:.6f} | {v["bias"][0]:+.6f} |')
lines+=['','| Observable diagnostic quartile | Error | 0× bias |','| --- | ---: | ---: |']
for field in ['recent_win','win_change','distribution_change','lag_strength']:
 for key,v in DG['groups'][field].items():lines.append(f'| {field}/{key} | {v["error"]:.6f} | {v["bias"][0]:+.6f} |')
lines+=['','All recent 40 windows are fully observed. Short histories contain 64 observed rows while long histories contain their full 192–256 tick prefixes. '
        'Recent/older categorical disagreement correlates 0.144 with error, but the signed win-change correlation is only 0.072. '
        'A confident diagnosis of slow adaptation or overreaction is not established. Local conditional lag-strength error correlation is negative (-0.306), '
        'so it does not support a general claim that strong observed lag responses are missed. Rare 20× probability is slightly underpredicted on average.','',
        '## Complementarity and fitting','',
        'Query-level squared-error correlations are approximately 0.657 for HMM/local,0.671 for HMM/observable CatBoost, '
        '0.756 for HMM/CatBoost+HMM, and 0.890 between the two CatBoost variants. Weight fitting additionally checks weighted signed residual similarity. '
        'Different errors justify the limited blend test, not automatic adoption. Fold weights are global, nonnegative and sum to one; '
        'condition labels only stratify/report and normalize the training objective. No prediction-time gating uses conditions.','',
        'The feature audit and fitting-data provenance are in phase2-protocol.md. Base models use only training outcomes. '
        'There are 768 actual queries and2,366 historical examples, with equal total episode/machine weight. Historical cutoffs and signal missingness '
        'are imperfect matches to real queries; there is no fabricated historical signal. Cross-fitted priors and HMMs exclude the entire owning episode. '
        'Early stopping uses nested train-only splits. All causal tests passed.','',
        'The fixed search is six local settings and two CatBoost configurations × two feature variants × two seeds. '
        'Eight early-stop and eight final training fits are complete. Some depth 6 runs hit 400 trees; the cap was not expanded. '
        'No additional search was launched after evaluation. A metadata-only resume fix preserves upstream provenance when cached feature matrices are reused; '
        'the exact originally executed scripts are retained under artifacts/phase2/source/.','']
runs=json.loads((O/'training_runs.json').read_text());prep=json.loads((O/'preparation.json').read_text())
lines+=['| CatBoost run | Trees | Train + early-stop seconds | Validation prediction seconds |','| --- | ---: | ---: | ---: |']
for v in runs:lines.append(f'| {v["name"]} | {v["trees"]} | {v["training_seconds"]:.3f} | {v["validation_inference_seconds"]:.4f} |')
lines += ['',f'Cross-fitted feature preparation took{prep["seconds"]:.2f} seconds on this host; tree inference timings exclude feature preparation. '
          'Prepared sequences, feature matrices and model predictions remain cached. Training is CPU-only, with CatBoost restricted to two threads.','',
          '## Final handoff','',
          f'Original CSV: `artifacts/final/submission.csv` — SHA256 `{reproduction["submission_sha256"]}`.',
          f'Original ZIP: `artifacts/final/entry.zip` — SHA256 `{reproduction["entry_sha256"]}`.','',
          f'A fresh extraction of that exact ZIP reproduced the CSV byte-for-byte in **{elapsed:.3f} seconds** with socket creation and third-party packages disabled. '
          'The prior organizer checker acceptance remains valid because every original file hash is unchanged. The checker was also rerun in this phase. '
          'This local macOS/Python3.13 test does not execute the organizer Linux container.','',
          'From the extracted original entry:', '', '```sh','python predict.py --data /absolute/path/to/public --out /output/submission.csv','```','',
          'To reproduce the bounded investigation from the project root with its pinned environment:', '', '```sh',
          '.venv/bin/python scripts/phase2_train.py', '.venv/bin/python scripts/phase2_evaluate.py', '.venv/bin/python scripts/phase2_verify.py',
          '.venv/bin/python scripts/phase2_report.py', '.venv/bin/python scripts/phase2_log.py', '```','',
          'The current scripts default to the supplied kit path. Executed snapshots are preserved separately. '
          'The manifest records original source HEAD plus content hashes, data/configuration/fold/dependency hashes, all saved checkpoints and predictions. '
          'Nothing was uploaded, and no Databricks infrastructure was provisioned. Local MLflow uses the existing tracking directory and records the parent champion.','']
Path('docs/phase2-results.md').write_text('\n'.join(lines))
print('Report complete; champion unchanged; fresh reproduction seconds',elapsed)
