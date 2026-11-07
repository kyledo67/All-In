"""Render the frozen experiment's observations; never select or fit a model."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'organizer'))
from slot_forecasting.contest import score_contest_files


def render(data):
    dev=json.loads((ROOT/'artifacts/experiment/development.json').read_text())
    hold=json.loads((ROOT/'artifacts/experiment/holdout.json').read_text())
    checks=json.loads((ROOT/'artifacts/verification/results.json').read_text())
    compare=json.loads((ROOT/'artifacts/verification/upstream_vs_baseline.json').read_text())
    overall=score_contest_files(ROOT/'artifacts/final/validation_predictions.csv',
        data/'validation/targets.csv',data/'validation/queries.csv',
        data/'validation/conditions.csv',data/'validation/reference.csv',max_points=60)
    (ROOT/'artifacts/final/validation_score.json').write_text(json.dumps(overall,indent=2)+'\n')
    lines=['# Round-one results','',
       'The machine-deviation patch is implemented and independently verified, but **rejected for final deployment**. '
       'Its predeclared paired holdout comparison does not establish a benefit over the shared model. '
       'The final CSV uses the unmodified shared HMM, exported through an equivalent standard-library forward pass.','',
       'The released kit specifies 60 main points and 40 adaptation points. All numbers below use the official '
       'episode- and condition-normalized scorer with a 60-point maximum; larger points and smaller ratios are better.','',
       '## Frozen comparisons','',
       'There are 48 development and 48 holdout episodes, 384 queries per partition, with 12 episodes per condition. '
       'The three positive precisions were compared on development only. Lambda=30 was frozen before holdout scoring. '
       'The saved rule retains the patch only if its paired 95% interval beats both the shared and unshrunk controls. '
       'The fallback (shared HMM) was also fixed in advance. The holdout was used for this confirmatory gate; no settings were changed afterward.','']
    conditions=['stable','short_history','drift','shared_change']
    for title,results in [('Development',dev['development']),('Holdout',hold['holdout'])]:
        lines += [f'### {title}','',
          '| Model | Points / 60 | Mean ratio | Stable ratio | Short-history ratio | Drift ratio | Shared-change ratio |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in ['baseline','upstream','unshrunk','pooled_0.3','pooled_3','pooled_30']:
            if name not in results:continue
            r=results[name]
            numbers=[r['technical_points'],r['mean_loss_ratio']]+[r['by_condition'][c]['loss_ratio'] for c in conditions]
            lines.append('| '+name+' | '+' | '.join(f'{x:.6f}' for x in numbers)+' |')
        lines.append('')
    lines += ['The JSON reports also retain absolute condition errors, reference errors and episode/query counts. '
              'All controls see identical queries and features. No CatBoost benchmark was present in the supplied project.','',
              '### Paired whole-episode bootstrap','',
              '2,000 resamples, stratified by condition, seed 314159. Each draw uses the same episodes for both models '
              'and the reference. Values are differences in normalized loss ratio; negative favors the first model.','',
              '| Comparison on holdout | 95% interval |','| --- | --- |']
    for name,v in hold['paired_bootstrap'].items():
        lo,hi=v['delta_loss_ratio_ci95']
        lines.append(f'| pooled_30 minus {name} | [{lo:.6f}, {hi:.6f}] |')
    lo,hi=compare['delta_loss_ratio_ci95']
    lines += [f'| upstream minus baseline | [{lo:.6f}, {hi:.6f}] |','',
              'Per-condition intervals are included in the corresponding JSON. These are unadjusted intervals '
              'for the frozen comparisons, not simultaneous guarantees. The shared HMM has a measured holdout '
              'gain over baseline, but the interval still includes zero. The experiment does **not** establish a '
              'statistically resolved competition improvement from the source modification.','',
              f'The delivered model scores **{overall["technical_points"]:.6f}/60** across all public validation '
              'episodes. This combines development and holdout, so it is descriptive, not a new independent test. '
              'The 1,536 test targets are hidden; no test score is claimed.','',
              '## Numerical verification','',
              f'- Tiny clean upstream filtering/smoothing/enumeration maximum error: {max(checks["tiny"].values()):.3g}.',
              f'- Actual inherited optimizer objective gradient versus finite differences: {checks["objective"]["finite_difference_max_abs"]:.3g}.',
              f'- Frozen population and transition parameter drift during adaptation: {checks["objective"]["frozen_parameter_max_abs"]:.1f}.',
              f'- Clean-source population replay maximum parameter difference: {checks["competition_replay"]["max_parameter_difference"]:.3g}.',
              '- Post-cutoff outcome changes produce identical forecasts. Missing lags are rejected; sampled older rows are never treated as adjacent.',
              '- Full validation/test probabilities agree between portable inference and Dynamax within 4.5e-16. Every output is finite, nonnegative, normalized and correctly ordered.',
              '- Fresh-package offline reproduction and resource usage are recorded in `artifacts/final/reproduction.json`.','',
              '### Synthetic known-probability cases','',
              '64 machines per case, fixed seeds, same population and adaptation budgets. These are independently '
              'constructed cases, not competition-simulator reconstructions. The population is correctly specified '
              'in the identical and changing-regime cases, so shared inference is already optimal under the supplied model.','',
              '| Case | Shared error | Unshrunk error | Pooled (30) error | Pooled deviation RMS |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name,r in checks['synthetic'].items():
        numbers=[r[n]['squared_probability_error'] for n in ['shared','unshrunk','pooled_30']]+[r['pooled_30']['deviation_rms']]
        lines.append('| '+name+' | '+' | '.join(f'{x:.6g}' for x in numbers)+' |')
    lines += ['','Shrinkage suppresses, but does not completely eliminate, spurious adaptation. It greatly reduces '
              'short-history overfitting. It helps relative to shared coefficients when the synthetic machines have '
              'opposite lag responses, but strong shrinkage underfits those responses and loses to unshrunk fitting there. '
              'It adds noise when a known shared model is already correct. These failures are part of the result.','',
              '## Approximations, compute and provenance','',
              'Training uses 3,768 nonoverlapping 32-tick observed windows, with eight earlier ticks required for lag '
              'inputs. Query inference uses the most recent 32 outcomes and their eight lags. Older complete windows '
              'can contribute to population training, but older query history is discarded. A known-regime synthetic '
              'sensitivity test found mean squared disagreement 0.000405 between 32-tick and full 128-tick inference '
              '(8 ticks: 0.002695; 16 ticks: 0.001370). This documents a cost of resetting the state prior; it was not '
              'used to tune the competition window. Unknown autoregressive inputs are avoided, not integrated.','',
              'Signal is observed only at the query cutoff. The HMM omits it rather than inventing a historical '
              'series. Machine type is assumed stable within an episode. Cross-machine evidence is not used. '
              'Lag features encode win/loss and four pairs; they do not model every payout-size interaction.','',
              'The population run uses 15 generalized EM iterations and 30 Adam steps per emission M-step; adaptation '
              'uses eight iterations with the same optimizer. The original population fit took approximately 4.36 seconds; '
              'validation shared inference 0.16 seconds; pooled adaptation about 1.53 seconds on this host, including '
              'JIT work for those paths. These are local timings, not two-core organizer benchmarks. Logged objectives '
              'did not decrease in this run, but the last population trace gain is about 42.08: convergence is not '
              'established. Parameter uncertainty is not integrated.','',
              'The incoming workspace already contained the source patch, pinned dependency environment, split list '
              'and frozen development finalist. They were preserved, audited and replayed; earlier split-creation '
              'history cannot be independently reconstructed from this session. No subsequent holdout tuning occurred. '
              'The organizer baseline replay prints full-validation baseline summaries after the finalist freeze; '
              'those summaries did not change candidate selection.','',
              'Dynamax supplied initialization, likelihoods, forward/backward inference, EM orchestration, priors, '
              'parameter freezing and optimization. The 38-line emission extension tests the pooling hypothesis '
              'without rewriting those components. No source change is credited with a final competition improvement. '
              'The rejected extension remains available for reproducible research; it is not in the deployed model.','',
              'No Databricks integration or credentials were configured in this workspace. The existing local MLflow '
              'fallback retains the experiment and final artifacts. No REA runtime was needed for available Python '
              'source. Source paths, exact dependency versions, licenses, data hashes and evidence boundaries are '
              'in the source map, ledger and artifact manifests.','']
    (ROOT/'docs/results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    render(ap.parse_args().data)
