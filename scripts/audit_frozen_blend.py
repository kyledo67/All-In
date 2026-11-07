"""Identity/rules and official OOF aggregation checks; does not score deployed mean."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'organizer'))
from slot_forecasting.io import read_submission,read_targets,read_rows,load_inputs
from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import subset_score
O=ROOT/'artifacts/blend_release';P=ROOT/'artifacts/phase2';D=Path('/Users/kyledo/Downloads/main-participants-60-40/public')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
E=json.loads((P/'evaluation.json').read_text());recipes=json.loads((O/'model/recipes.json').read_text());frozen=json.loads((O/'frozen_artifacts.json').read_text())
assert all(sha(O/p)==v for p,v in frozen['files'].items())
train={q.episode_id for q in load_inputs(D/'train')[0]};val={q.episode_id for q in load_inputs(D/'validation')[0]};test={q.episode_id for q in load_inputs(D/'test')[0]}
assert train.isdisjoint(val|test) and val.isdisjoint(test)
provenance=json.loads((P/'upstream_feature_provenance.json').read_text())
assert all(set(v['trained_episodes'])<=train for v in provenance.values())
assert json.loads((P/'causality_tests.json').read_text())['passed']
for recipe,old in zip(recipes,E['folds']):
 assert recipe['weights']==old['blend_weights'] and recipe['config_indices']==old['selection'] and recipe['fold']==old['fold']
 assert all(v>=0 for v in recipe['weights'].values()) and abs(sum(recipe['weights'].values())-1)<1e-12
validity={'all_four_recipes_valid_for_inference':True,'checkpoint_prior_schema_and_recipe_hashes_match':True,
 'recipes_match_original_evaluation':True,'base_fitting_data':'public/train observed outcomes; no validation target probabilities in base fits',
 'learned_upstream_features':'Owning training episode excluded; earlier cross-fitting provenance verified against disjoint train episodes.',
 'local_adaptation':'Pre-cutoff observations only; actual tick distances; no fabricated historical signals.',
 'weight_fitting':'Previously selected on other validation folds as permitted model selection; reused unchanged.',
 'prediction_inputs':'Test queries and histories; no test conditions, labels, targets or reference forecasts.',
 'inference_class_order':[0,1,2,5,20],'independent_validation_for_deployed_average':False,
 'source_evidence':['artifacts/phase2/causality_tests.json','artifacts/phase2/upstream_feature_provenance.json','docs/phase2-protocol.md'],
 'rules_review':'Released organizer/RULES.md permits organizer data, strictly prior observations, model selection and offline prediction. Existing audited causal feature code is reused.'}
save(O/'recipe_validity.json',validity)
pred=read_submission(O/'evaluated_oof_predictions.csv');targets=read_targets(D/'validation/targets.csv');groups={r['query_id']:r['condition'] for r in read_rows(D/'validation/conditions.csv',('query_id','condition'))};reference=read_submission(D/'validation/reference.csv')
score=contest_score(pred,targets,groups,reference,max_points=60,version=E['scores']['ensemble']['contest_version'])
assert abs(score['technical_points']-E['scores']['ensemble']['technical_points'])<1e-12
folds=json.loads((O/'model/validation_fold_map.json').read_text())
points=[subset_score(pred,targets,groups,reference,{e for e,f in folds.items() if f==i},60)['technical_points'] for i in range(4)]
combined=sum(points)/4
assert abs(score['technical_points']-combined)>1e-6
save(O/'oof_score_audit.json',{'official_score_on_combined_oof_predictions':score,'fold_points_for_audit_only':points,'naive_average_fold_points_not_used':combined,'matches_saved_combined_score':True,'deployed_equal_average_evaluated_on_validation':False})
print('Official combined OOF points:',score['technical_points'],'Naive fold average (NOT used):',combined)
print('All four recipe identities, stored parameters and original rule-compliance checks verified.')
