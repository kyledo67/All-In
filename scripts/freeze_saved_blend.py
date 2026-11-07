"""Preserve saved fold recipes and record the user-authorized policy revision. No fitting."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'artifacts/phase2';O=ROOT/'artifacts/blend_release'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
def main():
 if O.exists():raise SystemExit('Release already exists; preserve it.')
 E=json.loads((P/'evaluation.json').read_text());champ=json.loads((P/'champion/manifest.json').read_text())
 assert all(sha(ROOT/p)==h for p,h in champ['hashes'].items())
 O.mkdir();release=O/'model';release.mkdir()
 recipes=[{'fold':r['fold'],'config_indices':r['selection'],'weights':r['blend_weights']} for r in E['folds']]
 save(release/'recipes.json',recipes)
 save(release/'validation_fold_map.json',json.loads((P/'folds.json').read_text())['validation'])
 for name in ['protocol.json','population_prior.json','feature_schema.json']:shutil.copyfile(P/name,release/name)
 shutil.copyfile(ROOT/'artifacts/experiment/population.json',release/'population.json')
 protocol=json.loads((P/'protocol.json').read_text());models=set()
 for recipe in recipes:
  for variant in ['observable','hmm']:
   if recipe['weights']['cat_'+variant]>0:
    for seed in protocol['seeds']:models.add(f'cb_{variant}_{recipe["config_indices"][variant]}_{seed}.cbm')
 (release/'models').mkdir()
 for name in sorted(models):shutil.copyfile(P/'models'/name,release/'models'/name)
 shutil.copyfile(P/'predictions/oof_ensemble.csv',O/'evaluated_oof_predictions.csv')
 save(O/'policy_change.json',{
  'authorized_by':'User instruction: select the existing 31.702-point blend; compare complete candidate directly against deployed champion.',
  'previous_rule':'Required 0.25 additional points over the best individual plus its own paired interval; then compared against HMM.',
  'revised_rule':'Judge the complete existing blend directly against deployed HMM; do not require an incremental win over CatBoost. No training, optimization, configuration selection or search.',
  'post_evaluation_policy_change':True,'original_evaluation_preserved':True,'parent_champion':champ['champion_id'],
  'hmm_score':E['scores']['hmm'],'blend_oof_score':E['scores']['ensemble'],
  'gain_points':E['scores']['ensemble']['technical_points']-E['scores']['hmm']['technical_points'],
  'existing_blend_minus_hmm_interval':E['paired_vs_champion']['ensemble'],
  'folds_improved':sum(r['score']['ensemble']['technical_points']>r['score']['hmm']['technical_points'] for r in E['folds']),
  'all_condition_errors_improved':all(E['scores']['ensemble']['by_condition'][c]['error']<v['error'] for c,v in E['scores']['hmm']['by_condition'].items()),
  'uncertainty':'Exploratory, selected on previously inspected development episodes. Policy changed after results were known; no new independent evidence.',
  'deployment_resolution':'Pending user clarification: 31.702 is a concatenation of four held-fold recipes, not one global blend weight vector.',
  'preservation':'Every original fold weight, selected configuration, model checkpoint, seed and feature setting remains unchanged.',
  'fallback':{'csv':str(ROOT/'artifacts/final/submission.csv'),'entry':str(ROOT/'artifacts/final/entry.zip'),'hashes':champ['hashes']},
  'frozen_unix':time.time()})
 save(O/'frozen_artifacts.json',{'source_evaluation_sha256':sha(P/'evaluation.json'),'oof_predictions_sha256':sha(P/'predictions/oof_ensemble.csv'),'recipe_source':'artifacts/phase2/evaluation.json::folds','files':{str(p.relative_to(O)):sha(p) for p in release.rglob('*') if p.is_file()}})
 print('Frozen',len(recipes),'recipes and',len(models),'CatBoost checkpoints; all original champion hashes match.')
if __name__=='__main__':main()
