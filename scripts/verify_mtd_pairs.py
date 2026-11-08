"""Causal and probabilistic checks for pairwise MTD correction."""
import sys,json
from dataclasses import replace
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'organizer')]
import numpy as np
from slot_forecasting.io import load_inputs
from slot_model.mtd import MTDModel
from slot_model.mtd_pairs import PairMTDModel,LAG7_PAIRS
D=Path('/Users/kyledo/Downloads/main-participants-60-40/public');O=R/'artifacts/mtd_pairs_01'
queries,history=load_inputs(D/'validation');q=queries[0];model=PairMTDModel(base=MTDModel.load(R/'artifacts/mtd_01/change.json'))
original=model.predict([q],history)[q.query_id]
changed=[replace(row,payout=20) if row.episode_id==q.episode_id and row.tick>=q.cutoff else row for row in history]
assert model.predict([q],changed)[q.query_id]==original
assert np.isfinite(original).all() and min(original)>=0 and abs(sum(original)-1)<1e-12
base=PairMTDModel(pairs=(),base=model.base).predict([q],history)[q.query_id]
assert base==model.base.predict([q],history)[q.query_id]
(O/'verification.json').write_text(json.dumps({'post_cutoff_invariance':True,'normalized':True,'zero_pair_matches_additive':True,'pairs':LAG7_PAIRS},indent=2))
print('pairwise checks passed')
