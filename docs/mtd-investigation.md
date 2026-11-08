# MTDg investigation (in progress)

Upstream source and MIT license are preserved under research/mtd_reference;
commit.txt pins the inspected revision. In upstream.py, MTD._fit_one runs EM;
_calculate_log_likelihood_mtd and _ChainBase._create_markov implement the additive
mixture sum(lambda[j] * Q[j,x[j],y]). This is not a product model and cannot
represent pure XOR between two inputs without interaction variables.

The port in slot_model/mtd.py avoids upstream's full categorical tensor.
It uses row-level responsibilities and weighted categorical sufficient counts.
Upstream _expectation_step aggregates responsibilities over unobserved contexts
before _maximization_step multiplies by marginal counts. The port instead uses
the standard observed-row EM sufficient statistics; it is not a bit-for-bit
reproduction of that upstream update. Fixed initialization replaces random
restarts. The fitted parameters are point estimates with pseudocount smoothing.

Own lags use actual ticks 1..8 and category 5 for unknown. Neighbor inputs are
exchangeable win-fraction categories at actual lags 1..3, not individual neighbor
identities. Historical signals are unavailable: historical data fits type-only
unknown-signal models; known type/signal contexts use only released query labels.
Historical outcomes must never be assigned the current query's signal.

The empirical concentration is a method-of-moments estimate from training-machine
categorical proportions, clipped to [1,40/3]. Local fitting uses a cap of one
third effective history mass; context blending is also capped. These are two
separate smoothing mechanisms, not a single combined Dirichlet prior. The
changepoint flag uses the largest difference between two 20-actual-tick windows
with at least 15 observations each; a difference of .25 triggers quarter weight
for examples whose eight-tick input window crosses or precedes the shift.

Evaluation scripts compare plain/change variants and fixed weights .15,.25,.35
against the saved 31.702 combined OOF recipe evidence. The deployed equal average
has no independent validation score. All comparisons remain exploratory.

Current execution blocker: existing NumPy import stalls in importlib file reads
inside .venv. A same-version temporary install is being attempted. No score or
retention decision is established yet. Existing release artifacts are untouched.

## Completed first comparison

The plain MTD scored 30.2582/60 alone; changepoint MTD scored 30.5352.
The unchanged frozen OOF recipes scored 31.7023. With a fixed .25 MTD
contribution, combined OOF scores were 31.8291 plain and 31.9023 changepoint.
The changepoint candidate's normalized condition ratios were stable 0.8628,
short history 0.9448, drift 0.9087, shared change 0.8066. Shared change
regressed from the frozen stack's 0.8035. Three of four folds improved. A
paired whole-episode exploratory interval for candidate-minus-stack normalized
error is [-0.02425,+0.00098]. The validation episodes and fixed weight grid
were already inspected; this is not independent evidence. The deployed mean of
four original recipes is also not independently evaluated. The .15 and .35
weights in evaluation.json are descriptive sensitivity checks, not optimized
release weights.

MTD's type-only model ranks own lag 7 first for steady and balanced. The
saved observable CatBoost model also ranks `lag7_c0` among its top features,
so this suggests some shared structure. Its leading features additionally
include recent counts and conditional lag-pair estimates. In the
balanced/signal+1 MTD context, neighbor win lag 1 ranks first. These are
model importance signals, not proof of the simulator's hidden mechanism.

The experimental deployment interface is `predict_mtd_stack.py`; it combines
frozen mean stack probabilities with .25 changepoint MTD probabilities, preserving
all four internal recipe weights. It has no new runtime dependency beyond the
existing stack's NumPy/CatBoost requirements. The old blend release remains
available while the combined candidate is verified.

## Candidate release and limits

A separate candidate is saved in `artifacts/mtd_01/submission.csv` and
`artifacts/mtd_01/entry.zip`. The existing frozen blend and HMM fallback are
unchanged. The organizer checker accepted 1,536 test rows. A direct replay
returned byte-identical CSV in 100.1 seconds on the local host; a clean ZIP
extraction replay returned byte-identical CSV in 97.8 seconds. The `source_manifest.json` inside the candidate
ZIP verifies all 52 other members. The current inference command is:

```
python predict.py --data /data --out /output/submission.csv --model model --mtd mtd.json
```

The MTD source uses NumPy and the Python standard library. The combined stack
also needs its existing CatBoost dependency. No upstream `mtd-learn` package is
needed at inference. No test score is available; no competition submission was
made. The 31.902 OOF score belongs to the fold-routed evaluation procedure and
must not be assigned to the deployed mean-plus-MTD test ensemble.

The candidate CSV SHA-256 is `c8ac8e53f96bcca7aa1dc34fd67752b263555bd176e1117386338240fd93deeb`; ZIP SHA-256 is `5efcc9c7b10129cffec9a522c3f49eb92f70826aaab8dd4b211fd6aa862e374b`.
