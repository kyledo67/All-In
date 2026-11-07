# Dynamax source map

Pinned upstream: `probml/dynamax@ffc626bf89bf07c7ae7064922f6a373d013c4b17`.
Paths below are relative to `research/vendor/dynamax/`. Exact AST-derived definitions
are also saved in `artifacts/source_index.json`. The only upstream edit is the
38-line additive emission extension in `artifacts/dynamax.patch`; the original
classes have identical ASTs to the clean upstream file saved in
`research/upstream_categorical_glm_hmm.py`.

Let K=2 hidden regimes, C=5 payout categories, F=15 features, T=32 observed
ticks per window and N=3,768 training windows. Integer outcomes 0..4 map to
gross payouts **0, 1, 2, 5, 20**, respectively.

| Operation | Exact source entry | Implementation and shape |
| --- | --- | --- |
| Construct model | `dynamax/hidden_markov_model/models/categorical_glm_hmm.py:143`, `CategoricalRegressionHMM.__init__` | Composes `StandardHMMInitialState`, `StandardHMMTransitions`, and `CategoricalRegressionHMMEmissions`; inherits `HMM`, which inherits `SSM`. |
| Initialize emissions | same file, `CategoricalRegressionHMMEmissions.initialize:66` | Independent standard normal draws for W[K,C,F], b[K,C] unless supplied. The word `prior` names initialization; it does not imply a regularizing emission prior. |
| Emission probabilities | same file, `CategoricalRegressionHMMEmissions.distribution:110` | logits[k]=W[k]x+b[k]; categorical probabilities are softmax over C. Input x[F], distribution event shape scalar. |
| Existing emission prior | same file, `CategoricalRegressionHMMEmissions.log_prior:59` | Exactly 0.0. The class-level statement that the model has no prior is incomplete: initial and transition components do have Dirichlet priors. |
| Initial state | `dynamax/hidden_markov_model/models/initial.py:38,67,85` | Initialize pi[K]; Dirichlet concentrations 1.1; M-step uses posterior Dirichlet mode based on smoothed initial-state counts. |
| Transition state | `dynamax/hidden_markov_model/models/transitions.py:55,83,107` | A[K,K], row-stochastic, Dirichlet row priors 1.1, optional diagonal stickiness (0 here). M-step uses posterior mode of aggregated expected transitions. |
| Conditional log likelihood | `dynamax/hidden_markov_model/models/abstractions.py:428`, `HMMEmissions._compute_conditional_logliks` | Nested vmap over times and regimes; L[t,k]=log softmax(W[k]x[t]+b[k])[y[t]], shape [T,K]. |
| Inference wrappers | same file, `HMM._inference_args:601`, `filter:624`, `smoother:631` | Produces pi, A, L; passes these to the generic HMM routines. No autoregressive missing-data integration is supplied by this wrapper. |
| Forward filtering | `dynamax/hidden_markov_model/inference.py:86,104,114`, `_condition_on`, `_predict`, `hmm_filter` | f[t,k] proportional to predicted[t,k] exp(L[t,k]); next predicted=A.T@f[t]. Outputs filtered[T,K], pre-observation predicted[T,K], scalar marginal log likelihood. Uses log-scale stabilization and lax.scan. |
| Smoothing | same file, `hmm_two_filter_smoother:209`, `hmm_smoother:263` | Forward/backward message routines, smoothed gamma[T,K], expected transition counts[K,K]. Tiny exhaustive enumeration independently checks both filtering and the public smoother. |
| Actual E-step | `abstractions.py:639`, `HMM.e_step` | Calls **hmm_two_filter_smoother**, not the public `smoother` wrapper. Collects initial/transition counts and emission sufficient statistics (gamma[T,K], y[T], x[T,F]). |
| EM training driver | `dynamax/ssm.py:357`, `SSM.fit_em` | Adds batch axis, vmaps E-steps over N, then calls HMM.m_step. Logs log p(theta)+sum_n log p(y_n|x_n,theta) **before** each M-step. Fixed iteration budget, not a convergence certificate. |
| Emission M-step | `abstractions.py:470`, `HMMEmissions.m_step` | Minimizes -(log_prior + sum_n,t,k gamma[n,t,k] L[n,t,k])/scale; default scale=1.0. The penalty is counted once per adapted machine, not once per tick. |
| Optimizer reuse | `dynamax/utils/optimize.py:78`, `run_gradient_descent` | JAX value_and_grad + Optax Adam, carried optimizer state and lax.scan; approximate M-step with 30 updates. No custom EM or optimizer implementation. |
| Frozen parameters | `dynamax/parameters.py:111`, `from_unconstrained` | trainable=False applies stop_gradient. Initial/transition closed-form M-steps also honor trainable flags. Tested unchanged population, pi and A after adaptation. |
| Added emission | `categorical_glm_hmm.py:198,205,223,226` | ParamsPartiallyPooledEmissions and PartiallyPooledCategoricalEmissions. delta[C,3] shares deviations across states; logits=W[k]x+b[k]+delta@[1,x[3],x[11]]. log_prior=-lambda/2 sum(delta²). |

The model wrapper in `slot_model/model.py` initializes W as 0.05 times seeded
normal draws, b as log pooled training outcome frequencies, pi=(.5,.5), and A
with .95 diagonal. All are learned by the same 15-iteration upstream EM run.
The population fit is MAP for pi/A and unregularized likelihood fitting for
emissions. It is not a full posterior over coefficients.

`adaptation_model` freezes the trained population, pi and A, installs the added
emission component, and initializes delta to zero separately for every query
machine. Lambda=0 is the matched unshrunk control; lambda in {.3,3,30} is the
development-only tuning grid. Eight EM iterations, 30 Adam steps per M-step,
learning rate .03, K, A, features and history windows match across adaptation
conditions. All controls share the exact same population checkpoint; the
unmodified control performs no machine fitting. This is conditional MAP
adaptation with a development-selected prior precision, not Bayesian integration.

The 15 features are three machine-type indicators, centered binary win indicators
at each actual lag 1..8, and products for pairs (1,2), (2,4), (4,8), (1,8).
The **only** adapted terms are the intercept, lag-1 and lag-1 × lag-2, for each
category (15 extra coefficients per machine). This tests a small subset of
heterogeneity; it cannot establish conclusions for all machine-specific effects.

At cutoff c, `slot_model/model.py::next_probability` filters y[c-32:c], then
computes r=f[c-1]@A and p[c]=sum_k r[k]softmax(W[k]x[c]+b[k]+delta_m s[c]).
It averages probabilities, not logits, and uses no result at or after c.
`slot_model/portable.py::predict_shared` implements the same forward recurrence
with Python floats for the retained shared model. Entire validation/test arrays
are cross-checked against Dynamax; this avoids JAX dependencies for offline
submission inference. The original and modified JAX prediction paths remain
runnable in `predict.py --engine dynamax` and `scripts/experiment.py`.

Reused upstream initialization, categorical likelihoods, filtering, two-filter
smoothing, sufficient statistics, constrained parameter machinery, EM driver,
Dirichlet updates and Adam optimization account for nearly all modeling logic.
The extension needs no new inference algorithm. No JAX profiling was necessary:
source inspection and executable checks answered the specific questions.
