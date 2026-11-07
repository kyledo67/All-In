"""Upstream control and small empirical-Bayes-style MAP adaptation."""
from . import runtime
from .features import FEATURES, ADAPT_COLUMNS
import inspect
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from dynamax.hidden_markov_model.models.categorical_glm_hmm import (
    CategoricalRegressionHMM, PartiallyPooledCategoricalEmissions,
    ParamsPartiallyPooledEmissions,
)
from dynamax.parameters import ParameterProperties

SEED = 20261107
CONFIG = dict(states=2, classes=5, features=FEATURES, window=32, seed=SEED,
              population_em_iters=15, adaptation_em_iters=8, m_step_iters=30,
              learning_rate=0.03, deviation_columns=list(ADAPT_COLUMNS),
              precisions=[0.0, 0.3, 3.0, 30.0])

def new_control():
    return CategoricalRegressionHMM(2, 5, len(FEATURES),
        m_step_optimizer=optax.adam(CONFIG['learning_rate']),
        m_step_num_iters=CONFIG['m_step_iters'])

def fit_population(data):
    start = time.perf_counter()
    model = new_control()
    counts = np.bincount(data['y'].ravel(), minlength=5) + 0.5
    biases = jnp.tile(jnp.log(jnp.asarray(counts/counts.sum())), (2, 1))
    weights = jr.normal(jr.PRNGKey(SEED), (2, 5, len(FEATURES))) * 0.05
    params, props = model.initialize(jr.PRNGKey(SEED), emission_weights=weights,
        emission_biases=biases, initial_probs=jnp.array([0.5, 0.5]),
        transition_matrix=jnp.array([[0.95, 0.05], [0.05, 0.95]]))
    initial = params
    fitted, logp = model.fit_em(params, props, jnp.asarray(data['y']),
        jnp.asarray(data['x']), num_iters=CONFIG['population_em_iters'], verbose=False)
    logp = np.asarray(logp)
    return model, fitted, initial, dict(seconds=time.perf_counter()-start,
        log_joint=logp.tolist(), decreases=int(np.sum(np.diff(logp) < -1e-7)),
        final_gain=float(logp[-1]-logp[-2]),
        source=inspect.getfile(type(model)), sequences=len(data['y']))

def adaptation_model(population, precision):
    model = new_control()
    model.emission_component = PartiallyPooledCategoricalEmissions(
        2, 5, len(FEATURES), ADAPT_COLUMNS, precision,
        optax.adam(CONFIG['learning_rate']), CONFIG['m_step_iters'])
    _, props = model.initialize(jr.PRNGKey(SEED))
    props.initial.probs.trainable = False
    props.transitions.transition_matrix.trainable = False
    props = props._replace(emissions=ParamsPartiallyPooledEmissions(
        ParameterProperties(trainable=False), ParameterProperties(trainable=False),
        ParameterProperties()))
    params = population._replace(emissions=ParamsPartiallyPooledEmissions(
        population.emissions.weights, population.emissions.biases,
        jnp.zeros((5, 1+len(ADAPT_COLUMNS)))))
    return model, params, props

def next_probability(model, params, y, x, next_x):
    filtered = model.filter(params, y, x).filtered_probs[-1]
    next_state = filtered @ params.transitions.transition_matrix
    emissions = jax.vmap(lambda k: model.emission_distribution(params, k, next_x).probs_parameter())(jnp.arange(2))
    return next_state @ emissions

def predict_batch(population, data, precision=None, batch_size=64):
    start = time.perf_counter()
    if precision is None:
        model = new_control()
        def single(y, x, nx):
            return next_probability(model, population, y, x, nx), jnp.zeros((5,3)), jnp.zeros(8)
    else:
        model, initial, props = adaptation_model(population, precision)
        def single(y, x, nx):
            params, lp = model.fit_em(initial, props, y, x,
                num_iters=CONFIG['adaptation_em_iters'], verbose=False)
            return next_probability(model, params, y, x, nx), params.emissions.deviations, lp
    run = jax.jit(jax.vmap(single))
    results = []
    for start_idx in range(0, len(data['y']), batch_size):
        sl = slice(start_idx, start_idx+batch_size)
        result = run(jnp.asarray(data['y'][sl]), jnp.asarray(data['x'][sl]), jnp.asarray(data['next_x'][sl]))
        results.append(tuple(np.asarray(r) for r in result))
    probs, deviations, lp = [np.concatenate([r[i] for r in results]) for i in range(3)]
    assert np.isfinite(probs).all() and (probs >= 0).all()
    assert np.max(np.abs(probs.sum(1)-1)) < 1e-10
    return probs, deviations, dict(seconds=time.perf_counter()-start,
        log_joint=lp.tolist(), decreases=int(np.sum(np.diff(lp,axis=1) < -1e-7)),
        deviation_rms=float(np.sqrt(np.mean(deviations**2))),
        source=inspect.getfile(type(model.emission_component)))

def serialize(params):
    return dict(initial=np.asarray(params.initial.probs).tolist(),
                transition=np.asarray(params.transitions.transition_matrix).tolist(),
                weights=np.asarray(params.emissions.weights).tolist(),
                biases=np.asarray(params.emissions.biases).tolist())

def deserialize(value):
    model = new_control()
    p, _ = model.initialize(jr.PRNGKey(SEED),
        initial_probs=jnp.asarray(value['initial']), transition_matrix=jnp.asarray(value['transition']),
        emission_weights=jnp.asarray(value['weights']), emission_biases=jnp.asarray(value['biases']))
    return p
