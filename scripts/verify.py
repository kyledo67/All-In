"""Independent numerical checks. No validation targets or holdout scores read."""
import argparse
import ast
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slot_model import runtime
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from slot_model.features import FEATURES, PAYOUTS, prepare, feature_at, cached_prepare
from slot_model.model import (CONFIG, SEED, new_control, adaptation_model,
                              predict_batch, next_probability, serialize, deserialize)
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import Query, Play as HistoryRow


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def softmax(x):
    x = np.asarray(x)
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def reference_filter(pi, trans, likelihood):
    state = np.array(pi)
    filtered, loglik = [], 0.0
    for row in likelihood:
        joint = state * row
        loglik += np.log(joint.sum())
        state = joint / joint.sum()
        filtered.append(state)
        state = state @ trans
    return np.array(filtered), loglik


def source_checks():
    original = ROOT / 'research/upstream_categorical_glm_hmm.py'
    modified = ROOT / 'research/vendor/dynamax/dynamax/hidden_markov_model/models/categorical_glm_hmm.py'
    a, b = ast.parse(original.read_text()), ast.parse(modified.read_text())
    before = {n.name: ast.dump(n) for n in a.body if isinstance(n, ast.ClassDef)}
    after = {n.name: ast.dump(n) for n in b.body if isinstance(n, ast.ClassDef)}
    assert all(after[k] == v for k, v in before.items())
    spec = importlib.util.spec_from_file_location('clean_categorical', original)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert Path(inspect.getfile(new_control().__class__)).resolve() == modified.resolve()
    return module, dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        modified_sha256=hashlib.sha256(modified.read_bytes()).hexdigest(),
        unchanged_original_classes=list(before),
        original_source=str(original), modified_source=str(modified))


def tiny_control(module, out):
    start = time.perf_counter()
    m = module.CategoricalRegressionHMM(2, 5, 3,
        m_step_optimizer=optax.adam(.03), m_step_num_iters=30)
    x = jnp.array([[1., 0., -1.], [0., 1., 1.], [1., 0., 1.],
                   [0., 1., -1.], [1., 0., 1.]])
    y = jnp.array([0, 1, 0, 3, 2])
    p, props = m.initialize(jr.PRNGKey(SEED))
    fitted, lp = m.fit_em(p, props, y, x, num_iters=5, verbose=False)
    likelihood = softmax(np.einsum('kcf,tf->tkc', np.array(fitted.emissions.weights), x)
                         + np.array(fitted.emissions.biases)[None])
    obs = likelihood[np.arange(5), :, np.array(y)]
    filtered, ll = reference_filter(fitted.initial.probs, fitted.transitions.transition_matrix, obs)
    actual = m.filter(fitted, y, x)
    smooth = np.zeros((5, 2)); normalizer = 0.
    for states in itertools.product(range(2), repeat=5):
        weight = float(fitted.initial.probs[states[0]]) * obs[0, states[0]]
        for t in range(1, 5):
            weight *= float(fitted.transitions.transition_matrix[states[t-1], states[t]]) * obs[t, states[t]]
        normalizer += weight
        smooth[np.arange(5), states] += weight
    smooth /= normalizer
    errors = dict(filter_max_abs=float(np.max(np.abs(filtered - actual.filtered_probs))),
                  loglik_abs=float(abs(ll - actual.marginal_loglik)),
                  enumeration_loglik_abs=float(abs(np.log(normalizer)-ll)),
                  smoother_max_abs=float(np.max(np.abs(smooth-m.smoother(fitted,y,x).smoothed_probs))))
    # Separate recurrence reference checks next-spin state averaging.
    nx = jnp.array([1., 0., -1.])
    expected = (filtered[-1] @ np.asarray(fitted.transitions.transition_matrix)) @ softmax(
        np.asarray(fitted.emissions.weights) @ np.asarray(nx) + np.asarray(fitted.emissions.biases))
    errors['next_forecast_max_abs'] = float(np.max(np.abs(expected-next_probability(m,fitted,y,x,nx))))
    assert max(errors.values()) < 1e-10
    save(out/'tiny_control.json', dict(seed=SEED, initial=serialize(p), fitted=serialize(fitted),
         x=x.tolist(), y=y.tolist(), next_x=nx.tolist(), prediction=expected.tolist(),
         log_joint=lp.tolist(), seconds=time.perf_counter()-start, errors=errors))
    return errors


def objective_checks(population):
    """Intercept the real inherited optimizer closure, not a copied objective."""
    rng = np.random.default_rng(SEED)
    x = jnp.asarray(rng.normal(size=(7, len(FEATURES))))
    y = jnp.asarray([0, 1, 2, 0, 4, 3, 0])
    delta = jnp.asarray(rng.normal(size=(5, 3))*.1)
    captured = []
    for precision in (0., 30.):
        m, p, props = adaptation_model(population, precision)
        p = p._replace(emissions=p.emissions._replace(deviations=delta))
        posterior = m.smoother(p, y, x)
        stats = m.emission_component.collect_suff_stats(p.emissions, posterior, y, x)
        stats = jax.tree.map(lambda z: z[None], stats)
        def capture(fn, params, optimizer, optimizer_state=None, **kwargs):
            captured.append(lambda d: fn(params._replace(deviations=d)))
            return params, optimizer_state, jnp.zeros(1)
        with patch('dynamax.hidden_markov_model.models.abstractions.run_gradient_descent', capture):
            m.emission_component.m_step(p.emissions, props.emissions, stats,
                m.emission_component.initialize_m_step_state(p.emissions, props.emissions))
    loss0, loss = captured
    analytic = np.asarray(jax.grad(loss)(delta))
    finite = np.zeros_like(analytic); eps = 1e-5
    for index in np.ndindex(delta.shape):
        h = np.zeros_like(finite); h[index] = eps
        finite[index] = (float(loss(delta+h))-float(loss(delta-h)))/(2*eps)
    error = float(np.max(np.abs(analytic-finite)))
    penalty_error = float(abs((loss(delta)-loss0(delta))-15*jnp.sum(delta**2)))
    gradient_penalty_error = float(np.max(np.abs(jax.grad(loss)(delta)-jax.grad(loss0)(delta)-30*delta)))
    assert error < 1e-7 and penalty_error < 1e-10 and gradient_penalty_error < 1e-10
    # Confirm frozen population/transition parameters survive actual EM adaptation.
    fitted, _ = m.fit_em(p, props, y, x, num_iters=2, verbose=False)
    fixed_diff = max(float(np.max(np.abs(a-b))) for a,b in zip(
        jax.tree.leaves(population), jax.tree.leaves(fitted)[:4]))
    assert fixed_diff == 0
    return dict(finite_difference_max_abs=error, objective_penalty_error=penalty_error,
                gradient_penalty_error=gradient_penalty_error, frozen_parameter_max_abs=fixed_diff,
                actual_objective='HMMEmissions.m_step.neg_expected_log_joint', epsilon=eps)


def synthetic_case(kind, n=64):
    rng = np.random.default_rng(SEED + ['identical','opposite_lag','short_history','changing_regimes'].index(kind))
    p = new_control().initialize(jr.PRNGKey(SEED))[0]
    probs = np.array([[.55,.20,.12,.10,.03]]*2)
    if kind == 'changing_regimes':
        probs = np.array([[.82,.08,.06,.03,.01],[.25,.35,.22,.14,.04]])
    p = p._replace(initial=p.initial._replace(probs=jnp.array([.5,.5])),
        transitions=p.transitions._replace(transition_matrix=jnp.array([[.95,.05],[.05,.95]])),
        emissions=p.emissions._replace(weights=jnp.zeros((2,5,len(FEATURES))),biases=jnp.log(probs)))
    length = 8 if kind == 'short_history' else 32
    xs, ys, nxs, truth = [], [], [], []
    for i in range(n):
        ticks = {t:int(rng.choice(5,p=probs[0])) for t in range(8)}
        state = rng.integers(2)
        delta = np.zeros((5,3))
        if kind == 'opposite_lag':
            delta[0,1] = (1 if i%2 else -1)*1.5
            delta[1:,1] = -delta[0,1]/4
        xx, yy, likes = [], [], []
        for t in range(8, 8+length):
            x = np.asarray(feature_at(ticks,t,'balanced'))
            em = softmax(np.log(probs) + delta @ np.array([1,x[3],x[11]]))
            y = int(rng.choice(5,p=em[state]))
            xx.append(x); yy.append(y); likes.append(em[:,y]); ticks[t]=y
            state = int(rng.choice(2,p=np.asarray(p.transitions.transition_matrix[state])))
        nx = np.asarray(feature_at(ticks,8+length,'balanced'))
        filt,_ = reference_filter(p.initial.probs,p.transitions.transition_matrix,np.array(likes))
        em = softmax(np.log(probs) + delta @ np.array([1,nx[3],nx[11]]))
        truth.append((filt[-1] @ p.transitions.transition_matrix) @ em)
        xs.append(xx); ys.append(yy); nxs.append(nx)
    data = dict(x=np.array(xs),y=np.array(ys),next_x=np.array(nxs))
    result = {}
    for name,precision in [('shared',None),('unshrunk',0.),('pooled_30',30.)]:
        pred,d,diag = predict_batch(p,data,precision)
        result[name] = dict(squared_probability_error=float(np.mean(np.sum((pred-np.array(truth))**2,axis=1))),
            deviation_rms=diag['deviation_rms'], seconds=diag['seconds'],
            optimizer_decreases=diag['decreases'])
    result['shrinkage_reduces_deviations'] = result['pooled_30']['deviation_rms'] < result['unshrunk']['deviation_rms']
    result['pooling_beats_shared'] = result['pooled_30']['squared_probability_error'] < result['shared']['squared_probability_error']
    result['machines'],result['observations'] = n,length
    return result


def causal_checks(population):
    rng = np.random.default_rng(57)
    q = Query('q','episode','machine','steady',1,64)
    history = [HistoryRow('episode','machine',t,int(rng.choice(PAYOUTS))) for t in range(64)]
    first = prepare([q],history)
    altered = history + [HistoryRow('episode','machine',t,20) for t in range(64,72)]
    assert first == prepare([q],altered)
    assert first['x'] == prepare([replace(q,signal=-1)],history)['x']
    # Old sampled ticks can change or disappear without changing complete recent windows.
    sparse = [r for r in history if r.tick>=24 or r.tick in (1,7,13)]
    assert first == prepare([q],sparse)
    ticks = {0:0,2:1,3:0,4:0,5:0,6:0,7:0,8:0}
    assert feature_at(ticks,9,'steady') is None  # tick 1 is missing, not zero.
    missing_raises = False
    try:
        prepare([q],[r for r in history if r.tick!=60])
    except ValueError:
        missing_raises=True
    assert missing_raises
    # Same machine ID in an unrelated episode never changes the first sequence.
    q2 = replace(q,query_id='q2',episode_id='other')
    h2 = [HistoryRow('other','machine',t,20) for t in range(64)]
    combined = prepare([q,q2],history+h2)
    assert first['x'][0] == combined['x'][0] and first['y'][0] == combined['y'][0]
    a={k:np.asarray(v) for k,v in first.items()}; b={k:np.asarray(v) for k,v in prepare([q],altered).items()}
    p1,_,_=predict_batch(population,a,30.)
    p2,_,_=predict_batch(population,b,30.)
    diff=float(np.max(np.abs(p1-p2))); assert diff==0
    return dict(post_cutoff_probability_max_abs=diff, missing_lags_rejected=missing_raises,
        sparse_older_rows_ignored=True, historical_signal_unused=True, episode_identity_isolated=True,
        limitation='Discarding old observations loses information; it does not integrate unknown autoregressive inputs.')


def competition_replay(module, data_path, out):
    start=time.perf_counter()
    tq,th=load_inputs(data_path/'train'); vq,vh=load_inputs(data_path/'validation')
    train,_=cached_prepare(data_path/'train',tq,th,training=True)
    val,_=cached_prepare(data_path/'validation',vq,vh)
    m=module.CategoricalRegressionHMM(2,5,len(FEATURES),
        m_step_optimizer=optax.adam(CONFIG['learning_rate']),m_step_num_iters=CONFIG['m_step_iters'])
    value=json.loads((ROOT/'artifacts/experiment/initial.json').read_text())
    p,props=m.initialize(jr.PRNGKey(SEED),initial_probs=jnp.array(value['initial']),
        transition_matrix=jnp.array(value['transition']),emission_weights=jnp.array(value['weights']),emission_biases=jnp.array(value['biases']))
    fitted,lp=m.fit_em(p,props,jnp.asarray(train['y']),jnp.asarray(train['x']),
        num_iters=CONFIG['population_em_iters'],verbose=False)
    saved=deserialize(json.loads((ROOT/'artifacts/experiment/population.json').read_text()))
    diff=max(float(np.max(np.abs(a-b))) for a,b in zip(jax.tree.leaves(fitted),jax.tree.leaves(saved)))
    assert diff < 1e-10
    f=jax.jit(jax.vmap(lambda y,x,nx:next_probability(m,fitted,y,x,nx)))
    predictions=np.asarray(f(jnp.asarray(val['y']),jnp.asarray(val['x']),jnp.asarray(val['next_x'])))
    write_submission(out/'clean_upstream_validation.csv',dict(zip(val['owners'].tolist(),predictions.tolist())))
    save(out/'clean_upstream_population.json',serialize(fitted))
    return dict(source=inspect.getfile(type(m)), max_parameter_difference=diff,
                seconds=time.perf_counter()-start, log_joint=lp.tolist(),
                note='Clean upstream class trained on permitted train histories; no validation labels/targets read.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/verification')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    module,source=source_checks()
    pop=deserialize(json.loads((ROOT/'artifacts/experiment/population.json').read_text()))
    report=dict(source=source,tiny=tiny_control(module,args.out),objective=objective_checks(pop),causal=causal_checks(pop))
    save(args.out/'results.json',report)
    report['synthetic']={}
    for kind in ('identical','opposite_lag','short_history','changing_regimes'):
        print('Synthetic:',kind,flush=True)
        report['synthetic'][kind]=synthetic_case(kind)
        save(args.out/'results.json',report)
    report['competition_replay']=competition_replay(module,args.data,args.out)
    save(args.out/'results.json',report)
    print(json.dumps(report,indent=2))
