"""Dependency-free, float64 forward inference for the shared saved HMM.

This is the same recurrence independently checked against Dynamax, for stable
offline reproduction without a JAX runtime. Training remains in Dynamax.
"""
import math
from .features import prepare


def softmax(logits):
    peak = max(logits)
    values = [math.exp(v-peak) for v in logits]
    total = math.fsum(values)
    return [v/total for v in values]


def state_emissions(model, x):
    return [softmax([math.fsum(w*f for w,f in zip(row,x))+bias
                     for row,bias in zip(weights,biases)])
            for weights,biases in zip(model['weights'],model['biases'])]


def predict_shared(model, queries, history):
    data = prepare(queries, history)
    trans = model['transition']
    predictions = {}
    for owner, ys, xs, nx in zip(data['owners'],data['y'],data['x'],data['next_x']):
        state = model['initial'][:]
        for y,x in zip(ys,xs):
            emissions = state_emissions(model,x)
            posterior = [s*e[y] for s,e in zip(state,emissions)]
            norm = math.fsum(posterior)
            posterior = [v/norm for v in posterior]
            state = [math.fsum(posterior[j]*trans[j][k] for j in range(len(state)))
                     for k in range(len(state))]
        emissions = state_emissions(model,nx)
        predictions[owner] = [math.fsum(s*e[c] for s,e in zip(state,emissions)) for c in range(5)]
    return predictions
