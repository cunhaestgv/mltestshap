"""Faulty model versions (mutants).

Model-level operators follow DeepMutation (Ma et al., 2018) and change the
weights of the trained model directly:
    GF  Gaussian fuzzing: add noise (std of the layer) to a fraction of weights
    WS  weight shuffling: shuffle the incoming weights of a fraction of neurons
    NEB neuron effect blocking: zero the outgoing weights of a fraction of neurons
    NAI neuron activation inverse: negate the outgoing weights of a fraction of neurons
Data-level operators retrain the model on corrupted training data:
    LN  label noise: corrupt a fraction of the training targets
    FS  feature shuffle: break the relation between one important variable and the target
    DS  data slice removal: drop the training samples in the top quartile of the
        most important variable
"""
import numpy as np

MODEL_OPS = ("GF", "WS", "NEB", "NAI")
LEVELS = (0.02, 0.05, 0.1)


def _kernels(weights):
    return [i for i, w in enumerate(weights) if w.ndim == 2]


def mutate_weights(weights, op, rate, rng):
    w = [a.copy() for a in weights]
    kernels = _kernels(w)
    if op == "GF":
        for i in kernels:
            mask = rng.random(w[i].shape) < rate
            w[i][mask] += rng.normal(0, w[i].std(), mask.sum())
    elif op == "WS":
        for i in kernels:
            cols = rng.choice(w[i].shape[1], max(1, int(rate * w[i].shape[1])), replace=False)
            for c in cols:
                w[i][:, c] = rng.permutation(w[i][:, c])
    elif op in ("NEB", "NAI"):
        # rows of every kernel except the first are hidden neurons
        # (for an LSTM, the recurrent kernel and the dense layer)
        for i in kernels[1:]:
            rows = rng.choice(w[i].shape[0], max(1, int(rate * w[i].shape[0])), replace=False)
            w[i][rows] = 0.0 if op == "NEB" else -w[i][rows]
    else:
        raise ValueError(op)
    return w


def corrupt_data(data, op, param, rng):
    """Return (X_train, y_train) corrupted by a data-level operator.

    For LN, param is the fraction of corrupted targets. For FS and DS, param
    is the index of the variable (not of the flattened feature) to corrupt.
    """
    X, y = data["X_train"].copy(), data["y_train"].copy()
    if op == "LN":
        idx = rng.choice(len(y), int(param * len(y)), replace=False)
        if data["task"] == "classification":
            y[idx] = 1 - y[idx]
        else:
            y[idx] = y[rng.permutation(idx)]
    elif op == "FS":
        perm = rng.permutation(len(X))
        X[..., param] = X[perm][..., param]
    elif op == "DS":
        col = X[:, -1, param] if X.ndim == 3 else X[:, param]
        keep = col <= np.quantile(col, 0.75)
        if keep.all():                       # binary or skewed variable
            keep = col < col.max()
        X, y = X[keep], y[keep]
    else:
        raise ValueError(op)
    return X, y
