"""Model construction, training, evaluation and SHAP explanations.

TensorFlow runs in graph (v1) mode, as in the original notebooks, because
shap.DeepExplainer does not support LSTMs in eager mode. Every model is built
in a fresh graph (clear_session) from a list of numpy weights, so mutants and
retrained models never share state.
"""
import numpy as np
import tensorflow as tf
import shap
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

CONFIG = {
    # dataset: (units, batch size, max epochs)
    "bike": dict(units=32, batch=64, epochs=60),
    "diet": dict(units=75, batch=8, epochs=300),
    "diet_date": dict(units=75, batch=8, epochs=300),
    "adult": dict(units=32, batch=128, epochs=60),
    "synth": dict(units=32, batch=64, epochs=200),     # Stage 2 (stage2.py)
}


def build(data, name, seed):
    tf.keras.backend.clear_session()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    cfg = CONFIG[name]
    L = tf.keras.layers
    if data["kind"] == "seq":
        model = tf.keras.Sequential([
            L.LSTM(cfg["units"], input_shape=(data["lookback"], data["n_vars"])),
            L.Dense(1)])
        model.compile(optimizer="adam", loss="mse")
    else:
        reg = data["task"] == "regression"
        model = tf.keras.Sequential([
            L.Dense(cfg["units"], activation="relu", input_shape=(data["n_vars"],)),
            L.Dense(cfg["units"] // 2, activation="relu"),
            L.Dense(1, activation=None if reg else "sigmoid")])
        model.compile(optimizer="adam", loss="mse" if reg else "binary_crossentropy")
    return model


def train(data, name, seed, X=None, y=None):
    """Train a model and return its weights. X/y override the training data."""
    X = data["X_train"] if X is None else X
    y = data["y_train"] if y is None else y
    cfg = CONFIG[name]
    model = build(data, name, seed)
    stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X, y, validation_data=(data["X_val"], data["y_val"]), epochs=cfg["epochs"],
              batch_size=cfg["batch"], callbacks=[stop], verbose=0, shuffle=True)
    return model.get_weights()


def load(data, name, weights, seed=0):
    model = build(data, name, seed)
    model.set_weights(weights)
    return model


def predict(model, X, batch=4096):
    return model.predict(X, batch_size=batch, verbose=0).reshape(-1)


def metrics(data, model):
    p = predict(model, data["X_test"])
    y = data["y_test"]
    if data["task"] == "regression":
        span = data["y_max"] - data["y_min"]
        return dict(rmse=float(np.sqrt(mean_squared_error(y, p)) * span),
                    mae=float(mean_absolute_error(y, p) * span))
    return dict(accuracy=float(accuracy_score(y, p > 0.5)), auc=float(roc_auc_score(y, p)))


def shap_values(model, background, X, batch=500):
    """SHAP values (DeepSHAP) of X, flattened to shape (n, n_features)."""
    explainer = shap.DeepExplainer(model, background)
    out = []
    for i in range(0, len(X), batch):
        sv = explainer.shap_values(X[i:i + batch], check_additivity=False)
        sv = sv[0] if isinstance(sv, list) else sv
        out.append(np.asarray(sv).reshape(len(X[i:i + batch]), -1))
    return np.concatenate(out)


def output_effects(model, background, X, pairs, groups=None, chunk=20000):
    """Black-box contribution of feature f to the prediction of input x.

    effect(x, f) = M(x) - mean_b M(x with feature f replaced by b_f),
    over the background samples b. It needs only calls to predict(), and its
    sign plays the role of the SHAP sign in the black-box oracle.
    pairs: array of (row index into X, feature index).
    groups: boolean matrix (features x flattened columns) of the columns each
    feature replaces; default: feature f is flattened column f.
    """
    shape = X.shape[1:]
    Xf = X.reshape(len(X), -1)
    Bf = background.reshape(len(background), -1)
    groups = np.eye(Xf.shape[1], dtype=bool) if groups is None else groups
    nb = len(Bf)
    base = predict(model, X)
    effects = np.empty(len(pairs))
    per = max(1, chunk // nb)
    for start in range(0, len(pairs), per):
        part = pairs[start:start + per]
        batch = np.repeat(Xf[part[:, 0]], nb, axis=0)
        mask = np.repeat(groups[part[:, 1]], nb, axis=0)
        batch = np.where(mask, np.tile(Bf, (len(part), 1)), batch)
        p = predict(model, batch.reshape((-1,) + shape)).reshape(len(part), nb)
        effects[start:start + len(part)] = base[part[:, 0]] - p.mean(1)
    return effects
