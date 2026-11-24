"""Stage 3: do margins, SHAP values and trees choose the right fix? (see STAGE3.md)

For every case: train 5 base models, compute the signals on discovery data, try
every candidate fix on the test data and score what each level of rules would
have recommended.

    python3 -I src/stage3.py                  # all cases
    python3 -I src/stage3.py --shard 0/8      # one of 8 parallel workers
    python3 -I src/stage3.py --summarize      # tables from the finished cases

Results go to results/stage3/: one JSON per case, saved as soon as it is done,
so an interrupted run resumes where it stopped. No weights are kept.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.get_logger().setLevel("ERROR")

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

import datasets  # noqa: E402
import diagnose  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402

SYNTH = ["noeffect", "sharp", "inter", "complex", "nonmono", "missing", "noise", "sparse",
         "unstable", "spurious", "outliers", "imbalance"]
ONLY = {"outliers": "regression", "imbalance": "classification"}
REAL = {"adult": "classification", "bike": "regression"}
N_TR, N_VA, N_DI, N_TE = 3000, 500, 750, 750
N_SEEDS, N_CAND, N_EXTRA, N_BG, N_SHAP = 5, 3, 1000, 100, 1000
BASE, BIG = (32, 16), (128, 64, 32)
LEVELS = ("L1", "L2", "L3")
COPY_NOISE = 1.0                                       # Stage 4 raises it for harder spurious cases


def sg(u):
    return np.where(u > 0.5, 1.0, -1.0)


def signal(scn, X, z):
    """Noise-free target of a scenario."""
    x1 = X[:, 0]
    step = 0.5 * (x1 > 0.5)
    inter = 0.8 * (x1 - 0.5) * sg(X[:, 1])
    g = {"noeffect": 0 * x1, "inter": inter, "noise": inter,
         "complex": inter * sg(X[:, 5]) * sg(X[:, 6]), "nonmono": 0.5 * ((x1 > 0.3) & (x1 < 0.7)),
         "missing": 0.8 * (x1 - 0.5) * sg(z)}.get(scn, step)
    return 0.3 * X[:, 2] - 0.3 * X[:, 3] + 0.2 * X[:, 4] + g


def draw(scn, task, n, rng, cut):
    """Fresh samples: inputs, noisy and clean targets, before any train-only defect."""
    F = 27 if scn == "unstable" else 7
    X = rng.random((n, F))
    z = rng.random(n)
    f = signal(scn, X, z)
    if cut is None:
        cut = np.quantile(f, 0.9 if scn == "imbalance" else 0.5)
    y = f + rng.normal(0, 0.05, n)
    if task == "classification":
        y, f = (y > cut).astype(float), (f > cut).astype(float)
    Xz = X.copy()                                     # the inputs an audit would give
    if scn == "missing":
        Xz[:, 1] = z
        X[:, 1] = z + rng.normal(0, 0.15, n)
    return dict(X=X, Xz=Xz, y=y, f=f, x1=Xz[:, 0].copy(), cut=cut)


def copy_of_target(y, task, rng):
    if task == "regression":
        return np.clip((y + 0.3) / 1.3 + rng.normal(0, 0.05 * COPY_NOISE, len(y)), 0, 1)
    return np.clip(0.25 + 0.5 * y + rng.normal(0, 0.15 * COPY_NOISE, len(y)), 0, 1)


def corrupt(scn, task, d, rng, thin=True):
    """Train-time defects. Returns inputs, inputs for an audit, targets, clean targets."""
    X, Xz, y, y0 = d["X"].copy(), d["Xz"], d["y"].copy(), d["y"].copy()
    mid = (d["x1"] > 0.25) & (d["x1"] < 0.75)
    if scn == "noise":
        pick = np.flatnonzero(mid & (rng.random(len(y)) < 0.4))
        y[pick] = y[rng.permutation(pick)]
    if scn == "outliers":
        pick = rng.random(len(y)) < 0.05
        y[pick] += 3 * rng.choice([-1, 1], pick.sum())
    if scn == "spurious":
        X[:, 6] = copy_of_target(y, task, rng)
    keep = np.ones(len(y), bool)
    if thin and scn == "sparse":
        keep = ~mid | (rng.random(len(y)) < 0.2)
    if thin and scn == "unstable":
        keep[300:] = False
    return X[keep], Xz[keep], y[keep], y0[keep]


def synth_case(scn, task, rep):
    idx = SYNTH.index(scn)
    rng = np.random.default_rng(3000 + 100 * idx + 50 * (task == "classification") + rep)
    d = draw(scn, task, N_TR + N_VA + N_DI + N_TE, rng, None)
    part = lambda a, b: {k: (v[a:b] if k != "cut" else v) for k, v in d.items()}
    tr = part(0, N_TR)
    va = part(N_TR, N_TR + N_VA)
    di = part(N_TR + N_VA, N_TR + N_VA + N_DI)
    te = part(N_TR + N_VA + N_DI, None)
    X_tr, Xz_tr, y_tr, y0_tr = corrupt(scn, task, tr, rng)
    X_va = copy_of_target(va["y"], task, rng) if scn == "spurious" else None
    Xv = va["X"].copy()
    if X_va is not None:
        Xv[:, 6] = X_va
    F = X_tr.shape[1]
    names = ["x%d" % i for i in range(1, F + 1)]

    def pool(rule, n, seed):
        """Up to n fresh samples inside a slice, from the same collection process."""
        rng2 = np.random.default_rng(seed)
        X, y = [], []
        for _ in range(40):
            xp, _, yp, _ = corrupt(scn, task, draw(scn, task, 5000, rng2, d["cut"]), rng2, thin=False)
            m = rule(xp)
            X.append(xp[m]), y.append(yp[m])
            if sum(map(len, y)) >= n:
                break
        return np.concatenate(X)[:n].astype("float32"), np.concatenate(y)[:n].astype("float32")

    audit = None
    if scn == "missing":
        audit = dict(X_train=Xz_tr, X_val=va["Xz"], X_disc=di["Xz"], X_test=te["Xz"])
    if scn in ("noise", "outliers"):
        audit = dict(y_train=y0_tr)
    fl = lambda a: a.astype("float32")
    return dict(X_train=fl(X_tr), y_train=fl(y_tr), X_val=fl(Xv), y_val=fl(va["y"]),
                X_disc=fl(di["X"]), y_disc=fl(di["y"]), X_test=fl(te["X"]), y_test=fl(te["y"]),
                clean_test=te["f"], x1_disc=di["x1"], x1_test=te["x1"], audit=audit, pool=pool,
                task=task, kind="tab", n_vars=F, lookback=1, feature_names=names,
                batch=64, epochs=200, real=False)


def real_case(name, rep):
    data = dict(datasets.LOADERS[name]())
    n = len(data["X_test"])
    order = np.random.default_rng(rep).permutation(n) if name == "adult" else np.arange(n)
    a, b = order[:n // 2], order[n // 2:]
    y_te = data["y_test"]
    cfg = models.CONFIG[name]
    data.update(X_disc=data["X_test"][a], y_disc=y_te[a], X_test=data["X_test"][b], y_test=y_te[b],
                clean_test=y_te[b], audit=None, pool=None, batch=cfg["batch"], epochs=cfg["epochs"],
                real=True, seed0=1000 * rep)
    return data


def flat(X):
    return X.reshape(len(X), -1)


def build(data, seed, units=BASE, l2=0.0, drop=0.0, huber=False):
    tf.keras.backend.clear_session()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    L = tf.keras.layers
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    clf = data["task"] == "classification"
    model = tf.keras.Sequential()
    if data["kind"] == "seq":
        model.add(L.LSTM(units[0] * (4 if units is BIG else 1), kernel_regularizer=reg,
                         input_shape=(data["lookback"], data["n_vars"])))
        if drop:
            model.add(L.Dropout(drop))
    else:
        model.add(L.InputLayer(input_shape=(data["n_vars"],)))
        for u in units:
            model.add(L.Dense(u, activation="relu", kernel_regularizer=reg))
            if drop:
                model.add(L.Dropout(drop))
    model.add(L.Dense(1, activation="sigmoid" if clf else None))
    loss = "binary_crossentropy" if clf else tf.keras.losses.Huber(0.1) if huber else "mse"
    model.compile(optimizer="adam", loss=loss)
    return model


def fit(data, seed, keep=False, class_weight=None, **kw):
    """Train one model. Returns predictions; with keep=True, the model as well."""
    model = build(data, seed + data.get("seed0", 0), **kw)
    stop =tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
              epochs=data["epochs"], batch_size=data["batch"], callbacks=[stop], verbose=0,
              shuffle=True, class_weight=class_weight)
    out = {k: models.predict(model, data["X_" + k]) for k in ("disc", "test")}
    return (out, model) if keep else out


def balanced_error(y, p, thr=0.5):
    yhat = p > thr
    pos, neg = y > 0.5, y <= 0.5
    return 0.5 * ((~yhat[pos]).mean() + yhat[neg].mean())


def test_error(data, p, thr=0.5):
    y = data["clean_test"]
    return float(balanced_error(y, p, thr) if data["task"] == "classification" else np.abs(y - p).mean())


def best_threshold(y, p):
    grid = np.quantile(p, np.linspace(0.01, 0.99, 99))
    return float(grid[int(np.argmin([balanced_error(y, p, t) for t in grid]))])


def loss_of(data, y, p):
    if data["task"] == "regression":
        return (y - p) ** 2
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def loss_shapley(data, model, X, y, bg, rng, m=30, n=300):
    """Mean Shapley value of the loss per feature, by sampled permutations."""
    shape = X.shape[1:]
    Xf, Bf = flat(X), flat(bg)
    idx = rng.permutation(len(Xf))[:n]
    F = Xf.shape[1]
    rows = np.repeat(idx, m)
    perms = np.array([rng.permutation(F) for _ in rows])
    b = Bf[rng.integers(0, len(Bf), len(rows))]
    # coalition j holds the first j features of the permutation, from x; the rest from b
    pos = np.argsort(perms, axis=1)
    phi = np.zeros(F)
    prev = None
    for j in range(F + 1):
        batch = np.where(pos < j, Xf[rows], b)
        L = loss_of(data, y[rows], models.predict(model, batch.reshape((-1,) + shape)))
        if prev is not None:
            phi += np.bincount(perms[:, j - 1], weights=L - prev, minlength=F)
        prev = L
    return phi / len(rows), float(prev.mean())


def signals(data, names):
    """Base models and everything the rules read."""
    rng = np.random.default_rng(0)
    Xd, yd = data["X_disc"], data["y_disc"]
    sub = rng.permutation(len(Xd))[:N_SHAP]
    bg = data["X_train"][rng.permutation(len(data["X_train"]))[:N_BG]]
    tsub = rng.permutation(len(data["X_test"]))[:N_SHAP]
    P_d, P_t, S, res_tr = [], [], [], None
    for seed in range(N_SEEDS):
        out, model = fit(data, seed, keep=True)
        P_d.append(out["disc"]), P_t.append(out["test"])
        S.append(models.shap_values(model, bg, Xd[sub]))
        if seed == 0:
            phi, base_loss = loss_shapley(data, model, Xd, yd, bg, rng)
            S_te = models.shap_values(model, bg, data["X_test"][tsub])
            res_tr = data["y_train"] - models.predict(model, data["X_train"])
    P_d, P_t = np.array(P_d), np.array(P_t)

    err = np.abs(yd - P_d).mean(0)
    Xf = flat(Xd)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(20, int(0.05 * len(Xf))),
                                 random_state=0).fit(Xf, err)
    leaf_d = tree.apply(Xf)
    leaves = np.unique(leaf_d)
    worst = leaves[int(np.argmax([err[leaf_d == lf].mean() for lf in leaves]))]
    in_s = leaf_d == worst
    rule = lambda X: tree.apply(flat(X)) == worst
    tr_s = rule(data["X_train"])

    clf = data["task"] == "classification"
    gb = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(random_state=0)
    gb.fit(flat(data["X_train"]), data["y_train"])
    p_gb = gb.predict_proba(Xf[in_s])[:, 1] if clf else gb.predict(Xf[in_s])
    pos = np.maximum(phi, 0)
    top = int(np.argmax(phi))
    in_sub = in_s[sub]
    Sa = np.array(S)
    mean_s = np.abs(Sa[0][in_sub]).mean(0) if in_sub.any() else np.abs(Sa[0]).mean(0)
    top_shap = int(np.argmax(mean_s))
    rows_s = in_sub if in_sub.any() else np.ones(len(sub), bool)
    sign_pos = (np.sign(Sa[:, rows_s, top_shap]) > 0).mean(0)
    root = int(tree.tree_.feature[0])
    row = None
    if root >= 0:
        resid = [yd[sub] - p[sub] for p in P_d]
        row = int(diagnose.diagnose(Xf[sub], list(Sa), resid, names).iloc[root].row)
    y_tr = data["y_train"]
    return dict(
        ratio=float(err[in_s].mean() / err.mean()),
        density=float(tr_s.mean() / in_s.mean()),
        disagreement=float(np.abs(P_d[:, in_s] - P_d[:, in_s].mean(0)).mean() / err[in_s].mean()),
        gbm=float(np.abs(yd[in_s] - p_gb).mean() / err[in_s].mean()),
        outlier_share=float((np.abs(res_tr) > 5 * np.median(np.abs(res_tr))).mean()) if not clf else 0.0,
        prevalence=float(min(y_tr.mean(), 1 - y_tr.mean())) if clf else 0.5,
        top_share=float(pos[top] / max(pos.sum(), 1e-12)), top_rel=float(pos[top] / max(base_loss, 1e-12)),
        top_feature=top, shap_top=int(np.argmax(np.abs(Sa[0]).mean(0))), root=root, row=row,
        shap_total=float(mean_s.sum()), err_slice=float(err[in_s].mean()),
        sign_agreement=float(np.maximum(sign_pos, 1 - sign_pos).mean()),
    ), dict(P_d=P_d, P_t=P_t, rule=rule, S0=Sa[0], S_te=S_te, sub=sub, tsub=tsub, err=err, in_s=in_s)


def recommend(s, level):
    """The rules of STAGE3.md."""
    if s["prevalence"] < 0.2:
        return "threshold"
    if s["outlier_share"] >= 0.01:
        return "robust"
    if level != "L1" and s["top_share"] >= 0.5 and s["top_rel"] >= 0.1:
        return "drop_top"
    if level == "L3" and s["row"] is not None:
        if s["row"] == 5:
            return "audit"
        if s["row"] == 6:
            return "more_data"
        if s["row"] == 7:
            return "ensemble"
        if s["row"] in (1, 3, 4, 9) and s["ratio"] >= 1.5:
            return "capacity"
    if s["density"] < 0.5:
        return "more_data"
    if s["disagreement"] >= 0.5 or (level != "L1" and s["sign_agreement"] < diagnose.THRESHOLDS["agree"]):
        return "ensemble"
    if s["gbm"] <= 0.8:
        return "capacity"
    if s["ratio"] >= 1.5 or (level != "L1" and s["shap_total"] < s["err_slice"]):
        return "audit"
    return "none"


def candidates(data, s, aux):
    """Test error of every candidate that can be tried on this case."""
    P_t, P_d = aux["P_t"], aux["P_d"]
    err = {"none": float(np.mean([test_error(data, p) for p in P_t])),
           "ensemble": test_error(data, P_t.mean(0))}
    trained = lambda d=data, **kw: float(np.mean([test_error(d, fit(d, 100 + i, **kw)["test"]) for i in range(N_CAND)]))
    err["capacity"] = trained(units=BIG)
    err["regularize"] = trained(l2=1e-3, drop=0.2)
    keep_mean = data["X_train"].reshape(len(data["X_train"]), -1)[:, s["top_feature"]].mean()

    def neutral(X):
        Y = flat(X).copy()
        Y[:, s["top_feature"]] = keep_mean
        return Y.reshape(X.shape)
    err["drop_top"] = trained(dict(data, **{k: neutral(data[k]) for k in ("X_train", "X_val", "X_disc", "X_test")}))
    if data["task"] == "regression":
        err["robust"] = trained(huber=True)
    else:
        thr = [best_threshold(data["y_disc"], p) for p in P_d]
        err["threshold"] = float(np.mean([test_error(data, p, t) for p, t in zip(P_t, thr)]))
        w1 = float((data["y_train"] <= 0.5).mean() / max((data["y_train"] > 0.5).mean(), 1e-9))
        err["class_weight"] = trained(class_weight={0: 1.0, 1: w1})
    if not data["real"]:
        err["audit"] = trained(dict(data, **data["audit"])) if data["audit"] else err["none"]
        Xn, yn = data["pool"](aux["rule"], N_EXTRA, 7)
        err["more_data"] = trained(dict(data, X_train=np.concatenate([data["X_train"], Xn]),
                                        y_train=np.concatenate([data["y_train"], yn])))
    return err


def score(err, rec):
    """Regret and hit of a recommendation; None when it cannot be evaluated."""
    gains = {k: (err["none"] - v) / err["none"] for k, v in err.items()}
    best = max(gains.values())
    if rec not in gains:
        return None, None
    g = gains[rec]
    hit = g >= -0.01 if best < 0.01 else g >= 0.9 * best
    return float(best - g), bool(hit)


def slicing(data, s, aux):
    """Worst error-tree leaf against the worst k-means cluster of SHAP vectors."""
    S0, S_te = aux["S0"], aux["S_te"]
    km = KMeans(n_clusters=8, n_init=5, random_state=0).fit(S0)
    err = aux["err"][aux["sub"]]
    worst = max(range(8), key=lambda c: err[km.labels_ == c].mean() if (km.labels_ == c).any() else -1)
    Xt, tsub = data["X_test"], aux["tsub"]
    terr = np.abs(data["y_test"] - aux["P_t"]).mean(0)
    in_tree, in_clu = aux["rule"](Xt), np.zeros(len(Xt), bool)
    in_clu[tsub] = km.predict(S_te) == worst
    lift = lambda m, e: float(e[m].mean() / e.mean()) if m.any() else None
    out = dict(lift_tree=lift(in_tree, terr), lift_cluster=lift(in_clu[tsub], terr[tsub]))
    if "x1_test" in data:
        true = (data["x1_test"] > 0.25) & (data["x1_test"] < 0.75)
        f1 = lambda m, t: float(2 * (m & t).sum() / max(m.sum() + t.sum(), 1))
        out.update(f1_tree=f1(in_tree, true), f1_cluster=f1(in_clu[tsub], true[tsub]))
    return out


def one(key):
    scn, task, rep = key
    data = real_case(scn, rep) if scn in REAL else synth_case(scn, task, rep)
    names = list(data["feature_names"])
    s, aux = signals(data, names)
    err = candidates(data, s, aux)
    res = dict(scenario=scn, task=task, rep=rep, real=data["real"], signals=s, err=err,
               slicing=slicing(data, s, aux))
    for lv in LEVELS:
        rec = recommend(s, lv)
        regret, hit = score(err, rec)
        res[lv] = dict(rec=rec, regret=regret, hit=hit)
    return res


def all_keys():
    keys = [(s, t, r) for s in SYNTH for t in ("regression", "classification")
            if ONLY.get(s, t) == t for r in range(5)]
    return keys + [(s, t, r) for s, t in REAL.items() for r in range(3)]


def boot(a, b, n=2000):
    """Mean of a - b and its 95% paired bootstrap interval."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summarize(out):
    rows = []
    for n in sorted(os.listdir(out)):
        if n.endswith(".json"):
            with open(os.path.join(out, n)) as fh:
                rows.append(json.load(fh))
    lines = ["# Stage 3 results", "", "Design: ../../STAGE3.md. %d cases." % len(rows), ""]

    def table(sub, title):
        err = pd.DataFrame([r["err"] for r in sub])
        gains = err.apply(lambda c: (err["none"] - c) / err["none"])
        best = gains.max(axis=1)
        reg = {lv: [r[lv]["regret"] for r in sub] for lv in LEVELS}
        lines.extend(["## " + title, "", "| rule | mean regret | hit rate | evaluable |", "|---|---|---|---|"])
        for lv in LEVELS:
            v = [x for x in reg[lv] if x is not None]
            h = [r[lv]["hit"] for r in sub if r[lv]["hit"] is not None]
            lines.append("| %s | %.3f | %.0f%% | %d/%d |" % (lv, np.mean(v), 100 * np.mean(h), len(v), len(sub)))
        fixed = {c: best - gains[c] for c in gains}
        order = sorted((c for c in fixed if fixed[c].notna().all()), key=lambda c: fixed[c].mean())
        for c in order:
            lines.append("| always %s | %.3f | | %d/%d |" % (c, fixed[c].mean(), len(sub), len(sub)))
        rnd = (best.values[:, None] - gains.values)
        lines.extend(["| random candidate | %.3f | | %d/%d |" % (np.nanmean(rnd), len(sub), len(sub)), ""])
        return fixed[order[0]].values, order[0] if order else None

    syn = [r for r in rows if not r["real"]]
    if syn:
        best_fixed, name = table(syn, "Synthetic cases")
        full = [i for i, r in enumerate(syn) if all(r[lv]["regret"] is not None for lv in LEVELS)]
        reg = {lv: np.array([syn[i][lv]["regret"] for i in full]) for lv in LEVELS}
        lines.extend(["Paired differences in regret (95% bootstrap interval; negative = first is better):", ""])
        lines.append("- L1 vs always %s: %.3f [%.3f, %.3f]" % ((name,) + boot(reg["L1"], best_fixed[full])))
        lines.append("- L2 vs L1: %.3f [%.3f, %.3f]" % boot(reg["L2"], reg["L1"]))
        lines.append("- L3 vs L2: %.3f [%.3f, %.3f]" % boot(reg["L3"], reg["L2"]))
        lines.append("")
        df = pd.DataFrame([dict(scenario=r["scenario"], task=r["task"][:3], best=max(r["err"], key=lambda k: -r["err"][k]),
                                best_gain=1 - min(r["err"].values()) / r["err"]["none"],
                                **{lv: r[lv]["rec"] for lv in LEVELS},
                                **{lv + "_regret": r[lv]["regret"] for lv in LEVELS}) for r in syn])
        lines.extend(["## By scenario", "",
                      "| scenario | task | best fix (cases) | best gain | L1 recs | L2 recs | L3 recs | regret L1/L2/L3 |",
                      "|---|---|---|---|---|---|---|---|"])
        cnt = lambda s: ", ".join("%s×%d" % kv for kv in s.value_counts().items())
        for (sc, t), d in df.groupby(["scenario", "task"], sort=False):
            lines.append("| %s | %s | %s | %.0f%% | %s | %s | %s | %.2f / %.2f / %.2f |" % (
                sc, t, cnt(d.best), 100 * d.best_gain.mean(), cnt(d.L1), cnt(d.L2), cnt(d.L3),
                d.L1_regret.mean(), d.L2_regret.mean(), d.L3_regret.mean()))
        lines.append("")
        sp = [r for r in syn if r["scenario"] == "spurious"]
        if sp:
            c = lambda k: sum(r["signals"][k] == 6 for r in sp)
            lines.extend(["## Culprit feature (spurious, x7 ranked first)", "",
                          "Loss Shapley %d/%d, error-tree root %d/%d, |SHAP| %d/%d." % (
                              c("top_feature"), len(sp), c("root"), len(sp), c("shap_top"), len(sp)), ""])
    real = [r for r in rows if r["real"]]
    for task in ("classification", "regression"):
        sub = [r for r in real if r["task"] == task]
        if sub:
            table(sub, "Real data: %s (%s)" % (sub[0]["scenario"], task))
            lines.append("Recommendations: " + ", ".join("%s %s" % (lv, sorted({r[lv]["rec"] for r in sub})) for lv in LEVELS))
            lines.extend(["", "Mean gain per candidate: " + ", ".join("%s %.1f%%" % (
                k, 100 * np.mean([1 - r["err"][k] / r["err"]["none"] for r in sub])) for k in sub[0]["err"]), ""])
    sl = pd.DataFrame([r["slicing"] for r in rows])
    if len(sl):
        lines.extend(["## Slicing", "", "Mean error lift on test: error tree %.2f, SHAP clusters %.2f." % (
            sl.lift_tree.mean(), sl.lift_cluster.mean())])
        if "f1_tree" in sl:
            ids = [i for i, r in enumerate(rows) if r["scenario"] in ("sparse", "noise")]
            lines.append("F1 against the true region (sparse, noise): error tree %.2f, SHAP clusters %.2f." % (
                sl.f1_tree[ids].mean(), sl.f1_cluster[ids].mean()))
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(run.ROOT, "results", "stage3"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", nargs="*", help="scenario names")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reslice", action="store_true", help="redo only the slicing measure of finished cases")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.summarize:
        summarize(args.out)
        return
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    i, n = map(int, args.shard.split("/"))
    keys = [k for k in all_keys() if (not args.only or k[0] in args.only)
            and (args.reps is None or k[2] < args.reps) and not (args.reslice and k[0] in REAL)]
    for key in keys[i::n]:
        path = os.path.join(args.out, "%s_%s_%d.json" % key)
        if args.reslice:
            with open(path) as fh:
                res = json.load(fh)
            if not res.get("resliced"):
                data = synth_case(*key)
                s, aux = signals(data, list(data["feature_names"]))
                res.update(slicing=slicing(data, s, aux), resliced=True, signals_again=s)
                with open(path, "w") as fh:
                    json.dump(res, fh, indent=1, default=float)
                run.log("%s %s #%d: resliced" % key)
            continue
        if os.path.exists(path):
            continue
        t0 = time.time()
        res = one(key)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        run.log("%s %s #%d: %s (%.0fs)" % (key + ({lv: res[lv]["rec"] for lv in LEVELS}, time.time() - t0)))


if __name__ == "__main__":
    main()
