"""Diagnosis of uncertainty margins.

For every feature, classifies its SHAP-sign margin into one of the rows of the
diagnosis table, using a reference model and models retrained with other seeds:

    1 sharp boundary         5 missing feature or label noise
    2 no real effect         6 too little data
    3 simple interaction     7 unstable training
    4 complex interaction    8 explainer artifact (correlated features)
                             9 non-monotonic effect

    python3 -I src/diagnose.py bike      # uses the saved original + healthy models
    python3 -I src/diagnose.py bike --group-lags   # lagged copies as one variable

The thresholds below were fixed before looking at any diagnosis results. The
raw scores are saved too, so the sensitivity to each threshold can be reported.
Row 9 was added after Stage 1 (hr_t has two sharp boundaries, which Algorithm 1
cannot represent). It uses no new threshold: the effect is non-monotonic when a
tree of depth shallow_depth on the feature alone resolves its SHAP signs over
the whole range (score "self") and its leaves, in feature order, change sign at
least twice ("self_changes"). Two post hoc corrections, both to the definition
and not to a threshold: (1) the sign-change condition, added after the grouped
bike run labelled the binary yr non-monotonic (one change is a single boundary);
(2) both scores are computed over the whole range, not only inside the margin,
after Stage 2 showed that for a bump the margin is the pure middle, which has
no sign change inside it. Stage 2 reports the results before and after (2).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.get_logger().setLevel("ERROR")

import datasets  # noqa: E402
import method  # noqa: E402
import models  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

THRESHOLDS = dict(
    no_effect=0.1,       # mean |SHAP| of the feature / mean over features below this -> no effect
    agree=0.8,           # mean per-sample SHAP-sign agreement across seeds (0.5 = coin flip)
    leaf_majority=0.9,   # a tree leaf is pure if this share of its samples has the same sign
    resolved=0.8,        # share of margin samples in pure leaves for the tree to resolve the margin
    shallow_depth=2,
    deep_depth=6,
    err_ratio=1.5,       # mean |error| in the margin / mean |error| overall above this -> high
    resid_corr=0.7,      # mean pairwise correlation of residuals across seeds -> same failures
    sparse=0.5,          # share of samples / share of value range below this -> sparse region
    corr=0.9,            # |Pearson r| with another feature that also has a wide margin -> artifact
)

LABELS = {1: "sharp boundary", 2: "no real effect", 3: "simple interaction",
          4: "complex interaction", 5: "missing feature or label noise",
          6: "too little data", 7: "unstable training", 8: "explainer artifact",
          9: "non-monotonic effect"}


def _tree_resolves(X, y, depth, t):
    """Best share of samples in pure leaves over trees of depth 1..depth.

    Taking the best depth keeps the score monotonic: a deeper tree can split a
    leaf just above leaf_majority into children just below it.
    """
    if len(np.unique(y)) < 2:
        return 1.0
    best = 0.0
    for d in range(1, depth + 1):
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=max(3, len(y) // 50),
                                      random_state=0).fit(X, y)
        leaf = tree.apply(X)
        pure = 0
        for lf in np.unique(leaf):
            m = leaf == lf
            if max(y[m].mean(), 1 - y[m].mean()) >= t["leaf_majority"]:
                pure += m.sum()
        best = max(best, pure / len(y))
    return best


def _sign_changes(v, y, depth, t):
    """Sign changes along v of the majority signs of a 1-D tree's leaves."""
    if len(np.unique(y)) < 2:
        return 0
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(3, len(y) // 50),
                                  random_state=0).fit(v[:, None], y)
    leaf = tree.apply(v[:, None])
    order = sorted(np.unique(leaf), key=lambda lf: v[leaf == lf].mean())
    maj = [y[leaf == lf].mean() >= 0.5 for lf in order]
    return int(sum(a != b for a, b in zip(maj, maj[1:])))


def _mean_pairwise_corr(R):
    """Mean pairwise Pearson correlation between the rows of R (models x samples)."""
    if R.shape[0] < 2 or R.shape[1] < 3:
        return np.nan
    c = np.corrcoef(R)
    return float(np.nanmean(c[np.triu_indices_from(c, 1)]))


def margin_analysis(X, S, f, theta, wide_frac):
    """Single-feature analysis of feature f and the boolean mask of its margin."""
    v, s = X[:, f], np.sign(S[:, f])
    ok = s != 0
    a = method.analyse_feature(v[ok], s[ok], theta, wide_frac)
    if a["type"] != "wide":
        mask = np.zeros(len(v), bool)
    elif a["lo"] is None:
        mask = ok.copy()                      # no pure partition: the whole range is the margin
    else:
        mask = ok & (v > a["lo"]) & (v < a["hi"])
    return a, mask


def diagnose(X, S_list, resid_list, names, theta=0.1, wide_frac=0.1, t=THRESHOLDS):
    """Diagnose every feature.

    X: analysis inputs, flattened (n, F). S_list: SHAP values (n, F) of each
    model; S_list[0] is the reference model. resid_list: y - prediction of each
    model on X, in the same order.
    """
    S0 = S_list[0]
    imp = np.abs(S0).mean(0)
    signs = np.stack([np.sign(S[:, :]) for S in S_list])          # models x n x F
    resid = np.stack(resid_list)                                   # models x n
    err_all = np.abs(resid).mean(1)
    analyses = [margin_analysis(X, S0, f, theta, wide_frac) for f in range(X.shape[1])]
    wide = np.array([a["type"] == "wide" for a, _ in analyses])
    corr = np.nan_to_num(np.corrcoef(X.T))
    rows = []
    for f, (a, mask) in enumerate(analyses):
        r = dict(feature=names[f], type=a["type"], lo=a.get("lo"), hi=a.get("hi"),
                 shap_rel=float(imp[f] / imp.mean()), share=float(mask.mean()))
        if r["shap_rel"] < t["no_effect"]:
            r["row"] = 2
        elif a["type"] != "wide":
            r["row"] = 1
        else:
            v = X[:, f]
            span = v.max() - v.min()
            range_frac = 1.0 if a["lo"] is None else (a["hi"] - a["lo"]) / span if span > 0 else 1.0
            p = (signs[:, mask, f] > 0).mean(0)
            s_ref = signs[0, mask, f] > 0
            ok_f = signs[0, :, f] != 0
            s_all = signs[0, ok_f, f] > 0
            partners = [names[g] for g in np.flatnonzero((np.abs(corr[f]) > t["corr"]) & wide)
                        if g != f]
            r.update(
                n_margin=int(mask.sum()),
                density=float(r["share"] / max(range_frac, 1e-9)),
                entropy=float(method.entropy(s_ref.sum(), (~s_ref).sum())),
                agreement=float(np.maximum(p, 1 - p).mean()),
                self=_tree_resolves(X[ok_f][:, [f]], s_all, t["shallow_depth"], t),
                self_changes=_sign_changes(X[ok_f, f], s_all, t["shallow_depth"], t),
                shallow=_tree_resolves(X[mask], s_ref, t["shallow_depth"], t),
                deep=_tree_resolves(X[mask], s_ref, t["deep_depth"], t),
                err_ratio=float(np.median(np.abs(resid[:, mask]).mean(1) / err_all)),
                resid_corr=_mean_pairwise_corr(resid[:, mask]),
                partners=";".join(partners))
            if partners:
                r["row"] = 8
            elif r["agreement"] >= t["agree"]:
                if r["self"] >= t["resolved"] and r["self_changes"] >= 2:
                    r["row"] = 9
                elif r["err_ratio"] > t["err_ratio"] and r["resid_corr"] > t["resid_corr"]:
                    r["row"] = 5
                elif r["shallow"] >= t["resolved"]:
                    r["row"] = 3
                else:
                    r["row"] = 4
            elif r["density"] < t["sparse"] or r["err_ratio"] > t["err_ratio"]:
                r["row"] = 6
            else:
                r["row"] = 7
        r["diagnosis"] = LABELS[r["row"]]
        rows.append(r)
    return pd.DataFrame(rows)


def model_outputs(data, name, weights, bg, X, y, cache):
    """SHAP values and residuals of one model on X, cached as .npz."""
    if os.path.exists(cache):
        with np.load(cache) as z:
            return z["S"], z["resid"]
    model = models.load(data, name, weights)
    S = models.shap_values(model, bg, X)
    resid = y - models.predict(model, X)
    np.savez(cache, S=S, resid=resid)
    return S, resid


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.LOADERS))
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--group-lags", action="store_true")
    args = ap.parse_args()
    src = os.path.join(run.ROOT, "results", args.dataset)
    out = src + ("_grouped" if args.group_lags else "")
    os.makedirs(out, exist_ok=True)
    cfg = pd.read_json(os.path.join(src, "config.json"), typ="series")
    ddir = os.path.join(src, "diag")             # cached SHAP values are per flattened column
    os.makedirs(ddir, exist_ok=True)

    data = datasets.LOADERS[args.dataset]()
    gen_idx, bg_idx = run.sample_sets(data, int(cfg.n_gen), int(cfg.n_bg))
    Xg, yg, bg = data["X_train"][gen_idx], data["y_train"][gen_idx], data["X_train"][bg_idx]
    wdir = os.path.join(src, "weights")
    names = ["original"] + sorted((n[:-4] for n in os.listdir(wdir) if n.startswith("healthy_")),
                                  key=lambda n: int(n.split("_")[1]))
    S_list, resid_list = [], []
    for n in names:
        S, res = model_outputs(data, args.dataset, run.load_weights(os.path.join(wdir, n + ".npz")),
                               bg, Xg, yg, os.path.join(ddir, n + ".npz"))
        S_list.append(S)
        resid_list.append(res)
    run.log("%s: %d models" % (args.dataset, len(names)))

    groups, fnames = run.analysis_view(data, args.group_lags)
    Xv, _ = run.to_view(groups, Xg)
    S_list = [run.to_view(groups, S=S)[1] for S in S_list]
    df = diagnose(Xv, S_list, resid_list, fnames, theta=args.theta)
    for col in ("lo", "hi"):
        df[col + "_raw"] = [None if pd.isna(v) else datasets.raw_value(data, f, v)
                            for f, v in enumerate(df[col])]
    df.to_csv(os.path.join(out, "diagnosis.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["row", "diagnosis"]).size().rename("features").to_string())
    cols = ["feature", "diagnosis", "share", "shap_rel", "agreement", "self", "self_changes", "shallow", "deep",
            "err_ratio", "resid_corr", "density", "partners"]
    print(df[df.row > 2][[c for c in cols if c in df]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
