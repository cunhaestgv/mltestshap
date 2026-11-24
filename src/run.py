"""Fault-detection experiment for one dataset.

    python3 -I src/run.py --dataset bike

Steps:
 1. train the original model (seed 0) and compute its SHAP values on a
    generation set of training samples;
 2. generate test suites: ours (one per entropy threshold theta), random and
    classic boundary-value analysis (BVA) of the same size;
 3. build models to test: healthy retrains with other seeds, model-level
    mutants and data-level mutants;
 4. run every suite on every model with two oracles: the SHAP sign (as defined
    in the paper) and a black-box output effect that only calls predict().

With --group-lags, the lagged copies of a variable (var_t-2, var_t-1, var_t)
are analysed as one feature: its SHAP value is the sum over the time steps,
its value is the one at time t, and the black-box oracle replaces all of its
time steps at once. Results then go to results/<dataset>_grouped/.

Results go to results/<dataset>/. Each evaluated model is saved as soon as it
is done, so an interrupted run resumes where it stopped.
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

import datasets  # noqa: E402
import method  # noqa: E402
import models  # noqa: E402
import mutants  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_weights(path, weights):
    np.savez(path, *weights)


def load_weights(path):
    with np.load(path) as z:
        return [z["arr_%d" % i] for i in range(len(z.files))]


def cached_training(path, fn):
    if os.path.exists(path):
        return load_weights(path)
    w = fn()
    save_weights(path, w)
    return w


def sample_sets(data, n_gen, n_bg):
    """Indices into X_train of the generation set and the SHAP background."""
    rng = np.random.default_rng(0)
    n_tr = len(data["X_train"])
    gen_idx = np.sort(rng.choice(n_tr, min(n_gen, n_tr), replace=False))
    bg_idx = rng.choice(n_tr, min(n_bg, n_tr), replace=False)
    return gen_idx, bg_idx


def analysis_view(data, grouped):
    """Features the tests are about, as a boolean matrix (features x flattened
    columns) and their names. Ungrouped: one feature per flattened column."""
    n_flat = len(data["feature_names"])
    if not grouped or data["kind"] != "seq":
        return np.eye(n_flat, dtype=bool), list(data["feature_names"])
    groups = np.zeros((data["n_vars"], n_flat), dtype=bool)
    for v in range(data["n_vars"]):
        groups[v, v::data["n_vars"]] = True
    return groups, list(data["var_names"])


def to_view(groups, X=None, S=None):
    """Feature values (value of the last column of each group, i.e. time t)
    and SHAP values (sum over the group) in the analysis view."""
    last = [int(np.flatnonzero(g)[-1]) for g in groups]
    Xv = None if X is None else X.reshape(len(X), -1)[:, last]
    Sv = None if S is None else S @ groups.T.astype(S.dtype)
    return Xv, Sv


def make_suites(data, Xf, S0, names, args, out):
    suites = {}
    for theta in args.thetas:
        ours = method.generate(Xf, S0, names, theta=theta, wide_frac=args.wide_frac, k=args.k)
        n = len(ours["rows"])
        suites["ours@%g" % theta] = ours
        suites["bva@%g" % theta] = method.bva_suite(Xf, S0, n)
        for r in range(args.random_reps):
            suites["random@%g#%d" % (theta, r)] = method.random_suite(S0, n, np.random.default_rng(1000 + r))
        write_artifacts(data, Xf, ours, names, theta, out)
        log("theta=%g: %d tests (%s)" % (theta, n, pd.Series(ours["kind"]).value_counts().to_dict()))
    return suites


def write_artifacts(data, Xf, suite, names, theta, out):
    """Human-readable partitions, tree rules and tests (raw feature units)."""
    tag = "theta%g" % theta
    feats = pd.DataFrame(suite["features"])
    for col in ("min", "max", "lo", "hi"):
        feats[col + "_raw"] = [None if pd.isna(v) else datasets.raw_value(data, f, v)
                               for f, v in zip(feats.feature.map(names.index), feats[col])]
    feats.to_csv(os.path.join(out, "partitions_%s.csv" % tag), index=False)
    pd.DataFrame(suite["leaves"]).to_csv(os.path.join(out, "tree_leaves_%s.csv" % tag), index=False)
    with open(os.path.join(out, "trees_%s.txt" % tag), "w") as fh:
        for name, text in suite["trees"].items():
            fh.write("=== SHAP sign of %s ===\n%s\n" % (name, text))
    tests = pd.DataFrame({
        "row": suite["rows"], "feature": [names[f] for f in suite["feats"]],
        "kind": suite["kind"], "partition": suite["part"], "expected_sign": suite["expected"],
        "value_raw": [datasets.raw_value(data, f, Xf[r, f]) for r, f in zip(suite["rows"], suite["feats"])]})
    tests.to_csv(os.path.join(out, "tests_%s.csv" % tag), index=False)


def model_plan(args, top_vars):
    plan = [dict(name="original", group="original")]
    plan += [dict(name="healthy_%d" % s, group="healthy", seed=s) for s in range(1, args.healthy + 1)]
    for op in mutants.MODEL_OPS:
        for level in mutants.LEVELS:
            for s in range(args.seeds):
                plan.append(dict(name="%s_%g_%d" % (op, level, s), group="model", op=op, level=level, seed=s))
    data_ops = [("LN", 0.1), ("LN", 0.3)] + [("FS", v) for v in top_vars[:3]] + [("DS", top_vars[0])]
    for op, param in data_ops:
        for s in range(args.seeds_data):
            plan.append(dict(name="%s_%s_%d" % (op, param, s), group="data", op=op, level=param, seed=s))
    return plan


def get_weights(data, args, item, original, wdir):
    g = item["group"]
    if g == "original":
        return original
    if g == "healthy":
        return cached_training(os.path.join(wdir, item["name"] + ".npz"),
                               lambda: models.train(data, args.dataset, item["seed"]))
    if g == "model":
        rng = np.random.default_rng(1000 * mutants.MODEL_OPS.index(item["op"])
                                    + 100 * mutants.LEVELS.index(item["level"]) + item["seed"])
        return mutants.mutate_weights(original, item["op"], item["level"], rng)

    def fn():
        rng = np.random.default_rng(500 + item["seed"])
        X, y = mutants.corrupt_data(data, item["op"], item["level"], rng)
        return models.train(data, args.dataset, 100 + item["seed"], X, y)
    return cached_training(os.path.join(wdir, item["name"] + ".npz"), fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(datasets.LOADERS))
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--wide-frac", type=float, default=0.1,
                    help="margin wider than this fraction of samples -> decision tree")
    ap.add_argument("--k", type=int, default=3, help="test inputs per boundary/leaf")
    ap.add_argument("--n-gen", type=int, default=3000, help="training samples used to generate tests")
    ap.add_argument("--n-bg", type=int, default=100, help="SHAP background samples")
    ap.add_argument("--n-obg", type=int, default=30, help="background samples of the black-box oracle")
    ap.add_argument("--healthy", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3, help="seeds per model-level operator and level")
    ap.add_argument("--seeds-data", type=int, default=2, help="seeds per data-level operator")
    ap.add_argument("--random-reps", type=int, default=5)
    ap.add_argument("--group-lags", action="store_true",
                    help="analyse the time steps of each variable as one feature")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "results", args.dataset + ("_grouped" if args.group_lags else ""))
    wdir, edir = os.path.join(out, "weights"), os.path.join(out, "evals")
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(edir, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)

    data = datasets.LOADERS[args.dataset]()
    log("%s: train %s, val %s, test %s, %d features" % (
        args.dataset, data["X_train"].shape, data["X_val"].shape, data["X_test"].shape,
        len(data["feature_names"])))

    original = cached_training(os.path.join(wdir, "original.npz"),
                               lambda: models.train(data, args.dataset, 0))
    gen_idx, bg_idx = sample_sets(data, args.n_gen, args.n_bg)
    Xg, bg = data["X_train"][gen_idx], data["X_train"][bg_idx]
    obg = bg[:args.n_obg]

    s0_path = os.path.join(out, "shap_original.npy")
    if os.path.exists(s0_path):
        S0 = np.load(s0_path)
    else:
        log("SHAP of the original model on %d samples" % len(Xg))
        S0 = models.shap_values(models.load(data, args.dataset, original), bg, Xg)
        np.save(s0_path, S0)

    groups, names = analysis_view(data, args.group_lags)
    Xv, S0v = to_view(groups, Xg, S0)
    suites = make_suites(data, Xv, S0v, names, args, out)
    pairs = np.unique(np.concatenate([np.stack([s["rows"], s["feats"]], 1) for s in suites.values()]), axis=0)
    pair_id = {tuple(p): i for i, p in enumerate(pairs)}
    for s in suites.values():
        s["pair_idx"] = np.array([pair_id[(r, f)] for r, f in zip(s["rows"], s["feats"])])
    log("%d distinct (input, feature) pairs for the black-box oracle" % len(pairs))

    n_vars = data["n_vars"]
    imp = np.abs(S0).mean(0).reshape(-1, n_vars).sum(0)
    top_vars = [int(v) for v in np.argsort(-imp)]
    with open(os.path.join(out, "variable_importance.json"), "w") as fh:
        json.dump({data["var_names"][v]: float(imp[v]) for v in top_vars}, fh, indent=1)

    plan = model_plan(args, top_vars)
    main_key = "ours@%g" % args.thetas[len(args.thetas) // 2]
    for i, item in enumerate(plan):
        path = os.path.join(edir, item["name"] + ".json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        weights = get_weights(data, args, item, original, wdir)
        model = models.load(data, args.dataset, weights)
        result = dict(item, metrics=models.metrics(data, model))
        _, S = to_view(groups, S=models.shap_values(model, bg, Xg))
        effects = models.output_effects(model, obg, Xg, pairs, groups)
        fr = {}
        for key, s in suites.items():
            fr[key] = dict(
                shap=float(np.mean(np.sign(S[s["rows"], s["feats"]]) != s["expected"])),
                output=float(np.mean(np.sign(effects[s["pair_idx"]]) != s["expected"])),
                size=int(len(s["rows"])))
        result["failure_rate"] = fr
        if item.get("op") in ("FS", "DS"):
            result["variable"] = data["var_names"][item["level"]]
        with open(path, "w") as fh:
            json.dump(result, fh)
        log("[%d/%d] %s %s %s shap=%.3f output=%.3f (%.0fs)" % (
            i + 1, len(plan), item["name"], result["metrics"], main_key,
            fr[main_key]["shap"], fr[main_key]["output"], time.time() - t0))
    log("done")


if __name__ == "__main__":
    main()
