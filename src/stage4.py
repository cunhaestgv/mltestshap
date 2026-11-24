"""Stage 4: group-level fault detectors and a culprit screen (see STAGE4.md).

    python3 -I src/stage4.py shap --shard 0/8     # SHAP of every model, one per worker
    python3 -I src/stage4.py detect               # Part A
    python3 -I src/stage4.py culprit --shard 0/8  # Part B, one per worker
    python3 -I src/stage4.py summarize            # results/stage4/summary.md

Results go to results/stage4/. Finished models and cases are skipped, so an
interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.get_logger().setLevel("ERROR")

from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

import datasets  # noqa: E402
import method  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402
import stage3  # noqa: E402

OUT = os.path.join(run.ROOT, "results", "stage4")
DATASETS = ("adult", "bike", "diet")
VIEWS = (("adult", False), ("bike", False), ("diet", False), ("bike", True))
DELTAS, BS_SEEDS = (0.05, 0.1, 0.2), (0, 1)
THETA, WIDE, KEEP = 0.1, 0.1, 0.1
METHODS = ("D1", "D2", "D2-all", "ours", "bva", "random", "accuracy")


def setup(ds):
    res = os.path.join(run.ROOT, "results", ds)
    cfg = pd.read_json(os.path.join(res, "config.json"), typ="series")
    data = datasets.LOADERS[ds]()
    gen_idx, bg_idx = run.sample_sets(data, int(cfg.n_gen), int(cfg.n_bg))
    with open(os.path.join(res, "variable_importance.json")) as fh:
        top_vars = [list(data["var_names"]).index(v) for v in json.load(fh)]
    return res, cfg, data, data["X_train"][gen_idx], data["X_train"][bg_idx], top_vars


def bs_target(res, data):
    """The narrow feature of the reference model with most signal, and its variable."""
    S0 = np.load(os.path.join(res, "shap_original.npy"))
    p = pd.read_csv(os.path.join(res, "partitions_theta%g.csv" % THETA))
    p = p[(p.type == "narrow") & (p.n_values > 10)]
    names = list(data["feature_names"])
    f = max((names.index(n) for n in p.feature), key=lambda i: np.abs(S0[:, i]).mean())
    return f, f % data["n_vars"]


def shifted(data, v, delta):
    def shift(X):
        X = X.copy()
        X[..., v] = np.clip(X[..., v] + delta, 0, 1)
        return X
    return shift(data["X_train"]), shift(data["X_val"])


def plan_of(ds, cfg, top_vars):
    args = SimpleNamespace(healthy=int(cfg.healthy), seeds=int(cfg.seeds), seeds_data=int(cfg.seeds_data), dataset=ds)
    plan = [p for p in run.model_plan(args, top_vars) if p["group"] != "original"]
    plan += [dict(name="BS_%g_%d" % (d, s), group="bs", op="BS", level=d, seed=s) for d in DELTAS for s in BS_SEEDS]
    return args, plan


def bs_weights(data, ds, item, v, wdir):
    Xtr, Xva = shifted(data, v, item["level"])
    return run.cached_training(os.path.join(wdir, item["name"] + ".npz"),
                               lambda: models.train(dict(data, X_val=Xva), ds, 100 + item["seed"], Xtr))


def compute_shap(shard):
    i, n = map(int, shard.split("/"))
    for ds in DATASETS:
        res, cfg, data, Xg, bg, top_vars = setup(ds)
        args, plan = plan_of(ds, cfg, top_vars)
        f, v = bs_target(res, data)
        sdir, wdir = os.path.join(OUT, ds, "shap"), os.path.join(OUT, ds, "weights")
        os.makedirs(sdir, exist_ok=True), os.makedirs(wdir, exist_ok=True)
        with np.load(os.path.join(res, "weights", "original.npz")) as z:
            original = [z["arr_%d" % k] for k in range(len(z.files))]
        for item in plan[i::n]:
            path = os.path.join(sdir, item["name"] + ".npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            if item["group"] == "bs":
                w = bs_weights(data, ds, item, v, wdir)
            else:
                w = run.get_weights(data, args, item, original, os.path.join(res, "weights"))
            model = models.load(data, ds, w)
            S = models.shap_values(model, bg, Xg).astype("float32")
            np.savez_compressed(path, S=S, metrics=json.dumps(models.metrics(data, model)))
            run.log("%s %s (%.0fs)" % (ds, item["name"], time.time() - t0))


def regions_of(X, S0):
    """Regions of the reference model: (feature, mask, sign)."""
    signs = np.sign(S0)
    min_leaf = max(5, int(0.01 * len(X)))
    out = []
    for f in range(X.shape[1]):
        s, ok = signs[:, f], signs[:, f] != 0
        if ok.sum() < 6:
            continue
        v = X[:, f]
        a = method.analyse_feature(v[ok], s[ok], THETA, WIDE)
        if a["type"] == "single":
            out.append((f, ok.copy(), a["sign"]))
        elif a["type"] == "narrow":
            out.append((f, ok & (v <= a["lo"]), a["sign_lo"]))
            out.append((f, ok & (v >= a["hi"]), a["sign_hi"]))
        else:
            tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=min_leaf,
                                          random_state=0).fit(X[ok], s[ok] > 0)
            leaf = np.full(len(X), -1)
            leaf[ok] = tree.apply(X[ok])
            for lf in np.unique(leaf[ok]):
                m = leaf == lf
                npos, nneg = (s[m] > 0).sum(), (s[m] < 0).sum()
                if float(method.entropy(npos, nneg)) < THETA:
                    out.append((f, m, 1 if npos >= nneg else -1))
    return out


def violations(regions, S):
    return np.array([np.mean(np.sign(S[m, f]) != sg) for f, m, sg in regions])


def narrow_of(X, S0):
    out = {}
    for f in range(X.shape[1]):
        s = np.sign(S0[:, f])
        ok = s != 0
        if ok.sum() >= 6:
            a = method.analyse_feature(X[ok, f], s[ok], THETA, WIDE)
            if a["type"] == "narrow":
                out[f] = a
    return out


def drift(X, S, ref):
    """d_f of every narrow feature of the reference model."""
    d = []
    for f, a0 in ref.items():
        s = np.sign(S[:, f])
        ok = s != 0
        a = method.analyse_feature(X[ok, f], s[ok], THETA, WIDE) if ok.sum() >= 2 else dict(type="none")
        if a["type"] != "narrow" or a["sign_lo"] != a0["sign_lo"] or a["sign_hi"] != a0["sign_hi"]:
            d.append(1.0)
        else:
            span = max(a0["max"] - a0["min"], 1e-12)
            d.append(abs((a["lo"] + a["hi"]) / 2 - (a0["lo"] + a0["hi"]) / 2) / span)
    return np.array(d)


def detect():
    """Statistics of every model under every method and view."""
    rows, loc = [], []
    for ds, grouped in VIEWS:
        view = ds + ("_grouped" if grouped else "")
        res, cfg, data, Xg, bg, top_vars = setup(ds)
        _, plan = plan_of(ds, cfg, top_vars)
        groups, names = run.analysis_view(data, grouped)
        S0_raw = np.load(os.path.join(res, "shap_original.npy"))
        Xv, S0 = run.to_view(groups, Xg, S0_raw)
        ours = method.generate(Xv, S0, names, theta=THETA, wide_frac=WIDE, k=3)
        n = len(ours["rows"])
        suites = dict(ours=[ours], bva=[method.bva_suite(Xv, S0, n)],
                      random=[method.random_suite(S0, n, np.random.default_rng(1000 + r)) for r in range(5)])
        ref, regs = narrow_of(Xv, S0), regions_of(Xv, S0)

        items, Ss = [], {}
        for it in plan:
            with np.load(os.path.join(OUT, ds, "shap", it["name"] + ".npz"), allow_pickle=False) as z:
                Ss[it["name"]] = run.to_view(groups, S=z["S"].astype("float64"))[1]
                it = dict(it, metrics=json.loads(str(z["metrics"])))
            items.append(it)
        H = [it["name"] for it in items if it["group"] == "healthy"]
        V = {k: violations(regs, Ss[k]) for k in Ss}
        VH = np.array([V[k] for k in H])
        keep_all = VH.mean(0) <= KEEP
        dH = np.array([drift(Xv, Ss[k], ref) for k in H]) if ref else None
        metric, sign = ("rmse", 1) if "rmse" in items[0]["metrics"] else ("auc", -1)
        E = np.sign(S0)
        PH = {(su, r): np.array([np.sign(Ss[h][s["rows"], s["feats"]]) != s["expected"] for h in H])
              for su, lst in suites.items() for r, s in enumerate(lst)}
        PA = np.array([np.sign(Ss[h]) != E for h in H]).astype("float32")

        for it in items:
            k = it["name"]
            stat = {}
            for su, lst in suites.items():
                for r, s in enumerate(lst):
                    stat[su if len(lst) == 1 else "%s#%d" % (su, r)] = float(
                        np.mean(np.sign(Ss[k][s["rows"], s["feats"]]) != s["expected"]))
            # post hoc controls: the healthy filter of D2 on single tests and on all (input, feature) pairs
            others = [h for h in H if h != k]
            for su, lst in suites.items():
                for r, s in enumerate(lst):
                    kp = PH[su, r][[H.index(h) for h in others]].mean(0) <= KEEP
                    fail = np.sign(Ss[k][s["rows"], s["feats"]]) != s["expected"]
                    stat["F-%s" % su if len(lst) == 1 else "F-%s#%d" % (su, r)] = float(fail[kp].mean()) if kp.any() else None
            kp = PA[[H.index(h) for h in others]].mean(0) <= KEEP
            stat["F-pairs"] = float((np.sign(Ss[k]) != E)[kp].mean())
            keep = keep_all
            if k in H:
                others = [j for j, h in enumerate(H) if h != k]
                keep = VH[others].mean(0) <= KEEP
            stat["D2"] = float(V[k][keep].mean()) if keep.any() else None
            stat["D2-all"] = float(V[k].mean())
            stat["D1"] = float(drift(Xv, Ss[k], ref).mean()) if ref else None
            stat["accuracy"] = sign * it["metrics"][metric]
            for m, v in stat.items():
                rows.append(dict(view=view, name=k, group=it["group"], op=it.get("op", "healthy"), method=m, stat=v,
                                 n_regions=len(regs), kept=int(keep_all.sum()), n_narrow=len(ref)))
            if it["group"] == "bs" and ref and not grouped:
                f, v = bs_target(res, data)
                ex = drift(Xv, Ss[k], ref) - dH.max(0)
                top = list(ref)[int(np.argmax(ex))]
                a0, sg = method.analyse_feature, np.sign(Ss[k][:, f])
                c0, c1 = ((lambda a: None if a["type"] != "narrow" else (a["lo"] + a["hi"]) / 2)(
                    a0(Xv[q != 0, f], q[q != 0], THETA, WIDE)) for q in (np.sign(S0[:, f]), sg))
                loc.append(dict(view=view, name=k, delta=it["level"], hit=bool(top % data["n_vars"] == v),
                                feature=names[top], target=names[f], excess=float(ex.max()),
                                moved=None if c1 is None else c1 - c0,
                                healthy_max=float(dH[:, list(ref).index(f)].max())))
        run.log("%s: %d models, %d regions, %d kept, %d narrow" % (view, len(items), len(regs), keep_all.sum(), len(ref)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "detect.csv"), index=False)
    pd.DataFrame(loc).to_csv(os.path.join(OUT, "localize.csv"), index=False)


def neutral(data, f):
    mean = flat_col(data["X_train"], f).mean()

    def fn(X):
        Y = stage3.flat(X).copy()
        Y[:, f] = mean
        return Y.reshape(X.shape)
    return dict(data, **{k: fn(data[k]) for k in ("X_train", "X_val", "X_disc", "X_test")})


def flat_col(X, f):
    return X.reshape(len(X), -1)[:, f]


def try_drop(data, f, base_d):
    """Retrain without feature f: gain on discovery, and test error."""
    d = neutral(data, f)
    outs = [stage3.fit(d, 100 + i) for i in range(stage3.N_CAND)]
    disc = float(np.mean([np.abs(data["y_disc"] - o["disc"]).mean() for o in outs]))
    return dict(disc_gain=(base_d - disc) / base_d,
                test=float(np.mean([stage3.test_error(data, o["test"]) for o in outs])))


def culprit_keys():
    keys = [(s, t, r, 1.0) for s, t, r in stage3.all_keys()]
    for mult, first in ((2.0, 5), (4.0, 10)):
        keys += [("spurious", t, first + r, mult) for t in ("regression", "classification") for r in range(5)]
    return keys


def culprit_case(key):
    scn, task, rep, mult = key
    stage3.COPY_NOISE = mult
    real = scn in stage3.REAL
    data = stage3.real_case(scn, rep) if real else stage3.synth_case(scn, task, rep)
    s, aux = stage3.signals(data, list(data["feature_names"]))
    base = float(np.mean([stage3.test_error(data, p) for p in aux["P_t"]]))
    base_d = float(np.abs(data["y_disc"] - aux["P_d"]).mean())
    rule = s["top_feature"] if stage3.recommend(s, "L2") == "drop_top" else None
    feats = {f for f in (s["root"], s["shap_top"], rule) if f is not None and f >= 0}
    tried = {int(f): try_drop(data, f, base_d) for f in feats}
    return dict(scenario=scn, task=task, rep=rep, mult=mult, real=real, base=base, root=s["root"],
                shap_top=s["shap_top"], rule=rule, tried=tried)


def culprit(shard):
    i, n = map(int, shard.split("/"))
    out = os.path.join(OUT, "culprit")
    os.makedirs(out, exist_ok=True)
    for key in culprit_keys()[i::n]:
        path = os.path.join(out, "%s_%s_%d_%g.json" % key)
        if os.path.exists(path):
            continue
        t0 = time.time()
        res = culprit_case(key)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        run.log("%s %s #%d x%g: root %s shap %s (%.0fs)" % (key + (res["root"], res["shap_top"], time.time() - t0)))


def policy(r, name):
    """Feature dropped by a policy in a case, or None."""
    t = r["tried"]
    ok = lambda f: f is not None and f >= 0 and t[str(f)]["disc_gain"] >= 0.05
    if name == "screen":
        return r["root"] if r["root"] == r["shap_top"] and ok(r["root"]) else None
    if name == "root-only":
        return r["root"] if ok(r["root"]) else None
    if name == "shap-only":
        return r["shap_top"] if ok(r["shap_top"]) else None
    if name == "no-confirm":
        return r["root"] if r["root"] == r["shap_top"] and r["root"] >= 0 else None
    return r["rule"]


def mcnemar(a, b):
    """Exact two-sided p of paired detections, with the discordant counts."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return x, y, (1.0 if x + y == 0 else binomtest(x, x + y, 0.5).pvalue)


def summarize():
    df = pd.read_csv(os.path.join(OUT, "detect.csv"))
    lines = ["# Stage 4 results", "", "Design: ../../STAGE4.md.", "", "## Part A: detection", ""]
    hb = df[df.group == "healthy"].groupby(["view", "method"]).stat.max().rename("bar")
    mut = df[df.group != "healthy"].join(hb, on=["view", "method"])
    mut["det"] = mut.stat > mut.bar
    mut.loc[mut.stat.isna(), "det"] = np.nan
    mut["set"] = mut.group.map(dict(model="weight", data="data", bs="BS"))
    mut["det"] = mut.det.astype(float)
    mut["method"] = mut.method.str.split("#").str[0]                # random and F-random: repetitions averaged
    mut = mut.groupby(["view", "name", "group", "op", "set", "method"], as_index=False).det.mean()
    g = lambda d: d[d.set != "BS"]
    lines.extend(["| view | set | n | " + " | ".join(METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)])
    for view in [v + ("_grouped" if gr else "") for v, gr in VIEWS]:
        v = mut[mut.view == view]
        for lab, d in (("general", g(v)), ("weight", v[v.set == "weight"]), ("data", v[v.set == "data"]), ("BS", v[v.set == "BS"])):
            c = d.groupby("method").det.mean()
            lines.append("| %s | %s | %d | %s |" % (view, lab, d.name.nunique(), " | ".join(
                "–" if pd.isna(c.get(m)) else "%.0f%%" % (100 * c[m]) for m in METHODS)))
    lines.append("")
    pool = mut[mut.view.isin(DATASETS)]
    wide = pool.set_index(["view", "name", "method"]).det.unstack("method")
    ctrl = wide
    sets = pool.drop_duplicates(["view", "name"]).set_index(["view", "name"]).set
    lines.extend(["Pooled over Adult, Bike and Diet; α = 0.05/8. a/b = mutants only the detector / only the comparator found.", "",
                  "| set | test | a/b | p | detector | comparator |", "|---|---|---|---|---|---|"])
    for lab, idx in (("general", sets != "BS"), ("BS", sets == "BS")):
        w = wide[idx.reindex(wide.index)]
        for det in ("D1", "D2"):
            for cmp in ("ours", "bva"):
                u = w[[det, cmp]].dropna() > 0
                x, y, p = mcnemar(u[det], u[cmp])
                lines.append("| %s | %s vs %s | %d/%d | %.4f | %.0f%% | %.0f%% |" % (
                    lab, det, cmp, x, y, p, 100 * u[det].mean(), 100 * u[cmp].mean()))
    lines.append("")
    silent = wide[wide["accuracy"] == 0]
    lines.extend(["Mutants accuracy misses (pooled, n=%d): " % len(silent) + ", ".join(
        "%s %.0f%%" % (m, 100 * silent[m].astype(float).mean()) for m in METHODS if m != "accuracy"), ""])
    lines.extend(["AUROC, mutants vs healthy (general / data + BS):", "",
                  "| view | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)])
    for view in df.view.unique():
        cells = []
        for m in METHODS:
            d = df[(df.view == view) & (df.method.str.split("#").str[0] == m)].dropna(subset=["stat"])
            d = d.groupby(["name", "group"], as_index=False).stat.mean()
            if d.empty:
                cells.append("–")
                continue
            auc = lambda q: roc_auc_score(q.group != "healthy", q.stat)
            cells.append("%.2f / %.2f" % (auc(d[d.group != "bs"]), auc(d[d.group != "model"])))
        lines.append("| %s | %s |" % (view, " | ".join(cells)))
    info = df.drop_duplicates("view")[["view", "n_regions", "kept", "n_narrow"]]
    lines.extend(["", "Regions, kept regions, narrow features: " + "; ".join(
        "%s %d, %d, %d" % tuple(r) for r in info.values), ""])
    loc = pd.read_csv(os.path.join(OUT, "localize.csv"))
    if len(loc):
        loc["hit"] = loc.hit & (loc.excess > 0)       # with no excess, nothing was localized
        lines.extend(["Localization of BS mutants (hit = the top drifting feature belongs to the shifted variable): " + ", ".join(
            "%s %d/%d" % (v, d.hit.sum(), len(d)) for v, d in loc.groupby("view")),
            "By δ: " + ", ".join("%g %d/%d" % (v, d.hit.sum(), len(d)) for v, d in loc.groupby("delta")), ""])

        lines.extend(["Did BS mutants move the boundary? Midpoint shift of the target feature, scaled units:", "",
                      "| view | target | δ | moved (seeds 0, 1) | healthy max d_f |", "|---|---|---|---|---|"])
        for (v, dl), d in loc.groupby(["view", "delta"]):
            lines.append("| %s | %s | %g | %s | %.3f |" % (v, d.target.iloc[0], dl, ", ".join(
                "not narrow" if pd.isna(x) else "%+.3f" % x for x in d.moved), d.healthy_max.iloc[0]))
        lines.append("")
    lines.extend(["## Post hoc controls (not pre-registered)", "",
                  "The healthy filter of D2 applied without regions. Cross-fitted like D2. Pooled over Adult, Bike and Diet.", "",
                  "| set | D2 | D2-all | ours | F-ours | bva | F-bva | random | F-random | F-pairs |", "|---|---|---|---|---|---|---|---|---|---|"])
    for lab, idx in (("general", sets != "BS"), ("BS", sets == "BS")):
        c = ctrl[idx.reindex(ctrl.index)].mean()
        lines.append("| %s | %s |" % (lab, " | ".join("%.0f%%" % (100 * c[m]) for m in (
            "D2", "D2-all", "ours", "F-ours", "bva", "F-bva", "random", "F-random", "F-pairs"))))
    u = ctrl[(sets != "BS").reindex(ctrl.index)]
    x, y, q = mcnemar(u["D2"] > 0, u["F-pairs"] > 0)
    lines.extend(["", "D2 vs F-pairs, general: %d/%d, p = %.4f." % (x, y, q), ""])

    cdir = os.path.join(OUT, "culprit")
    if os.path.isdir(cdir):
        R = []
        for n in sorted(os.listdir(cdir)):
            with open(os.path.join(cdir, n)) as fh:
                R.append(json.load(fh))
        lines.extend(["## Part B: culprit screen", "", "%d cases." % len(R), "",
                      "| policy | recall (spurious) | regression | classification | harm (others) | gain spurious | gain others |",
                      "|---|---|---|---|---|---|---|"])
        spur = [r for r in R if r["scenario"] == "spurious"]
        rest = [r for r in R if r["scenario"] != "spurious" and not r["real"]]

        def gain(r, f):
            return 0.0 if f is None else (r["base"] - r["tried"][str(f)]["test"]) / r["base"]
        for p in ("screen", "root-only", "shap-only", "no-confirm", "stage3-rule"):
            hit = [policy(r, p) == 6 for r in spur]
            t = np.array([r["task"] == "regression" for r in spur])
            drops = [(r, policy(r, p)) for r in rest]
            harm = sum(gain(r, f) < -0.01 for r, f in drops if f is not None)
            lines.append("| %s | %d/%d | %d/%d | %d/%d | %d/%d | %.1f%% | %.1f%% |" % (
                p, sum(hit), len(hit), np.sum(np.array(hit)[t]), t.sum(), np.sum(np.array(hit)[~t]), (~t).sum(),
                harm, len(rest), 100 * np.mean([gain(r, policy(r, p)) for r in spur]),
                100 * np.mean([gain(r, f) for r, f in drops])))
        lines.append("")
        lines.append("Screen recall by copy noise: " + ", ".join("×%g %d/%d" % (
            m, sum(policy(r, "screen") == 6 for r in spur if r["mult"] == m), sum(r["mult"] == m for r in spur))
            for m in sorted({r["mult"] for r in spur})))
        lines.append("")
        for r in [q for q in R if q["real"]]:
            lines.append("Real: %s #%d root %s, shap %s, " % (r["scenario"], r["rep"], r["root"], r["shap_top"]) + ", ".join(
                "drop %s: discovery gain %.1f%%, test gain %.1f%%" % (
                    f, 100 * v["disc_gain"], 100 * (r["base"] - v["test"]) / r["base"]) for f, v in r["tried"].items()))
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("shap", "detect", "culprit", "summarize"))
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    dict(shap=lambda: compute_shap(args.shard), detect=detect,
         culprit=lambda: culprit(args.shard), summarize=summarize)[args.step]()


if __name__ == "__main__":
    main()
