"""Stage 6: confirming detectors based on a retraining reference (see STAGE6.md).

    python3 -I src/stage6.py models       # train/mutate the confirmation mutants and compute their SHAP
    python3 -I src/stage6.py detect       # statistics on the exploration and confirmation sets
    python3 -I src/stage6.py background   # background sweep of the narrow features
    python3 -I src/stage6.py summarize    # results/stage6/summary.md

Results go to results/stage6/. Finished models are skipped.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import method  # noqa: E402
import models  # noqa: E402
import mutants  # noqa: E402
import run  # noqa: E402
import stage4  # noqa: E402
from stage5_localize import majority  # noqa: E402

OUT = os.path.join(run.ROOT, "results", "stage6")
THETA, WIDE, KEEP = stage4.THETA, stage4.WIDE, stage4.KEEP
W_SEEDS, D_SEEDS = (3, 4, 5), (2, 3)
NEW = ("F-pairs-H", "Z-pairs", "M-var")
METHODS = ("F-pairs",) + NEW + ("W-margin", "ours", "bva", "random", "accuracy", "acc+F-pairs")


def confirm_plan(top_vars):
    plan = [dict(name="%s_%g_%d" % (op, lv, s), group="model", op=op, level=lv, seed=s)
            for op in mutants.MODEL_OPS for lv in mutants.LEVELS for s in W_SEEDS]
    data_ops = [("LN", 0.1), ("LN", 0.3)] + [("FS", v) for v in top_vars[:3]] + [("DS", top_vars[0])]
    plan += [dict(name="%s_%s_%d" % (op, p, s), group="data", op=op, level=p, seed=s) for op, p in data_ops for s in D_SEEDS]
    return plan


def build_models():
    for ds in stage4.DATASETS:
        res, cfg, data, Xg, bg, top_vars = stage4.setup(ds)
        args, _ = stage4.plan_of(ds, cfg, top_vars)
        sdir, wdir = os.path.join(OUT, ds, "shap"), os.path.join(OUT, ds, "weights")
        os.makedirs(sdir, exist_ok=True), os.makedirs(wdir, exist_ok=True)
        original = run.load_weights(os.path.join(res, "weights", "original.npz"))
        for item in confirm_plan(top_vars):
            path = os.path.join(sdir, item["name"] + ".npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            model = models.load(data, ds, run.get_weights(data, args, item, original, wdir))
            S = models.shap_values(model, bg, Xg).astype("float32")
            np.savez_compressed(path, S=S, metrics=json.dumps(models.metrics(data, model)))
            run.log("%s %s %s (%.0fs)" % (ds, item["name"], models.metrics(data, model), time.time() - t0))


def margins(X, S):
    w = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        s = np.sign(S[:, f])
        ok = s != 0
        if ok.sum() < 2:
            w[f] = 1.0
            continue
        w[f] = method.analyse_feature(X[ok, f], s[ok], THETA, WIDE)["margin_frac"]
    return w


def load_set(path_dir, names):
    out = {}
    for n in names:
        with np.load(os.path.join(path_dir, n + ".npz")) as z:
            out[n] = (z["S"].reshape(len(z["S"]), -1).astype("float64"), json.loads(str(z["metrics"])))
    return out


def detect():
    rows = []
    for ds in stage4.DATASETS:
        res, cfg, data, Xg, bg, top_vars = stage4.setup(ds)
        _, plan4 = stage4.plan_of(ds, cfg, top_vars)
        X = Xg.reshape(len(Xg), -1)
        nv = data["n_vars"]
        var_of = np.arange(X.shape[1]) % nv
        S0 = np.load(os.path.join(res, "shap_original.npy")).reshape(len(X), -1)
        E = np.sign(S0)
        names = list(data["feature_names"])
        ours = method.generate(X, S0, names, theta=THETA, wide_frac=WIDE, k=3)
        n = len(ours["rows"])
        suites = dict(ours=[ours], bva=[method.bva_suite(X, S0, n)],
                      random=[method.random_suite(S0, n, np.random.default_rng(1000 + r)) for r in range(5)])
        H = [p["name"] for p in plan4 if p["group"] == "healthy"]
        sets = {"healthy": load_set(os.path.join(stage4.OUT, ds, "shap"), H),
                "exploration": load_set(os.path.join(stage4.OUT, ds, "shap"),
                                        [p["name"] for p in plan4 if p["group"] in ("model", "data")]),
                "confirmation": {}}
        cnames = [p["name"] for p in confirm_plan(top_vars)]
        cdir = os.path.join(OUT, ds, "shap")
        if all(os.path.exists(os.path.join(cdir, c + ".npz")) for c in cnames):
            sets["confirmation"] = load_set(cdir, cnames)
        info = {p["name"]: p for p in plan4}
        info.update({("c", p["name"]): p for p in confirm_plan(top_vars)})
        SH = np.stack([sets["healthy"][h][0] for h in H])
        sgH = np.sign(SH)
        absH = np.stack([np.array([np.abs(S[:, var_of == v]).mean(0).sum() for v in range(nv)]) for S in SH])
        WH = np.stack([margins(X, S) for S in SH])
        metric, msign = ("rmse", 1) if "rmse" in sets["healthy"][H[0]][1] else ("auc", -1)
        eps = 0.01 * np.abs(SH).mean()

        def stats(S, met, ref):
            """ref: indices of the healthy models used as reference."""
            st = {}
            sg = np.sign(S)
            for su, lst in suites.items():
                for r, s in enumerate(lst):
                    st[su if len(lst) == 1 else "%s#%d" % (su, r)] = float(np.mean(sg[s["rows"], s["feats"]] != s["expected"]))
            kp = (sgH[ref] != E).mean(0) <= KEEP
            st["F-pairs"] = float((sg != E)[kp].mean())
            maj, h = majority(sgH[ref])
            kp = h <= KEEP
            st["F-pairs-H"] = float((sg != maj)[kp].mean())
            mu, sd = SH[ref].mean(0), SH[ref].std(0)
            st["Z-pairs"] = float(np.minimum(np.abs(S - mu) / (sd + eps), 10).mean())
            a = np.array([np.abs(S[:, var_of == v]).mean(0).sum() for v in range(nv)])
            ab = absH[ref].mean(0)
            rel = np.abs(a - ab) / np.maximum(ab, 1e-12)
            st["M-var"] = float(rel.max())
            st["M-var-top"] = int(np.argmax(rel))
            st["W-margin"] = float(np.abs(margins(X, S) - WH[ref].mean(0)).mean())
            st["accuracy"] = msign * met[metric]
            return st

        recs = []
        allidx = list(range(len(H)))
        for k, h in enumerate(H):
            recs.append(dict(stats(SH[k], sets["healthy"][h][1], [j for j in allidx if j != k]),
                             set="healthy", name=h, group="healthy", op="healthy", var=None))
        for sname in ("exploration", "confirmation"):
            for nm, (S, met) in sets[sname].items():
                p = info[nm] if sname == "exploration" else info[("c", nm)]
                recs.append(dict(stats(S, met, allidx), set=sname, name=nm, group=p["group"], op=p["op"],
                                 var=p["level"] if p["op"] in ("FS", "DS") else None))
        df = pd.DataFrame(recs)
        # acc+F-pairs: z-scores against healthy (cross-fitted for healthy models)
        hd = df[df.set == "healthy"]
        comb = []
        for i, r in df.iterrows():
            ref = hd[hd.name != r["name"]] if r["set"] == "healthy" else hd
            z = [(r[m] - ref[m].mean()) / max(ref[m].std(ddof=0), 1e-12) for m in ("accuracy", "F-pairs")]
            comb.append(max(z))
        df["acc+F-pairs"] = comb
        df.insert(0, "view", ds)
        df["n_vars"] = nv
        rows.append(df)
        run.log("%s: %d tests, %d models" % (ds, n, len(df)))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(OUT, "detect.csv"), index=False)


def background():
    def mid(v, s):
        ok = s != 0
        if ok.sum() < 2:
            return None
        a = method.analyse_feature(v[ok], s[ok], THETA, WIDE)
        return (a["lo"] + a["hi"]) / 2 if a["type"] == "narrow" else None

    rows = []
    for ds in stage4.DATASETS:
        res, cfg, data, Xg, bg, _ = stage4.setup(ds)
        model = models.load(data, ds, run.load_weights(os.path.join(res, "weights", "original.npz")))
        Xf, S0 = Xg.reshape(len(Xg), -1), np.load(os.path.join(res, "shap_original.npy")).reshape(len(Xg), -1)
        Xtr = data["X_train"].reshape(len(data["X_train"]), -1)
        names = list(data["feature_names"])
        p = pd.read_csv(os.path.join(res, "partitions_theta%g.csv" % THETA))
        for nm in p[p.type == "narrow"].feature:
            f = names.index(nm)
            v = Xf[:, f]
            med = np.median(Xtr[:, f])
            rng = np.random.default_rng(0)
            rec = dict(dataset=ds, feature=nm, n_values=int(len(np.unique(v))), boundary=mid(v, np.sign(S0[:, f])),
                       bg_mean=float(bg.reshape(len(bg), -1)[:, f].mean()))
            for tag, m in (("low", Xtr[:, f] <= med), ("high", Xtr[:, f] > med)):
                idx = rng.permutation(np.flatnonzero(m))[:100]
                if len(idx) == 0:
                    rec["bg_" + tag], rec["boundary_" + tag] = None, None
                    continue
                S = models.shap_values(model, data["X_train"][idx], Xg).reshape(len(Xg), -1)
                rec["bg_" + tag] = float(Xtr[idx, f].mean())
                rec["boundary_" + tag] = mid(v, np.sign(S[:, f]))
            bl, bh = rec["boundary_low"], rec["boundary_high"]
            gl, gh = rec["bg_low"], rec["bg_high"]
            if gl is None or gh is None or abs(gh - gl) < 1e-9:
                rec["label"], rec["ratio"] = "no contrast", None
            elif bl is None or bh is None:
                rec["label"], rec["ratio"] = "not stable", None
            else:
                rec["ratio"] = abs(bh - bl) / abs(gh - gl)
                rec["label"] = "reference crossing" if rec["ratio"] >= 0.5 else "learned step"
            rows.append(rec)
            run.log("%s %s %s %s" % (ds, nm, rec["label"], rec["ratio"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "background.csv"), index=False)


def detected(df):
    """Boolean detection for every non-healthy row and method (bar = max over healthy of the same view)."""
    out = []
    for ds, g in df.groupby("view"):
        h = g[g.set == "healthy"]
        m = g[g.set != "healthy"].copy()
        for meth in METHODS:
            if meth == "random":
                reps = [c for c in g.columns if c.startswith("random#")]
                m["det_random"] = np.mean([m[c] > h[c].max() for c in reps], axis=0)
            else:
                m["det_" + meth] = m[meth] > h[meth].max()
        hm = h.copy()
        for meth in METHODS:
            if meth != "random":
                hm["det_" + meth] = [r[meth] > h[h.name != r["name"]][meth].max() for _, r in h.iterrows()]
        out.append(pd.concat([m, hm]))
    return pd.concat(out, ignore_index=True)


def auroc(df, meth):
    vals = []
    for ds, g in df.groupby("view"):
        y = (g.set != "healthy").astype(int)
        if meth == "random":
            reps = [c for c in g.columns if c.startswith("random#")]
            vals.append(np.mean([roc_auc_score(y, g[c]) for c in reps]))
        else:
            vals.append(roc_auc_score(y, g[meth]))
    return np.mean(vals)


def holm(ps):
    order = np.argsort(ps)
    adj, run_max = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run_max = max(run_max, (len(ps) - i) * ps[k])
        adj[k] = min(1.0, run_max)
    return adj


def summarize():
    df = pd.read_csv(os.path.join(OUT, "detect.csv"))
    d = detected(df)
    L = ["# Stage 6 results", "", "Design: `../../STAGE6.md`. Statistics: `detect.csv`.", ""]
    with open(os.path.join(OUT, "choice.json")) as fh:
        choice = json.load(fh)["chosen"]
    for sname in ("exploration", "confirmation"):
        mut = d[d.set == sname]
        L += ["## %s set (%d general mutants)%s" % (sname.capitalize(), len(mut), "" if sname == "confirmation" else ", exploratory"), "",
              "| method | pooled | adult | bike | diet | weight | data | missed by accuracy | AUROC |", "|---|---|---|---|---|---|---|---|---|"]
        miss = ~mut.det_accuracy.astype(bool)
        sub = pd.concat([df[df.set == "healthy"], df[df.set == sname]])
        for meth in METHODS:
            c = mut["det_" + meth]
            L.append("| %s | %.0f%% | %s | %.0f%% | %.0f%% | %.0f%% (%d) | %.2f |" % (
                meth, 100 * c.mean(), " | ".join("%.0f%%" % (100 * c[mut.view == v].mean()) for v in stage4.DATASETS),
                100 * c[mut.group == "model"].mean(), 100 * c[mut.group == "data"].mean(),
                100 * c[miss].mean() if miss.any() else float("nan"), miss.sum(), auroc(sub, meth)))
        fs = df[(df.set == sname) & (df.op == "FS")]
        L += ["", "M-var localization on FS mutants: %d/%d." % ((fs["M-var-top"] == fs["var"]).sum(), len(fs)), ""]
    hm = d[d.set == "healthy"]
    L += ["Healthy models flagged (cross-fitted, 30): " + ", ".join(
        "%s %d" % (m, hm["det_" + m].sum()) for m in METHODS if m != "random"), ""]
    mut = d[d.set == "confirmation"]
    tests = [("F-pairs", "ours"), ("F-pairs", "bva"), (choice, "F-pairs"), ("acc+F-pairs", "accuracy"), ("W-margin", "ours")]
    res = [stage4.mcnemar(mut["det_" + a].astype(bool), mut["det_" + b].astype(bool)) for a, b in tests]
    adj = holm(np.array([r[2] for r in res]))
    L += ["## Pre-registered tests (confirmation set, Holm over 5)", "",
          "| test | A only | B only | p | Holm p | supported |", "|---|---|---|---|---|---|"]
    sup = []
    for (a, b), (x, y, p), q in zip(tests, res, adj):
        ok = bool(q < 0.05 and x > y)
        sup.append(ok)
        L.append("| %s > %s | %d | %d | %.4f | %.4f | %s |" % (a, b, x, y, p, q, "yes" if ok else "no"))
    L += ["", "Claim 1 (F-pairs > ours and bva): %s." % ("supported" if sup[0] and sup[1] else "not supported"),
          "Claim 2 (%s > F-pairs): %s." % (choice, "supported" if sup[2] else "not supported"),
          "Claim 3 (acc+F-pairs > accuracy): %s; healthy flagged %d/30." % (
              "supported" if sup[3] else "not supported", hm["det_acc+F-pairs"].sum()),
          "Claim 4 (W-margin > ours): %s." % ("supported" if sup[4] else "not supported"), ""]
    acc = df[df.set != "healthy"].groupby(["view", "set"]).accuracy.agg(["min", "median", "max"])
    L += ["## Accuracy of mutants (RMSE, or -AUC for Adult)", "", "```\n" + acc.round(3).to_string() + "\n```", ""]
    bgp = os.path.join(OUT, "background.csv")
    if os.path.exists(bgp):
        b = pd.read_csv(bgp)
        L += ["## Background sweep", "", "```\n" + b.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_string() + "\n```", ""]
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def choose():
    """Pick the claim-2 detector on the exploration set only."""
    df = pd.read_csv(os.path.join(OUT, "detect.csv"))
    d = detected(df)
    mut = d[d.set == "exploration"]
    sub = df[df.set.isin(["healthy", "exploration"])]
    score = {m: (mut["det_" + m].mean(), auroc(sub, m)) for m in NEW}
    chosen = max(NEW, key=lambda m: score[m])
    with open(os.path.join(OUT, "choice.json"), "w") as fh:
        json.dump(dict(chosen=chosen, scores=score), fh, indent=1)
    print(chosen, score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["models", "detect", "choose", "background", "summarize"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(models=build_models, detect=detect, choose=choose, background=background, summarize=summarize)[a.step]()


if __name__ == "__main__":
    main()
