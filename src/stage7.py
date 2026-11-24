"""Stage 7: diagnosing the type of fault against a retraining reference (see STAGE7.md).

    python3 -I src/stage7.py features --dataset adult   # also bike, diet: fingerprints of every model
    python3 -I src/stage7.py summarize                  # classifiers, tests, tables -> results/stage7/summary.md

Uses the SHAP values and weights saved by Stages 4 and 6; no training.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp, spearmanr  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import method  # noqa: E402
import models  # noqa: E402
import mutants  # noqa: E402
import run  # noqa: E402
import stage4  # noqa: E402
import stage6  # noqa: E402
from stage5_localize import majority  # noqa: E402

OUT = os.path.join(run.ROOT, "results", "stage7")
THETA, WIDE, KEEP = stage4.THETA, stage4.WIDE, stage4.KEEP
CLASSES = ["weights", "LN", "FS", "DS"]
A = ["acc_z", "drop_max", "scale"]
B = ["F-pairs-H", "Z-pairs", "var_conc", "ratio_sd", "region_conc"]
C = ["W-margin", "margin_shift", "margin_conc", "tree_res", "tree_leaves"]
D = ["slope", "r2", "resid_var", "resid_region"]
E = ["own_fit_z", "label_dis", "coverage", "relation"]
SETS = {"S0": A, "S1": A + B, "S2": A + B + C, "S3": A + B + D, "S4": A + B + C + D, "S5": A + B + C + D + E}


def cls_of(op):
    return "weights" if op in mutants.MODEL_OPS else op


def last(X, v):
    return X[:, -1, v] if X.ndim == 3 else X[:, v]


def region(r, Xg, nv):
    """Max over variables and tails (bottom/top quartile) of mean r in the tail / mean r elsewhere."""
    best = (-np.inf, None, None)
    for v in range(nv):
        x = last(Xg, v)
        lo, hi = np.quantile(x, [0.25, 0.75])
        for tag, m in (("bottom", x <= lo), ("top", x >= hi)):
            if 0 < m.sum() < len(x):
                ratio = r[m].mean() / max(r[~m].mean(), 1e-12)
                if ratio > best[0]:
                    best = (float(ratio), v, tag)
    return best


def spear(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    c = spearmanr(a, b).correlation
    return 0.0 if np.isnan(c) else float(c)


def tree_stats(X, S, names):
    g = method.generate(X, S, names, theta=THETA, wide_frac=WIDE, k=3)
    sg = np.sign(S)
    wide = [names.index(f["feature"]) for f in g["features"] if f["type"] == "wide"]
    n_wide = int(sum((sg[:, f] != 0).sum() for f in wide))
    covered = sum(lf["samples"] for lf in g["leaves"])
    return (covered / n_wide if n_wide else 1.0), len(g["leaves"])


def loss(task, y, p):
    if task == "classification":
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean((y - p) ** 2))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def own_data(data, it):
    if it["group"] != "data":
        return data["X_train"], data["y_train"]
    return mutants.corrupt_data(data, it["op"], it["level"], np.random.default_rng(500 + it["seed"]))


def weights_of(data, args, it, original, wdir):
    if it["group"] in ("healthy", "data") and not os.path.exists(os.path.join(wdir, it["name"] + ".npz")):
        raise FileNotFoundError("missing weights for %s (no training in Stage 7)" % it["name"])
    return run.get_weights(data, args, it, original, wdir)


def features(ds):
    res, cfg, data, Xg, bg, top_vars = stage4.setup(ds)
    args, plan4 = stage4.plan_of(ds, cfg, top_vars)
    X = Xg.reshape(len(Xg), -1)
    nv, names, task = data["n_vars"], list(data["feature_names"]), data["task"]
    var_of = np.arange(X.shape[1]) % nv
    original = run.load_weights(os.path.join(res, "weights", "original.npz"))
    items = [dict(p, set="healthy") for p in plan4 if p["group"] == "healthy"]
    items += [dict(p, set="exploration") for p in plan4 if p["group"] in ("model", "data")]
    items += [dict(p, set="confirmation") for p in stage6.confirm_plan(top_vars)]
    nH = sum(it["set"] == "healthy" for it in items)
    own = {}
    for it in items:
        if it["group"] == "data":
            own[(it["set"], it["name"])] = own_data(data, it)
    Xtr, ytr = data["X_train"], data["y_train"]

    # pass 1: raw quantities of every model
    raw = []
    for it in items:
        t0 = time.time()
        conf = it["set"] == "confirmation"
        sdir = os.path.join(stage6.OUT if conf else stage4.OUT, ds, "shap")
        wdir = os.path.join(stage6.OUT, ds, "weights") if conf else os.path.join(res, "weights")
        with np.load(os.path.join(sdir, it["name"] + ".npz")) as z:
            S = z["S"].reshape(len(z["S"]), -1).astype("float64")
            met = json.loads(str(z["metrics"]))
        model = models.load(data, ds, weights_of(data, args, it, original, wdir))
        Xo, yo = own.get((it["set"], it["name"]), (Xtr, ytr))
        r = dict(it=it, S=S, err=met["rmse"] if "rmse" in met else 1 - met["auc"],
                 out=models.predict(model, Xg), own_loss=loss(task, yo, models.predict(model, Xo)))
        if it["set"] == "healthy":
            r["clean_pred"] = models.predict(model, Xtr)
            r["own_pred"] = {k: models.predict(model, v[0]) for k, v in own.items()}
        r["tree_res"], r["tree_n"] = tree_stats(X, S, names)
        r["w"] = stage6.margins(X, S)
        raw.append(r)
        run.log("stage7 %s %s %s (%.0fs)" % (ds, it["set"], it["name"], time.time() - t0))

    H = raw[:nH]
    SH = np.stack([r["S"] for r in H])
    sgH = np.sign(SH)
    absH = np.stack([[np.abs(S[:, var_of == v]).mean(0).sum() for v in range(nv)] for S in SH])
    errH = np.array([r["err"] for r in H])
    WH = np.stack([r["w"] for r in H])
    resH = np.array([r["tree_res"] for r in H])
    nlH = np.array([r["tree_n"] for r in H])
    outH = np.stack([r["out"] for r in H])
    lossH = np.array([r["own_loss"] for r in H])
    cleanH = np.stack([r["clean_pred"] for r in H])
    eps = 0.01 * np.abs(SH).mean()

    def share_dis(y, pH, ref):
        if task == "classification":
            pg = np.where(y > 0.5, pH, 1 - pH)
            return float(np.mean(pg < 0.1))
        thr = 3 * np.mean(np.sqrt([lossH[j] for j in ref]))
        return float(np.mean(np.abs(y - pH) > thr))

    rows = []
    for k, r in enumerate(raw):
        it, S = r["it"], r["S"]
        ref = [j for j in range(nH) if j != k] if it["set"] == "healthy" else list(range(nH))
        f = dict(view=ds, set=it["set"], name=it["name"], group=it["group"], op=it.get("op", "healthy"),
                 level=it.get("level"), cls="healthy" if it["set"] == "healthy" else cls_of(it["op"]))
        eh = errH[ref]
        f["acc_z"] = (r["err"] - eh.mean()) / max(eh.std(), 1e-12)
        a = np.array([np.abs(S[:, var_of == v]).mean(0).sum() for v in range(nv)])
        ab = absH[ref].mean(0)
        drop = (ab - a) / np.maximum(ab, 1e-12)
        f["drop_max"], f["drop_var"] = float(drop.max()), int(np.argmax(drop))
        f["scale"] = float(np.log(a.sum() / ab.sum()))
        maj, h = majority(sgH[ref])
        f["F-pairs-H"] = float((np.sign(S) != maj)[h <= KEEP].mean())
        mu, sd = SH[ref].mean(0), SH[ref].std(0)
        f["Z-pairs"] = float(np.minimum(np.abs(S - mu) / (sd + eps), 10).mean())
        dS = np.abs(S - mu)
        dv = np.array([dS[:, var_of == v].sum(1).mean() for v in range(nv)])
        f["var_conc"] = float(dv.max() / max(dv.sum(), 1e-12))
        f["ratio_sd"] = float(np.std(np.log(np.maximum(a, 1e-12) / np.maximum(ab, 1e-12))))
        f["region_conc"], f["region_var"], f["region_tail"] = region(dS.sum(1), Xg, nv)
        dw = r["w"] - WH[ref].mean(0)
        f["W-margin"] = float(np.abs(dw).mean())
        f["margin_shift"] = float(dw.mean())
        f["margin_conc"] = float(np.abs(dw).max() / np.abs(dw).sum()) if np.abs(dw).sum() > 0 else 0.0
        f["tree_res"] = r["tree_res"] - resH[ref].mean()
        f["tree_leaves"] = float(np.log((r["tree_n"] + 1) / (nlH[ref].mean() + 1)))
        yh = outH[ref].mean(0)
        yo = r["out"]
        if task == "classification":
            yh, yo = logit(yh), logit(yo)   # logit of the mean probability of H
        b, a0 = np.polyfit(yh, yo, 1)
        resid = yo - (a0 + b * yh)
        f["slope"] = float(b)
        f["r2"] = float(1 - resid.var() / max(yo.var(), 1e-12))
        f["resid_var"] = max(abs(spear(resid, last(Xg, v))) for v in range(nv))
        f["resid_region"], f["resid_region_var"], f["resid_region_tail"] = region(np.abs(yo - yh), Xg, nv)
        lh = lossH[ref]
        f["own_fit_z"] = (r["own_loss"] - lh.mean()) / max(lh.std(), 1e-12)
        key = (it["set"], it["name"])
        Xo, yv = own.get(key, (Xtr, ytr))
        clean_share = share_dis(ytr, cleanH[ref].mean(0), ref)
        pH = np.mean([H[j]["own_pred"][key] for j in ref], 0) if key in own else cleanH[ref].mean(0)
        f["label_dis"] = share_dis(yv, pH, ref) - clean_share
        if key in own:
            f["coverage"] = max(float(ks_2samp(last(Xo, v), last(Xtr, v)).statistic) for v in range(nv))
            f["relation"] = max(abs(spear(last(Xtr, v), ytr)) - abs(spear(last(Xo, v), yv)) for v in range(nv))
        else:
            f["coverage"], f["relation"] = 0.0, 0.0
        rows.append(f)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "features_%s.csv" % ds), index=False)
    run.log("stage7 %s: %d models" % (ds, len(rows)))


# ---------------------------------------------------------------- classification

def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


def rule(r):
    if r["drop_max"] >= 0.5 and r["var_conc"] >= 0.4:
        return "FS"
    if r["region_conc"] >= 2.0:
        return "DS"
    if r["scale"] <= np.log(0.8) and r["ratio_sd"] <= 0.25:
        return "LN"
    return "weights"


def bal_acc(y, p, classes=CLASSES):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean([np.mean(p[y == c] == c) for c in classes if (y == c).any()]))


def predict_all(tr, te):
    P = {}
    for s, cols in SETS.items():
        P[s] = lr().fit(tr[cols], tr.cls).predict(te[cols])
    P["rules"] = np.array([rule(r) for _, r in te.iterrows()])
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0)
    P["RF-S4"] = rf.fit(tr[SETS["S4"]], tr.cls).predict(te[SETS["S4"]])
    return P


def load_features():
    return pd.concat([pd.read_csv(os.path.join(OUT, "features_%s.csv" % ds)) for ds in stage4.DATASETS],
                     ignore_index=True)


def checks(df):
    s6 = pd.read_csv(os.path.join(stage6.OUT, "detect.csv"))
    m = df.merge(s6, on=["view", "set", "name"], suffixes=("", "_s6"))
    return len(m), {c: float(np.abs(m[c] - m[c + "_s6"]).max()) for c in ("F-pairs-H", "Z-pairs", "W-margin")}


def table(df, cols, fmt="%.3f"):
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(fmt % r[c] if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return L


def summarize():
    df = load_features()
    mut = df[df.set != "healthy"]
    tr, te = mut[mut.set == "exploration"].reset_index(drop=True), mut[mut.set == "confirmation"].reset_index(drop=True)
    P = predict_all(tr, te)
    y = te.cls.to_numpy()
    pd.DataFrame(dict(te[["view", "name", "op", "cls"]], **{k: v for k, v in P.items()})).to_csv(
        os.path.join(OUT, "predictions.csv"), index=False)

    # stratified bootstrap
    rng = np.random.default_rng(0)
    idx = {c: np.flatnonzero(y == c) for c in CLASSES}
    corr = {k: (v == y) for k, v in P.items()}
    boot = {k: np.empty(10000) for k in P}
    for b in range(10000):
        smp = {c: rng.choice(ix, len(ix), replace=True) for c, ix in idx.items()}
        for k in P:
            boot[k][b] = np.mean([corr[k][smp[c]].mean() for c in CLASSES])
    tests = [("S1 > S0", boot["S1"] - boot["S0"], bal_acc(y, P["S1"]) - bal_acc(y, P["S0"])),
             ("S2 > S1", boot["S2"] - boot["S1"], bal_acc(y, P["S2"]) - bal_acc(y, P["S1"])),
             ("S3 > S1", boot["S3"] - boot["S1"], bal_acc(y, P["S3"]) - bal_acc(y, P["S1"])),
             ("rules > 0.25", boot["rules"] - 0.25, bal_acc(y, P["rules"]) - 0.25)]
    ps = np.array([float(np.mean(d <= 0)) for _, d, _ in tests])
    adj = stage6.holm(ps)

    n, chk = checks(df)
    L = ["# Stage 7 results", "", "Design: `../../STAGE7.md`. Fingerprints: `features_<dataset>.csv`; predictions: `predictions.csv`.", "",
         "Check (Stage 6 statistics reproduced, %d models): max |difference| %s." % (n, ", ".join("%s %.2g" % kv for kv in chk.items())), "",
         "## Pre-registered tests (confirmation set, %d mutants; balanced accuracy; Holm over 4)" % len(te), "",
         "| test | difference | 95% CI | p | Holm p | supported |", "|---|---|---|---|---|---|"]
    for (nm, d, obs), p, pa in zip(tests, ps, adj):
        lo, hi = np.quantile(d, [0.025, 0.975])
        L.append("| %s | %+.3f | [%+.3f, %+.3f] | %.4f | %.4f | %s |" % (nm, obs, lo, hi, p, pa, "yes" if pa < 0.05 and obs > 0 else "no"))

    L += ["", "## Balanced accuracy and recall per class (confirmation set)", "",
          "| classifier | balanced acc. | weights | LN | FS | DS | adult | bike | diet |", "|---|---|---|---|---|---|---|---|---|"]
    for k, p in P.items():
        rec = [np.mean(p[y == c] == c) for c in CLASSES]
        per = [bal_acc(y[te.view == ds], p[te.view == ds]) for ds in stage4.DATASETS]
        L.append("| %s | %.3f | %s | %s |" % (k, bal_acc(y, p), " | ".join("%.2f" % v for v in rec), " | ".join("%.2f" % v for v in per)))

    L += ["", "## Confusion matrices (rows = true, columns = predicted; order %s)" % ", ".join(CLASSES), ""]
    for k in ("S0", "S1", "S2", "S3", "S4", "S5", "rules"):
        L += ["**%s**" % k, "```", str(confusion_matrix(y, P[k], labels=CLASSES)), "```", ""]

    # exploratory
    L += ["## Exploratory", "", "| analysis | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    cv = {s: np.empty(len(tr), dtype=object) for s in SETS}
    for a, b in skf.split(tr, tr.cls):
        for s, cols in SETS.items():
            cv[s][b] = lr().fit(tr.iloc[a][cols], tr.iloc[a].cls).predict(tr.iloc[b][cols])
    L.append("| exploration, 5-fold CV | " + " | ".join("%.3f" % bal_acc(tr.cls, cv[s]) for s in SETS) + " |")
    for ds in stage4.DATASETS:
        a, b = tr[tr.view != ds], te[te.view == ds]
        L.append("| leave out %s | " % ds + " | ".join(
            "%.3f" % bal_acc(b.cls, lr().fit(a[c], a.cls).predict(b[c])) for c in SETS.values()) + " |")
    s6 = stage6.detected(pd.read_csv(os.path.join(stage6.OUT, "detect.csv")))
    s6 = s6[s6.set == "confirmation"]
    flagged = set(s6[s6["det_accuracy"] | s6["det_F-pairs-H"]].apply(lambda r: (r["view"], r["name"]), axis=1))
    fm = np.array([(r["view"], r["name"]) in flagged for _, r in te.iterrows()])
    L.append("| flagged by accuracy OR F-pairs-H (%d) | " % fm.sum() + " | ".join("%.3f" % bal_acc(y[fm], P[s][fm]) for s in SETS) + " |")

    fs = te[te.cls == "FS"]
    dsm = te[te.cls == "DS"]
    L += ["", "**Localization (confirmation):** FS top-drop variable = shuffled variable %d/%d. DS: SHAP region variable = removed "
          "variable %d/%d (top tail %d); output region variable %d/%d (top tail %d)." % (
              (fs.drop_var == fs.level.astype(float).astype(int)).sum(), len(fs),
              (dsm.region_var == dsm.level.astype(float).astype(int)).sum(), len(dsm),
              ((dsm.region_var == dsm.level.astype(float).astype(int)) & (dsm.region_tail == "top")).sum(),
              (dsm.resid_region_var == dsm.level.astype(float).astype(int)).sum(), len(dsm),
              ((dsm.resid_region_var == dsm.level.astype(float).astype(int)) & (dsm.resid_region_tail == "top")).sum())]
    tw, ew = tr[tr.cls == "weights"], te[te.cls == "weights"]
    pw = lr().fit(tw[SETS["S4"]], tw.op).predict(ew[SETS["S4"]])
    L += ["", "**Weight operators (GF/WS/NEB/NAI), S4, confirmation:** balanced accuracy %.3f (chance 0.25)." %
          bal_acc(ew.op, pw, list(mutants.MODEL_OPS))]

    feats = A + B + C + D + E
    g = df.groupby("cls")[feats].median().reindex(["healthy"] + CLASSES).reset_index()
    L += ["", "## Median fingerprint per class (all sets; healthy cross-fitted)", ""] + table(g, ["cls"] + feats, "%.2f")
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "summarize"])
    ap.add_argument("--dataset", choices=stage4.DATASETS)
    a = ap.parse_args()
    if a.step == "features":
        for ds in ([a.dataset] if a.dataset else stage4.DATASETS):
            features(ds)
    else:
        summarize()


if __name__ == "__main__":
    main()
