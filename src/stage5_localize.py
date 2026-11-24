"""Stage 5: locate known faults from SHAP disagreement (see STAGE5.md).

    python3 -I src/stage5_localize.py

Uses the SHAP values saved by Stage 4; no training. Writes results/stage5/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

import stage4  # noqa: E402

OUT = os.path.join(stage4.run.ROOT, "results", "stage5")


def majority(signs):
    """Majority sign over models (axis 0) and share that differ from it."""
    pos, neg = (signs > 0).sum(0), (signs < 0).sum(0)
    maj = np.where(pos >= neg, 1.0, -1.0)
    h = (signs != maj).mean(0)
    return maj, h


def true_cut(data, v):
    X = data["X_train"]
    col = X[:, -1, v] if X.ndim == 3 else X[:, v]
    q = np.quantile(col, 0.75)
    if (col <= q).all():
        vals = np.unique(col)
        return float((vals[-1] + vals[-2]) / 2), True
    return float(q), False


def analyse(ds):
    res, cfg, data, Xg, bg, top_vars = stage4.setup(ds)
    _, plan = stage4.plan_of(ds, cfg, top_vars)
    nv, names = data["n_vars"], list(data["feature_names"])
    X = Xg.reshape(len(Xg), -1)
    nf = X.shape[1]
    var_of = np.arange(nf) % nv
    t_col = {v: int(np.flatnonzero(var_of == v)[-1]) for v in range(nv)}
    sdir = os.path.join(stage4.OUT, ds, "shap")
    load = lambda n: np.load(os.path.join(sdir, n + ".npz"))["S"].reshape(len(X), -1)  # noqa: E731
    healthy = [p["name"] for p in plan if p["group"] == "healthy"]
    SH = np.stack([load(n) for n in healthy])
    sig_h = np.sign(SH)
    absH = np.abs(SH).mean(1)                           # models x features
    ds_var = next(p["level"] for p in plan if p.get("op") == "DS")
    fs_vars = sorted({p["level"] for p in plan if p.get("op") == "FS"})
    cut, skewed = true_cut(data, ds_var)
    above = X[:, t_col[ds_var]] > cut
    leaf = max(5, int(0.02 * len(X)))

    rows = []
    for p in plan:
        if p["group"] == "bs":
            continue
        if p["group"] == "healthy":
            k = healthy.index(p["name"])
            others = [j for j in range(len(healthy)) if j != k]
            S = SH[k]
            maj, h = majority(sig_h[others])
            ref_abs = absH[others].mean(0)
        else:
            S = load(p["name"])
            maj, h = majority(sig_h)
            ref_abs = absH.mean(0)
        e = (np.sign(S) != maj) - h                       # rows x features
        # FS view: variable ranking
        var_ex = np.array([e[:, var_of == v].mean() for v in range(nv)])
        a = np.abs(S).mean(0)
        drop = np.array([((ref_abs - a)[var_of == v].sum() / max(ref_abs[var_of == v].sum(), 1e-12)) for v in range(nv)])
        top_e, top_a = int(np.argmax(var_ex)), int(np.argmax(drop))
        # DS view: region tree
        target = e.mean(1)
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=leaf, random_state=0).fit(X, target)
        rf, rt = int(tree.tree_.feature[0]), float(tree.tree_.threshold[0])
        r = dict(view=ds, name=p["name"], group=p["group"], op=p.get("op", "healthy"), var=p.get("level"),
                 mean_excess=float(e.mean()), top_var_excess=top_e, top_var_absdrop=top_a,
                 root_feature=names[rf] if rf >= 0 else None, root_var=int(var_of[rf]) if rf >= 0 else None,
                 root_is_t=bool(rf == t_col[var_of[rf]]) if rf >= 0 else None, root_thr=rt,
                 ds_var=ds_var, ds_cut=cut, ds_skewed=skewed,
                 lift=float(target[above].mean() - target[~above].mean()) if above.any() else np.nan,
                 n_vars=nv)
        if p.get("op") == "FS":
            v = p["level"]
            r["rank_excess"] = int((var_ex > var_ex[v]).sum() + 1)
            r["rank_absdrop"] = int((drop > drop[v]).sum() + 1)
        for v in fs_vars:
            r["top_is_fs%d" % v] = top_e == v
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df):
    L = ["# Stage 5: fault localization from SHAP disagreement", "", "Design: `STAGE5.md`. Per-model rows: `localize.csv`.", ""]
    hl = df[df.group == "healthy"]
    L.append("Sanity: mean excess, healthy (cross-fitted) %.4f; mutants %.4f." % (
        hl.mean_excess.mean(), df[df.group != "healthy"].mean_excess.mean()))
    fs = df[df.op == "FS"]
    hit_e = (fs.top_var_excess == fs["var"]).sum()
    hit_a = (fs.top_var_absdrop == fs["var"]).sum()
    L += ["", "## FS: which variable", "",
          "| view | n | sign-excess top-1 | |SHAP|-drop top-1 | chance | median rank (excess) | healthy null top=v |",
          "|---|---|---|---|---|---|---|"]
    for ds, g in fs.groupby("view"):
        h = df[(df.view == ds) & (df.group == "healthy")]
        null = np.mean([h["top_is_fs%d" % v].mean() for v in g["var"].unique()])
        L.append("| %s | %d | %d | %d | %.2f | %g | %.2f |" % (
            ds, len(g), (g.top_var_excess == g["var"]).sum(), (g.top_var_absdrop == g["var"]).sum(),
            1 / g.n_vars.iloc[0], g.rank_excess.median(), null))
    L.append("| all | %d | %d | %d | | %g | |" % (len(fs), hit_e, hit_a, fs.rank_excess.median()))
    dsm = df[df.op == "DS"].copy()
    dsm["root_hit"] = dsm.root_var == dsm.ds_var
    dsm["thr_hit"] = dsm.root_hit & ((dsm.root_thr - dsm.ds_cut).abs() <= 0.05)
    L += ["", "## DS: which region", "",
          "| view | model | root feature | threshold | true cut | root on v | threshold hit | lift |", "|---|---|---|---|---|---|---|---|"]
    for _, r in dsm.iterrows():
        L.append("| %s | %s | %s | %.3f | %.3f%s | %s | %s | %+.3f |" % (
            r["view"], r["name"], r.root_feature, r.root_thr, r.ds_cut, " (skewed)" if r.ds_skewed else "",
            r.root_hit, r.thr_hit, r.lift))
    L += ["", "Root on v: %d/%d; threshold hit: %d/%d." % (dsm.root_hit.sum(), len(dsm), dsm.thr_hit.sum(), len(dsm)), "",
          "Control, share of models whose root splits on the DS variable, and mean lift:", "",
          "| view | healthy | LN | weight | FS | DS |", "|---|---|---|---|---|---|"]
    df = df.assign(root_on_ds=df.root_var == df.ds_var,
                   fam=np.where(df.op.isin(["GF", "WS", "NEB", "NAI"]), "weight", df.op))
    for ds, g in df.groupby("view"):
        cell = lambda f: "%.2f (%+.3f)" % (g[g.fam == f].root_on_ds.mean(), g[g.fam == f].lift.mean())  # noqa: E731
        L.append("| %s | %s | %s | %s | %s | %s |" % (ds, cell("healthy"), cell("LN"), cell("weight"), cell("FS"), cell("DS")))
    hr = df[df.fam == "healthy"].root_on_ds.mean()
    crit = [("FS top-1 >= 12/18 and >= |SHAP|-drop", hit_e >= 12 and hit_e >= hit_a),
            ("DS root on v >= 4/6 and healthy rate <= 0.30", dsm.root_hit.sum() >= 4 and hr <= 0.30),
            ("DS threshold hit >= 3/6", dsm.thr_hit.sum() >= 3)]
    L += ["", "## Criteria", ""] + ["- %s: %s" % (c, "met" if ok else "not met") for c, ok in crit]
    L.append("- healthy root-on-v rate, pooled: %.2f" % hr)
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.concat([analyse(ds) for ds in stage4.DATASETS], ignore_index=True)
    df.to_csv(os.path.join(OUT, "localize.csv"), index=False)
    text = summarize(df)
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
