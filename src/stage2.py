"""Stage 2: controlled validation of the margin diagnosis (see STAGE2.md).

Synthetic regression data with one injected cause per scenario. For every
scenario and replicate: train a reference MLP and 4 retrains, diagnose the
target feature x1, apply the scenario's fix and measure the margin share and
the local error before and after it.

    python3 -I src/stage2.py                 # all scenarios, 5 replicates
    python3 -I src/stage2.py --scenarios sharp --reps 1

Results go to results/stage2/: one JSON per (scenario, replicate), saved as
soon as it is done, so an interrupted run resumes where it stopped, and
results.csv / summary.md with all of them.
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

import diagnose  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402

SCENARIOS = {   # name: (expected row, fix)
    "sharp": (1, None), "noeffect": (2, "drop"), "inter": (3, None), "complex": (4, None),
    "nonmono": (9, None), "copy": (8, "group"), "missing": (5, "retrain"),
    "noise": (5, "retrain"), "sparse": (6, "retrain"),
}
N, N_GEN, N_BG, N_SEEDS = 4000, 3000, 100, 5


def make_data(scn, rep, fixed=False):
    """Data of one scenario and replicate. fixed=True applies the fix (same
    random draws, so the data differ only by the fix)."""
    rng = np.random.default_rng(1000 * list(SCENARIOS).index(scn) + rep)
    X = rng.random((N, 7))
    z = rng.random(N)
    eps = rng.normal(0, 0.05, N)
    copy_noise = rng.normal(0, 0.02, N)
    w_noise = rng.normal(0, 0.15, N)
    perm = rng.permutation(N)
    u_noise, u_sparse = rng.random(N), rng.random(N)

    sg = lambda u: np.where(u > 0.5, 1.0, -1.0)
    x1 = X[:, 0]
    g = {"sharp": 0.5 * (x1 > 0.5), "noeffect": 0 * x1,
         "inter": 0.8 * (x1 - 0.5) * sg(X[:, 1]),
         "complex": 0.8 * (x1 - 0.5) * sg(X[:, 1]) * sg(X[:, 5]) * sg(X[:, 6]),
         "nonmono": 0.5 * ((x1 > 0.3) & (x1 < 0.7)), "copy": 0.5 * (x1 > 0.5),
         "missing": 0.8 * (x1 - 0.5) * sg(z), "noise": 0.8 * (x1 - 0.5) * sg(X[:, 1]),
         "sparse": 0.5 * (x1 > 0.5)}[scn]
    f = 0.3 * X[:, 2] - 0.3 * X[:, 3] + 0.2 * X[:, 4] + g
    y = f + eps

    inputs, names = X.copy(), ["x%d" % i for i in range(1, 8)]
    if scn == "copy":
        inputs = np.column_stack([inputs, x1 + copy_noise])
        names.append("x8")
    if scn == "missing":
        inputs[:, 1] = z if fixed else z + w_noise
        names[1] = "z" if fixed else "w"
    if scn == "noeffect" and fixed:
        inputs, names = inputs[:, 1:], names[1:]

    a, b = int(0.7 * N), int(0.85 * N)
    tr, va, te = perm[:a], perm[a:b], perm[b:]
    y_tr = y[tr].copy()
    mid = (x1[tr] > 0.25) & (x1[tr] < 0.75)
    if scn == "noise" and not fixed:
        pick = np.flatnonzero(mid & (u_noise[tr] < 0.4))
        y_tr[pick] = y_tr[np.random.default_rng(rep).permutation(pick)]
    if scn == "sparse" and not fixed:
        keep = ~mid | (u_sparse[tr] < 0.2)
        tr, y_tr = tr[keep], y_tr[keep]

    F = inputs.shape[1]
    return dict(X_train=inputs[tr].astype("float32"), y_train=y_tr.astype("float32"),
                X_val=inputs[va].astype("float32"), y_val=y[va].astype("float32"),
                X_test=inputs[te].astype("float32"), y_test=y[te].astype("float32"),
                f_test=f[te], x1_test=x1[te], feature_names=names, var_names=names,
                task="regression", kind="tab", n_vars=F, lookback=1,
                x_min=np.zeros(F), x_max=np.ones(F), y_min=0.0, y_max=1.0)


def outputs(data, seed, wpath):
    """Train (or load) one model; its SHAP values and residuals on the generation set."""
    weights = run.cached_training(wpath, lambda: models.train(data, "synth", seed))
    model = models.load(data, "synth", weights)
    gen_idx, bg_idx = run.sample_sets(data, N_GEN, N_BG)
    Xg, yg = data["X_train"][gen_idx], data["y_train"][gen_idx]
    S = models.shap_values(model, data["X_train"][bg_idx], Xg)
    pred_test = models.predict(model, data["X_test"])
    return dict(S=S, resid=yg - models.predict(model, Xg), Xg=Xg, pred_test=pred_test)


def region(a, x1_test):
    """Test samples whose x1 lies in the reference margin (None: no margin)."""
    if a["type"] == "single":
        return None
    if a["lo"] is None:
        return np.ones(len(x1_test), bool)
    return (x1_test > a["lo"]) & (x1_test < a["hi"])


def local_error(data, pred, reg):
    if reg is None or not reg.any():
        return None
    return float(np.abs(data["f_test"][reg] - pred[reg]).mean())


def one(scn, rep, wdir):
    expected, fix = SCENARIOS[scn]
    data = make_data(scn, rep)
    outs = [outputs(data, s, os.path.join(wdir, "%s_%d_%d.npz" % (scn, rep, s))) for s in range(N_SEEDS)]
    Xg = outs[0]["Xg"]
    names = data["feature_names"]
    df = diagnose.diagnose(Xg, [o["S"] for o in outs], [o["resid"] for o in outs], names)
    r = df.iloc[0].to_dict()                          # x1 is always column 0
    a, _ = diagnose.margin_analysis(Xg, outs[0]["S"], 0, 0.1, 0.1)
    reg = region(a, data["x1_test"])
    err_all = float(np.abs(data["f_test"] - outs[0]["pred_test"]).mean())
    res = dict(scenario=scn, rep=rep, expected=expected, row=int(r["row"]), diagnosis=r["diagnosis"],
               correct=int(r["row"]) == expected, type=r["type"], share_before=float(r["share"]),
               region_test=None if reg is None else float(reg.mean()), err_all=err_all,
               local_err_before=local_error(data, outs[0]["pred_test"], reg),
               scores={k: (None if pd.isna(v) else v) for k, v in r.items()
                       if k not in ("feature", "type", "diagnosis", "row")})
    res["rows_all"] = dict(zip(df.feature, df.row.astype(int)))

    if fix == "group":
        groups = np.eye(len(names), dtype=bool)[:-1]
        groups[0, -1] = True                           # x1 and x8 as one feature
        Xv, _ = run.to_view(groups, Xg)
        Sv = [run.to_view(groups, S=o["S"])[1] for o in outs]
        dg = diagnose.diagnose(Xv, Sv, [o["resid"] for o in outs], ["x1+x8"] + names[1:-1])
        res.update(row_after=int(dg.iloc[0].row), diagnosis_after=dg.iloc[0].diagnosis,
                   share_after=float(dg.iloc[0].share), local_err_after=None)
    elif fix is not None:
        fdata = make_data(scn, rep, fixed=True)
        o = outputs(fdata, 0, os.path.join(wdir, "%s_%d_0_fix.npz" % (scn, rep)))
        if fix == "drop":
            share_after = None
        else:
            share_after = float(diagnose.margin_analysis(o["Xg"], o["S"], 0, 0.1, 0.1)[1].mean())
        res.update(share_after=share_after, local_err_after=local_error(fdata, o["pred_test"], reg),
                   err_all_after=float(np.abs(fdata["f_test"] - o["pred_test"]).mean()))
    return res


def summarize(out):
    rows = []
    for n in sorted(os.listdir(out)):
        if n.endswith(".json"):
            with open(os.path.join(out, n)) as fh:
                rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    flat = df.drop(columns=["scores", "rows_all"])
    flat.to_csv(os.path.join(out, "results.csv"), index=False)
    order = [s for s in SCENARIOS if s in set(df.scenario)]
    lines = ["# Stage 2 results", "", "Design: ../../STAGE2.md. Target feature x1; %d models per replicate." % N_SEEDS, ""]
    lines += ["| scenario | expected | correct | diagnoses | margin before | margin after | local err before | local err after | err all |",
              "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda s: "–" if s.isna().all() else "%.2f" % s.mean()
    for s in order:
        d = df[df.scenario == s]
        diag = ", ".join("%s ×%d" % (k, v) for k, v in d.diagnosis.value_counts().items())
        lines.append("| %s | %d %s | %d/%d | %s | %s | %s | %s | %s | %s |" % (
            s, d.expected.iloc[0], diagnose.LABELS[d.expected.iloc[0]], d.correct.sum(), len(d), diag,
            fmt(d.share_before), fmt(d.get("share_after", pd.Series(dtype=float))),
            fmt(d.local_err_before), fmt(d.get("local_err_after", pd.Series(dtype=float))), fmt(d.err_all)))
    if "row_after" in df:
        d = df[df.scenario == "copy"]
        lines += ["", "**copy, after grouping x1 and x8:** " + ", ".join(
            "%s ×%d" % (k, v) for k, v in d.diagnosis_after.value_counts().items())]
    lines += ["", "Margin = share of generation samples in the margin of x1 (0 when it is not wide). "
              "Local error = mean |f(x) − ŷ| of the reference model on test samples in the margin region "
              "(noise-free targets); after = model trained on the fixed data, same region. Means over replicates.",
              "", "Per-replicate scores (agreement, self, shallow, deep, err_ratio, resid_corr, density): results.csv and *.json."]
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(run.ROOT, "results", "stage2"))
    args = ap.parse_args()
    wdir = os.path.join(args.out, "weights")
    os.makedirs(wdir, exist_ok=True)
    for scn in args.scenarios:
        for rep in range(args.reps):
            path = os.path.join(args.out, "%s_%d.json" % (scn, rep))
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = one(scn, rep, wdir)
            with open(path, "w") as fh:
                json.dump(res, fh, indent=1, default=float)
            run.log("%s #%d: %s (expected %s) share %.2f -> %s (%.0fs)" % (
                scn, rep, res["diagnosis"], diagnose.LABELS[res["expected"]], res["share_before"],
                res.get("share_after"), time.time() - t0))
    summarize(args.out)


if __name__ == "__main__":
    main()
