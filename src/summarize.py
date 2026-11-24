"""Summarize fault-detection results into tables.

    python3 -I src/summarize.py bike adult

A mutant is detected by a suite when the suite's failure rate on it is higher
than on every healthy retrained model (same oracle). Accuracy-based detection
is the conventional check: the test-set metric is worse than on every healthy
model. Writes results/summary.md and one CSV per table.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ORACLES = ("shap", "output")
OP_ORDER = ["GF", "WS", "NEB", "NAI", "LN", "FS", "DS"]


def md(frame):
    """DataFrame -> markdown table (avoids the optional tabulate dependency)."""
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ["/".join(map(str, c)) for c in frame.columns]
    idx = frame.index.names
    head = ["/".join(str(i) for i in idx if i is not None) or ""] + [str(c) for c in frame.columns]
    rows = [["/".join(map(str, i)) if isinstance(i, tuple) else str(i)] + [str(v) for v in r]
            for i, r in zip(frame.index, frame.values)]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def load(dataset):
    rows = []
    for path in glob.glob(os.path.join(ROOT, dataset, "evals", "*.json")):
        with open(path) as fh:
            r = json.load(fh)
        base = dict(dataset=dataset, name=r["name"], group=r["group"], op=r.get("op"),
                    level=r.get("level"), **r["metrics"])
        for key, fr in r["failure_rate"].items():
            family, theta, rep = re.match(r"(\w+)@([\d.]+)(?:#(\d+))?", key).groups()
            for oracle in ORACLES:
                rows.append(dict(base, suite=family, theta=float(theta), rep=int(rep or 0),
                                 oracle=oracle, fr=fr[oracle], size=fr["size"]))
    return pd.DataFrame(rows)


def detection(df):
    healthy = df[df.group == "healthy"]
    thr = healthy.groupby(["suite", "theta", "rep", "oracle"]).fr.max().rename("thr")
    mut = df[df.group.isin(["model", "data"])].join(thr, on=["suite", "theta", "rep", "oracle"])
    mut["detected"] = mut.fr > mut.thr
    metric, worse = ("rmse", np.greater) if "rmse" in df else ("auc", np.less)
    hm = healthy.drop_duplicates("name")[metric]
    limit = hm.max() if worse is np.greater else hm.min()
    mut["acc_detected"] = worse(mut[metric], limit)
    return mut, healthy


def pct(x):
    return "%.0f%%" % (100 * x)


def tables(dataset, df, main_theta):
    mut, healthy = detection(df)
    lines = ["## %s" % dataset, ""]

    models = df.drop_duplicates("name")
    metric_cols = [c for c in ("rmse", "mae", "accuracy", "auc") if c in models]
    orig = models[models.group == "original"][metric_cols].iloc[0]
    hl = models[models.group == "healthy"][metric_cols]
    lines.append("**Model quality (test set).** Original: " + ", ".join(
        "%s %.4g" % (c, orig[c]) for c in metric_cols) + ". Healthy retrains (n=%d): " % len(hl) +
        ", ".join("%s %.4g ± %.2g" % (c, hl[c].mean(), hl[c].std()) for c in metric_cols) + ".")
    lines.append("")

    sizes = df[(df.group == "original") & (df.suite == "ours") & (df.oracle == "shap")]
    lines.append("**Suite size per theta:** " + ", ".join(
        "θ=%g: %d tests" % (t, s) for t, s in zip(sizes.theta, sizes["size"])) + ".")
    lines.append("")

    lines.append("**False-alarm level** (mean failure rate on healthy retrains, θ=%g):" % main_theta)
    lines.append("")
    hm = healthy[healthy.theta == main_theta].groupby(["suite", "oracle"]).fr.agg(["mean", "max"])
    lines.append(md(hm.round(3)))
    lines.append("")

    m = mut[mut.theta == main_theta]
    det = m.groupby(["op", "suite", "oracle"]).detected.mean().unstack(["suite", "oracle"])
    det.loc["All"] = m.groupby(["suite", "oracle"]).detected.mean()
    acc = m.drop_duplicates("name").groupby("op").acc_detected.mean()
    acc.loc["All"] = m.drop_duplicates("name").acc_detected.mean()
    det[("accuracy", "-")] = acc
    n = m.drop_duplicates("name").groupby("op").size()
    n.loc["All"] = n.sum()
    det[("n", "mutants")] = n
    det = det.reindex([o for o in OP_ORDER if o in det.index] + ["All"])
    det.columns = ["%s/%s" % c for c in det.columns]
    det.to_csv(os.path.join(ROOT, "%s_detection.csv" % dataset))
    lines.append("**Mutants detected (θ=%g).** Random is averaged over its repetitions." % main_theta)
    lines.append("")
    lines.append(md(det.apply(lambda c: c if c.name == "n/mutants" else c.map(pct))))
    lines.append("")

    silent = m[~m.acc_detected]
    if len(silent):
        sd = silent.groupby(["suite", "oracle"]).detected.mean().unstack("oracle")
        lines.append("**Faults that accuracy misses** (%d mutants whose test metric stays within the "
                     "healthy range), detected by:" % silent.name.nunique())
        lines.append("")
        lines.append(md(sd.applymap(pct)))
        lines.append("")

    sens = mut.groupby(["theta", "suite", "oracle"]).detected.mean().unstack(["suite", "oracle"])
    sens.columns = ["%s/%s" % c for c in sens.columns]
    sens.to_csv(os.path.join(ROOT, "%s_theta_sensitivity.csv" % dataset))
    lines.append("**Sensitivity to θ** (all mutants detected):")
    lines.append("")
    lines.append(md(sens.applymap(pct)))
    lines.append("")
    return lines


def main():
    datasets = sys.argv[1:] or sorted(d for d in os.listdir(ROOT)
                                      if os.path.isdir(os.path.join(ROOT, d, "evals")) and not d.startswith("_"))
    lines = ["# Fault-detection results", ""]
    for d in datasets:
        df = load(d)
        if df.empty or not (df.group == "healthy").any():
            lines += ["## %s" % d, "", "(no results yet)", ""]
            continue
        thetas = sorted(df.theta.unique())
        lines += tables(d, df, thetas[len(thetas) // 2])
    with open(os.path.join(ROOT, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
