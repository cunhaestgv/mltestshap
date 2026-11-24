"""Figures of the case study (Section "Case Study" of the paper), from a finished run.

    python3 -I src/figures.py diet --out ../pdf_images

Writes <dataset>_violin.pdf, <dataset>_margin_<feature>.pdf for the features
given with --margins, and <dataset>_tree_<feature>.pdf for --trees, all in
raw feature units. Needs matplotlib (not used by the experiments).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

import datasets  # noqa: E402
import run  # noqa: E402


def raw_inputs(data, X):
    """Flattened inputs X in raw units."""
    Xf = X.reshape(len(X), -1)
    return np.column_stack([datasets.raw_value(data, f, Xf[:, f]) for f in range(Xf.shape[1])])


def margin_plot(name, v, s, part, path):
    """Feature values per sample, marked by SHAP sign, with the partition boundaries
    and the uncertainty margin shaded (the style of the original figures)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    idx = np.arange(len(v))
    ax.scatter(idx[s < 0], v[s < 0], marker="x", c="k", label="Negative SHAP")
    ax.scatter(idx[s > 0], v[s > 0], marker="+", c="k", s=60, label="Positive SHAP")
    if part["type"] == "wide" and pd.isna(part["lo_raw"]):
        ax.axhspan(v.min(), v.max(), color="lightgrey", zorder=0, label="Uncertainty margin")
    elif part["type"] != "single":
        lo, hi = part["lo_raw"], part["hi_raw"]
        ax.axhline(lo, ls="--", c="k", lw=1, label="Lower boundary (%.4g)" % lo)
        ax.axhline(hi, ls=":", c="k", lw=1, label="Upper boundary (%.4g)" % hi)
        if hi > lo:
            ax.axhspan(lo, hi, color="lightgrey", zorder=0)
    ax.set_xlabel("Sample")
    ax.set_ylabel(name)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tree_plot(name, Xr, s, names, cfg, path):
    """The decision tree of method.tree_partitions, refitted on raw units (an affine
    rescaling of each feature does not change the tree, only its thresholds)."""
    ok = s != 0
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0,
                                  min_samples_leaf=max(5, int(0.01 * ok.sum())))
    tree.fit(Xr[ok], s[ok] > 0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_tree(tree, feature_names=names, class_names=["negative", "positive"], impurity=True,
              filled=False, rounded=True, fontsize=8, precision=4, ax=ax)
    ax.set_title("SHAP sign of %s" % name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return export_text(tree, feature_names=list(names), decimals=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--margins", nargs="*", default=[])
    ap.add_argument("--trees", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = os.path.join(run.ROOT, "results", args.dataset)
    cfg = pd.read_json(os.path.join(res, "config.json"), typ="series")
    data = datasets.LOADERS[args.dataset]()
    gen_idx, _ = run.sample_sets(data, int(cfg.n_gen), int(cfg.n_bg))
    Xr = raw_inputs(data, data["X_train"][gen_idx])
    S = np.load(os.path.join(res, "shap_original.npy"))
    names = data["feature_names"]
    parts = pd.read_csv(os.path.join(res, "partitions_theta%g.csv" % args.theta)).set_index("feature")
    os.makedirs(args.out, exist_ok=True)
    tag = args.dataset

    plt.figure()
    shap.summary_plot(S, Xr, feature_names=names, plot_type="violin", show=False, plot_size=(7, 9))
    plt.savefig(os.path.join(args.out, "%s_violin.pdf" % tag), bbox_inches="tight")
    plt.close("all")

    for name in args.margins:
        f = names.index(name)
        margin_plot(name, Xr[:, f], np.sign(S[:, f]), parts.loc[name],
                    os.path.join(args.out, "%s_margin_%s.pdf" % (tag, name)))
    for name in args.trees:
        f = names.index(name)
        print(tree_plot(name, Xr, np.sign(S[:, f]), names, cfg,
                        os.path.join(args.out, "%s_tree_%s.pdf" % (tag, name))))
    print("written to", args.out)


if __name__ == "__main__":
    main()
