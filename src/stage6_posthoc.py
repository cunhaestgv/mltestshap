"""Post hoc checks for Stage 6 (see STAGE6.md, Outcome): one-sided |SHAP| drop localization and union with accuracy.

    python3 -I src/stage6_posthoc.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, stage4, stage6
# 1. one-sided relative drop (Stage 5 definition) on FS mutants of both sets
for sname, base in (("exploration", stage4.OUT), ("confirmation", stage6.OUT)):
    hit = n = 0
    for ds in stage4.DATASETS:
        res, cfg, data, Xg, bg, tv = stage4.setup(ds)
        nv = data["n_vars"]; X = Xg.reshape(len(Xg), -1); var_of = np.arange(X.shape[1]) % nv
        L = lambda d, nm: np.load(os.path.join(d, ds, "shap", nm + ".npz"))["S"].reshape(len(X), -1)
        ref = np.mean([np.abs(L(stage4.OUT, "healthy_%d" % i)).mean(0) for i in range(1, 11)], 0)
        seeds = (0, 1) if sname == "exploration" else stage6.D_SEEDS
        for v in tv[:3]:
            for s in seeds:
                a = np.abs(L(base, "FS_%d_%d" % (v, s))).mean(0)
                drop = [(ref - a)[var_of == u].sum() / ref[var_of == u].sum() for u in range(nv)]
                hit += int(np.argmax(drop) == v); n += 1
    print(sname, "one-sided drop localization %d/%d" % (hit, n))
# 2. union of accuracy and a detector, each with its own rule
df = pd.read_csv(os.path.join(stage6.OUT, "detect.csv")); d = stage6.detected(df)
for sname in ("exploration", "confirmation"):
    m = d[d.set == sname]
    for x in ("F-pairs", "F-pairs-H", "Z-pairs"):
        u = m.det_accuracy.astype(bool) | m["det_" + x].astype(bool)
        h = d[d.set == "healthy"]; hu = h.det_accuracy.astype(bool) | h["det_" + x].astype(bool)
        print(sname, "accuracy OR %s: %.1f%% (%d/144), healthy flagged %d/30" % (x, 100 * u.mean(), u.sum(), hu.sum()))
    print(sname, "accuracy alone %d/144" % m.det_accuracy.sum())
