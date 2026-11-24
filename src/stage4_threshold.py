"""Post hoc check for Stage 4: a real step against a smooth trend, under three SHAP backgrounds.

y = g(x1) + 0.3 x2 + noise, with g a step at 0.7 or a straight line. x ~ U(0,1), so the mean of x1 is
not at the step. Background: all, below 0.4 or above 0.6 in x1. Writes results/stage4/threshold.csv.

    python3 -I src/stage4_threshold.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import models  # noqa: E402
import run  # noqa: E402
import stage4  # noqa: E402

rows = []
for kind in ("step", "line"):
    for rep in range(5):
        rng = np.random.default_rng(rep)
        X = rng.random((5000, 4)).astype("float32")
        g = 0.5 * (X[:, 0] > 0.7) if kind == "step" else 0.5 * X[:, 0]
        y = (g + 0.3 * X[:, 1] + rng.normal(0, 0.03, 5000)).astype("float32")
        data = dict(X_train=X[:3500], y_train=y[:3500], X_val=X[3500:4000], y_val=y[3500:4000], X_test=X[4000:], y_test=y[4000:],
                    task="regression", kind="tab", n_vars=4, lookback=1, y_min=0.0, y_max=1.0)
        model = models.load(data, "synth", models.train(data, "synth", rep))
        Xg = X[:2000]
        for tag, m in (("all", np.ones(3500, bool)), ("low", X[:3500, 0] < 0.4), ("high", X[:3500, 0] > 0.6)):
            b = X[:3500][m][:100]
            s = np.sign(models.shap_values(model, b, Xg)[:, 0])
            rows.append(dict(kind=kind, rep=rep, bg=tag, bg_mean=float(b[:, 0].mean()),
                             **dict(zip(("lo", "hi"), (lambda a: (a.get("lo"), a.get("hi")))(
                                 stage4.method.analyse_feature(Xg[s != 0, 0], s[s != 0], stage4.THETA, stage4.WIDE))))))
df = pd.DataFrame(rows)
df["mid"] = (df.lo + df.hi) / 2
df.to_csv(os.path.join(stage4.OUT, "threshold.csv"), index=False)
print(df.groupby(["kind", "bg"])[["bg_mean", "mid"]].agg(["mean", "min", "max"]).round(3).to_string())
