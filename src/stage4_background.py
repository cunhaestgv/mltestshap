"""Post hoc check for Stage 4: do SHAP-sign boundaries follow the model or the SHAP background?

Same reference model, same samples: only the background changes (samples below, or above, the
median of the feature). If a boundary is a property of the model, it stays. Writes
results/stage4/background.csv.

    python3 -I src/stage4_background.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import method  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402
import stage4  # noqa: E402


def mid(v, s):
    a = method.analyse_feature(v[s != 0], s[s != 0], stage4.THETA, stage4.WIDE)
    return (a["lo"] + a["hi"]) / 2 if a["type"] == "narrow" else None


rows = []
for ds in stage4.DATASETS:
    res, cfg, data, Xg, bg, _ = stage4.setup(ds)
    model = models.load(data, ds, run.load_weights(os.path.join(res, "weights", "original.npz")))
    Xf, S0 = Xg.reshape(len(Xg), -1), np.load(os.path.join(res, "shap_original.npy"))
    Xtr = data["X_train"].reshape(len(data["X_train"]), -1)
    names = list(data["feature_names"])
    p = pd.read_csv(os.path.join(res, "partitions_theta%g.csv" % stage4.THETA))
    for n in p[(p.type == "narrow") & (p.n_values > 10)].feature:
        f = names.index(n)
        v = Xf[:, f]
        med = np.median(Xtr[:, f])
        rng = np.random.default_rng(0)
        rec = dict(dataset=ds, feature=n, boundary=mid(v, np.sign(S0[:, f])),
                   bg_mean=float(bg.reshape(len(bg), -1)[:, f].mean()), slope=None)
        for tag, m in (("low", Xtr[:, f] <= med), ("high", Xtr[:, f] > med)):
            idx = rng.permutation(np.flatnonzero(m))[:100]
            b = data["X_train"][idx]
            S = models.shap_values(model, b, Xg)
            rec["bg_" + tag] = float(Xtr[idx, f].mean())
            rec["boundary_" + tag] = mid(v, np.sign(S[:, f]))
        rows.append(rec)
        run.log("%s %s %s" % (ds, n, {k: None if x is None else round(x, 3) for k, x in rec.items() if k.startswith(("bound", "bg"))}))
df = pd.DataFrame(rows).drop(columns="slope")
df.to_csv(os.path.join(stage4.OUT, "background.csv"), index=False)
