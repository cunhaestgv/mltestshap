"""Stage 3 checks: repeatability, rule behaviour and the post hoc sensitivity in STAGE3.md "Outcome".

    python3 -I src/stage3_check.py
"""
import glob, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT + "/src")
import numpy as np, pandas as pd
import stage3 as S3
R = [json.load(open(p)) for p in sorted(glob.glob(ROOT + "/results/stage3/*.json"))]
syn = [r for r in R if not r["real"]]
# 1. repeatability of signals between run and reslice
keys = ["ratio", "density", "disagreement", "gbm", "outlier_share", "top_share", "top_rel"]
d = {k: np.mean([abs(r["signals"][k] - r["signals_again"][k]) for r in syn]) for k in keys}
print("mean abs diff run vs rerun:", {k: round(v, 3) for k, v in d.items()})
same = {lv: np.mean([S3.recommend(r["signals"], lv) == S3.recommend(r["signals_again"], lv) for r in syn]) for lv in S3.LEVELS}
print("same recommendation on rerun:", same)
# 2. how often does each rule fire, by task
df = pd.DataFrame([dict(task=r["task"][:3], scn=r["scenario"], **{lv: r[lv]["rec"] for lv in S3.LEVELS}, out=r["signals"]["outlier_share"], gbm=r["signals"]["gbm"], D=r["signals"]["disagreement"], dens=r["signals"]["density"], ratio=r["signals"]["ratio"], top_share=r["signals"]["top_share"], top_rel=r["signals"]["top_rel"], row=r["signals"]["row"], agr=r["signals"]["sign_agreement"], err=min(r["err"], key=r["err"].get)) for r in syn])
for t, g in df.groupby("task"):
    print(t, "L1", g.L1.value_counts().to_dict())
    print(t, "L3", g.L3.value_counts().to_dict())
    print(t, "best", g.err.value_counts().to_dict())
print("outlier share regression: by scenario\n", df[df.task == "reg"].groupby("scn").out.mean().round(3).to_dict())
print("median signals:", df[["gbm", "D", "dens", "ratio", "agr"]].median().round(2).to_dict(), "row counts", df.row.value_counts().to_dict())
# 3. exploratory: rule 2 off
def rec_no_robust(s, lv):
    return S3.recommend(dict(s, outlier_share=0.0), lv)
def rec_big(s, lv):
    return S3.recommend(dict(s, outlier_share=s["outlier_share"] if s["outlier_share"] >= 0.05 else 0.0), lv)
for name, fn in [("rule 2 needs share>=5%", rec_big), ("rule 2 off", rec_no_robust)]:
    reg = {lv: np.array([S3.score(r["err"], fn(r["signals"], lv))[0] for r in syn]) for lv in S3.LEVELS}
    hit = {lv: np.mean([S3.score(r["err"], fn(r["signals"], lv))[1] for r in syn]) for lv in S3.LEVELS}
    print(name, {lv: (round(reg[lv].mean(), 3), round(hit[lv], 2)) for lv in S3.LEVELS}, "L2-L1 %.3f [%.3f, %.3f]" % S3.boot(reg["L2"], reg["L1"]), "L3-L2 %.3f [%.3f, %.3f]" % S3.boot(reg["L3"], reg["L2"]))
reg1 = np.array([S3.score(r["err"], S3.recommend(r["signals"], "L1"))[0] for r in syn])
# 4. spurious details
for r in syn:
    if r["scenario"] == "spurious":
        s = r["signals"]; print(r["task"][:3], r["rep"], "top", s["top_feature"], "share %.2f rel %.2f" % (s["top_share"], s["top_rel"]), "root", s["root"], "shap", s["shap_top"], "gain drop %.0f%%" % (100 * (1 - r["err"]["drop_top"] / r["err"]["none"])))
# 5. slicing paired
sl = pd.DataFrame([r["slicing"] for r in R]); sl["real"] = [r["real"] for r in R]
print("lift paired cluster-tree %.3f [%.3f, %.3f]" % S3.boot(sl.lift_cluster.fillna(sl.lift_tree), sl.lift_tree), "cluster wins", (sl.lift_cluster > sl.lift_tree).sum(), "/", len(sl), "medians", sl.lift_tree.median().round(2), sl.lift_cluster.median().round(2))
