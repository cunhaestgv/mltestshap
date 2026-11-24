"""Overview figure of the methodology (Figure 1): pdf_images/MethodologyOverview.pdf"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
INK, MUTED, A, B = "#1a1a1a", "#555555", "#e8eef7", "#f3ece0"

def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=INK)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=0, shrinkB=0))

W, H = 1.62, 0.9
ax.text(0.1, 4.35, "Part 1: partitions and boundary tests from one model", fontsize=10, weight="bold", color=INK)
y1 = 3.15
xs = [0.1, 2.1, 4.1, 6.1, 8.1]
t1 = ["Trained model,\ntraining data,\nbackground", "SHAP values\nand their signs", "Partitions\nper feature",
      "Uncertainty\nmargin\n(entropy)", "Boundary\ntest cases"]
for x, t in zip(xs, t1):
    box(x, y1, W, H, t, A)
for i in range(4):
    arrow(xs[i] + W + 0.03, y1 + H / 2, xs[i + 1] - 0.03, y1 + H / 2)
ax.text(8.1 + W / 2, y1 - 0.2, "wide margin: decision tree", ha="center", fontsize=8, color=MUTED, style="italic")

ax.text(0.1, 2.25, "Part 2: reference of retrained models", fontsize=10, weight="bold", color=INK)
y2 = 1.05
t2 = ["R retrains\n(other seeds)", "SHAP values\nof every model", "Stable pairs,\nmajority sign",
      "Detection\nstatistics,\nflag", "Localization\nand diagnosis"]
for x, t in zip(xs, t2):
    box(x, y2, W, H, t, B)
for i in range(4):
    arrow(xs[i] + W + 0.03, y2 + H / 2, xs[i + 1] - 0.03, y2 + H / 2)
ax.text(5.0, 0.55, "Model under test compared with the reference; complements an accuracy check",
        ha="center", fontsize=8, color=MUTED, style="italic")
# link: margins of part 1 feed the margin-width statistic and the margin diagnosis
arrow(6.1 + W / 2, y1 - 0.03, 6.1 + W / 2, y2 + H + 0.03)
ax.text(6.1 + W / 2 + 0.08, 2.5, "margin widths", fontsize=8, color=MUTED, style="italic", va="center")
fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
