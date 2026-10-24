"""Redraw the SHAP figure with labelled axes, units and sample sizes (vector PDF only).

Figure 1 is drawn by 23_make_framework_figure.py."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
RESULTS = os.path.join(BASE, "..", "results")
FIG = os.path.join(BASE, "..", "figures")

top = json.load(open(os.path.join(RESULTS, "shap_top_tokens.json")))["top_tokens"][:10]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.05))
toks = [t[0] for t in top][::-1]
vals = [t[1] for t in top][::-1]
cnt = [t[2] for t in top][::-1]
ax.barh(range(len(toks)), vals, color="#1f4e79", height=0.6)
for i, (v, c) in enumerate(zip(vals, cnt)):
    ax.text(v + 0.0008, i, f"n={c}", va="center", fontsize=6.5)
ax.set_yticks(range(len(toks)))
ax.set_yticklabels([f"“{t}”" for t in toks])
ax.set_xlabel("Mean |SHAP value| for the threat class (probability units)")
ax.set_ylabel("WordPiece token")
ax.set_xlim(0, max(vals) * 1.22)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_shap_summary_v3.pdf"))

print("done")
