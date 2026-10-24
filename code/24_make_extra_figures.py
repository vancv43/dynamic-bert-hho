"""User-audit figure of the leakage audit, drawn from saved files only (vector PDF).

The per-seed score figure that this script first drew is now produced by 29_analyze_final.py.

fig_user_audit : rows per user in the original training split, and malicious test users already seen in
                 training under each protocol (data_processed/*.csv).
fig_seed_scores: per-seed recall against ROC-AUC, and the median threat score of the 69 malicious LOSO
                 test rows, for the seven BERT models (results/loso_ablation/*.npy)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
D = os.path.join(BASE, "..", "data_processed")
ABL = os.path.join(BASE, "..", "results", "loso_ablation")
FIG = os.path.join(BASE, "..", "figures")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
MAL, BEN = "#c0504d", "#1f4e79"


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- user audit
tr = pd.read_csv(os.path.join(D, "train.csv"))
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw={"width_ratios": [1.15, 1]})
bins = np.logspace(0, np.log10(130), 12)
for lab, col, name in ((0, BEN, "Benign users"), (1, MAL, "Malicious users")):
    c = tr[tr.label == lab].groupby("user").size()
    h, e = np.histogram(c, bins=bins)
    k = h > 0  # bars start at 0.8 so no path runs off the log axis
    a.bar(e[:-1][k], h[k] - 0.8, bottom=0.8, width=np.diff(e)[k], align="edge", color=col, alpha=0.75,
          label=f"{name}, n = {len(c)}", edgecolor="white", lw=0.3)
a.set_xscale("log")
a.set_yscale("log")
a.set_ylim(0.8, 2000)
a.minorticks_off()
a.set_xlabel("Training rows per user", labelpad=2)
a.set_ylabel("Number of users")
a.legend(frameon=False, fontsize=6, loc="upper right")
a.set_title("(a) Original training split", fontsize=7.5, loc="left")

prot = ["Original", "Grouped", "LOSO"]
pairs = [("train.csv", "test.csv"), ("train_grouped.csv", "test_grouped.csv"), ("train_loso.csv", "test_loso.csv")]
seen, unseen = [], []
for t1, t2 in pairs:
    x = pd.read_csv(os.path.join(D, t1)); y = pd.read_csv(os.path.join(D, t2))
    mtr, mte = set(x[x.label == 1].user), set(y[y.label == 1].user)
    seen.append(len(mte & mtr)); unseen.append(len(mte - mtr))
xs = np.arange(3)
b.bar(xs, seen, color=MAL, width=0.6, label="Seen in training")
b.bar(xs, unseen, bottom=seen, color="#d9d9d9", width=0.6, label="Unseen")
for i in xs:
    b.text(i, seen[i] + unseen[i] + 1, f"{seen[i]}/{seen[i] + unseen[i]}", ha="center", va="bottom", fontsize=6.5)
b.set_xticks(xs)
b.set_xticklabels(prot, fontsize=6.5)
b.set_ylabel("Malicious test users")
b.set_ylim(0, 55)
b.legend(frameon=False, fontsize=6, loc="upper right")
b.set_title("(b) Test users by protocol", fontsize=7.5, loc="left")
for ax_ in (a, b):
    ax_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3, w_pad=0.8)
save(fig, "fig_user_audit")

print("seen", seen, "unseen", unseen)
