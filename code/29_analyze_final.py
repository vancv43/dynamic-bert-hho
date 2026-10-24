"""Final statistics, tables and figures of the second-round revision, on the leakage-free rerun.

Reads results/rerun_loso (searches on the LOSO split, ten-seed LOSO ablation, five independent
random searches). Writes results/final_summary.json and vector PDF figures:
  fig_search_loss.pdf   loss of every evaluation, HHO and the five random searches
  fig_loso_ablation.pdf per-run LOSO precision, recall and F1 with mean and 95% interval
  fig_seed_scores.pdf   recall against ROC-AUC, recall of each selected configuration, threat scores
  fig_confusion.pdf     LOSO confusion matrices, mean and SD counts
The random-search control is pooled over five searches (seeds 1-5) x ten training seeds = 50 runs.
The seed-42 search shares its random stream with the HHO initialization and selects the HHO
configuration, so it is not used as a control. Intervals come from a two-level bootstrap that
resamples runs and then test rows (10,000 replicates, rng 2026). Paired two-sided Wilcoxon tests run
over the ten training seeds (the random-search value of a seed is its mean over the five searches),
Holm-corrected within each metric."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, cohen_kappa_score

BASE = os.path.dirname(__file__)
R = os.path.join(BASE, "..", "results", "rerun_loso")
ABL = os.path.join(R, "loso_ablation")
FIG = os.path.join(BASE, "..", "figures")
SEEDS = list(range(1, 11))
RS = [f"bert_rs_s{k}" for k in range(1, 6)]
MODELS = [("tfidf_xgboost", "TF-IDF + XGBoost"), ("word2vec_xgboost", "Word2Vec + XGBoost"),
          ("bert_default", "BERT, default"), ("bert_manual", "BERT, manual"),
          ("bert_rs", "BERT, random search"), ("bert_hho", "Dynamic BERT-HHO"),
          ("bert_default_meta", "BERT-Meta, default"),
          ("bert_hho_meta_defaultfusion", "BERT-HHO-Meta, default fusion"),
          ("bert_hho_meta", "Dynamic BERT-HHO-Meta")]
PAIRS = [("bert_hho", "bert_default"), ("bert_hho", "bert_manual"), ("bert_hho", "bert_rs"),
         ("bert_hho_meta", "bert_hho"), ("bert_default_meta", "bert_default"),
         ("bert_hho_meta", "bert_hho_meta_defaultfusion"), ("bert_hho_meta", "tfidf_xgboost")]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})
BLUE, RED, ORANGE = "#1f6fd1", "#d62728", "#e07b00"

y = np.load(os.path.join(ABL, "test_labels.npy"))
raw = json.load(open(os.path.join(ABL, "raw_runs.json")))
n = len(y)


def runs(m):
    """List of (model, seed) runs behind a table row."""
    if m == "bert_rs":
        return [(r, s) for r in RS for s in SEEDS]
    return [(m, s) for s in SEEDS]


def prob(m, s):
    return np.load(os.path.join(ABL, f"{m}_seed{s}.npy"))


def met(p):
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": pr, "recall": rc, "f1": f1,
            "roc_auc": roc_auc_score(y, p), "kappa": cohen_kappa_score(y, pred)}


def fast(yy, pp):
    tp = np.sum((pp == 1) & (yy == 1)); fp = np.sum((pp == 1) & (yy == 0)); fn = np.sum((pp == 0) & (yy == 1))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1.0, run)
    return adj


rng = np.random.default_rng(2026)
B = 10000
summary = {}
for m, name in MODELS:
    rr = runs(m)
    P = [prob(a, s) for a, s in rr]
    M = [met(p) for p in P]
    preds = np.array([(p >= 0.5).astype(int) for p in P])
    boot = np.zeros((B, 3))
    for b in range(B):
        vals = []
        for u in rng.integers(0, len(rr), len(rr)):
            idx = rng.integers(0, n, n)
            vals.append(fast(y[idx], preds[u][idx]))
        boot[b] = np.mean(vals, axis=0)
    d = {"name": name, "n_runs": len(rr)}
    for j, k in enumerate(["precision", "recall", "f1"]):
        d.setdefault(k, {})["ci95"] = [float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "kappa"]:
        v = np.array([x[k] for x in M])
        d.setdefault(k, {}).update({"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": v.tolist()})
        # per-training-seed value used for paired tests (mean over searches for random search)
        d[k]["per_seed"] = [float(np.mean([x[k] for (a, s), x in zip(rr, M) if s == t])) for t in SEEDS]
    tr = [raw[a][str(s)]["train_seconds"] for a, s in rr]
    inf = [raw[a][str(s)]["infer_seconds"] / raw[a][str(s)]["n_test"] * 1000 for a, s in rr]
    d["train_seconds"] = {"mean": float(np.mean(tr)), "sd": float(np.std(tr, ddof=1))}
    d["infer_ms_per_request"] = {"mean": float(np.mean(inf)), "sd": float(np.std(inf, ddof=1))}
    cms = np.array([[[np.sum((y == i) & ((p >= 0.5).astype(int) == j)) for j in (0, 1)] for i in (0, 1)] for p in P], float)
    d["confusion_mean"], d["confusion_sd"] = cms.mean(0).tolist(), cms.std(0, ddof=1).tolist()
    summary[m] = d

tests = {}
for k in ["precision", "recall", "f1", "roc_auc"]:
    rows, ps = [], []
    for a, b in PAIRS:
        va, vb = np.array(summary[a][k]["per_seed"]), np.array(summary[b][k]["per_seed"])
        dd = va - vb
        p = 1.0 if np.allclose(dd, 0) else float(wilcoxon(va, vb).pvalue)
        ps.append(p)
        rows.append({"a": a, "b": b, "mean_diff": float(dd.mean()), "wins": int((dd > 0).sum()),
                     "losses": int((dd < 0).sum()), "p": p})
    for r_, pa in zip(rows, holm(ps)):
        r_["p_holm"] = float(pa)
    tests[k] = rows

hho = json.load(open(os.path.join(R, "hho_search_result.json")))
meta = json.load(open(os.path.join(R, "hho_meta_search_result.json")))
rsj = [json.load(open(os.path.join(R, "rs_seeds", f"random_search_seed{k}.json"))) for k in range(1, 6)]
per_cfg = {m: summary["bert_hho"]["recall"]["values"] if m == "bert_hho" else
           [raw[m][str(s)]["metrics"]["recall"] for s in SEEDS] for m in ["bert_hho"] + RS}
search = {
    "hho": {"evals": hho["n_evaluations"], "seconds": hho["elapsed_seconds"],
            "n_zero": int(sum(e["fitness"] == 0 for e in hho["evaluations"])),
            "selected_eval": next(i for i, e in enumerate(hho["evaluations"]) if e["fitness"] == hho["best_fitness"]) + 1,
            "test_overlap_rows": hho["test_overlap_rows"], "best_hp": hho["best_hp"]},
    "rs": {"searches": 5, "evals": 51, "seconds": float(np.mean([j["elapsed_seconds"] for j in rsj])),
           "n_zero": [int(sum(e["fitness"] == 0 for e in j["evaluations"])) for j in rsj],
           "selected_draw": [next(i for i, e in enumerate(j["evaluations"]) if e["fitness"] == j["best_fitness"]) + 1 for j in rsj],
           "best_hp": [j["best_hp"] for j in rsj]},
    "hho_meta_fusion": {"evals": meta["n_evaluations"], "seconds": meta["elapsed_seconds"],
                        "best_fusion_hp": meta["best_fusion_hp"]},
    "config_mean_recall": {m: float(np.mean(v)) for m, v in per_cfg.items()},
}
json.dump({"models": summary, "tests": tests, "search": search, "n_test": int(n), "n_pos": int(y.sum()),
           "seeds": SEEDS}, open(os.path.join(BASE, "..", "results", "final_summary.json"), "w"), indent=2)


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)


# ---- search loss per evaluation
fig, ax = plt.subplots(figsize=(3.45, 2.2))
fh = [e["fitness"] for e in hho["evaluations"]]
for j in rsj:
    f = [e["fitness"] for e in j["evaluations"]]
    ax.plot(np.arange(1, 52), f, "x", ms=2.6, color=ORANGE, alpha=0.75, mew=0.6,
            label="Random search, seeds 1 to 5" if j is rsj[0] else None)
ax.plot(np.arange(1, len(fh) + 1), fh, "o", ms=2.6, color=BLUE, label="HHO")
ax.set_xlabel("Evaluation index, 2,000 training and 600 validation rows")
ax.set_ylabel(r"Validation loss $L=1-S$")
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
save(fig, "fig_search_loss")

# ---- per-run LOSO precision, recall, F1
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
ypos = np.arange(len(MODELS))[::-1]
jr = np.random.default_rng(7)
for ax, k, lab in zip(axes, ["precision", "recall", "f1"], ["Precision", "Recall", "F1-score"]):
    for yy, (m, _) in zip(ypos, MODELS):
        v = summary[m][k]["values"]
        ax.scatter(v, yy + jr.uniform(-0.2, 0.2, len(v)), s=7, color=BLUE, alpha=0.5, lw=0, zorder=2)
        lo, hi = summary[m][k]["ci95"]
        ax.plot([lo, hi], [yy, yy], color=RED, lw=1.5, zorder=3)
        ax.plot(summary[m][k]["mean"], yy, "o", ms=3.8, color=RED, mec="white", mew=0.4, zorder=4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(f"{lab} at threshold 0.5")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([nm for _, nm in MODELS])
fig.text(0.99, 0.005, f"LOSO test set: {n} requests, {int(y.sum())} malicious. Blue dots: single runs, ten per model and "
         "fifty for random search. Red: mean with 95% bootstrap interval.", ha="right", va="bottom", fontsize=6.3)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "fig_loso_ablation")

# ---- recall vs AUC, recall per selected configuration, threat scores
bert = [("bert_default", "BERT, default", "#9e9e9e", "o"), ("bert_manual", "BERT, manual", "#5f5f5f", "s"),
        ("bert_rs", "BERT, random search", ORANGE, "D"), ("bert_hho", "Dynamic BERT-HHO", BLUE, "^"),
        ("bert_default_meta", "BERT-Meta, default", "#d9a441", "v"),
        ("bert_hho_meta_defaultfusion", "BERT-HHO-Meta, default fusion", "#b07cc6", "P"),
        ("bert_hho_meta", "Dynamic BERT-HHO-Meta", RED, "X")]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1, 0.8, 1.25]})
for m, name, col, mk in bert:
    a.scatter(summary[m]["roc_auc"]["values"], summary[m]["recall"]["values"], s=10, color=col, marker=mk,
              lw=0, alpha=0.8, label=name)
a.set_xlabel("ROC-AUC")
a.set_ylabel("Recall at threshold 0.5")
a.set_xlim(0.8, 1.005); a.set_ylim(-0.05, 1.05)
a.legend(fontsize=5.4, loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1,
         handletextpad=0.2, borderaxespad=0.2)
a.set_title("(a) Ranking quality and recall", fontsize=7.5, loc="left")
cfgs = ["bert_hho"] + RS
labs = ["HHO"] + [f"RS{k}" for k in range(1, 6)]
for i, m in enumerate(cfgs):
    v = per_cfg[m]
    col = BLUE if m == "bert_hho" else ORANGE
    b.scatter(i + jr.uniform(-0.18, 0.18, len(v)), v, s=9, color=col, alpha=0.6, lw=0)
    b.plot([i - 0.28, i + 0.28], [np.mean(v)] * 2, color=RED, lw=1.4)
b.set_xticks(range(len(cfgs))); b.set_xticklabels(labs, fontsize=6.5)
b.set_ylabel("Recall at threshold 0.5"); b.set_ylim(-0.05, 1.05)
b.set_title("(b) Configurations with zero validation loss", fontsize=7.5, loc="left")
for i, (m, name, col, mk) in enumerate(bert):
    med = [np.median(prob(a_, s)[y == 1]) for a_, s in runs(m)]
    c.scatter(i + jr.uniform(-0.3, 0.3, len(med)), med, s=8, color=col, marker=mk, lw=0, alpha=0.8)
c.axhline(0.5, color="#444444", lw=0.7, ls="--")
c.set_yscale("log"); c.minorticks_off(); c.set_ylim(1e-5, 1.5)
c.set_xticks(range(len(bert)))
c.set_xticklabels(["Def.", "Man.", "RS", "HHO", "Meta", "HHO-\nMeta,\nd.f.", "HHO-\nMeta"], fontsize=6,
                  linespacing=1.3)
c.set_ylabel("Median threat score,\nmalicious test rows")
c.set_title("(c) Threat scores; dashed line, threshold 0.5", fontsize=7.5, loc="left")
for ax_ in (a, b, c):
    ax_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3, w_pad=1.0)
save(fig, "fig_seed_scores")

# ---- confusion matrices
CM = [("tfidf_xgboost", "(a) TF-IDF + XGBoost"), ("bert_rs", "(b) BERT, random search"),
      ("bert_hho", "(c) Dynamic BERT-HHO"), ("bert_hho_meta", "(d) Dynamic BERT-HHO-Meta")]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))
for ax, (m, title) in zip(axes, CM):
    mean, sd = np.array(summary[m]["confusion_mean"]), np.array(summary[m]["confusion_sd"])
    share = mean / mean.sum(1, keepdims=True)
    # vector cells (pcolormesh) instead of imshow, so the PDF holds no raster image
    ax.pcolormesh(np.arange(3) - 0.5, np.arange(3) - 0.5, share, cmap="Blues", vmin=0, vmax=1,
                  edgecolors="white", linewidth=0.8)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5); ax.set_aspect("equal")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mean[i, j]:.1f}\n$\\pm${sd[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if share[i, j] > 0.55 else "#1a1a1a", linespacing=1.35)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Malicious"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Malicious"], rotation=90, va="center")
    ax.set_title(title, fontsize=7.5, loc="left"); ax.tick_params(length=0)
    ax.set_xlabel("Predicted class")
axes[0].set_ylabel("True class")
fig.tight_layout(pad=0.3, w_pad=1.0)
save(fig, "fig_confusion")

print(json.dumps({m: {k: round(summary[m][k]["mean"], 3) for k in ["precision", "recall", "f1", "roc_auc"]}
                  for m, _ in MODELS}, indent=1))
for k in ["recall", "f1", "roc_auc"]:
    for r_ in tests[k]:
        print(k, r_["a"], r_["b"], round(r_["mean_diff"], 3), r_["wins"], r_["losses"], round(r_["p_holm"], 3))
print({k: v for k, v in search.items() if k != "rs"}, search["rs"]["n_zero"], search["rs"]["selected_draw"],
      round(search["rs"]["seconds"] / 60, 1))
