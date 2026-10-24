# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Final statistics and tables of the study, on the leakage-free rerun.

Reads results/rerun_loso (searches on the LOSO split, ten-seed LOSO ablation, five independent
random searches). Writes results/final_summary.json.
The random-search control is pooled over five searches (seeds 1-5) x ten training seeds = 50 runs.
The seed-42 search shares its random stream with the HHO initialization and selects the HHO
configuration, so it is not used as a control. Intervals come from a two-level bootstrap that
resamples runs and then test rows (10,000 replicates, rng 2026). Paired two-sided Wilcoxon tests run
over the ten training seeds (the random-search value of a seed is its mean over the five searches),
Holm-corrected within each metric."""
import json
import os

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, cohen_kappa_score

BASE = os.path.dirname(__file__)
R = os.path.join(BASE, "..", "results", "rerun_loso")
ABL = os.path.join(R, "loso_ablation")
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


print(json.dumps({m: {k: round(summary[m][k]["mean"], 3) for k in ["precision", "recall", "f1", "roc_auc"]}
                  for m, _ in MODELS}, indent=1))
for k in ["recall", "f1", "roc_auc"]:
    for r_ in tests[k]:
        print(k, r_["a"], r_["b"], round(r_["mean_diff"], 3), r_["wins"], r_["losses"], round(r_["p_holm"], 3))
print({k: v for k, v in search.items() if k != "rs"}, search["rs"]["n_zero"], search["rs"]["selected_draw"],
      round(search["rs"]["seconds"] / 60, 1))
