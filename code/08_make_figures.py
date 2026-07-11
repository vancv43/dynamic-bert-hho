"""Generate all real figures (PNG) from the saved experiment results."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
RESULTS = os.path.join(BASE, "..", "results")
FIGDIR = os.path.join(BASE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"font.size": 10})


def load_json(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def main():
    baseline = load_json("baseline_metrics.json")
    final = load_json("final_model_metrics.json")
    hho_search = load_json("hho_search_result.json")

    models = ["tfidf_xgboost", "word2vec_xgboost", "bert_no_hpo", "bert_manual_tuning", "dynamic_bert_hho"]
    labels = ["TF-IDF+\nXGBoost", "Word2Vec+\nXGBoost", "BERT", "BERT+\nManual Tuning", "Dynamic\nBERT-HHO"]
    metrics_all = {**baseline, **{k: v["metrics"] for k, v in final.items()}}

    accs = [metrics_all[m]["accuracy"] for m in models]

    # Fig: accuracy comparison bar chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, accs, color=["#888888", "#888888", "#6699cc", "#6699cc", "#cc5533"])
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.002, f"{a:.4f}", ha="center", fontsize=8)
    plt.ylabel("Accuracy")
    plt.ylim(min(accs) - 0.02, max(accs) + 0.02)
    plt.title("Accuracy comparison among baseline models and Dynamic BERT-HHO")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_accuracy_comparison.png"), dpi=150)
    plt.close()

    # Fig: HHO convergence curve
    history = hho_search["history"]
    fitness_curve = [1 - h for h in history]  # convert back to "score" for readability
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(fitness_curve)), fitness_curve, marker="o")
    plt.xlabel("HHO iteration")
    plt.ylabel("Best weighted fitness score (Acc/F1/Kappa)")
    plt.title("HHO convergence curve during BERT hyperparameter tuning")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_hho_convergence.png"), dpi=150)
    plt.close()

    # Fig: confusion matrix (final Dynamic BERT-HHO model)
    y_true = pd.read_csv(os.path.join(RESULTS, "test_labels_final.csv"))["label"].values
    y_pred = np.load(os.path.join(RESULTS, "bert_hho_test_pred.npy"))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(4.2, 4))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i,j]}\n{cm_norm[i,j]:.1f}%", ha="center", va="center",
                      color="white" if cm_norm[i, j] > 50 else "black", fontsize=9)
    plt.xticks([0, 1], ["Normal", "Threat"])
    plt.yticks([0, 1], ["Normal", "Threat"])
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Confusion matrix: Dynamic BERT-HHO")
    plt.colorbar(label="Row-normalised (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_confusion_matrix.png"), dpi=150)
    plt.close()

    # Fig: ROC curves
    plt.figure(figsize=(6, 5))
    prob_files = {
        "TF-IDF+XGBoost": "tfidf_xgb_test_prob.npy",
        "Word2Vec+XGBoost": "word2vec_xgb_test_prob.npy",
        "BERT": "bert_no_hpo_test_prob.npy",
        "Dynamic BERT-HHO": "bert_hho_test_prob.npy",
    }
    roc_summary = {}
    for name, fname in prob_files.items():
        prob = np.load(os.path.join(RESULTS, fname))
        fpr, tpr, _ = roc_curve(y_true, prob)
        plt.plot(fpr, tpr, label=name)
        roc_summary[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_roc_curve.png"), dpi=150)
    plt.close()

    # Fig: Precision-Recall curves
    plt.figure(figsize=(6, 5))
    for name, fname in prob_files.items():
        prob = np.load(os.path.join(RESULTS, fname))
        prec, rec, _ = precision_recall_curve(y_true, prob)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve comparison")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_pr_curve.png"), dpi=150)
    plt.close()

    # Fig: SHAP summary bar chart
    shap_top = load_json("shap_top_tokens.json")["top_tokens"][:10]
    toks = [t[0] for t in shap_top][::-1]
    vals = [t[1] for t in shap_top][::-1]
    plt.figure(figsize=(6, 4.5))
    plt.barh(toks, vals, color="#cc5533")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top tokens by mean |SHAP value| (Dynamic BERT-HHO)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_shap_summary.png"), dpi=150)
    plt.close()

    print("all figures saved to", FIGDIR)

    # dump a consolidated summary for LaTeX table generation
    summary = {
        "baseline_metrics": baseline,
        "final_metrics": {k: v["metrics"] for k, v in final.items()},
        "final_hp": {k: v["hp"] for k, v in final.items()},
        "final_train_seconds": {k: v["train_seconds"] for k, v in final.items()},
        "hho_search": {k: v for k, v in hho_search.items() if k != "history"},
        "confusion_matrix": cm.tolist(),
        "shap_top_tokens": shap_top,
    }
    with open(os.path.join(RESULTS, "summary_for_paper.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
