"""Train the final BERT (manual-tuning baseline) and BERT-HHO (best-found config) models
on the full training set and evaluate on the held-out test set."""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from common import train_and_eval

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "data_processed")
RESULTS = os.path.join(BASE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# Reasonable manually chosen configuration (BERT + Manual Tuning baseline)
MANUAL_HP = {
    "lr": 2e-5, "batch_size": 16, "dropout": 0.1, "weight_decay": 0.01,
    "epochs": 2, "trainable_layers": 4,
}


def main():
    train = pd.read_csv(os.path.join(OUT, "train.csv")).fillna("")
    test = pd.read_csv(os.path.join(OUT, "test.csv")).fillna("")

    with open(os.path.join(RESULTS, "hho_search_result.json")) as fh:
        hho_result = json.load(fh)
    best_hp = hho_result["best_hp"]

    results = {}

    print("training BERT (no HPO, fixed BERT-base defaults, 1 epoch, all layers)...")
    t0 = time.time()
    default_hp = {"lr": 5e-5, "batch_size": 16, "dropout": 0.1, "weight_decay": 0.0, "epochs": 1, "trainable_layers": 12}
    model_bert, m_bert, prob_bert, pred_bert = train_and_eval(
        train["text"], train["label"], test["text"], test["label"], default_hp, verbose=True,
    )
    print("BERT (no HPO):", m_bert, f"({time.time()-t0:.0f}s)")
    results["bert_no_hpo"] = {"metrics": m_bert, "hp": default_hp, "train_seconds": time.time() - t0}
    np.save(os.path.join(RESULTS, "bert_no_hpo_test_prob.npy"), np.array(prob_bert))

    print("training BERT + Manual Tuning...")
    t0 = time.time()
    model_manual, m_manual, prob_manual, pred_manual = train_and_eval(
        train["text"], train["label"], test["text"], test["label"], MANUAL_HP, verbose=True,
    )
    print("BERT + Manual Tuning:", m_manual, f"({time.time()-t0:.0f}s)")
    results["bert_manual_tuning"] = {"metrics": m_manual, "hp": MANUAL_HP, "train_seconds": time.time() - t0}
    np.save(os.path.join(RESULTS, "bert_manual_test_prob.npy"), np.array(prob_manual))

    print("training Dynamic BERT-HHO (best HHO config) on full train set...")
    t0 = time.time()
    model_hho, m_hho, prob_hho, pred_hho = train_and_eval(
        train["text"], train["label"], test["text"], test["label"], best_hp, verbose=True,
    )
    train_seconds_hho = time.time() - t0
    print("Dynamic BERT-HHO:", m_hho, f"({train_seconds_hho:.0f}s)")
    results["dynamic_bert_hho"] = {"metrics": m_hho, "hp": best_hp, "train_seconds": train_seconds_hho}
    np.save(os.path.join(RESULTS, "bert_hho_test_prob.npy"), np.array(prob_hho))
    np.save(os.path.join(RESULTS, "bert_hho_test_pred.npy"), np.array(pred_hho))

    test["label"].to_csv(os.path.join(RESULTS, "test_labels_final.csv"), index=False)

    with open(os.path.join(RESULTS, "final_model_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # save the final HHO model + tokenizer for the SHAP step
    save_dir = os.path.join(RESULTS, "final_bert_hho_model")
    model_hho.save_pretrained(save_dir)
    from common import get_tokenizer
    get_tokenizer().save_pretrained(save_dir)

    print("all final models trained and saved.")


if __name__ == "__main__":
    main()
