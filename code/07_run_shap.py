# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""SHAP explainability analysis on the final Dynamic BERT-HHO model."""
import json
import os

import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from common import DEVICE

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "data_processed")
RESULTS = os.path.join(BASE, "..", "results")


def main():
    model_dir = os.path.join(RESULTS, "final_bert_hho_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    pipe = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, device=0 if DEVICE.type != "cpu" else -1,
        top_k=None, truncation=True, max_length=96,
    )

    test = pd.read_csv(os.path.join(OUT, "test.csv")).fillna("")
    # balanced small sample for tractable SHAP computation
    pos = test[test["label"] == 1].sample(n=min(20, (test["label"] == 1).sum()), random_state=42)
    neg = test[test["label"] == 0].sample(n=min(20, (test["label"] == 0).sum()), random_state=42)
    sample = pd.concat([pos, neg]).reset_index(drop=True)

    explainer = shap.Explainer(pipe)
    shap_values = explainer(sample["text"].tolist())

    # mean |SHAP| per token, aggregated across the sample, for the "insider threat" (label 1) output
    label_names = [str(x) for x in pipe.model.config.id2label.values()]
    threat_idx = 1 if len(label_names) > 1 else 0

    token_scores = {}
    for i in range(len(shap_values)):
        vals = shap_values[i].values[:, threat_idx] if shap_values[i].values.ndim > 1 else shap_values[i].values
        toks = shap_values[i].data
        for tok, val in zip(toks, vals):
            tok_clean = tok.strip()
            if not tok_clean or tok_clean in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            token_scores.setdefault(tok_clean, []).append(abs(float(val)))

    agg = [(tok, float(np.mean(v)), len(v)) for tok, v in token_scores.items() if len(v) >= 2]
    agg.sort(key=lambda x: -x[1])
    top15 = agg[:15]

    with open(os.path.join(RESULTS, "shap_top_tokens.json"), "w") as fh:
        json.dump({"top_tokens": top15}, fh, indent=2)
    print("top SHAP tokens:", top15)

    # save one illustrative waterfall example (highest-confidence true-positive)
    pos_texts = pos["text"].tolist()
    if pos_texts:
        example = pos_texts[0]
        ex_shap = explainer([example])
        vals = ex_shap[0].values[:, threat_idx] if ex_shap[0].values.ndim > 1 else ex_shap[0].values
        toks = ex_shap[0].data
        pairs = sorted(zip(toks, vals), key=lambda x: -abs(x[1]))[:8]
        with open(os.path.join(RESULTS, "shap_waterfall_example.json"), "w") as fh:
            json.dump({
                "text": example,
                "base_value": float(ex_shap[0].base_values[threat_idx] if np.ndim(ex_shap[0].base_values) else ex_shap[0].base_values),
                "top_contributions": [(t, float(v)) for t, v in pairs],
            }, fh, indent=2)

    print("SHAP analysis complete.")


if __name__ == "__main__":
    main()
