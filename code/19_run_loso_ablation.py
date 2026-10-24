# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Multi-seed LOSO ablation with saved probabilities and timing (second-round revision).

Protocol (locked before running):
  split   : LOSO (train r4.2-2 + r4.2-3, test unseen r4.2-1), files *_loso.csv
  seeds   : 1..10, every model retrained from scratch per seed
  models  : TF-IDF+XGBoost, Word2Vec+XGBoost,
            BERT-default (no HPO), BERT-manual, BERT-RS (equal-budget random search),
            BERT-HHO (Dynamic BERT-HHO),
            BERT-default+Meta (default fusion 32/0.2, no HHO anywhere),
            BERT-HHO+Meta (HHO backbone, default fusion 32/0.2),
            BERT-HHO-Meta (HHO backbone + HHO fusion = Dynamic BERT-HHO-Meta)
  output  : per-seed test probabilities (results/loso_ablation/*.npy), metrics at the fixed
            0.5 threshold, training wall time and test inference time per model/seed.
No threshold, hyperparameter or seed is chosen from test results. Runs are resumable:
a (model, seed) pair whose probability file exists is skipped."""
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from common import build_model as build_text_model, get_tokenizer, HttpTextDataset, evaluate as evaluate_text, DEVICE
from common_meta import build_model as build_meta_model, HttpMetaDataset, evaluate as evaluate_meta, get_tokenizer as get_tok_meta

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "data_processed")
RESULTS = os.path.join(BASE, "..", "results")
# REVISION_RESULTS selects a rerun folder (e.g. rerun_loso) holding its own search results
HP_DIR = os.path.join(RESULTS, os.environ.get("REVISION_RESULTS", ""))
ABL = os.path.join(HP_DIR, "loso_ablation")
os.makedirs(ABL, exist_ok=True)

SEEDS = list(range(1, 11))
HP_DEFAULT = {"lr": 5e-5, "batch_size": 16, "dropout": 0.1, "weight_decay": 0.0, "epochs": 1, "trainable_layers": 12}
HP_MANUAL = {"lr": 2e-5, "batch_size": 16, "dropout": 0.1, "weight_decay": 0.01, "epochs": 2, "trainable_layers": 4}
FUSION_DEFAULT = {"fusion_hidden": 32, "fusion_dropout": 0.2}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def metrics_from(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, pred), "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": roc_auc_score(y_true, prob), "kappa": cohen_kappa_score(y_true, pred)}


def run_tfidf_xgb(train, test, seed):
    t0 = time.time()
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(train["text"])
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=seed)
    clf.fit(Xtr, train["label"])
    t_train = time.time() - t0
    t0 = time.time()
    prob = clf.predict_proba(vec.transform(test["text"]))[:, 1]
    return prob, t_train, time.time() - t0


def doc_vector(tokens, w2v, dim):
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def run_word2vec_xgb(train, test, seed, dim=100):
    t0 = time.time()
    train_tokens = [t.split() for t in train["text"]]
    w2v = Word2Vec(sentences=train_tokens, vector_size=dim, window=5, min_count=1, workers=4, seed=seed)
    Xtr = np.vstack([doc_vector(t, w2v, dim) for t in train_tokens])
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=seed)
    clf.fit(Xtr, train["label"])
    t_train = time.time() - t0
    t0 = time.time()
    Xte = np.vstack([doc_vector(t.split(), w2v, dim) for t in test["text"]])
    prob = clf.predict_proba(Xte)[:, 1]
    return prob, t_train, time.time() - t0


def run_bert_text(train, test, hp):
    tok = get_tokenizer()
    train_ds = HttpTextDataset(train["text"], train["label"], tok)
    model = build_text_model(hp["dropout"], hp["trainable_layers"])
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=hp["weight_decay"])
    loader = torch.utils.data.DataLoader(train_ds, batch_size=int(round(hp["batch_size"])), shuffle=True)
    t0 = time.time()
    model.train()
    for _ in range(max(1, int(round(hp["epochs"])))):
        for b in loader:
            optim.zero_grad()
            out = model(input_ids=b["input_ids"].to(DEVICE), attention_mask=b["attention_mask"].to(DEVICE), labels=b["labels"].to(DEVICE))
            out.loss.backward()
            optim.step()
    sync()
    t_train = time.time() - t0
    t0 = time.time()
    test_ds = HttpTextDataset(test["text"], test["label"], tok)
    _, prob, _ = evaluate_text(model, test_ds)
    sync()
    return np.array(prob), t_train, time.time() - t0


def run_bert_meta(train, test, meta_cols, hp):
    tok = get_tok_meta()
    train_ds = HttpMetaDataset(train["text"], train[meta_cols].values, train["label"], tok)
    model = build_meta_model(hp["dropout"], hp["trainable_layers"], len(meta_cols), hp["fusion_hidden"], hp["fusion_dropout"])
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(train_ds, batch_size=int(round(hp["batch_size"])), shuffle=True)
    t0 = time.time()
    model.train()
    for _ in range(max(1, int(round(hp["epochs"])))):
        for b in loader:
            optim.zero_grad()
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["meta"].to(DEVICE))
            loss_fn(logits, b["labels"].to(DEVICE)).backward()
            optim.step()
    sync()
    t_train = time.time() - t0
    t0 = time.time()
    test_ds = HttpMetaDataset(test["text"], test[meta_cols].values, test["label"], tok)
    _, prob, _ = evaluate_meta(model, test_ds)
    sync()
    return np.array(prob), t_train, time.time() - t0


def main():
    train = pd.read_csv(os.path.join(OUT, "train_loso.csv")).fillna("")
    test = pd.read_csv(os.path.join(OUT, "test_loso.csv")).fillna("")
    meta_cols = [c for c in train.columns if c not in ("text", "label", "user", "scenario")]
    y = test["label"].values
    np.save(os.path.join(ABL, "test_labels.npy"), y)

    with open(os.path.join(HP_DIR, "hho_search_result.json")) as fh:
        hp_hho = json.load(fh)["best_hp"]
    with open(os.path.join(HP_DIR, "hho_meta_search_result.json")) as fh:
        fusion_hho = json.load(fh)["best_fusion_hp"]
    with open(os.path.join(HP_DIR, "random_search_result.json")) as fh:
        hp_rs = json.load(fh)["best_hp"]

    models = {
        "tfidf_xgboost": lambda s: run_tfidf_xgb(train, test, s),
        "word2vec_xgboost": lambda s: run_word2vec_xgb(train, test, s),
        "bert_default": lambda s: run_bert_text(train, test, HP_DEFAULT),
        "bert_manual": lambda s: run_bert_text(train, test, HP_MANUAL),
        "bert_rs": lambda s: run_bert_text(train, test, hp_rs),
        "bert_hho": lambda s: run_bert_text(train, test, hp_hho),
        "bert_default_meta": lambda s: run_bert_meta(train, test, meta_cols, {**HP_DEFAULT, **FUSION_DEFAULT}),
        "bert_hho_meta_defaultfusion": lambda s: run_bert_meta(train, test, meta_cols, {**hp_hho, **FUSION_DEFAULT}),
        "bert_hho_meta": lambda s: run_bert_meta(train, test, meta_cols, {**hp_hho, **fusion_hho}),
    }

    raw_path = os.path.join(ABL, "raw_runs.json")
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    for seed in SEEDS:
        for name, fn in models.items():
            pf = os.path.join(ABL, f"{name}_seed{seed}.npy")
            if os.path.exists(pf):
                continue
            set_seed(seed)
            prob, t_train, t_inf = fn(seed)
            np.save(pf, prob)
            m = metrics_from(y, prob)
            raw.setdefault(name, {})[str(seed)] = {"metrics": m, "train_seconds": t_train,
                                                   "infer_seconds": t_inf, "n_test": int(len(y))}
            with open(raw_path, "w") as fh:
                json.dump(raw, fh, indent=2)
            print(f"[seed {seed}] {name}: rec={m['recall']:.3f} prec={m['precision']:.3f} f1={m['f1']:.3f} "
                  f"auc={m['roc_auc']:.4f} train={t_train:.0f}s infer={t_inf:.2f}s", flush=True)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
