# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Baseline 1: TF-IDF + XGBoost. Baseline 2: Word2Vec + XGBoost."""
import json
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
)
from xgboost import XGBClassifier

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "data_processed")
RESULTS = os.path.join(BASE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def metrics_from_preds(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    kappa = cohen_kappa_score(y_true, y_pred)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "kappa": kappa}


def run_tfidf_xgb(train, val, test):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(train["text"])
    Xte = vec.transform(test["text"])
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss")
    clf.fit(Xtr, train["label"])
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return metrics_from_preds(test["label"], pred, prob), prob.tolist(), pred.tolist()


def doc_vector(tokens, w2v, dim):
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


def run_word2vec_xgb(train, val, test, dim=100):
    train_tokens = [t.split() for t in train["text"]]
    test_tokens = [t.split() for t in test["text"]]
    w2v = Word2Vec(sentences=train_tokens, vector_size=dim, window=5, min_count=1, workers=4, seed=42)
    Xtr = np.vstack([doc_vector(t, w2v, dim) for t in train_tokens])
    Xte = np.vstack([doc_vector(t, w2v, dim) for t in test_tokens])
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss")
    clf.fit(Xtr, train["label"])
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return metrics_from_preds(test["label"], pred, prob), prob.tolist(), pred.tolist()


def main():
    train = pd.read_csv(os.path.join(OUT, "train.csv")).fillna("")
    val = pd.read_csv(os.path.join(OUT, "val.csv")).fillna("")
    test = pd.read_csv(os.path.join(OUT, "test.csv")).fillna("")

    results = {}
    print("running TF-IDF + XGBoost...")
    m1, prob1, pred1 = run_tfidf_xgb(train, val, test)
    print("TF-IDF+XGBoost:", m1)
    results["tfidf_xgboost"] = m1

    print("running Word2Vec + XGBoost...")
    m2, prob2, pred2 = run_word2vec_xgb(train, val, test)
    print("Word2Vec+XGBoost:", m2)
    results["word2vec_xgboost"] = m2

    with open(os.path.join(RESULTS, "baseline_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    np.save(os.path.join(RESULTS, "tfidf_xgb_test_prob.npy"), np.array(prob1))
    np.save(os.path.join(RESULTS, "word2vec_xgb_test_prob.npy"), np.array(prob2))
    test["label"].to_csv(os.path.join(RESULTS, "test_labels.csv"), index=False)
    print("saved baseline results")


if __name__ == "__main__":
    main()
