"""Shortcut checks on the user-grouped split, quoted in the leakage audit (Section 4.2).

All models are fitted on train_grouped.csv and scored once on test_grouped.csv:
  * TF-IDF (5,000 uni- and bigrams) + logistic regression
  * metadata only (the 12 request features) + logistic regression and + XGBoost
  * TF-IDF + logistic regression after deleting the 80 tokens with the largest absolute
    coefficients of the first model from every text
Writes results/grouped_split_checks.json."""
import json
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

BASE = os.path.dirname(__file__)
D = os.path.join(BASE, "..", "data_processed")
tr = pd.read_csv(os.path.join(D, "train_grouped.csv")).fillna("")
te = pd.read_csv(os.path.join(D, "test_grouped.csv")).fillna("")
meta = [c for c in tr.columns if c not in ("text", "label", "user", "scenario")]
out = {}


def score(name, y, p):
    out[name] = {"accuracy": float(accuracy_score(y, p)), "f1": float(f1_score(y, p))}
    print(name, out[name])


vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
Xtr, Xte = vec.fit_transform(tr["text"]), vec.transform(te["text"])
lr = LogisticRegression(max_iter=2000).fit(Xtr, tr["label"])
score("tfidf_logreg", te["label"], lr.predict(Xte))

score("meta_logreg", te["label"], LogisticRegression(max_iter=2000).fit(tr[meta], tr["label"]).predict(te[meta]))
score("meta_xgboost", te["label"],
      XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42)
      .fit(tr[meta], tr["label"]).predict(te[meta]))

names = vec.get_feature_names_out()
order = abs(lr.coef_[0]).argsort()[::-1]
top = []
for i in order:  # unigrams only, so every removed item is a single token
    if " " not in names[i]:
        top.append(names[i])
    if len(top) == 80:
        break
pat = re.compile(r"\b(" + "|".join(map(re.escape, top)) + r")\b")
mtr, mte = tr["text"].str.replace(pat, " ", regex=True), te["text"].str.replace(pat, " ", regex=True)
vec2 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
lr2 = LogisticRegression(max_iter=2000).fit(vec2.fit_transform(mtr), tr["label"])
score("tfidf_logreg_mask80", te["label"], lr2.predict(vec2.transform(mte)))
out["masked_tokens"] = top
json.dump(out, open(os.path.join(BASE, "..", "results", "grouped_split_checks.json"), "w"), indent=2)
