"""Combine malicious + benign HTTP rows into a cleaned, split classification dataset."""
import json
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "data_processed")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def clean_text(url, content):
    text = f"{url} {content}"
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"https?://", " ", text)         # drop scheme (keep host/path tokens)
    text = re.sub(r"[^a-zA-Z0-9./_\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def main():
    mal = load_jsonl(os.path.join(OUT, "malicious_http.jsonl"))
    ben = load_jsonl(os.path.join(OUT, "benign_http_sample.jsonl"))
    print("malicious:", len(mal), "benign:", len(ben))

    rows = []
    for r in mal:
        rows.append({"text": clean_text(r["url"], r["content"]), "label": 1, "user": r["user"]})
    for r in ben:
        rows.append({"text": clean_text(r["url"], r["content"]), "label": 0, "user": r["user"]})

    df = pd.DataFrame(rows)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print("final dataset size:", len(df), "positive rate:", df["label"].mean())

    # subject-level-ish split: since insiders' benign activity may also appear, do a stratified
    # random split at the row level (paper protocol: 70/15/15 train/val/test)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    train_df.to_csv(os.path.join(OUT, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT, "test.csv"), index=False)

    print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
    print("train pos rate:", train_df["label"].mean(), "val:", val_df["label"].mean(), "test:", test_df["label"].mean())


if __name__ == "__main__":
    main()
