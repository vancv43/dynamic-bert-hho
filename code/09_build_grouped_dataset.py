"""Build a user-grouped (leakage-free) train/val/test split with structural/behavioural
metadata features, for the Dynamic BERT-HHO-Meta revision.

Motivation (see manuscript revision, Section: User-Grouped Evaluation Protocol):
the original row-level stratified split (02_extract_http_sample.py + 03_build_dataset.py)
allows the same insider user's repeated incident text to appear in both train and test
(43 of 68 malicious users in the original train set also appear in the original test set;
malicious users average ~40 rows/user in train vs ~1.4 rows/user for benign users). This
inflates held-out accuracy for every model. This script instead groups by `user` so that no
user's HTTP rows span more than one split, and additionally extracts structural/behavioural
metadata features (content length, digit ratio, URL structure, time-of-day) that avoid
re-encoding host or user identity (which would simply reintroduce the same leakage in a
different form; see diagnostic notes below and Experimental Setup revision).
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "data_processed")
RESULTS = os.path.join(BASE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def clean_text(url, content):
    text = f"{url} {content}"
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://", " ", text)
    text = re.sub(r"[^a-zA-Z0-9./_\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def host_of(url):
    m = re.search(r"https?://([^/]+)", url)
    return m.group(1) if m else ""


def tld_of(host):
    if not host:
        return "other"
    ext = host.split(".")[-1].lower()
    return ext if ext in {"com", "org", "net", "gov"} else "other"


def parse_hour_dow(date_str):
    try:
        # format: MM/DD/YYYY HH:MM:SS
        date_part, time_part = date_str.split(" ")
        mm, dd, yyyy = [int(x) for x in date_part.split("/")]
        hh = int(time_part.split(":")[0])
        import datetime
        dow = datetime.date(yyyy, mm, dd).weekday()
        return hh, dow
    except Exception:
        return -1, -1


def build_rows(records, label):
    rows = []
    for r in records:
        url, content = r.get("url", ""), r.get("content", "")
        text = clean_text(url, content)
        if not text:
            continue
        host = host_of(url)
        hh, dow = parse_hour_dow(r.get("date", ""))
        digit_ratio = (sum(c.isdigit() for c in content) / len(content)) if content else 0.0
        rows.append({
            "text": text,
            "label": label,
            "user": r.get("user", ""),
            "content_len": len(content),
            "content_word_count": len(content.split()),
            "digit_ratio": digit_ratio,
            "url_len": len(url),
            "url_path_depth": url.count("/") - 2 if url.count("/") >= 2 else 0,
            "hour": hh,
            "dow": dow,
            "tld": tld_of(host),
        })
    return rows


def grouped_split(df, seed=42, test_frac=0.30, second_frac=0.5):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["user"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=second_frac, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["user"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def main():
    mal = load_jsonl(os.path.join(OUT, "malicious_http.jsonl"))
    ben = load_jsonl(os.path.join(OUT, "benign_http_sample.jsonl"))

    rows = build_rows(mal, 1) + build_rows(ben, 0)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print("de-duplicated dataset size:", len(df), "positive rate:", df["label"].mean())

    # verify no user spans both classes (sanity check for group split validity)
    user_label_counts = df.groupby("user")["label"].nunique()
    mixed_users = (user_label_counts > 1).sum()
    print("users with rows in both classes:", mixed_users, "/", df["user"].nunique())

    # try a handful of seeds, keep the one whose realized positive rates are closest across splits
    best = None
    for seed in range(10):
        train_df, val_df, test_df = grouped_split(df, seed=seed)
        rates = [train_df["label"].mean(), val_df["label"].mean(), test_df["label"].mean()]
        sizes = [len(train_df), len(val_df), len(test_df)]
        if min(sizes) < 300:  # avoid degenerate tiny splits
            continue
        spread = max(rates) - min(rates)
        if best is None or spread < best[0]:
            best = (spread, seed, train_df, val_df, test_df, rates, sizes)

    spread, seed, train_df, val_df, test_df, rates, sizes = best
    print(f"selected split seed={seed}, sizes={sizes}, positive rates={[round(r, 4) for r in rates]}")

    # user-overlap sanity check across the selected split
    tr_u, va_u, te_u = set(train_df["user"]), set(val_df["user"]), set(test_df["user"])
    print("train/test user overlap (must be 0):", len(tr_u & te_u))
    print("train/val user overlap (must be 0):", len(tr_u & va_u))
    print("val/test user overlap (must be 0):", len(va_u & te_u))

    # standardize numeric metadata features using TRAIN statistics only
    num_cols = ["content_len", "content_word_count", "digit_ratio", "url_len", "url_path_depth", "hour", "dow"]
    mu = train_df[num_cols].mean()
    sigma = train_df[num_cols].std().replace(0, 1.0)
    for split_df in (train_df, val_df, test_df):
        for c in num_cols:
            split_df[c + "_z"] = (split_df[c] - mu[c]) / sigma[c]

    # one-hot tld (fit categories from train)
    tld_cats = sorted(train_df["tld"].unique().tolist())
    for split_df in (train_df, val_df, test_df):
        for cat in tld_cats:
            split_df[f"tld_{cat}"] = (split_df["tld"] == cat).astype(float)

    meta_cols = [c + "_z" for c in num_cols] + [f"tld_{c}" for c in tld_cats]

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        keep = ["text", "label", "user"] + meta_cols
        split_df[keep].to_csv(os.path.join(OUT, f"{name}_grouped.csv"), index=False)

    meta = {
        "meta_cols": meta_cols,
        "tld_categories": tld_cats,
        "selected_seed": seed,
        "sizes": {"train": sizes[0], "val": sizes[1], "test": sizes[2]},
        "positive_rates": {"train": rates[0], "val": rates[1], "test": rates[2]},
        "mixed_label_users": int(mixed_users),
        "note": "grouped by `user`; no user spans more than one split. host/user identity "
                "intentionally excluded from metadata to avoid re-encoding the same leakage.",
    }
    with open(os.path.join(RESULTS, "grouped_split_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("saved grouped split + metadata feature columns:", meta_cols)


if __name__ == "__main__":
    main()
