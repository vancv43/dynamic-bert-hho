"""Stream http.csv directly out of archive.zip (Kaggle mirror of CERT r4.2) without ever
extracting the full ~84GB file to disk, and take a random sample of benign rows to pair
with the ground-truth malicious rows already parsed from the CMU answer files."""
import csv
import io
import json
import os
import subprocess
import random
import sys

BASE = os.path.dirname(__file__)
RAW = os.path.join(BASE, "..", "data_raw")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "..", "data_processed")
os.makedirs(OUT, exist_ok=True)

ZIP_PATH = os.path.join(ROOT, "archive.zip")
N_BENIGN = 20000


def main():
    malicious_ids = set()
    malicious_users = set()
    with open(os.path.join(OUT, "malicious_http.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            malicious_ids.add(rec["event_id"])
            malicious_users.add(rec["user"])
    print("malicious ids to exclude:", len(malicious_ids), "malicious users:", len(malicious_users))

    proc = subprocess.Popen(
        ["unzip", "-p", ZIP_PATH, "http.csv"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    text_stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
    reader = csv.reader(text_stream)
    header = next(reader, None)
    print("header:", header)
    # expected: id,date,user,pc,url,activity,content

    random.seed(42)
    reservoir = []
    seen = 0
    total_rows = 0
    for row in reader:
        total_rows += 1
        if len(row) < 5:
            continue
        event_id = row[0]
        user = row[2] if len(row) > 2 else ""
        if event_id in malicious_ids:
            continue
        seen += 1
        rec = {
            "event_id": event_id,
            "date": row[1] if len(row) > 1 else "",
            "user": user,
            "pc": row[3] if len(row) > 3 else "",
            "url": row[4] if len(row) > 4 else "",
            "activity": row[5] if len(row) > 5 else "",
            "content": row[6] if len(row) > 6 else "",
            "label": 0,
        }
        if len(reservoir) < N_BENIGN:
            reservoir.append(rec)
        else:
            j = random.randint(0, seen - 1)
            if j < N_BENIGN:
                reservoir[j] = rec
        if total_rows % 2000000 == 0:
            print(f"  scanned {total_rows} rows, sampled pool size {len(reservoir)}", flush=True)

    proc.stdout.close()
    proc.wait()

    print("total http rows scanned:", total_rows)
    print("non-malicious rows seen:", seen)
    print("sampled benign rows:", len(reservoir))

    with open(os.path.join(OUT, "benign_http_sample.jsonl"), "w", encoding="utf-8") as fh:
        for r in reservoir:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved benign_http_sample.jsonl")


if __name__ == "__main__":
    main()
