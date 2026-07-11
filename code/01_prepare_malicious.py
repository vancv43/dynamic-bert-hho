"""Extract malicious HTTP rows from CERT r4.2 insider answer files (ground truth)."""
import csv
import glob
import json
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "data_raw", "answers", "answers")
OUT = os.path.join(os.path.dirname(__file__), "..", "data_processed")
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(BASE, "r4.2-*", "*.csv")))
print(f"insider incident files: {len(files)}")

rows = []
for f in files:
    scenario_dir = os.path.basename(os.path.dirname(f))
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.startswith("http,"):
                continue
            parts = line.split(",", 5)
            if len(parts) < 6:
                continue
            _, event_id, date, user, pc, rest = parts
            if "," in rest:
                url, content = rest.split(",", 1)
            else:
                url, content = rest, ""
            rows.append({
                "event_id": event_id,
                "date": date,
                "user": user,
                "pc": pc,
                "url": url,
                "content": content,
                "scenario": scenario_dir,
                "label": 1,
            })

print(f"malicious http rows: {len(rows)}")
with open(os.path.join(OUT, "malicious_http.jsonl"), "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("saved to data_processed/malicious_http.jsonl")
