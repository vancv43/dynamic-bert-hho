# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Independent random-search controls for the HHO search (second-round revision).

Why: 26_rerun_searches_loso.py seeds the random search with default_rng(42), the same stream that
initializes the HHO hawks, so random draw k equals initial hawk k and both tuners can select the
same configuration. This script adds random searches whose streams are independent of HHO.

Protocol (locked before running):
  data      : the LOSO search subsample of 26_rerun_searches_loso.py (2,000 train_loso / 600 val_loso
              rows, random_state=42); no LOSO test row
  space     : hho.BOUNDS; fitness L = 1 - (0.4 Acc + 0.4 F1 + 0.2 max(kappa, 0)), minimized
  searches  : seeds 1, 2, 3, 4, 5; 51 uniform draws each; the first configuration with the lowest
              loss is kept (same rule as the seed-42 control)
  evaluation: each selected configuration is retrained on the full LOSO training set with training
              seeds 1..10 and scored on the LOSO test set at threshold 0.5, exactly as the models of
              19_run_loso_ablation.py (model names bert_rs_s1 .. bert_rs_s5)
Resumable: finished searches and (model, seed) runs are skipped."""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import train_and_eval
from hho import KEYS, LB, UB

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data_processed")
OUT = os.path.join(BASE, "..", "results", "rerun_loso")
RSD = os.path.join(OUT, "rs_seeds")
ABL = os.path.join(OUT, "loso_ablation")
os.makedirs(RSD, exist_ok=True)
RS_SEEDS = [1, 2, 3, 4, 5]
TRAIN_SEEDS = list(range(1, 11))

spec = importlib.util.spec_from_file_location("abl", os.path.join(BASE, "19_run_loso_ablation.py"))
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)


def main():
    train = pd.read_csv(os.path.join(DATA, "train_loso.csv")).fillna("")
    val = pd.read_csv(os.path.join(DATA, "val_loso.csv")).fillna("")
    test = pd.read_csv(os.path.join(DATA, "test_loso.csv")).fillna("")
    tr = train.sample(n=2000, random_state=42).reset_index(drop=True)
    va = val.sample(n=600, random_state=42).reset_index(drop=True)

    for rs in RS_SEEDS:
        p = os.path.join(RSD, f"random_search_seed{rs}.json")
        if os.path.exists(p):
            continue
        rng = np.random.default_rng(rs)
        best_hp, best_fit, evals, t0 = None, np.inf, [], time.time()
        for i in range(51):
            hp = dict(zip(KEYS, (LB + rng.uniform(0, 1, size=len(KEYS)) * (UB - LB)).tolist()))
            _, m, _, _ = train_and_eval(tr["text"], tr["label"], va["text"], va["label"], hp)
            f = 1 - (0.4 * m["accuracy"] + 0.4 * m["f1"] + 0.2 * max(m["kappa"], 0))
            if f < best_fit:
                best_fit, best_hp = f, hp
            evals.append({"hp": hp, "fitness": f, "metrics": m, "elapsed": time.time() - t0})
            print(f"  RS{rs} eval#{i+1} fit={f:.4f} best={best_fit:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        json.dump({"rs_seed": rs, "best_hp": best_hp, "best_fitness": best_fit, "evaluations": evals,
                   "n_evaluations": 51, "elapsed_seconds": time.time() - t0}, open(p, "w"), indent=2)
        print(f"RS_SEED_DONE {rs}", best_hp, best_fit, flush=True)

    y = test["label"].values
    raw_path = os.path.join(ABL, "raw_runs.json")
    raw = json.load(open(raw_path))
    for rs in RS_SEEDS:
        hp = json.load(open(os.path.join(RSD, f"random_search_seed{rs}.json")))["best_hp"]
        name = f"bert_rs_s{rs}"
        for s in TRAIN_SEEDS:
            pf = os.path.join(ABL, f"{name}_seed{s}.npy")
            if os.path.exists(pf):
                continue
            abl.set_seed(s)
            prob, t_train, t_inf = abl.run_bert_text(train, test, hp)
            np.save(pf, prob)
            m = abl.metrics_from(y, prob)
            raw = json.load(open(raw_path))  # re-read in case another writer finished meanwhile
            raw.setdefault(name, {})[str(s)] = {"metrics": m, "train_seconds": t_train, "infer_seconds": t_inf,
                                                "n_test": int(len(y))}
            json.dump(raw, open(raw_path, "w"), indent=2)
            print(f"[seed {s}] {name}: rec={m['recall']:.3f} prec={m['precision']:.3f} f1={m['f1']:.3f} "
                  f"auc={m['roc_auc']:.4f} train={t_train:.0f}s", flush=True)
    print("RS_SEEDS_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
