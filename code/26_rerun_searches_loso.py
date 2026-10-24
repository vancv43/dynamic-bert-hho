# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Leakage-free rerun of all hyperparameter searches on the LOSO split (second-round revision).

Why: the first searches (05_run_hho_search.py, 18_run_random_search.py) drew their 2,000/600 subsample
from the original row-level split, which shares 186 rows (30 malicious, 21 users) with the LOSO test set.

Protocol (locked before running, unchanged except for the data source):
  data     : 2,000 rows of train_loso.csv and 600 rows of val_loso.csv, random_state=42
             (scenarios r4.2-2 and r4.2-3 plus grouped benign users; no LOSO test row)
  space    : hho.BOUNDS for BERT, META_BOUNDS for the fusion head
  fitness  : L = 1 - (0.4 Acc + 0.4 F1 + 0.2 max(kappa, 0)), minimized
  stage 1  : HHO, 6 hawks, 4 iterations, seed 42           -> hho_search_result.json
  control  : random search, 51 evaluations, default_rng(42)  -> random_search_result.json
  stage 2  : HHO over fusion width and dropout, 4 hawks, 2 iterations, seed 42, on the new
             stage-1 BERT settings                           -> hho_meta_search_result.json
Outputs go to results/rerun_loso/; earlier results are left untouched. Stages already written
are skipped, so the script can be resumed."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import train_and_eval
from common_meta import train_and_eval_meta
from hho import hho_optimize, KEYS, LB, UB

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data_processed")
OUT = os.path.join(BASE, "..", "results", "rerun_loso")
os.makedirs(OUT, exist_ok=True)

ALPHA, BETA, GAMMA = 0.4, 0.4, 0.2
N_SEARCH_TRAIN, N_SEARCH_VAL, N_RS = 2000, 600, 51
META_BOUNDS = {"fusion_hidden": (8, 128), "fusion_dropout": (0.0, 0.5)}


def loss(m):
    return 1 - (ALPHA * m["accuracy"] + BETA * m["f1"] + GAMMA * max(m["kappa"], 0))


def main():
    train = pd.read_csv(os.path.join(DATA, "train_loso.csv")).fillna("")
    val = pd.read_csv(os.path.join(DATA, "val_loso.csv")).fillna("")
    test = pd.read_csv(os.path.join(DATA, "test_loso.csv")).fillna("")
    meta_cols = [c for c in train.columns if c not in ("text", "label", "user", "scenario")]
    tr = train.sample(n=min(N_SEARCH_TRAIN, len(train)), random_state=42).reset_index(drop=True)
    va = val.sample(n=min(N_SEARCH_VAL, len(val)), random_state=42).reset_index(drop=True)
    overlap = int(pd.concat([tr, va])["text"].isin(set(test["text"])).sum())
    print(f"search subsample {len(tr)}/{len(va)}, positives {tr.label.mean():.3f}/{va.label.mean():.3f}, "
          f"rows identical to a LOSO test row: {overlap}", flush=True)

    # ---------------- stage 1: HHO over the six BERT settings
    p1 = os.path.join(OUT, "hho_search_result.json")
    if not os.path.exists(p1):
        evals, t0 = [], time.time()

        def fit_bert(hp):
            _, m, _, _ = train_and_eval(tr["text"], tr["label"], va["text"], va["label"], hp)
            f = loss(m)
            evals.append({"hp": hp, "fitness": f, "metrics": m, "elapsed": time.time() - t0})
            print(f"  HHO eval#{len(evals)} fit={f:.4f} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                  f"kappa={m['kappa']:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
            return f

        best_hp, best_fit, history = hho_optimize(fit_bert, n_hawks=6, n_iter=4, seed=42,
                                                  log=lambda s: print("[HHO]", s, flush=True))
        json.dump({"best_hp": best_hp, "best_fitness": best_fit, "history": history, "evaluations": evals,
                   "n_evaluations": len(evals), "search_train_size": len(tr), "search_val_size": len(va),
                   "test_overlap_rows": overlap, "elapsed_seconds": time.time() - t0}, open(p1, "w"), indent=2)
        print("STAGE1_DONE", best_hp, best_fit, flush=True)
    hho_hp = json.load(open(p1))["best_hp"]

    # ---------------- equal-budget random search
    p2 = os.path.join(OUT, "random_search_result.json")
    if not os.path.exists(p2):
        rng = np.random.default_rng(42)
        best_hp, best_fit, history, evals, t0 = None, np.inf, [], [], time.time()
        for i in range(N_RS):
            hp = dict(zip(KEYS, (LB + rng.uniform(0, 1, size=len(KEYS)) * (UB - LB)).tolist()))
            _, m, _, _ = train_and_eval(tr["text"], tr["label"], va["text"], va["label"], hp)
            f = loss(m)
            if f < best_fit:
                best_fit, best_hp = f, hp
            history.append(best_fit)
            evals.append({"hp": hp, "fitness": f, "metrics": m, "elapsed": time.time() - t0})
            print(f"  RS eval#{i+1} fit={f:.4f} best={best_fit:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        json.dump({"best_hp": best_hp, "best_fitness": best_fit, "history": history, "evaluations": evals,
                   "n_evaluations": N_RS, "search_train_size": len(tr), "search_val_size": len(va),
                   "test_overlap_rows": overlap, "elapsed_seconds": time.time() - t0}, open(p2, "w"), indent=2)
        print("RS_DONE", best_hp, best_fit, flush=True)

    # ---------------- stage 2: HHO over the fusion head on the new BERT settings
    p3 = os.path.join(OUT, "hho_meta_search_result.json")
    if not os.path.exists(p3):
        evals, t0 = [], time.time()

        def fit_fusion(fhp):
            hp = {**hho_hp, **fhp}
            _, m, _, _ = train_and_eval_meta(tr["text"], tr[meta_cols].values, tr["label"],
                                             va["text"], va[meta_cols].values, va["label"], hp)
            f = loss(m)
            evals.append({"hp": fhp, "fitness": f, "metrics": m, "elapsed": time.time() - t0})
            print(f"  META eval#{len(evals)} fit={f:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
            return f

        best, best_fit, history = hho_optimize(fit_fusion, n_hawks=4, n_iter=2, seed=42, bounds=META_BOUNDS,
                                               log=lambda s: print("[HHO-Meta]", s, flush=True))
        json.dump({"best_fusion_hp": best, "best_fitness": best_fit, "history": history, "evaluations": evals,
                   "n_evaluations": len(evals), "elapsed_seconds": time.time() - t0,
                   "base_bert_hp_reused": hho_hp}, open(p3, "w"), indent=2)
        print("STAGE2_DONE", best, best_fit, flush=True)
    print("SEARCHES_DONE", flush=True)


if __name__ == "__main__":
    main()
