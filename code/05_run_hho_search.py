# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Run the HHO hyperparameter search for the BERT classifier on a subsample."""
import json
import os
import time

import numpy as np
import pandas as pd

from common import train_and_eval
from hho import hho_optimize

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "data_processed")
RESULTS = os.path.join(BASE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ALPHA, BETA, GAMMA = 0.4, 0.4, 0.2  # weights for Acc, F1, Kappa (paper Eq. fitness)

# Search subsample size, kept small to limit search cost
N_SEARCH_TRAIN = 2000
N_SEARCH_VAL = 600


def main():
    train = pd.read_csv(os.path.join(OUT, "train.csv")).fillna("")
    val = pd.read_csv(os.path.join(OUT, "val.csv")).fillna("")

    rng = np.random.default_rng(42)
    train_sub = train.sample(n=min(N_SEARCH_TRAIN, len(train)), random_state=42).reset_index(drop=True)
    val_sub = val.sample(n=min(N_SEARCH_VAL, len(val)), random_state=42).reset_index(drop=True)

    eval_count = {"n": 0}
    t0 = time.time()

    def fitness_fn(hp):
        eval_count["n"] += 1
        _, metrics, _, _ = train_and_eval(
            train_sub["text"], train_sub["label"], val_sub["text"], val_sub["label"], hp,
        )
        score = ALPHA * metrics["accuracy"] + BETA * metrics["f1"] + GAMMA * max(metrics["kappa"], 0)
        fit = 1 - score
        elapsed = time.time() - t0
        print(f"  eval#{eval_count['n']} hp={ {k: round(v,5) for k,v in hp.items()} } "
              f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} kappa={metrics['kappa']:.4f} "
              f"fit={fit:.4f} elapsed={elapsed:.0f}s", flush=True)
        return fit

    def log(msg):
        print("[HHO]", msg, flush=True)

    best_hp, best_fit, history = hho_optimize(fitness_fn, n_hawks=6, n_iter=4, seed=42, log=log)

    out = {
        "best_hp": best_hp,
        "best_fitness": best_fit,
        "history": history,
        "n_evaluations": eval_count["n"],
        "search_train_size": len(train_sub),
        "search_val_size": len(val_sub),
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(RESULTS, "hho_search_result.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("HHO search done:", out)


if __name__ == "__main__":
    main()
