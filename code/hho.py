# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Harris Hawks Optimization for BERT hyperparameter search (matches paper Algorithm 1)."""
import math
import numpy as np

# search space bounds, matching Table "Hyperparameter search space" in the manuscript
BOUNDS = {
    "lr": (1e-6, 5e-5),
    "batch_size": (8, 32),
    "dropout": (0.1, 0.5),
    "weight_decay": (0.0, 0.1),
    "epochs": (1, 3),          # reduced upper bound to limit search cost
    "trainable_layers": (2, 8),  # reduced upper bound to limit search cost
}
KEYS = list(BOUNDS.keys())
LB = np.array([BOUNDS[k][0] for k in KEYS])
UB = np.array([BOUNDS[k][1] for k in KEYS])


def levy_flight(dim, beta=1.5):
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.abs(v) ** (1 / beta))


def hp_from_vector(z):
    z = np.clip(z, 0, 1)
    x = LB + z * (UB - LB)
    return dict(zip(KEYS, x.tolist()))


def hho_optimize(fitness_fn, n_hawks=6, n_iter=4, seed=42, log=None, bounds=None):
    """fitness_fn(hp_dict) -> scalar fitness to MINIMIZE (1 - weighted score).
    Returns best_hp, best_fitness, history (per-iteration best fitness).
    `bounds`: optional dict overriding the default BOUNDS (e.g. a reduced search space for
    searching only a subset of hyperparameters while others are held fixed by the caller)."""
    if bounds is None:
        keys, lb, ub, to_hp = KEYS, LB, UB, hp_from_vector
    else:
        keys = list(bounds.keys())
        lb = np.array([bounds[k][0] for k in keys])
        ub = np.array([bounds[k][1] for k in keys])

        def to_hp(z):
            z = np.clip(z, 0, 1)
            x = lb + z * (ub - lb)
            return dict(zip(keys, x.tolist()))

    rng = np.random.default_rng(seed)
    dim = len(keys)
    X = rng.uniform(0, 1, size=(n_hawks, dim))
    fitness = np.array([fitness_fn(to_hp(X[i])) for i in range(n_hawks)])
    best_idx = int(np.argmin(fitness))
    rabbit = X[best_idx].copy()
    rabbit_fit = fitness[best_idx]
    history = [rabbit_fit]
    if log:
        log(f"init best fitness={rabbit_fit:.4f} hp={to_hp(rabbit)}")

    for t in range(n_iter):
        E0 = 2 * rng.uniform(0, 1, n_hawks) - 1
        E = 2 * E0 * (1 - t / n_iter)
        for i in range(n_hawks):
            Xi = X[i]
            if abs(E[i]) >= 1:
                q = rng.uniform()
                if q >= 0.5:
                    rand_idx = rng.integers(0, n_hawks)
                    X_rand = X[rand_idx]
                    r1, r2 = rng.uniform(), rng.uniform()
                    Xi = X_rand - r1 * np.abs(X_rand - 2 * r2 * Xi)
                else:
                    Xm = X.mean(axis=0)
                    r3, r4 = rng.uniform(), rng.uniform()
                    Xi = (rabbit - Xm) - r3 * (0 + r4 * (1 - 0))
            else:
                r = rng.uniform()
                J = 2 * (1 - rng.uniform())
                delta = rabbit - Xi
                if abs(E[i]) >= 0.5:
                    Xi_new = delta - E[i] * np.abs(J * rabbit - Xi)
                else:
                    Xi_new = rabbit - E[i] * np.abs(delta)
                cand_fit = fitness_fn(to_hp(Xi_new))
                if cand_fit >= fitness[i]:
                    lf = levy_flight(dim)
                    if abs(E[i]) >= 0.5:
                        Xi_new = rabbit - E[i] * np.abs(J * rabbit - Xi) + rng.uniform(size=dim) * lf
                    else:
                        Xm = X.mean(axis=0)
                        Xi_new = rabbit - E[i] * np.abs(J * rabbit - Xm) + rng.uniform(size=dim) * lf
                Xi = Xi_new

            Xi = np.clip(Xi, 0, 1)
            new_fit = fitness_fn(to_hp(Xi))
            X[i] = Xi
            fitness[i] = new_fit

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < rabbit_fit:
            rabbit = X[best_idx].copy()
            rabbit_fit = fitness[best_idx]
        history.append(rabbit_fit)
        if log:
            log(f"iter {t+1}/{n_iter} best fitness={rabbit_fit:.4f} hp={to_hp(rabbit)}")

    return to_hp(rabbit), rabbit_fit, history
