"""Machine-learning validation study of Section 5.

Trains two surrogates on noisy observations of the varrho-level endpoints and
checks whether the learned models still satisfy the Hermite-Hadamard inclusion.

Run:  python scripts/run_ml_experiment.py
Writes data/train_samples.csv, data/test_samples.csv,
       results/ml_metrics.csv, results/ml_margins.csv, results/fig_ml.png
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fqp import hh_terms, inclusion_margins

# ------------------------------------------------------------------ settings
SEED, N_TRAIN, N_TEST, SIGMA = 0, 3000, 800, 0.10
XI1, XI2, ETA, MU, GAMMA, LAM, Q = 0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.5
CORE, LEVELS, TOL = 4.0, np.linspace(0, 1, 11), 5e-4
H = lambda z: 1.0 / z if z > 0 else float("inf")
INV_H = lambda z: z

# ground truth = Example A of Section 4
f_low = lambda k, r: (1 - r) * k ** 2 + CORE * r
f_up = lambda k, r: (1 - r) * (9 - k ** 2) + CORE * r


def sample(n, rng):
    k = rng.uniform(XI1, XI2, n)
    r = rng.uniform(0, 1, n)
    return k, r


def wrap(model):
    def g(k, r):
        k = np.atleast_1d(np.asarray(k, float))
        p = model.predict(np.column_stack([k, np.full_like(k, r)]))
        return p[0] if p.size == 1 else p
    return g


def n_params(model):
    if isinstance(model, MLPRegressor):
        return sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    ridge = model.named_steps["ridge"]
    return ridge.coef_.size + 1


def evaluate(g_low, g_up):
    marg, terms = [], []
    for r in LEVELS:
        L, M, R = hh_terms(g_low, g_up, r, XI1, XI2, ETA, MU, GAMMA, LAM, Q, H, INV_H)
        marg.append(min(inclusion_margins(L, M, R)))
        terms.append((L, M, R))
    marg = np.array(marg)
    return marg, int(np.sum(marg >= -TOL)), terms


def main():
    os.makedirs("data", exist_ok=True); os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- Step 1 : noisy observations -------------------------------------
    kt, rt = sample(N_TRAIN, rng)
    Xtr = np.column_stack([kt, rt])
    ytr_lo = f_low(kt, rt) + rng.normal(0, SIGMA, N_TRAIN)
    ytr_up = f_up(kt, rt) + rng.normal(0, SIGMA, N_TRAIN)
    kv, rv = sample(N_TEST, rng)
    Xte = np.column_stack([kv, rv])
    yte_lo, yte_up = f_low(kv, rv), f_up(kv, rv)

    for path, K, R, lo, up in [("data/train_samples.csv", kt, rt, ytr_lo, ytr_up),
                               ("data/test_samples.csv", kv, rv, yte_lo, yte_up)]:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["kappa", "varrho", "lower", "upper"])
            w.writerows(zip(K, R, lo, up))
    print(f"Step 1: {N_TRAIN} training and {N_TEST} test samples written to data/")

    # ---- Step 2 : surrogates ---------------------------------------------
    models = {
        "Flexible neural network": (
            MLPRegressor(hidden_layer_sizes=(80, 80), max_iter=5000, random_state=1).fit(Xtr, ytr_lo),
            MLPRegressor(hidden_layer_sizes=(80, 80), max_iter=5000, random_state=2).fit(Xtr, ytr_up)),
        "Structure-aware model": (
            make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3)).fit(Xtr, ytr_lo),
            make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3)).fit(Xtr, ytr_up)),
    }

    # ---- Steps 3 and 4 : operator and diagnostic -------------------------
    metrics, margin_rows, keep = [], [], {}
    for name, (m_lo, m_up) in models.items():
        g_lo, g_up = wrap(m_lo), wrap(m_up)
        rmse = (float(np.sqrt(mean_squared_error(yte_lo, m_lo.predict(Xte)))),
                float(np.sqrt(mean_squared_error(yte_up, m_up.predict(Xte)))))
        marg, ok, terms = evaluate(g_lo, g_up)
        npar = n_params(m_lo) + n_params(m_up)
        keep[name] = (g_lo, g_up, marg, terms)
        metrics.append(dict(model=name, rmse_lower=round(rmse[0], 4), rmse_upper=round(rmse[1], 4),
                            trainable_parameters=npar, levels_satisfied=f"{ok}/{len(LEVELS)}",
                            smallest_margin=round(float(marg.min()), 4)))
        for r, mm in zip(LEVELS, marg):
            margin_rows.append(dict(model=name, level=round(float(r), 2), margin=round(float(mm), 6)))
        print(f"Steps 2-4: {name:26s}
