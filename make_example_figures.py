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
        print(f"Steps 2-4: {name:26s} RMSE=({rmse[0]:.4f},{rmse[1]:.4f})  "
              f"params={npar:6d}  inclusion {ok}/{len(LEVELS)}  min margin {marg.min():+.4f}")

    marg_true, ok_true, _ = evaluate(f_low, f_up)
    metrics.append(dict(model="Exact function (reference)", rmse_lower="", rmse_upper="",
                        trainable_parameters=0, levels_satisfied=f"{ok_true}/{len(LEVELS)}",
                        smallest_margin=round(float(marg_true.min()), 4)))
    print(f"reference: exact function            inclusion {ok_true}/{len(LEVELS)}  "
          f"min margin {marg_true.min():+.4f}")
    for r, mm in zip(LEVELS, marg_true):
        margin_rows.append(dict(model="Exact function", level=round(float(r), 2),
                                margin=round(float(mm), 6)))

    for path, rows in [("results/ml_metrics.csv", metrics), ("results/ml_margins.csv", margin_rows)]:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print("written: results/ml_metrics.csv, results/ml_margins.csv")

    make_figure(keep, marg_true)
    print("written: results/fig_ml.png")


def make_figure(keep, marg_true):
    C = dict(true="#333333", nn="#c0392b", sa="#1b7837")
    CC = dict(left="#1b7837", mid="#e08214", right="#2166ac")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linestyle": ":", "axes.axisbelow": True})
    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.35))
    g_lo_sa, g_up_sa, marg_sa, terms_sa = keep["Structure-aware model"]
    g_lo_nn, g_up_nn, marg_nn, _ = keep["Flexible neural network"]

    kk = np.linspace(XI1, XI2, 120)
    for r, ls in [(0.2, "-"), (0.6, "--")]:
        ax[0].plot(kk, f_low(kk, r), color=C["true"], lw=1.8, ls=ls)
        ax[0].plot(kk, f_up(kk, r), color=C["true"], lw=1.8, ls=ls)
        ax[0].plot(kk, g_lo_sa(kk, r), color=C["sa"], lw=0, marker="o", ms=3.2, alpha=.75)
        ax[0].plot(kk, g_up_sa(kk, r), color=C["sa"], lw=0, marker="o", ms=3.2, alpha=.75)
        ax[0].plot(kk, g_lo_nn(kk, r), color=C["nn"], lw=0, marker="x", ms=3.2, alpha=.7)
        ax[0].plot(kk, g_up_nn(kk, r), color=C["nn"], lw=0, marker="x", ms=3.2, alpha=.7)
    hs = [Line2D([], [], color=C["true"], lw=1.8, ls="-"),
          Line2D([], [], color=C["true"], lw=1.8, ls="--"),
          Line2D([], [], color=C["sa"], lw=0, marker="o", ms=4),
          Line2D([], [], color=C["nn"], lw=0, marker="x", ms=4)]
    ax[0].legend(hs, [r"exact, $\varrho=0.2$", r"exact, $\varrho=0.6$",
                      "structure-aware", "neural network"],
                 loc="center", framealpha=1, fancybox=True, borderpad=.6, fontsize=8.5)
    ax[0].set_xlabel(r"$\kappa$"); ax[0].set_ylabel("endpoint value")
    ax[0].set_title(r"(a) surrogate fit to $\varrho$-level endpoints")
    ax[0].margins(y=.12)

    ax[1].plot(LEVELS, marg_nn, color=C["nn"], lw=1.8, marker="x", ms=5, label="neural network")
    ax[1].plot(LEVELS, marg_sa, color=C["sa"], lw=1.8, marker="o", ms=5, label="structure-aware")
    ax[1].plot(LEVELS, marg_true, color=C["true"], lw=1.2, ls=":", label="exact function")
    ax[1].axhline(0, color="0.25", lw=1.2)
    ax[1].fill_between(LEVELS, np.minimum(marg_nn, 0), 0, color=C["nn"], alpha=.13)
    ax[1].set_xlabel(r"$\varrho$  (membership level)")
    ax[1].set_ylabel("smallest inclusion margin")
    ax[1].set_title("(b) theory-consistency diagnostic")
    ax[1].legend(loc="lower left", framealpha=1, fancybox=True, borderpad=.6, fontsize=8.5)
    y0, y1 = ax[1].get_ylim(); ax[1].set_ylim(y0 - .22 * (y1 - y0), y1 + .10 * (y1 - y0))

    for r, (L, M, R) in zip(LEVELS, terms_sa):
        ax[2].plot([L[0], L[1]], [r, r], color=CC["left"], lw=7, alpha=.55, solid_capstyle="butt")
        ax[2].plot([M[0], M[1]], [r, r], color=CC["mid"], lw=4, alpha=.85, solid_capstyle="butt")
        ax[2].plot([R[0], R[1]], [r, r], color=CC["right"], lw=1.8, solid_capstyle="butt")
    for k, lw, a, nm in [("left", 7, .55, "Left term"), ("mid", 4, .85, "Middle term"),
                         ("right", 1.8, 1, "Right term")]:
        ax[2].plot([], [], color=CC[k], lw=lw, alpha=a, label=nm)
    ax[2].legend(loc="upper left", framealpha=1, fancybox=True, borderpad=.6, fontsize=8.5)
    x0, x1 = ax[2].get_xlim(); ax[2].set_xlim(x0 - .42 * (x1 - x0), x1 + .05 * (x1 - x0))
    ax[2].set_ylim(-.06, 1.16)
    ax[2].set_xlabel("interval bound"); ax[2].set_ylabel(r"$\varrho$")
    ax[2].set_title("(c) inclusion recovered from learned model")

    plt.tight_layout(); plt.savefig("results/fig_ml.png", bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
