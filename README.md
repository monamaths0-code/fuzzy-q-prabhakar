# Fuzzy $q$-Prabhakar Operators — Reproducibility Package

Code and data accompanying the paper *New Refinements of Hermite–Hadamard Inequalities
through $q$-Fuzzy Prabhakar Operators and their Consequences*.

Everything in the paper's Sections 4 and 5 can be regenerated from this repository.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/ML_validation.ipynb)

> Replace `USERNAME/REPO` in the badge URL with your GitHub path after uploading.

## What this package does

1. Implements the $q$-Prabhakar machinery from scratch ($q$-Pochhammer symbol, $q$-Gamma
   function, $q$-Prabhakar function, and the left $q$-Prabhakar fractional integral).
2. Reproduces the three numerical examples of Section 4 and checks every inclusion.
3. Runs the machine-learning validation study of Section 5 and writes all figures and tables.

## Installation

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
```

Python 3.9 or later. No GPU required; the full study runs in under two minutes on a laptop.

## Usage

```bash
python scripts/verify_examples.py      # Section 4: examples A, B, C
python scripts/run_ml_experiment.py    # Section 5: ML validation study
```

Or open `notebooks/ML_validation.ipynb` in Google Colab — no local installation needed.

## Repository layout

```
fqp/
  qcalculus.py      q-Pochhammer, q-Gamma, q-Prabhakar function
  operators.py      left q-Prabhakar fractional integral; kernel constants K0 and I_R
  hh.py             the three Hermite-Hadamard terms and the inclusion margins
scripts/
  verify_examples.py     regenerates Section 4
  run_ml_experiment.py   regenerates Section 5 (writes data/ and results/)
notebooks/
  ML_validation.ipynb    the same study, runnable in Colab
data/       noisy samples used for training and testing (written by the script)
results/    metrics, margins and figures (written by the scripts)
```

## Correctness check

The implementation reproduces the original Example 4.1 of the manuscript exactly,
including $\Gamma_{1/2}(3) = 3/2$ and the middle-term value $32\varrho$. Run
`scripts/verify_examples.py` and compare against `results/examples.csv`.

## Two points worth knowing

**The parameter $\gamma = 1$ is degenerate.** Since $(\gamma;q)_k = \prod_{j<k}(1-\gamma q^j)$
vanishes for $k \ge 1$ when $\gamma = 1$, the $q$-Prabhakar function collapses to
$1/\Gamma_q(\mu)$ and the operator reduces to the Riemann–Liouville $q$-integral *for every*
$\lambda$. Example B uses $\gamma = 1/2$ so that the Prabhakar structure is genuinely
exercised.

**Use the kernel constant $K_0$, not the closed form.** The left-hand normaliser
$h(1/2)\,E^{\gamma}_{\eta,\mu+1}(\lambda)$ equals the kernel integral $K_0$ only when
$\gamma = 1$. For $\gamma = 1/2$, $\lambda = 1$, $\eta = 1$, $\mu = 2$, $q = 1/2$ one gets
$K_0 = 1.2258$ against $E^{\gamma}_{\eta,\mu+1}(\lambda) = 1.4627$, and the inclusion fails
with the latter. `kernel_K0` in `fqp/operators.py` computes the correct quantity.

## Machine-learning results

Ground truth is the fuzzy interval-valued function of Example A; $3000$ noisy observations
of its $\varrho$-level endpoints are generated with $\sigma = 0.10$, and $800$ further
samples are held out for testing.

| Model | Test RMSE (lower, upper) | Trainable parameters | Inclusion satisfied | Smallest margin |
|---|---|---|---|---|
| Flexible neural network (80, 80) | (0.0341, 0.0336) | 13,602 | 5 / 11 | −0.1528 |
| Structure-aware (degree 2, ridge) | (0.0217, 0.0217) | 14 | 11 / 11 | +0.0795 |
| Exact function (reference) | — | — | 11 / 11 | 0.0000 |

The network has more parameters than there are training samples and fits the noise; despite a
low fitting error it violates the inclusion at six of the eleven membership levels. The
violations occur where the theoretical margin of the exact function has itself become small,
so the inequality is most discriminating exactly where it is tightest.

## Reproducibility

All randomness is seeded (`SEED = 0` in `scripts/run_ml_experiment.py`). Re-running the
scripts reproduces `results/ml_metrics.csv` and `results/ml_margins.csv` byte for byte on the
same library versions. If you change the seed, the noise level, or the operator parameters,
regenerate the figures so the reported numbers stay consistent.

## Citation

@article{haseeb_siddiq_ali_qprabhakar,
  author  = {Haseeb, Muhammad and Siddiq, Mamoona and Ali, Rana Safdar},
  title   = {New Refinements of Hermite--Hadamard Inequalities through
             $q$-Fuzzy Prabhakar Operators with an Application to
             Machine Learning Validation},
  year    = {2026},
  note    = {Preprint. Code and data: \url{https://github.com/USERNAME/REPO}}
}
## License

Released under the MIT License; see `LICENSE`.
