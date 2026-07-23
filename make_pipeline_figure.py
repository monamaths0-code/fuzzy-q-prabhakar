"""Reproduce the numerical examples of Section 4 and check every inclusion.

Run:  python scripts/verify_examples.py
Writes results/examples.csv
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fqp import q_gamma, hh_terms, inclusion_margins, kernel_K0, kernel_IR
from fqp.operators import qprab_integral, _unit_kernel

H = lambda z: 1.0 / z if z > 0 else float('inf')
INV_H = lambda z: z                      # 1/h for h(z) = 1/z
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TOL = 1e-9

EXAMPLES = {
    "A (Thm 3.2, gamma=1, lam=0)": dict(
        f_low=lambda k, r: (1 - r) * k ** 2 + 4 * r,
        f_up=lambda k, r: (1 - r) * (9 - k ** 2) + 4 * r,
        xi1=0.0, xi2=1.0, eta=1.0, mu=2.0, gamma=1.0, lam=0.0, q=0.5),
    "B (Thm 3.2, gamma=1/2, lam=1)": dict(
        f_low=lambda k, r: (1 - r) * k ** 2 + 4 * r,
        f_up=lambda k, r: (1 - r) * (9 - k ** 2) + 4 * r,
        xi1=0.0, xi2=1.0, eta=1.0, mu=2.0, gamma=0.5, lam=1.0, q=0.5),
}


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    print(f"sanity: Gamma_(1/2)(3) = {q_gamma(3, 0.5):.6f}   (exact 1.5)\n")

    for name, p in EXAMPLES.items():
        K0 = kernel_K0(p["eta"], p["mu"], p["gamma"], p["lam"], p["q"])
        IR = kernel_IR(p["eta"], p["mu"], p["gamma"], p["lam"], p["q"], INV_H)
        print(f"=== Example {name}")
        print(f"    K0 = {K0:.6f}   I_R = {IR:.6f}")
        all_ok = True
        for r in LEVELS:
            L, M, R = hh_terms(p["f_low"], p["f_up"], r, p["xi1"], p["xi2"],
                               p["eta"], p["mu"], p["gamma"], p["lam"], p["q"], H, INV_H)
            m = inclusion_margins(L, M, R)
            ok = all(x >= -TOL for x in m)
            all_ok &= ok
            print(f"    rho={r:.2f}  L=[{L[0]:7.4f},{L[1]:8.4f}]  "
                  f"M=[{M[0]:7.4f},{M[1]:8.4f}]  R=[{R[0]:7.4f},{R[1]:8.4f}]  "
                  f"inclusion={'OK' if ok else 'FAIL'}")
            rows.append(dict(example=name, level=r,
                             left_lo=L[0], left_up=L[1], mid_lo=M[0], mid_up=M[1],
                             right_lo=R[0], right_up=R[1],
                             min_margin=min(m), inclusion=int(ok)))
        print(f"    -> inclusion holds at all levels: {all_ok}\n")

    # Example C : product inequality (Theorem 3.5)
    print("=== Example C (Thm 3.5, product, gamma=1, lam=0)")
    xi1, xi2, eta, mu, gamma, lam, q = 0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.5
    T = _unit_kernel(eta, mu, gamma, lam, q, lambda z: z * z + (1 - z) ** 2)
    Rc = _unit_kernel(eta, mu, gamma, lam, q, lambda z: 2 * z * (1 - z))
    print(f"    T = {T:.6f}   R = {Rc:.6f}")
    Fl = lambda k, r: (1 - r) * k ** 2 + 3 * r
    Fu = lambda k, r: (1 - r) * (4 - k ** 2) + 3 * r
    Gl = lambda k, r: (1 - r) * k ** 2 + 3 * r
    Gu = lambda k, r: (1 - r) * (6 - k ** 2) + 3 * r
    mirror = lambda g: (lambda k: g(xi1 + xi2 - k))
    all_ok = True
    for r in LEVELS:
        pl = lambda k: Fl(k, r) * Gl(k, r)
        pu = lambda k: Fu(k, r) * Gu(k, r)
        both = lambda g: (qprab_integral(g, xi1, xi2, eta, mu, gamma, lam, q)
                          + qprab_integral(mirror(g), xi1, xi2, eta, mu, gamma, lam, q)) / (xi2 - xi1) ** mu
        lhs = (both(pl), both(pu))
        rhs_lo = T * (Fl(xi1, r) * Gl(xi1, r) + Fl(xi2, r) * Gl(xi2, r)) \
                 + Rc * (Fl(xi1, r) * Gl(xi2, r) + Fl(xi2, r) * Gl(xi1, r))
        rhs_up = T * (Fu(xi1, r) * Gu(xi1, r) + Fu(xi2, r) * Gu(xi2, r)) \
                 + Rc * (Fu(xi1, r) * Gu(xi2, r) + Fu(xi2, r) * Gu(xi1, r))
        ok = (lhs[0] <= rhs_lo + TOL) and (rhs_up <= lhs[1] + TOL)
        all_ok &= ok
        print(f"    rho={r:.2f}  LHS=[{lhs[0]:8.4f},{lhs[1]:8.4f}]  "
              f"RHS=[{rhs_lo:8.4f},{rhs_up:8.4f}]  inclusion={'OK' if ok else 'FAIL'}")
        rows.append(dict(example="C (Thm 3.5, product)", level=r,
                         left_lo=lhs[0], left_up=lhs[1], mid_lo="", mid_up="",
                         right_lo=rhs_lo, right_up=rhs_up,
                         min_margin=min(rhs_lo - lhs[0], lhs[1] - rhs_up),
                         inclusion=int(ok)))
    print(f"    -> inclusion holds at all levels: {all_ok}")

    with open("results/examples.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print("\nwritten: results/examples.csv")


if __name__ == "__main__":
    main()
