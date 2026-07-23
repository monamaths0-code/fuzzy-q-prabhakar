"""Hermite-Hadamard terms of Theorem 3.2 and the inclusion diagnostic."""
from .operators import qprab_integral, kernel_K0, kernel_IR


def hh_terms(f_low, f_up, rho_level, xi1, xi2, eta, mu, gamma, lam, q, h, inv_h):
    """Return (left, middle, right), each a (lower, upper) pair, at one level.

    f_low(kappa, level), f_up(kappa, level) are the varrho-level endpoints.
    """
    L = xi2 - xi1
    mid = 0.5 * (xi1 + xi2)
    fl = lambda k: f_low(k, rho_level)
    fu = lambda k: f_up(k, rho_level)
    mirror = lambda g: (lambda k: g(xi1 + xi2 - k))

    K0 = kernel_K0(eta, mu, gamma, lam, q)
    IR = kernel_IR(eta, mu, gamma, lam, q, inv_h)

    left = (h(0.5) * K0 * fl(mid), h(0.5) * K0 * fu(mid))
    both = lambda g: (qprab_integral(g, xi1, xi2, eta, mu, gamma, lam, q)
                      + qprab_integral(mirror(g), xi1, xi2, eta, mu, gamma, lam, q)) / L ** mu
    middle = (both(fl), both(fu))
    right = ((fl(xi1) + fl(xi2)) * IR, (fu(xi1) + fu(xi2)) * IR)
    return left, middle, right


def inclusion_margins(left, middle, right):
    """Four quantities whose non-negativity is equivalent to Left > Middle > Right."""
    return (middle[0] - left[0],
            right[0] - middle[0],
            middle[1] - right[1],
            left[1] - middle[1])
