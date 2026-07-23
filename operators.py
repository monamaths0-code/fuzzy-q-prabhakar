"""Basic q-calculus: q-Pochhammer symbol, q-Gamma function, q-Prabhakar function."""


def q_pochhammer(a, q, n=None, n_inf=4000):
    """(a;q)_n.  If n is None the infinite product is returned (truncated)."""
    if n is None:
        prod = 1.0
        for k in range(n_inf):
            term = a * q ** k
            prod *= (1.0 - term)
            if abs(term) < 1e-18:
                break
        return prod
    prod = 1.0
    for k in range(n):
        prod *= (1.0 - a * q ** k)
    return prod


def q_gamma(x, q, n_inf=4000):
    """Gamma_q(x) = (q;q)_inf / (q^x;q)_inf * (1-q)^(1-x)."""
    num = q_pochhammer(q, q, None, n_inf)
    den = q_pochhammer(q ** x, q, None, n_inf)
    return num / den * (1.0 - q) ** (1.0 - x)


def q_prabhakar_E(z, gamma, eta, mu, q, k_max=60, tol=1e-16):
    """q-Prabhakar function E^gamma_{eta,mu,q}(z).

    Because (gamma;q)_k = prod_{j<k} (1 - gamma q^j) vanishes for k >= 1 when
    gamma = 1, the series collapses to 1/Gamma_q(mu) in that case and the
    operator degenerates to the Riemann-Liouville q-integral for every lambda.
    """
    s = 0.0
    for k in range(k_max):
        coeff = q_pochhammer(gamma, q, k) / q_pochhammer(q, q, k)
        term = coeff * (z ** k) / q_gamma(eta * k + mu, q)
        s += term
        if k > 3 and abs(term) < tol:
            break
    return s
