"""Left q-Prabhakar fractional integral and the kernel constants K0, I_R."""
from .qcalculus import q_pochhammer, q_gamma, q_prabhakar_E


def _kernel_weight(n, mu, q):
    """(q^{n+1};q)_inf / (q^{mu+n};q)_inf  -- the q-power kernel factor."""
    return q_pochhammer(q ** (n + 1), q, None) / q_pochhammer(q ** (mu + n), q, None)


def _E_factor(n, L, eta, mu, gamma, lam, q):
    if lam == 0:
        return 1.0 / q_gamma(mu, q)
    z = lam * (L ** eta) * q_pochhammer(q ** (n + 1), q, None) \
        / q_pochhammer(q ** (eta + n + 1), q, None)
    return q_prabhakar_E(z, gamma, eta, mu, q)


def qprab_integral(f, xi1, xi2, eta, mu, gamma, lam, q, n_max=4000):
    """Left q-Prabhakar fractional integral  ( _{xi1} E^{eta,mu}_{gamma,lam;q} f )(xi2).

    Nodes are t_n = xi1 + q^n (xi2 - xi1).
    """
    L = xi2 - xi1
    total = 0.0
    for n in range(n_max):
        w = q ** n
        if w < 1e-16:
            break
        E = _E_factor(n, L, eta, mu, gamma, lam, q)
        total += w * _kernel_weight(n, mu, q) * E * f(xi1 + w * L)
    return (1.0 - q) * (L ** mu) * total


def _unit_kernel(eta, mu, gamma, lam, q, weight, n_max=4000):
    """Integral over [0,1] of the kernel times an optional weight w(zeta)."""
    total = 0.0
    for n in range(n_max):
        w = q ** n
        if w < 1e-16:
            break
        x = q ** (n + 1)
        poch = q_pochhammer(x, q, None)
        base_mu = poch / q_pochhammer(x * q ** (mu - 1), q, None)
        if lam == 0:
            E = 1.0 / q_gamma(mu, q)
        else:
            base_eta = poch / q_pochhammer(x * q ** eta, q, None)
            E = q_prabhakar_E(lam * base_eta, gamma, eta, mu, q)
        total += w * base_mu * E * (1.0 if weight is None else weight(q ** n))
    return (1.0 - q) * total


def kernel_K0(eta, mu, gamma, lam, q):
    """K0: the kernel constant normalising the left-hand term of Theorem 3.2.

    Equals E^gamma_{eta,mu+1}(lam) only when gamma = 1.
    """
    return _unit_kernel(eta, mu, gamma, lam, q, None)


def kernel_IR(eta, mu, gamma, lam, q, inv_h):
    """I_R: kernel constant of the right-hand term; inv_h(z) = 1/h(z)."""
    return _unit_kernel(eta, mu, gamma, lam, q,
                        lambda z: inv_h(z) + inv_h(1.0 - z))
