"""Fuzzy q-Prabhakar fractional operators and Hermite-Hadamard inclusions."""
from .qcalculus import q_pochhammer, q_gamma, q_prabhakar_E
from .operators import qprab_integral, kernel_K0, kernel_IR
from .hh import hh_terms, inclusion_margins
__all__ = ["q_pochhammer", "q_gamma", "q_prabhakar_E", "qprab_integral",
           "kernel_K0", "kernel_IR", "hh_terms", "inclusion_margins"]
__version__ = "1.0.0"
