"""Research-oriented graph spectral and nonlinear dynamics experiments."""
from .core import (Parameters, growth_rates, integrate, ipr, laplacian_spectrum,
                   make_graph, modal_matrix, nonlinear_rhs, selected_mode,
                   spectral_metrics)
__all__ = ["Parameters", "growth_rates", "integrate", "ipr", "laplacian_spectrum",
           "make_graph", "modal_matrix", "nonlinear_rhs", "selected_mode", "spectral_metrics"]
