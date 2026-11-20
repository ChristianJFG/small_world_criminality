"""Reproducible spectral analysis and crime-inspired network reaction--diffusion model.

The negative-semidefinite Laplacian convention throughout is L = A - D.
Code is a clean reimplementation of methods in the original research notebooks;
outputs are newly computed, not exact reproductions of archived thesis figures.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Parameters:
    alpha: float = 0.0050415
    beta: float = 0.000013047
    delta: float = 0.0079639
    d_o: float = 0.0017783
    I: float = 0.62

    @property
    def rho_star(self) -> float:
        return self.alpha * self.delta / (self.alpha + self.beta)

    @property
    def o_star(self) -> float:
        return (self.alpha + self.beta) / self.delta

    def threshold_I(self) -> float:
        a, b, d, do = self.alpha, self.beta, self.delta, self.d_o
        return (2 * d * d * b + do * (a + b) ** 2
                + 2 * (a + b) * d * np.sqrt(2 * do * (a + b))) / (2 * a * d * d)

    def near_threshold(self, target_lambda: float, epsilon_I: float = 0.01) -> "Parameters":
        if abs(target_lambda) < 1e-9:
            raise ValueError("target_lambda cannot be zero")
        do = (self.alpha + self.beta) / (2 * target_lambda**2)
        p = Parameters(self.alpha, self.beta, self.delta, do, 0.0)
        return Parameters(p.alpha, p.beta, p.delta, p.d_o, p.threshold_I() + epsilon_I)


def make_graph(topology: str, p: float, seed: int, n: int = 36, side: int = 6) -> nx.Graph:
    """Generate an undirected connected regular graph plus Newman--Watts-like shortcuts.

    Ring: n nodes with k=4. Square: side x side *periodic* grid.
    Triangular: finite non-periodic lattice; node count differs from side**2.
    For square/triangular, each original edge independently triggers a shortcut
    with probability p; no existing edge is removed. This is *not* bond percolation.
    """
    if not 0 <= p <= 1:
        raise ValueError("Shortcut probability p must lie in [0,1]")
    if topology == "ring":
        if n < 6:
            raise ValueError("Ring requires n >= 6")
        graph = nx.watts_strogatz_graph(n, 4, 0, seed=seed)
    elif topology == "square":
        if side < 3:
            raise ValueError("side must be >= 3")
        graph = nx.grid_2d_graph(side, side, periodic=True)
    elif topology == "triangular":
        if side < 3:
            raise ValueError("side must be >= 3")
        graph = nx.triangular_lattice_graph(side, side)
    else:
        raise ValueError(f"Unsupported topology {topology!r}")
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    original_edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    for u, _ in original_edges:
        if rng.random() < p:
            available = [v for v in nodes if v != u and not graph.has_edge(u, v)]
            if available:
                graph.add_edge(u, int(rng.choice(available)))
    return graph


def laplacian_spectrum(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending eigenvalues and normalized eigenvectors as columns."""
    L = -nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).toarray().astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors


def ipr(eigenvectors: np.ndarray) -> np.ndarray:
    """Basis-dependent for exactly/nearly degenerate eigenvalues."""
    squared = np.abs(eigenvectors) ** 2
    return np.sum(squared ** 2, axis=0) / np.sum(squared, axis=0) ** 2


def spectral_metrics(graph: nx.Graph) -> dict[str, np.ndarray]:
    values, vectors = laplacian_spectrum(graph)
    node_ids = sorted(graph.nodes())
    degrees = np.array([graph.degree(v) for v in node_ids], dtype=float)
    clustering = np.array([nx.clustering(graph, v) for v in node_ids], dtype=float)
    mean_degree = degrees.mean()
    weights = np.abs(vectors) ** 2
    return {
        "eigenvalue": values,
        "eigenvalue_over_mean_degree": values / mean_degree,
        "ipr": ipr(vectors),
        "weighted_degree_over_mean_degree": weights.T @ degrees / mean_degree,
        "weighted_clustering": weights.T @ clustering,
    }


def modal_matrix(lam: float, params: Parameters) -> np.ndarray:
    """2x2 linearization: consistent with the nonlinear RHS below."""
    a, b, d, do, I = params.alpha, params.beta, params.delta, params.d_o, params.I
    return np.array([
        [2 * lam - (a + b) / d,
         -2 * I * a * d**2 / (a + b)**2 * lam - a * d / (a + b)],
        [(a + b) / d, do * lam - b * d / (a + b)],
    ])


def growth_rates(eigenvalues: np.ndarray, params: Parameters) -> np.ndarray:
    return np.array([np.linalg.eigvals(modal_matrix(float(v), params)).real.max()
                     for v in eigenvalues])


def selected_mode(graph: nx.Graph, params: Parameters, target_quantile: float = 0.5,
                  epsilon_I: float = 0.01) -> dict:
    """Choose d_o from a nonzero spectral quantile, then find fastest growing mode."""
    if not 0 <= target_quantile <= 1:
        raise ValueError("target_quantile must lie in [0,1]")
    values, vectors = laplacian_spectrum(graph)
    candidates = np.flatnonzero(np.abs(values) > 1e-9)
    if not len(candidates):
        raise ValueError("No non-homogeneous Laplacian modes")
    index = int(candidates[round(target_quantile * (len(candidates) - 1))])
    p = params.near_threshold(float(values[index]), epsilon_I)
    rates = growth_rates(values, p)
    critical = int(candidates[np.argmax(rates[candidates])])
    return {
        "params": p, "eigenvalues": values, "eigenvectors": vectors,
        "growth_rates": rates, "target_index": index,
        "mode_index": critical, "max_growth": float(rates[critical]),
        "unstable_modes": int(np.sum(rates[candidates] > 0)),
    }


def nonlinear_rhs(t: float, state: np.ndarray, adjacency: np.ndarray,
                  params: Parameters) -> np.ndarray:
    """Full network equations; works for any graph size (no 16-node hardcoding)."""
    n = len(adjacency)
    states = state.reshape(n, 2)
    rho, opportunity = states[:, 0], states[:, 1]
    # Guard against nonphysical negative/zero opportunity in solver trial states.
    o = np.clip(opportunity, 1e-12, None)
    L = adjacency - np.diag(adjacency.sum(axis=1))
    log_ratio = np.log(o[:, None]) - np.log(o[None, :])
    powers = np.exp(np.clip(params.I * log_ratio, -100, 100))
    nonlinear_flow = (adjacency * (powers * rho[None, :]
                                  - (1.0 / powers) * rho[:, None])).sum(axis=1)
    drho = -rho * o + params.alpha + L @ rho + nonlinear_flow
    do = -params.delta * o + params.beta + rho * o + params.d_o * (L @ o)
    return np.column_stack([drho, do]).ravel()


def integrate(graph: nx.Graph, params: Parameters, t_final: float = 2000.0,
              epsilon: float = 1e-3, seed: int = 0):
    """Return scipy OdeResult. Finite-time endpoint is not proven stationary."""
    if t_final <= 0 or epsilon <= 0:
        raise ValueError("t_final and epsilon must be positive")
    A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1, 1, size=(len(graph), 2))
    baseline = np.array([params.rho_star, params.o_star])
    initial = (baseline * (1 + epsilon * noise)).ravel()
    solution = solve_ivp(nonlinear_rhs, (0, t_final), initial,
                         args=(A, params), method="BDF", rtol=1e-7, atol=1e-10,
                         t_eval=np.linspace(0, t_final, 101))
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution
