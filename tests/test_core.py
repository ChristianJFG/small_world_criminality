import networkx as nx
import numpy as np
import pytest

from turing_networks.core import (Parameters, growth_rates, integrate, ipr,
                                  laplacian_spectrum, make_graph, modal_matrix,
                                  nonlinear_rhs, selected_mode, spectral_metrics)


@pytest.mark.parametrize("topology", ["ring", "square", "triangular"])
def test_graphs_are_connected_deterministic(topology):
    a = make_graph(topology, 0.3, 7)
    b = make_graph(topology, 0.3, 7)
    assert nx.is_connected(a)
    assert set(map(frozenset, a.edges())) == set(map(frozenset, b.edges()))


def test_ring_shortcuts_preserve_edges():
    original = make_graph("ring", 0, 42, n=20)
    shortcut = make_graph("ring", 0.9, 42, n=20)
    assert all(shortcut.has_edge(*edge) for edge in original.edges())
    assert shortcut.number_of_edges() > original.number_of_edges()


def test_spectrum_and_ipr():
    graph = make_graph("ring", 0.2, 4, n=20)
    values, vectors = laplacian_spectrum(graph)
    assert np.all(values <= 1e-10)
    assert values[-1] == pytest.approx(0, abs=1e-10)
    assert np.allclose(vectors.T @ vectors, np.eye(len(graph)), atol=1e-10)
    assert np.all((ipr(vectors) >= 1 / len(graph) - 1e-10) & (ipr(vectors) <= 1 + 1e-10))
    fields = spectral_metrics(graph)
    assert all(len(x) == len(graph) for x in fields.values())


def test_homogeneous_equilibrium_rhs():
    graph = make_graph("ring", 0.1, 3, n=16)
    A = nx.to_numpy_array(graph)
    p = Parameters()
    state = np.tile([p.rho_star, p.o_star], len(graph))
    assert np.linalg.norm(nonlinear_rhs(0, state, A, p), ord=np.inf) < 1e-12


def test_linearization_agrees_with_finite_difference():
    graph = make_graph("ring", 0.4, 2, n=12)
    A = nx.to_numpy_array(graph)
    p = Parameters()
    values, vectors = laplacian_spectrum(graph)
    index = 2
    lam, phi = float(values[index]), vectors[:, index]
    baseline = np.tile([p.rho_star, p.o_star], len(graph))
    h = 1e-6
    for basis in [np.array([1., 0.]), np.array([0., 1.])]:
        perturb = np.outer(phi, basis).ravel()
        numerical = (nonlinear_rhs(0, baseline + h * perturb, A, p) -
                     nonlinear_rhs(0, baseline - h * perturb, A, p)) / (2 * h)
        predicted = np.outer(phi, modal_matrix(lam, p) @ basis).ravel()
        assert np.allclose(numerical, predicted, rtol=2e-4, atol=2e-7)


def test_selected_mode_and_nonlinear_integration():
    graph = make_graph("ring", 0.1, 2, n=12)
    selected = selected_mode(graph, Parameters())
    assert selected["params"].I > selected["params"].threshold_I()
    assert np.all(np.isfinite(selected["growth_rates"]))
    assert np.max(growth_rates(selected["eigenvalues"], selected["params"])) >= selected["max_growth"]
    solution = integrate(graph, selected["params"], t_final=8, seed=2)
    assert solution.success
    assert np.all(np.isfinite(solution.y))


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        make_graph("square", -0.2, 0)
    with pytest.raises(ValueError):
        Parameters().near_threshold(0)
