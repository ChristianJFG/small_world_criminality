"""Independent analytical/finite-difference checks of scientific calculations."""
import numpy as np
import networkx as nx
import pytest
from turing_thesis.networks import make_graph
from turing_thesis.spectral import mode_metrics, default_model_parameters, modal_matrix, growth_rates
from turing_thesis.dynamics import rhs, integrate


@pytest.mark.parametrize('topology', ['ring', 'square', 'triangular'])
def test_eigenpairs_and_weighted_observables(topology):
    graph = make_graph(topology, 20, .3, 4)
    values = mode_metrics(graph)
    lam, vectors = values['eigenvalues'], values['eigenvectors']
    laplacian = -nx.laplacian_matrix(graph).toarray().astype(float)
    np.testing.assert_allclose(laplacian @ vectors, vectors * lam, atol=1e-12)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(20), atol=1e-12)
    assert np.isclose(lam.sum(), -2 * graph.number_of_edges())
    # Uniform zero mode: exact graph averages and minimum possible IPR.
    assert np.isclose(values['ipr'][-1], 1 / 20)
    assert np.isclose(values['weighted_degree'][-1], values['k_avg'])
    assert np.isclose(values['weighted_clustering'][-1], values['avg_clustering'])
    degree = np.array([d for _, d in graph.degree()])
    clustering = np.array(list(nx.clustering(graph).values()))
    for i in range(20):
        weights = vectors[:, i] ** 2
        assert np.isclose(values['weighted_degree'][i], sum(weights * degree))
        assert np.isclose(values['weighted_clustering'][i], sum(weights * clustering))


def test_rhs_matches_explicit_notebook_equations():
    graph = make_graph('ring', 8, .2, 3)
    adjacency = nx.to_numpy_array(graph)
    params = default_model_parameters()
    rng = np.random.default_rng(8)
    state = np.array([params['rho_ss'], params['O_ss']]) * rng.uniform(.8, 1.2, (8, 2))
    expected = np.zeros_like(state)
    for i, (rho, opportunity) in enumerate(state):
        expected[i, 0] = -rho * opportunity + params['alpha']
        expected[i, 1] = -params['delta'] * opportunity + params['beta'] + rho * opportunity
        for j, (rho_j, opportunity_j) in enumerate(state):
            expected[i, 0] += adjacency[i, j] * (rho_j - rho +
                (opportunity / opportunity_j) ** params['I'] * rho_j -
                (opportunity_j / opportunity) ** params['I'] * rho)
            expected[i, 1] += params['d_o'] * adjacency[i, j] * (opportunity_j - opportunity)
    np.testing.assert_allclose(rhs(0, state.ravel(), adjacency, params).reshape(8, 2), expected, atol=1e-14)


@pytest.mark.parametrize('topology', ['ring', 'square', 'triangular'])
def test_nonuniform_mode_linearization(topology):
    graph = make_graph(topology, 20, .2, 9)
    adjacency = nx.to_numpy_array(graph)
    modes = mode_metrics(graph)
    params = default_model_parameters()
    base = np.tile([params['rho_ss'], params['O_ss']], (20, 1))
    h = 1e-7
    for index in (0, 10, 19):
        vector = modes['eigenvectors'][:, index]
        observed = np.zeros((2, 2))
        for variable in range(2):
            perturbation = np.zeros_like(base)
            perturbation[:, variable] = h * vector
            plus = rhs(0, (base + perturbation).ravel(), adjacency, params).reshape(20, 2)
            minus = rhs(0, (base - perturbation).ravel(), adjacency, params).reshape(20, 2)
            observed[:, variable] = vector @ ((plus - minus) / (2 * h))
        np.testing.assert_allclose(observed, modal_matrix(modes['eigenvalues'][index], params), rtol=1e-6, atol=1e-8)
    predicted = growth_rates(modes['eigenvalues'], params)
    assert np.isfinite(predicted).all()


def test_integration_repeatability_and_fixed_point():
    graph = make_graph('ring', 8, .2, 3)
    params = default_model_parameters()
    fixed = integrate(graph, t_final=2, relative_noise=0)
    np.testing.assert_allclose(fixed.y, np.tile([params['rho_ss'], params['O_ss']], (8, 1)).reshape(-1, 1) + np.zeros_like(fixed.y), atol=1e-12)
    first = integrate(graph, t_final=2, seed=9)
    second = integrate(graph, t_final=2, seed=9)
    np.testing.assert_array_equal(first.y, second.y)
    assert np.all(first.y > 0)
