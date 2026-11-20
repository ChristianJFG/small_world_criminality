import numpy as np
import networkx as nx
from turing_thesis.networks import make_graph, closest_square_factors, best_triangular_dims
from turing_thesis.spectral import mode_metrics, default_model_parameters, modal_matrix, growth_rates
from turing_thesis.experiments import historical_p_values,run_sweep


def test_original_graph_shapes():
    for topology in ['ring','square','triangular']:
        G=make_graph(topology,200,0.1,42)
        assert G.number_of_nodes()==200
        assert nx.is_connected(G)


def test_base_topologies_do_not_use_periodic_square():
    G=make_graph('square',200,0,3)
    assert G.number_of_edges()==(10-1)*20 + 10*(20-1)


def test_exact_historical_sampling():
    for topology in ('ring','square','triangular'):
        p=historical_p_values(topology,105)
        assert len(p)==105
    assert np.isclose(historical_p_values('ring',105)[0],0)
    assert np.isclose(historical_p_values('square',105)[0],1e-4)
    assert np.isclose(historical_p_values('triangular',105)[-1],1)


def test_deterministic_graphs():
    for t in ['ring','square','triangular']:
        g1=make_graph(t,200,.1,7)
        g2=make_graph(t,200,.1,7)
        assert sorted(map(sorted,g1.edges()))==sorted(map(sorted,g2.edges()))


def test_laplacian_modes_and_ipr():
    g=make_graph('ring',20,.1,4)
    m=mode_metrics(g)
    assert len(m['eigenvalues'])==20
    assert abs(m['eigenvalues'][-1])<1e-9
    assert np.all(m['ipr']>=1/20-1e-9)
    assert np.all(m['ipr']<=1+1e-9)
    assert np.all(m['eigenvalues']<=1e-9)


def test_regimes_and_histogram_rows():
    graphs,modes=run_sweep('ring',preset='quick',n=20,seeds=1,p_points=1)
    assert len(graphs)==3
    assert len(modes)==3*20
    assert set(modes.p_region)=={'Regular','Small-world','Shortcut-rich'}


def test_modal_matrix_homogeneous_stability():
    p=default_model_parameters()
    eig=np.linalg.eigvals(modal_matrix(0,p))
    assert max(eig.real)<0


def test_shortcuts_never_remove_original_edges():
    for topology in ['ring','square','triangular']:
        baseline=make_graph(topology,200,0,42)
        augmented=make_graph(topology,200,.5,42)
        assert set(map(frozenset,baseline.edges())).issubset(set(map(frozenset,augmented.edges())))


def test_homogeneous_fixed_point_and_numerical_linearization():
    from turing_thesis.dynamics import rhs
    from turing_thesis.spectral import modal_matrix
    g=make_graph('ring',12,.3,9)
    p=default_model_parameters()
    A=nx.to_numpy_array(g,dtype=float)
    baseline=np.tile([p['rho_ss'],p['O_ss']],len(g))
    assert np.max(np.abs(rhs(0,baseline,A,p)))<1e-12
    # Derivative at any node of a homogeneous perturbation agrees with M(0)
    idx=2; n=len(g)
    jac=np.zeros((2,2)); h=1e-7
    for j in range(2):
        dr=np.tile([0.,0.],n)
        dr[j::2]=h
        plus=rhs(0,baseline+dr,A,p).reshape(n,2)[idx]
        minus=rhs(0,baseline-dr,A,p).reshape(n,2)[idx]
        jac[:,j]=(plus-minus)/(2*h)
    np.testing.assert_allclose(jac,modal_matrix(0,p),atol=1e-6)
