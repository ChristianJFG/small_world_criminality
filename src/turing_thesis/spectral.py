"""Mode observables from the original get_dynamics(..., integration=False).

The negative-semidefinite Laplacian L=A-D matches the notebooks.
All mode weights are computed from squared normalized eigenvectors.
"""
import numpy as np
import networkx as nx


def mode_metrics(graph: nx.Graph) -> dict:
    nodes=list(graph.nodes())
    L= -nx.laplacian_matrix(graph,nodelist=nodes).toarray().astype(float)
    L=(L+L.T)/2
    eig,vec=np.linalg.eigh(L)
    weights=vec**2 / np.sum(vec**2,axis=0,keepdims=True)
    degree=np.array([graph.degree(v) for v in nodes],dtype=float)
    clustering_dict=nx.clustering(graph)
    clustering=np.array([clustering_dict[v] for v in nodes],dtype=float)
    avg_degree=float(degree.mean())
    avg_clustering=float(clustering.mean())
    return {
        "eigenvalues":eig,
        "eigenvectors":vec,
        "ipr":np.sum(weights**2,axis=0),
        "weighted_degree":weights.T @ degree,
        "weighted_clustering":weights.T @ clustering,
        "k_avg":avg_degree,
        "avg_clustering":avg_clustering,
    }


def default_model_parameters() -> dict:
    # ring.ipynb cell 5, square.ipynb cell 6, triangular.ipynb cell 6
    a,b,d,d_o=0.0050415,0.000013047,0.0079639,0.0017783
    I_c=(2*d*d*b + d_o*(a+b)**2 + 2*(a+b)*d*np.sqrt(2*d_o*(a+b)))/(2*a*d*d)
    I=I_c+0.01
    lambda_c=(d_o*(a+b)**2+2*b*d*d-2*I*a*d*d)/(4*d_o*d*(a+b))
    return dict(alpha=a,beta=b,delta=d,d_o=d_o,I=I,I_c=I_c,lambda_c=lambda_c,
                rho_ss=a*d/(a+b),O_ss=(a+b)/d)


def modal_matrix(lam:float,p:dict) -> np.ndarray:
    a,b,d,do,I=[p[k] for k in ('alpha','beta','delta','d_o','I')]
    return np.array([[2*lam-(a+b)/d, -2*I*a*d*d/(a+b)**2*lam-a*d/(a+b)],
                     [(a+b)/d, do*lam-b*d/(a+b)]])


def growth_rates(eigenvalues:np.ndarray,params:dict) -> np.ndarray:
    return np.array([np.linalg.eigvals(modal_matrix(float(l),params)).real.max() for l in eigenvalues])
