"""Original two-variable network dynamics, extracted from the thesis notebooks.

The old square/triangular versions hardcoded np.arange(16) in the ODE RHS,
which fails for the n=200 network sweeps. This vectorized implementation
fixes that indexing error while retaining the stated equations.
"""
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from .spectral import default_model_parameters


def rhs(t: float,state: np.ndarray,adjacency:np.ndarray,params:dict)->np.ndarray:
    n=adjacency.shape[0]
    y=state.reshape(n,2)
    rho=y[:,0]
    O=np.maximum(y[:,1],1e-12)
    I=params['I']; a=params['alpha'];b=params['beta'];d=params['delta'];d_o=params['d_o']
    L=adjacency-np.diag(adjacency.sum(axis=1))
    # Notebook original: sum_j A_ij [(O_i/O_j)^I rho_j - (O_j/O_i)^I rho_i]
    logratio=I*(np.log(O[:,None])-np.log(O[None,:]))
    ratio=np.exp(np.clip(logratio,-100,100))
    cross=(adjacency*(ratio*rho[None,:]-(1/ratio)*rho[:,None])).sum(axis=1)
    drho=-rho*O+a+L@rho+cross
    dO=-d*O+b+rho*O+d_o*(L@O)
    return np.stack((drho,dO),axis=1).reshape(-1)


def integrate(graph:nx.Graph,params:dict|None=None,t_final:float=1000,
              seed:int=0,relative_noise:float=0.01):
    if params is None: params=default_model_parameters()
    nodes=list(graph.nodes())
    adjacency=nx.to_numpy_array(graph,nodelist=nodes,dtype=float)
    baseline=np.array([params['rho_ss'],params['O_ss']])
    rng=np.random.default_rng(seed)
    initial=(baseline[None,:]*(1+rng.uniform(-relative_noise,relative_noise,(len(nodes),2)))).reshape(-1)
    solution=solve_ivp(rhs,(0,t_final),initial,args=(adjacency,params),method='BDF',
                       rtol=1e-7,atol=1e-10,t_eval=np.linspace(0,t_final,300))
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(solution.message)
    return solution
