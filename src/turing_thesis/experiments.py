"""Historical parameter sweep and manuscript observables.

Source notebooks: ring.ipynb (cells 14,15,22-31), square.ipynb (cells
15,16,23-32), triangular.ipynb (cells 15,16,23-32).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from .networks import make_graph
from .spectral import mode_metrics

P_REGIONS={
    'ring':[(0,.05,'Regular'),(.05,.5,'Small-world'),(.5,1.1,'Shortcut-rich')],
    'square':[(0,.05,'Regular'),(.05,.5,'Small-world'),(.5,1.1,'Shortcut-rich')],
    'triangular':[(0,.002,'Regular'),(.002,.2,'Small-world'),(.2,1.1,'Shortcut-rich')],
}


def historical_p_values(topology:str,n_experiments:int=105)->np.ndarray:
    from .config import integer
    integer('n_experiments', n_experiments, 1)
    if topology in ('ring', 'triangular') and n_experiments % 3:
        raise ValueError('n_experiments must be a multiple of 3 for ring/triangular')
    if topology=='ring':
        g=n_experiments//3
        return np.concatenate([np.linspace(0,.01,g),np.linspace(.09,.2,g),np.linspace(.9,1.,g)])
    if topology=='square':
        return np.logspace(-4,0,n_experiments)
    if topology=='triangular':
        g=n_experiments//3
        return np.concatenate([np.linspace(0,.001,g),np.linspace(.05,.1,g),np.linspace(.9,1.,g)])
    raise ValueError(topology)


def p_region(topology:str,p:float) ->str:
    for lo,hi,label in P_REGIONS[topology]:
        if lo<=p<hi:
            return label
    raise ValueError(p)


def run_one(topology:str,n:int,p:float,seed:int,ring_k:int=4)->tuple[dict,list[dict]]:
    G=make_graph(topology,n,p,seed,ring_k=ring_k)
    values=mode_metrics(G)
    k_avg=values['k_avg']; c_avg=values['avg_clustering']
    graph_row=dict(topology=topology,p=float(p),seed=seed,n_nodes=len(G),n_edges=len(G.edges()),
        k_avg=k_avg,avg_clustering=c_avg,p_region=p_region(topology,p),
        # Not used in the figures; computing all-pairs geodesics on 3150 graphs
        # would make full replication unnecessarily expensive.
        transitivity=float(nx.transitivity(G)))
    rows=[]
    for index,lam in enumerate(values['eigenvalues']):
        w_k=float(values['weighted_degree'][index]);w_c=float(values['weighted_clustering'][index])
        rows.append(dict(topology=topology,p=float(p),seed=seed,mode_index=index,
          eigenvalue=float(lam),eigenvalue_over_k=float(lam)/k_avg,
          ipr=float(values['ipr'][index]),weighted_degree=w_k,weighted_degree_over_k=w_k/k_avg,
          weighted_clustering=w_c,
          weighted_clustering_over_avg=(w_c/c_avg if c_avg>1e-13 else np.nan),
          p_region=graph_row['p_region']))
    return graph_row,rows


def sweep_probabilities(topology, preset='quick', p_points=None, p_values=None):
    from .config import integer, validate_probabilities
    if preset not in ('quick', 'manuscript'):
        raise ValueError(preset)
    historical = historical_p_values(topology, 105)
    if p_points is not None:
        integer('p_points', p_points, 1)
        if preset != 'quick':
            raise ValueError('p_points applies only to the quick preset')
        limit = 105 if topology == 'square' else 35
        if p_points > limit:
            raise ValueError(f'p_points must be <= {limit}')
    if p_values is not None:
        validate_probabilities(p_values)
        return np.asarray(p_values, dtype=float)
    if preset == 'manuscript':
        return historical
    if topology in ('ring', 'triangular'):
        pp = 3 if p_points is None else p_points
        return np.concatenate([b[np.linspace(0, len(b)-1, pp).astype(int)]
                               for b in np.array_split(historical, 3)])
    if p_points is not None:
        return historical[np.linspace(0, len(historical)-1, p_points).astype(int)]
    target = np.array([1e-4, .002, .01, .1, .35, .8, 1.])
    return historical[np.unique(np.abs(historical[:, None]-target).argmin(axis=0))]


def run_sweep(topology: str, preset: str='quick', n: int=200, seeds: int | None=None,
              p_points: int | None=None, ring_k: int=4, progress: bool=False,
              seed_start: int=0, p_values: list[float] | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .config import ExperimentConfig
    config = ExperimentConfig(preset=preset, topologies=[topology], nodes=n,
                              seeds=seeds, seed_start=seed_start, ring_k=ring_k,
                              p_points=p_points,
                              probabilities={} if p_values is None else {topology: p_values}).validate()
    probabilities = sweep_probabilities(topology, preset, p_points, p_values)
    graph_rows, mode_rows = [], []
    total = len(probabilities) * config.seed_count
    for idx, p in enumerate(probabilities):
        for offset in range(config.seed_count):
            seed = seed_start + offset
            if progress:
                print(f'{topology}: {idx*config.seed_count+offset+1}/{total} (p={p:.5g}, seed={seed})', flush=True)
            g, rows = run_one(topology, n, float(p), seed, ring_k)
            graph_rows.append(g)
            mode_rows.extend(rows)
    return pd.DataFrame(graph_rows), pd.DataFrame(mode_rows)
