"""Graph generators extracted from ring.ipynb and square/triangular.ipynb.

Provenance:
- ring.ipynb, process_graph: NetworkX newman_watts_strogatz_graph.
- square.ipynb and triangular.ipynb, cell defining newman_watts_lattice_graph.

Square and triangular grids are OPEN (non-periodic) as in the originals.
For every original edge, one possible shortcut is considered, using random.Random(seed).
No edge is removed. This is not Watts-Strogatz rewiring or percolation.
"""
import math
import random
import networkx as nx


def closest_square_factors(n: int) -> tuple[int,int]:
    for rows in range(int(math.sqrt(n)), 1, -1):
        if n % rows == 0:
            return rows, n // rows
    raise ValueError(f"n={n} cannot be factored into a 2D grid with both sides >1")


def triangular_count(m: int, k: int) -> int:
    return math.ceil((m + 1) * (k + 2) / 2)


def best_triangular_dims(n: int) -> tuple[int,int]:
    best = None
    for m in range(1, n + 1):
        k0 = max(1, int(round((2*n)/(m+1)-2)))
        for k in range(max(1,k0-3), k0+4):
            if triangular_count(m,k) == n:
                score = abs(m-k)
                if best is None or score < best[0]:
                    best = (score,m,k)
    if best is None:
        raise ValueError(f"Could not make exactly {n} nodes in a triangular lattice")
    return best[1],best[2]


def make_graph(topology: str, n: int, p: float, seed: int, ring_k: int=4) -> nx.Graph:
    if not 0 <= p <= 1:
        raise ValueError("shortcut probability must be in [0,1]")
    if topology == "ring":
        return nx.newman_watts_strogatz_graph(n=n,k=ring_k,p=p,seed=seed)
    if n < 3:
        raise ValueError("n must be >=3")
    rng = random.Random(seed)
    if topology == "square":
        rows, cols = closest_square_factors(n)
        graph = nx.grid_2d_graph(rows, cols, periodic=False)
        pos = {v:(v[1],-v[0]) for v in graph.nodes()}
        nx.set_node_attributes(graph,pos,"pos")
    elif topology == "triangular":
        m,k = best_triangular_dims(n)
        graph = nx.triangular_lattice_graph(m,k,periodic=False,with_positions=True)
        if len(graph)!=n:
            raise RuntimeError(f"NetworkX lattice node count differs from notebook's formula: {len(graph)} != {n}")
    else:
        raise ValueError(f"Unknown topology: {topology}")
    graph = nx.convert_node_labels_to_integers(graph,ordering="default",label_attribute="coord")
    base_edges = list(graph.edges())
    nodes = list(graph.nodes())
    for u,_ in base_edges:
        if rng.random() < p:
            candidates = [w for w in nodes if w != u and not graph.has_edge(u,w)]
            if candidates:
                graph.add_edge(u, rng.choice(candidates))
    return graph
