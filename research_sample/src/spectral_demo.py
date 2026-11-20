"""Minimal reproducible Laplacian spectral analysis for shortcut-enhanced networks.

This is a standalone illustration of the spectral methods used in the project.
It does NOT reproduce the original paper figures or simulate nonlinear dynamics.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def shortcut_lattice(topology: str, p: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    if topology == "ring":
        return nx.newman_watts_strogatz_graph(n=100, k=4, p=p, seed=seed)
    if topology == "square":
        graph = nx.grid_2d_graph(10, 10)
    elif topology == "triangular":
        graph = nx.triangular_lattice_graph(9, 9)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    graph = nx.convert_node_labels_to_integers(graph)
    nodes = list(graph.nodes)
    for u, _ in list(graph.edges):
        if rng.random() >= p:
            continue
        candidates = [w for w in nodes if w != u and not graph.has_edge(u, w)]
        if candidates:
            graph.add_edge(u, int(rng.choice(candidates)))
    return graph


def spectral_metrics(graph: nx.Graph) -> pd.DataFrame:
    L = -nx.laplacian_matrix(graph).toarray().astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    weights = np.square(eigenvectors)
    degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
    clustering = np.asarray([nx.clustering(graph, n) for n in graph.nodes])
    mean_degree = float(degrees.mean())
    return pd.DataFrame({
        "eigenvalue": eigenvalues,
        "eigenvalue_over_mean_degree": eigenvalues / mean_degree,
        "ipr": np.sum(np.square(weights), axis=0),
        "weighted_degree_over_mean_degree": weights.T @ degrees / mean_degree,
        "weighted_clustering": weights.T @ clustering,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["ring", "square", "triangular"], default="ring")
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("spectral_demo.csv"))
    args = parser.parse_args()
    if not 0 <= args.p <= 1:
        parser.error("--p must be between 0 and 1")
    graph = shortcut_lattice(args.topology, args.p, args.seed)
    data = spectral_metrics(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Topology={args.topology}, p={args.p}, nodes={len(graph)}, edges={graph.number_of_edges()}")
    print(f"Saved {len(data)} eigenmodes to {args.output}")


if __name__ == "__main__":
    main()
