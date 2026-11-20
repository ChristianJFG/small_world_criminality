"""Command-line experiments. All outputs can be regenerated from the stated seeds."""
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from .core import (Parameters, growth_rates, integrate, make_graph,
                   selected_mode, spectral_metrics)


def _metadata(destination: Path, command: str, config: dict) -> None:
    meta = {"command": command, "config": config,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "networkx": nx.__version__,
                         "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
            "provenance": "Newly computed figures; not original thesis data."}
    (destination / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def sweep(args: argparse.Namespace) -> None:
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    mode_rows, graph_rows = [], []
    for topology in args.topologies:
        for p in args.p:
            for seed in range(args.seeds):
                graph = make_graph(topology, p, seed, n=args.n, side=args.side)
                metrics = spectral_metrics(graph)
                graph_rows.append({"topology": topology, "p": p, "seed": seed,
                                   "nodes": len(graph), "edges": graph.number_of_edges(),
                                   "mean_degree": np.mean([d for _, d in graph.degree()]),
                                   "connected": nx.is_connected(graph)})
                for i in range(len(graph)):
                    mode_rows.append({"topology": topology, "p": p, "seed": seed,
                                      "mode": i, **{k: float(v[i]) for k, v in metrics.items()}})
    modes = pd.DataFrame(mode_rows)
    graphs = pd.DataFrame(graph_rows)
    modes.to_csv(destination / "modes.csv", index=False, float_format="%.12g")
    graphs.to_csv(destination / "graphs.csv", index=False, float_format="%.12g")
    _metadata(destination, "sweep", vars(args) | {"output": str(destination)})

    features = [
        ("ipr", "Inverse participation ratio", "ipr_vs_eigenvalue.png"),
        ("weighted_degree_over_mean_degree", "Mode-weighted degree / mean degree", "degree_vs_eigenvalue.png"),
        ("weighted_clustering", "Mode-weighted local clustering", "clustering_vs_eigenvalue.png"),
    ]
    for feature, label, filename in features:
        fig, ax = plt.subplots(figsize=(9, 5))
        for topology in args.topologies:
            for p in args.p:
                sample = modes[(modes.topology == topology) & (modes.p == p)]
                ax.scatter(sample.eigenvalue_over_mean_degree, sample[feature],
                           alpha=0.38, s=9, label=f"{topology}, p={p:g}")
        ax.set_xlabel("Laplacian eigenvalue / mean degree (L = A - D)")
        ax.set_ylabel(label)
        ax.legend(markerscale=2, fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(destination / (Path(filename).stem + "." + args.format), dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for topology in args.topologies:
        for p in args.p:
            sample = modes[(modes.topology == topology) & (modes.p == p)]
            ax.hist(sample.eigenvalue_over_mean_degree, bins=28, density=True,
                    histtype="step", linewidth=1.4, label=f"{topology}, p={p:g}")
    ax.set_xlabel("Laplacian eigenvalue / mean degree (L = A - D)")
    ax.set_ylabel("Estimated spectral density")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(destination / ("spectral_density." + args.format), dpi=160)
    plt.close(fig)
    print(f"Saved {len(graphs)} graph realizations and {len(modes)} eigenmodes to {destination}")


def validate(args: argparse.Namespace) -> None:
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    graph = make_graph(args.topology, args.p, args.seed, n=args.n, side=args.side)
    prediction = selected_mode(graph, Parameters(), args.quantile, args.epsilon_I)
    parameters = prediction["params"]
    solution = integrate(graph, parameters, args.t_final, seed=args.seed)
    n = len(graph)
    final = solution.y[:, -1].reshape(n, 2)
    centered = final - np.array([parameters.rho_star, parameters.o_star])
    phi = prediction["eigenvectors"][:, prediction["mode_index"]]
    # Eigenvectors are defined only up to sign. The correlation below is sign invariant.
    corr = float(abs(np.corrcoef(phi, centered[:, 0])[0, 1])) if np.std(centered[:, 0]) > 1e-12 else None
    results = {"topology": args.topology, "p": args.p, "seed": args.seed,
               "node_count": n, "target_eigenvalue": float(prediction["eigenvalues"][prediction["target_index"]]),
               "selected_eigenvalue": float(prediction["eigenvalues"][prediction["mode_index"]]),
               "max_growth_rate": prediction["max_growth"], "unstable_modes": prediction["unstable_modes"],
               "d_o": parameters.d_o, "I": parameters.I,
               "final_t": float(solution.t[-1]), "solver_success": bool(solution.success),
               "abs_corr_mode_vs_final_centered_rho": corr,
               "relative_final_change": float(np.linalg.norm(solution.y[:, -1] - solution.y[:, -2]) /
                                              max(np.linalg.norm(solution.y[:, -1]), 1e-12))}
    (destination / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    _metadata(destination, "validate", vars(args) | {"output": str(destination)})
    pd.DataFrame({"node": np.arange(n), "critical_eigenvector": phi,
                  "centered_rho_final": centered[:, 0],
                  "centered_opportunity_final": centered[:, 1]}).to_csv(
                      destination / "final_pattern.csv", index=False)
    values = prediction["eigenvalues"]
    rates = growth_rates(values, parameters)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(values, rates, s=25)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("Largest real growth rate")
    ax.set_title("Modal dispersion relation (discrete graph spectrum)")
    fig.tight_layout()
    fig.savefig(destination / "dispersion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    phi_aligned = phi if (corr is None or np.corrcoef(phi, centered[:, 0])[0, 1] >= 0) else -phi
    ax.plot(phi_aligned / max(np.max(np.abs(phi_aligned)), 1e-12), label="Critical eigenvector (rescaled)")
    ax.plot(centered[:, 0] / max(np.max(np.abs(centered[:, 0])), 1e-12), label="Final centered density (rescaled)")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Relative amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "mode_vs_pattern.png", dpi=160)
    plt.close(fig)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible network dynamics research")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sweep", help="Compute spectral observables on shortcut networks")
    s.add_argument("--topologies", nargs="+", choices=["ring", "square", "triangular"],
                   default=["ring", "square", "triangular"])
    s.add_argument("--p", nargs="+", type=float, default=[0.0, 0.1, 0.5])
    s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--n", type=int, default=36, help="Ring size")
    s.add_argument("--side", type=int, default=6, help="Lattice side parameter")
    s.add_argument("--format", choices=["png", "svg"], default="png",
                   help="Plot format; SVG makes GitHub-friendly vector graphics")
    s.add_argument("--output", default="outputs/sweep")
    v = sub.add_parser("validate", help="Compare linear mode with finite-time nonlinear integration")
    v.add_argument("--topology", choices=["ring", "square", "triangular"], default="ring")
    v.add_argument("--p", type=float, default=0.1)
    v.add_argument("--seed", type=int, default=3)
    v.add_argument("--n", type=int, default=20)
    v.add_argument("--side", type=int, default=5)
    v.add_argument("--quantile", type=float, default=0.5)
    v.add_argument("--epsilon-I", dest="epsilon_I", type=float, default=0.01)
    v.add_argument("--t-final", type=float, default=1200)
    v.add_argument("--output", default="outputs/validation")
    args = parser.parse_args()
    if args.command == "sweep":
        if args.seeds <= 0:
            parser.error("--seeds must be positive")
        sweep(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
