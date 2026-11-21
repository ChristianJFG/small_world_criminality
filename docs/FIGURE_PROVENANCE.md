# Figure provenance: original notebooks → reproducible files

Sources checked in the original `master` branch of `ChristianJFG/small_world_criminality`:

| Notebook | Network-generator cells | Dynamics/spectral statistics | Experiment sampling | Manuscript figure commands |
| --- | --- | --- | --- | --- |
| `ring.ipynb` | Cell 14, `process_graph` using `newman_watts_strogatz_graph` | Cell 11 `get_dynamics`, `integration=False` | Cell 14 + cell 15: 200 nodes, 4 neighbors, 105 probabilities, 10 seeds | Cells 26–31 |
| `square.ipynb` | Cell 3 `newman_watts_lattice_graph`, `type='square'` | Cell 12 `get_dynamics` | Cells 15–16: 200 nodes, 105 log-spaced probabilities, 10 seeds | Cells 27–32 |
| `triangular.ipynb` | Cell 3 `newman_watts_lattice_graph`, `type='triangular'` | Cell 12 `get_dynamics` | Cells 15–16: 200 nodes, 105 probabilities in three ranges, 10 seeds | Cells 27–32 |

Indices are zero-based notebook cell indices, including Markdown cells. The original `master` contains the intact notebooks. The equations and mode statistics in the cleaned package were compared with their implementations, not inferred from the filenames. The old `functions.py` on master is **not** taken as authoritative because it differs from the updated notebooks (notably on simulation behavior and indexing).

### Mapping of the six PDF figure panels

1. `density_ring.png` and `density_triangular.png`: pooled Laplacian eigenvalues, smoothed KDE, hue by the notebook's three topology regimes; common hue normalization preserved (critical: separate KDE calls would produce a 3× density error).
2. `ipr_ring.png` and `ipr_triangular.png`: IPR of **every** eigenvector versus `lambda / <k>` from each sampled network realization; the plotting wrapper changed for speed, not the numerical quantity.
3. `degree_regression_triangular.png`: quadratic least-squares curves of normalized eigenvector-weighted degree versus normalized eigenvalue, separated by regime.
4. `clustering_regression_triangular.png`: corresponding normalized eigenvector-weighted local clustering; the base square lattice has zero clustering in its zero-shortcut limit, so its ratio is explicitly marked undefined there.

### Reproduction status (2026-11-20)

- The full *manuscript* preset was executed locally for **ring, square, and triangular**: 1,050 network realizations and 210,000 eigenmode records **per topology**.
- Nine automated checks passed, covering graph structure, deterministic seeding, spectral bounds, IPR and consistency of the full nonlinear RHS linearization at the homogeneous state.
- The six corresponding figures in the revised three-page PDF were **redrawn from the newly computed mode tables** and labeled in English. They closely track the qualitative and quantitative behavior of the original Spanish thesis plots.
- Exact archival identity was not established: original raw numerical `experiment_df` tables and 2026 installed library versions are unavailable; one deliberately removed cosmetic layer is the stochastic regression bootstrap confidence band.
