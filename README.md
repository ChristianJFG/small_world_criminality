# Turing patterns on complex networks — reproducible thesis figures

**Christian Jesús Flores Gómez** · Graduate research in complexity science / nonlinear dynamics

This branch is a **source-based reconstruction** of my original `ring.ipynb`, `square.ipynb` (rectangular grid), and `triangular.ipynb` notebooks. It generates the **same six *types* of figures used in my three-page research report** from new numerical experiments, rather than unrelated example plots.

## Quick start

Python 3.10+ (recommended: 3.11–3.13), with NumPy, SciPy, NetworkX, pandas, Matplotlib, and seaborn:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q

# Fast smoke run: same graph generators, observables and plotting definitions,
# fewer parameter values / realizations than the published manuscript.
python -m turing_thesis --preset quick

# Historical sampling settings from the original notebooks (more expensive):
python -m turing_thesis --preset manuscript

# Rebuild all figures from the saved numerical tables, without rerunning eigenproblems:
python -m turing_thesis --preset manuscript --plot-only
```

`--preset manuscript` runs **200 nodes, 105 shortcut probabilities, and 10 network realizations per probability, separately for each topology** (3,150 graphs and 630,000 eigenmode records total). It generates twelve English-labeled plots: spectral density, IPR, quadratic degree fit and quadratic clustering fit for all three geometries. Six of these are used in the three-page research report:

| Report figure | File created by the code |
| --- | --- |
| Ring spectral density | `results/ring/density_ring.png` |
| Triangular spectral density | `results/triangular/density_triangular.png` |
| Ring IPR | `results/ring/ipr_ring.png` |
| Triangular IPR | `results/triangular/ipr_triangular.png` |
| Triangular weighted-degree trend | `results/triangular/degree_regression_triangular.png` |
| Triangular weighted-clustering trend | `results/triangular/clustering_regression_triangular.png` |

The figures in the **revised PDF** were rerun using the manuscript preset, not relabeled screenshots or unrelated demonstrations. They reproduce the original analysis specification and closely match the trend and scale of the corresponding thesis images. Exact pixel-for-pixel duplication is **not asserted** because the original raw per-realization tables and original package versions were not archived.

For efficient reruns, set a single BLAS thread on machines where it helps:

```bash
# Linux / macOS
OPENBLAS_NUM_THREADS=1 python -m turing_thesis --preset manuscript
# Windows PowerShell
$env:OPENBLAS_NUM_THREADS="1"; python -m turing_thesis --preset manuscript
```

## What was preserved from the original notebooks

- **Network generation:** `nx.newman_watts_strogatz_graph(n=200,k=4,p,seed)` for ring; **open** (nonperiodic) rectangular `grid_2d_graph` and finite triangular `triangular_lattice_graph` for the others. Every original edge independently triggers a *possible added shortcut* with probability `p`, implemented using Python `random.Random(seed)` as in the originals; pre-existing edges are never removed.
- **Source parameter grids:** ring: 35 values each in `[0,0.01]`, `[0.09,0.2]`, `[0.9,1]`; square: 105 logarithmically spaced values in `[1e-4,1]`; triangular: 35 values each in `[0,0.001]`, `[0.05,0.1]`, `[0.9,1]`. Ten seeds per probability in the original figure-generation notebooks.
- **Regime labels:** original notebook-defined boundaries, not a universal physical critical point. They were translated from Spanish to English.
- **Spectral convention:** negative semidefinite graph Laplacian `L=A-D`. Symmetric eigendecomposition (`numpy.linalg.eigh`). IPR: `sum(phi_i**4)/(sum(phi_i**2)**2)`. Mode-weighted degree/clustering are based on normalized squared eigenvector weights.
- **Plots:** `seaborn.kdeplot` with *joint/common density normalization*; IPR against `lambda / mean_degree`; mode-weighted degree and clustering ratios regressed against normalized eigenvalue using a quadratic polynomial (`order=2`).

## What changed, and why

This is a reproducibility refactor, not a claim that all legacy notebooks executed successfully as-is:

- The original ring, square, and triangular notebooks contain duplicated functions, notebook-global state, and graph-size-specific integration code. They remain preserved on the **master** branch, while this branch exposes the tested implementation in `src/turing_thesis/`.
- Expensive `sns.lineplot` aggregation is replaced with an equivalent direct Matplotlib line/marker rendering of the **same calculated mode data**. The quadratic OLS coefficients use `numpy.polyfit` rather than a seaborn wrapper; **unseeded bootstrap confidence bands were removed**, so the revised PDF has deterministic curves but not exactly identical pixels.
- `square.ipynb` and `triangular.ipynb` had a hardcoded `np.arange(16)` in their *nonlinear* ODE implementation. `dynamics.py` uses array operations to handle arbitrary node counts; this correction does not change spectral plots, because the original figure pipeline called `get_dynamics(..., integration=False)`.
- The original notebook also computes graph-level summary statistics that are not used in the six paper panels. To reduce expensive repeated all-pairs shortest paths, this reimplementation omits that extra computation from the default figure sweep; the Laplacian and plotted mode observables are unchanged.
- Spectral eigenvectors within exactly degenerate eigenspaces are basis-dependent. NumPy/BLAS versions can therefore affect individual IPR values even when the eigenvalue distribution agrees.

## Project structure

```text
README.md
pyproject.toml
src/turing_thesis/
    networks.py        # original ring/rectangular/triangular shortcut generators
    spectral.py        # Laplacian eigenmodes, IPR and stability matrix
    experiments.py     # original p grids, 105 x 10 sweeps, mode tables
    plots.py           # all four original figure families in English
    dynamics.py        # full nonlinear two-variable RHS, corrected n indexing
    __main__.py        # single-command plotting/CSV pipeline
tests/test_reproduction.py
research/FIGURE_PROVENANCE.md
paper/research_report.tex
```

To rebuild the PDF from its six source figures, run the same full analysis directly into the paper folder, then compile LaTeX:

```bash
python -m turing_thesis --preset manuscript --output paper/figures_en
cd paper
pdflatex research_report.tex
```

`results/` is created automatically and contains `modes.csv`, `graphs.csv`, `metadata.json` and all figures for each topology. Plot-only mode consumes the saved `modes.csv`, so you can iterate on figure formatting without recomputing eigenvectors.

**Scope:** these are synthetic network simulations motivated by a crime-hotspot reaction–diffusion model, **not** an empirical crime-prediction system or a demonstration that observed crime clusters are Turing patterns. The research and mathematics are the author's; the repository is an English-language reproducibility refactor of his original work.
