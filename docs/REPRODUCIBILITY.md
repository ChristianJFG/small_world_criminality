# Reproducibility and numerical verification

Checked on 2026-11-21, on branch `new_version`, based on research commit `0c5563830e593540d35a6de339332b79102e3f73`.

## Outcome

The spectral reproduction pipeline runs successfully in both the existing environment and a fresh virtual environment installed from the former pinned requirements snapshot (now archived at `local/legacy/environment/requirements-reproducible.txt`). The nine pre-existing tests passed before changes. The expanded suite passes **46 tests** in both environments.

A full manuscript sweep was executed in the fresh environment: **1,050 graphs and 210,000 mode observations per topology**, totaling **3,150 graphs, 630,000 mode observations, and 12 figures**. The 200-node quick sweep was executed independently in both environments: **50 graphs and 10,000 mode observations** in total. Both CSV files and all four PNG figures for each topology were byte-identical across those two quick runs.

This verifies the implemented numerical calculations and local repeatability. It does not establish archival identity with historical manuscript images or empirically validate the model of crime.

## Native uv verification

The project also includes `.python-version` (3.10.12) and a native `uv.lock`, generated with uv **0.11.32**. The lock was seeded from the previously tested Python 3.10 dependency versions. Native `uv sync --locked --extra test` succeeded in `.venv-repro`, preserving the original `.venv`, and `uv run --locked --extra test python -m pytest -q` passed all **46 tests**. Source provenance now includes hashes of the uv lockfile and Python version pin. Other resolved Python/platform combinations remain unverified.

## Changes and issues addressed

- Added a commented `config.toml` with explicit loading, CLI overrides, and validation before computation. Experiment settings remain separate from numerical equations.
- Exposed ring degree, seed offset, quick grid size, and optional per-topology exact probability lists through the CLI/configuration.
- Fixed `p_points` being silently ignored for quick square sweeps. Supplying it now determines the total square sample count; ring/triangular use points per historical region.
- Rejected `p_points` for manuscript sweeps rather than silently ignoring it, and prevented oversampling the historical grids with duplicate selections.
- Rejected nonpositive seed counts instead of returning empty tables, invalid probabilities, configuration typos, odd/oversized ring degrees, and impossible lattices. Historical ring/triangular sample counts must be multiples of three rather than silently rounding down.
- Preserved historical default probability grids, graph construction, mode definitions, and plotting computations.
- Saved exact resolved settings as runnable TOML and JSON, dependency/platform/backend information, source hashes, and output hashes. Git commit alone is insufficient when working files are modified; source hashes identify the actual Python files used.
- Marked run status before computation, with `complete` or `failed` recorded after it. Abrupt process interruption may leave `running`.
- Preserved original metadata during plot-only runs and recorded separate plotting provenance.
- Added a pinned runtime/test dependency closure, excluding unrelated notebook packages found in the existing environment.

## Numerical evidence

The automated suite includes:

1. Exact node counts, connectivity, open square boundaries, seeded graph construction, and preservation of base edges when adding shortcuts.
2. Historical sampling and settings validation, seed offsets, custom probabilities, and effective ring-degree/quick-grid controls.
3. Laplacian eigenpair residuals, orthonormality, nonpositive spectrum, and the trace identity `sum(lambda) = -2 * number_of_edges`.
4. IPR bounds and the uniform zero mode's IPR of `1/n`; independent sums verify mode-weighted degree and clustering.
5. Agreement of the nonlinear RHS with independently written, explicit notebook equations at positive heterogeneous states.
6. Homogeneous fixed-point consistency and numerical linearization against the modal matrix for both homogeneous and nonuniform modes on all three topologies.
7. Short seeded BDF integrations: repeatability, positivity for the tested case, and preservation of the exact homogeneous fixed point.
8. Identical repeated sweep tables, CLI output hashes, replay from a saved snapshot, preservation of provenance on replot, and failed-run metadata.

The full manuscript output was additionally checked for 105 unique probabilities and 10 seeds per topology, exactly 200 modes per graph, finite defined observables, spectral/IPR bounds, the trace identity for every graph, and matching hashes for every table/figure.

## Environment and commands

Verified with CPython **3.10.12**, Linux x86_64, NumPy **2.1.3**, SciPy **1.14.1**, NetworkX **3.4.2**, pandas **2.3.3**, Matplotlib **3.9.2**, seaborn **0.13.2**, and pytest **9.2.0**. The maintained dependency resolution is now `uv.lock`; the earlier pip snapshot is retained only in the ignored local archive. Python 3.11+ parsing is supported in code but was not separately executed in this environment.

Fresh installation and tests (run from the repository root):

```bash
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --locked --extra test
UV_PROJECT_ENVIRONMENT=.venv-repro OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked --extra test python -m pytest -q
```

Full sweep:

```bash
UV_PROJECT_ENVIRONMENT=.venv-repro OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m turing_thesis --preset manuscript --output results/reproducibility-manuscript
```

Local verification outputs (ignored by Git):

- `results/reproducibility-check-a/`: quick run in the existing environment
- `results/reproducibility-check-b/`: matching quick run in the fresh environment
- `results/reproducibility-manuscript/`: full sweep, per-topology metadata, and `verification.json`

## Limits and scientific interpretation

- Historical raw tables and exact historical library versions are unavailable. This is a rerun of the reconstructed definitions, not proof of archival numerical identity.
- Identical seeds are insufficient for universal bitwise reproducibility. BLAS/LAPACK, platform, library versions, and repeated-eigenvalue basis choices can change individual eigenvector observables. The verified byte identity applies to the two tested environments on this machine. Compare across other environments with appropriate numerical tolerances.
- The CLI is a spectral pipeline: it does not integrate nonlinear dynamics or validate final stationary patterns. Physical reaction parameters therefore do not appear as sweep settings.
- The dynamics helper's defaults differ from the original notebook in integration duration, output sampling, solver tolerances, and random generator. The short integration tests do not establish long-time convergence. Its positive-state floor and exponent clipping are numerical regularizations, so fidelity at extreme states is not established by the moderate-state equation check.
- The historical region names are fixed probability labels, not measured small-world indices or dynamical transitions. Graph CSV transitivity is the raw NetworkX value; it is not the historically normalized ring transitivity. Shortest-path and sigma measures are intentionally absent from this figure pipeline.
- Polynomial bootstrap confidence bands are intentionally omitted to avoid the original unseeded plotting variation. This was an existing documented choice.
- Existing output directories are overwritten when reused. Preserve separate output directories, completed metadata, settings, and the matching code revision for experiments you want to keep.
