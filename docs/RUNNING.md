# Turing patterns on networks

Reconstructs the spectral figure pipeline from the ring, square, and triangular network notebooks. Computes negative graph Laplacian spectra, inverse participation ratios (IPR), and mode-weighted network measures.

- `src/turing_thesis/`: graph generation, spectral analysis, nonlinear dynamics, and plots
- `config.toml`: editable settings for the spectral sweep
- `tests/`: numerical, configuration, and reproducibility checks
- [Verification and limitations](REPRODUCIBILITY.md)
- [Figure provenance](FIGURE_PROVENANCE.md)
- [Visual introduction (PDF)](../paper/Turing_Patterns_on_Networks.pdf)
- [Technical report (PDF)](../paper/Christian_Flores_Turing_Reproduced_Report.pdf)

## Install with uv (recommended)

Run these commands from the repository directory in WSL/Ubuntu:

```bash
uv sync --locked --extra test
uv run --locked python -m turing_thesis --config config.toml
uv run --locked --extra test python -m pytest -q
```

No manual environment activation is needed. `.python-version` selects the tested Python **3.10.12**; `uv.lock` records the resolved dependency versions and package hashes. The Python 3.10 lock selections retain the versions verified below. The lock also resolves other supported Python/platform combinations, which have not all been tested.

As described in the [uv documentation](https://docs.astral.sh/uv/concepts/projects/sync/), `--locked` rejects an outdated lockfile instead of silently updating it. Keep `uv.lock` and `.python-version` in version control. After an intentional dependency change, run `uv lock` and rerun the tests. For intentional upgrades, use `uv lock --upgrade-package NAME`, then verify the results before adopting the new lock.

Examples:

```bash
uv run --locked python -m turing_thesis --config config.toml --nodes 20 --seeds 1 --output results/trial
uv run --locked python -m turing_thesis --preset manuscript --seeds 10 --output results/manuscript
```

`uv sync` normally uses `.venv` and removes undeclared packages there. To preserve an existing notebook environment, select another environment, using the same variable for every command (Linux/WSL):

```bash
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --locked --extra test
UV_PROJECT_ENVIRONMENT=.venv-repro uv run --locked python -m turing_thesis --config config.toml
```

The commands below using plain `python` also work through `uv run --locked python`.

## Optional pip export

`uv.lock` is the sole maintained dependency snapshot. If another tool requires pip requirements, generate an ignored export rather than maintaining a second lock by hand:

```bash
uv export --locked --extra test --no-emit-project --output-file requirements-export.txt
```

This follows [uv's guidance on exporting lockfiles](https://docs.astral.sh/uv/concepts/projects/export/). The export can be recreated whenever needed. The earlier manually pinned file has been preserved at `local/legacy/environment/requirements-reproducible.txt` only as a historical record.

## Change experiment parameters

A separate, version-controlled configuration file is appropriate here: it separates experimental choices from equations and makes settings easy to review. We use [TOML](https://docs.python.org/3/library/tomllib.html) because it supports comments and is supported by Python's standard library from 3.11; Python 3.10 uses the declared `tomli` fallback. No configuration framework is needed for this project.

Edit `config.toml`, then run:

```bash
python -m turing_thesis --config config.toml
```

| Setting | Meaning |
| --- | --- |
| `topologies` | Any nonempty subset of `ring`, `square`, `triangular` |
| `nodes` | Exact node count; square/triangular require a constructible lattice (200 and 20 both work) |
| `preset` | `quick` samples the historical grids; `manuscript` uses all 105 probabilities per topology |
| `seeds` | Number of graph realizations **per probability**; omitted: 2 for quick, 10 for manuscript |
| `seed_start` | First integer seed; e.g. 7 with 2 seeds runs seeds 7 and 8 |
| `ring_k` | Even base ring degree, at least 2 and strictly below node count; applies to the ring |
| `p_points` | Optional quick size: per region for ring/triangular (1–35), total for square (1–105) |
| `[probabilities]` | Optional explicit list for each selected topology; overrides its preset grid |
| `output` | Output directory; relative paths in the file start at the file's directory |
| `format` | `png`, `svg`, or `pdf` |

Leave optional settings commented out to preserve the original sampling. Unknown keys, duplicate probabilities, invalid seeds/degrees, and impossible lattice sizes fail before a sweep starts. The three region thresholds remain the historical definitions in `experiments.py`; they are descriptive labels, not dynamical phase boundaries.

Explicit command-line values override the file. Output paths passed on the CLI start at the current working directory. Without `--config`, the historical built-in defaults apply; a nearby file is never silently loaded.

```bash
# Small trial, without editing the file
python -m turing_thesis --config config.toml --nodes 20 --seeds 1 --output results/trial

# Historical full sweep: 105 probabilities x 10 seeds x 3 topologies
python -m turing_thesis --preset manuscript --seeds 10 --output results/manuscript
```

The example file explicitly sets `seeds = 2`: changing only its preset keeps that value. Set it to 10 or override with `--seeds 10` for the historical manuscript sample size. A custom probability grid replaces the historical grid even when the preset is `manuscript`.

To use your own probabilities, uncomment the entire table at the **end** of `config.toml`:

```toml
[probabilities]
ring = [0.0, 0.1, 1.0]
square = [0.0001, 0.01, 0.1, 1.0]
triangular = [0.0, 0.05, 1.0]
```

Top-level settings must stay above `[probabilities]` because TOML keys below a table heading belong to that table. Supply lists only for selected topologies; missing lists retain preset sampling.

## Repeat and inspect a run

Each topology directory contains:

- `graphs.csv` and `modes.csv`: graph and eigenmode observations
- four figures: density, IPR, degree regression, clustering regression
- `resolved_config.toml`: runnable snapshot with exact probabilities and resolved seed count
- `resolved_config.json`: settings including explicit seed values
- `metadata.json`: completion status, settings, source hashes/Git revision, installed package versions, NumPy numerical backend, thread settings, and SHA-256 hashes of tables and figures

Replay the saved snapshot into a new directory:

```bash
python -m turing_thesis --config results/trial/ring/resolved_config.toml --output results/replay
```

A snapshot retains **all selected topologies** from the original invocation. It uses exact saved grids. Existing output files are overwritten on rerun; use a new `--output` directory for each experiment you want to keep. Failed computations mark metadata as `failed`; interrupted processes can leave `running`, so use only outputs with `status = "complete"`.

Replot saved data without recomputing spectra:

```bash
python -m turing_thesis --config config.toml --output results/trial --plot-only --format svg
```

This preserves original simulation metadata and adds `replot_metadata.json` with input/output hashes and plotting provenance. It uses the selected topology directories' saved tables; the supplied sweep parameters do not recompute or relabel those data.

For the closest match to our numerical environment, set these **before starting Python** (Linux/WSL):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m turing_thesis --config config.toml
```

Identical seeds do not guarantee identical eigenvector observables across numerical backends: repeated eigenvalues permit different orthonormal bases. Preserve settings, code, and the recorded environment; compare numerical results with tolerances across platforms. PNG byte identity was verified locally, not promised for every system or image format.

## Scope of the pipeline

The CLI reproduces **spectral observables**, not time-evolved crime/opportunity patterns. Its results do not depend on `alpha`, `beta`, `delta`, `d_o`, or `I`, so these are not presented as sweep controls. Nonlinear equations are separately available in `dynamics.integrate`, and their default physical parameters are in `spectral.default_model_parameters`. Model changes must keep the derived steady state and modal matrix consistent.

The dynamics helper uses seeded relative perturbations, BDF integration, a default final time of 1000, and tighter tolerances than the original notebook. It does not reproduce the notebook's million-time-unit integration or its legacy random stream exactly, and a successful integration alone does not establish a stationary pattern. See the verification report for the limits of the checks.
