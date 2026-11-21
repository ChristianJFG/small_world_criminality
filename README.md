# Turing patterns on networks

Reproducible spectral analysis of ring, square, and triangular networks. Start here; the parameter guide is in [docs/RUNNING.md](docs/RUNNING.md).

## Run with uv

From the repository directory in WSL/Ubuntu:

```bash
uv sync --locked --extra test
uv run --locked python -m turing_thesis --config config.toml
uv run --locked --extra test python -m pytest -q
```

Edit **`config.toml`** to change the experiment. No manual environment activation is required. Python is pinned in `.python-version`; dependencies are locked in `uv.lock`.

`uv sync` manages `.venv`. To keep an existing notebook environment intact, prefix each command with `UV_PROJECT_ENVIRONMENT=.venv-repro` as described in the [running guide](docs/RUNNING.md).

## Repository map

```text
config.toml             Editable experiment settings
pyproject.toml          Package and dependency declarations
uv.lock                 Exact dependency resolution; maintain this one lock
.python-version         Tested Python version
src/turing_thesis/      Reusable calculation and plotting code
tests/                  Scientific and reproducibility checks
docs/                   Running guide, storage policy, verification, provenance
paper/                  Report source and two curated final PDFs
results/                Generated runs; local, ignored by Git
local/legacy/           Preserved old files; local, ignored by Git
.venv*/                 Installed environments; local, ignored by Git
```

**Git keeps the recipe and the small final documents.** Routine CSVs, figures, HTML exports, environments, caches, and logs stay local and are regenerated as needed. Important historical data need a separate backup because ignored files are absent from clones. See [what to keep and what to regenerate](docs/STORAGE.md).

The earlier loose CSVs, images, HTML files, scratch script, and empty notebook were preserved in `local/legacy/`; their paths, sizes, and checksums are recorded in [the inventory](docs/LEGACY_INVENTORY.json). No research files were deleted.

## Common tasks

```bash
# Small trial
uv run --locked python -m turing_thesis --config config.toml --nodes 20 --seeds 1 --output results/trial

# Full historical sample size
uv run --locked python -m turing_thesis --preset manuscript --seeds 10 --output results/manuscript

# Replot the saved full run without recomputing spectra
uv run --locked python -m turing_thesis --output results/manuscript --plot-only
```

Use a new output directory for experiments you want to keep; rerunning into an existing directory overwrites outputs. Ordinary runs write their settings, hashes, and environment metadata beside the tables. A run is usable when its `metadata.json` says `complete`.

- [Verification and limitations](docs/REPRODUCIBILITY.md)
- [Figure provenance](docs/FIGURE_PROVENANCE.md)
- [Visual introduction](paper/Turing_Patterns_on_Networks.pdf)
- [Technical report](paper/Christian_Flores_Turing_Reproduced_Report.pdf)
