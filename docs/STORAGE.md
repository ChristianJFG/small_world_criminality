# What is kept, generated, and backed up

There are three different kinds of persistence:

1. **Versioned in Git:** the recipe, tests, documentation, and selected final documents. A clone receives these once committed and pushed.
2. **Kept locally but ignored:** run outputs, installed environments, and the historical archive. Git does not back up these files.
3. **Archived externally when scientifically important:** the exact data and metadata supporting a published result, or historical data that cannot yet be recreated. No external upload or backup was performed during cleanup.

## Decisions for this repository

| Material | Location | Git policy | Recommendation |
| --- | --- | --- | --- |
| Python code and tests | `src/`, `tests/` | Keep | Required to regenerate and verify results |
| Experiment settings | `config.toml` | Keep | Review parameter changes with the code |
| Python/dependency recipe | `.python-version`, `pyproject.toml`, `uv.lock` | Keep | One maintained lockfile; regenerate environments |
| Guides and small provenance inventories | `docs/` | Keep | Explain what was run and where local files went |
| Report source and two curated final PDFs | `paper/` | Keep | Small intentional reader-facing deliverables; update for a report revision |
| Quick/full sweep CSV tables | `results/<run>/` | Ignore | Cheap to regenerate; cache locally while iterating |
| Routine PNG/SVG/PDF/HTML plots | `results/<run>/` | Ignore | Replot saved CSVs when useful; regenerate when code or data change |
| Per-run metadata and resolved settings | Beside outputs in `results/<run>/` | Ignore with the run | Keep with locally cached outputs; include in an external scientific archive |
| Historical large CSVs and validation exports | `local/legacy/data/`, `local/legacy/figures/` | Ignore | Preserve and back up externally; current pipeline does not recreate all of these |
| Personal scratch script and empty notebook | `local/legacy/tools/`, `local/legacy/notebooks/` | Ignore | Preserved for inspection, not part of the reproducible application |
| Earlier requirements snapshot | `local/legacy/environment/` | Ignore | Historical record only; use `uv.lock` going forward |
| Environments, caches, generated pip exports, LaTeX intermediates | `.venv*/`, cache paths, `requirements-export.txt`, `paper/` | Ignore | Recreate with uv or the relevant build command |

`.gitignore` uses output directories, not blanket `*.csv`, `*.png`, or `*.pdf` exclusions. Real input datasets and small numerical test fixtures may legitimately need versioning later. Add those deliberately under a clearly documented input/fixture directory rather than mixing them with output files.

## Measured cost, rather than a file-extension rule

Measurements on 2026-11-21 used Python 3.10.12 on this Linux/WSL machine, with single-thread OpenBLAS/OMP. Times below sum per-topology computation, table writing, and plotting; startup and dependency installation are excluded. The observed quick command took about 6 seconds including startup. Other hardware or parameters will differ.

| Run | Work | Recorded run time | CSV size | Four figures per topology | Total run size |
| --- | --- | --- | --- | --- | --- |
| Quick, 200 nodes, 2 seeds | 50 graphs; 10,000 modes | 3.14 seconds | 1.57 MiB | 1.10 MiB | 2.69 MiB |
| Manuscript, 200 nodes, 10 seeds | 3,150 graphs; 630,000 modes | 69.46 seconds | 102.97 MiB | 1.26 MiB | 104.32 MiB |

Exact measured bytes, timings, and code identities are in [RUN_BENCHMARK.json](RUN_BENCHMARK.json). These outputs are fast enough to regenerate after changing settings. It is still useful to keep local CSVs so a plot-only iteration does not rerun the spectra. Do not recompute every existing run just because you opened the project.

The two historical `experiment_data_...csv` files are **343,770,790 and 343,767,709 bytes** (about **328 MiB each**). Their headers include complete eigenvectors, eigenvalues, and additional historical observables. They are not equivalent to the cleaned pipeline's mode tables, and their original execution times are unknown. They must not be treated as disposable caches until their generation and scientific use are understood.

GitHub [blocks regular Git files larger than 100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). For these historical files, use an external dataset archive or object storage with checksums. Git LFS is an option only if collaborators specifically need those datasets versioned alongside code; it adds storage and workflow overhead and is not required for today's cheap regenerated outputs.

## Preserve a result that matters

For a publication or expensive/unique run, archive the entire selected run directory together, including its CSVs, figures, `metadata.json`, and resolved configuration files. Include the matching code revision (or source snapshot if it was not committed), `uv.lock`, and `.python-version`. Save the archive in backed-up storage or a dataset release, and add the archive URL/DOI and checksum to a small tracked note under `docs/`.

Preserving exact numerical outputs is useful even for a cheap run: different numerical backends can choose different bases for degenerate eigenspaces. Code and seeds alone do not guarantee identical per-eigenvector observables across every environment.

Do not archive all exploratory runs forever. Keep the published/selected run and any unique inputs, and regenerate routine trials when needed. No dataset manager, compression dependency, LFS migration, or Git history rewrite is needed for this cleanup.

## Where the old files went

All 15 loose historical/scratch files were moved without changing their contents. [LEGACY_INVENTORY.json](LEGACY_INVENTORY.json) records each original path, new local path, byte size, and SHA-256 checksum. The inventory itself is versioned; the archived payload is ignored and absent from fresh clones. The empty `pipeline_2.ipynb` was preserved as found; it is not a runnable notebook.

Existing `results/` runs were retained in place. Cleanup changed organization and Git visibility, not disk usage: about 665 MiB of legacy files still exist locally, along with generated outputs and environments. The existing Git history was not rewritten. Ignore rules apply to future additions; they do not remove files from other branches or past commits.

## Regenerate

Run commands from the repository root:

```bash
uv sync --locked --extra test
uv run --locked python -m turing_thesis --config config.toml
uv run --locked python -m turing_thesis --preset manuscript --seeds 10 --output results/manuscript
uv run --locked python -m turing_thesis --output results/manuscript --plot-only
```

Use a new output directory to retain another experiment. Reusing a directory overwrites its outputs. Only use runs whose `metadata.json` says `complete`.

To create a pip export when an external tool requires it:

```bash
uv export --locked --extra test --no-emit-project --output-file requirements-export.txt
```

The export is derived, ignored, and can be regenerated from `uv.lock`, following the [uv export guidance](https://docs.astral.sh/uv/concepts/projects/export/).
