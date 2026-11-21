# Paper assets

The two small final PDF reports are intentional versioned deliverables, retained so readers can open them without installing LaTeX. Ordinary experiment images belong in ignored `results/`, not here. Replace a final PDF only for a deliberate report revision.

`research_report.tex` is the source of the technical report. Its six image references now point at `../results/manuscript/`, replacing the missing historical `figures_en/` directory. From the repository root, generate the figures first:

```bash
uv run --locked python -m turing_thesis --preset manuscript --seeds 10 --output results/manuscript
```

With a LaTeX installation, compile from `paper/`:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error research_report.tex
```

The command creates `research_report.pdf`, which is a local draft and ignored. The two named published PDFs are retained as supplied; they were not rebuilt during repository cleanup. LaTeX is an external tool and is not installed by uv. A rebuilt report may differ from the curated PDF because the figures are recomputed.
