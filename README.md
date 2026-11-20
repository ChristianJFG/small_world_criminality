# Spectral Localization and Turing Instabilities in Complex Networks

**Christian Jesús Flores Gómez** — Graduate research in complexity science and nonlinear dynamics

> **Research sample for the Principles of Intelligence application.** Start with the [concise technical summary](research_sample/RESEARCH_SUMMARY.md), then inspect the [tested spectral example](research_sample/src/spectral_demo.py). The concise technical summary is readable directly on GitHub; a separate illustrated English PDF and LaTeX package has also been prepared for the application.

## Research question

How does the topology of a complex network influence which Laplacian eigenmodes can become unstable in reaction–diffusion dynamics, and how spatially localized might the resulting patterns be?

The model is inspired by mathematical crime-hotspot formation but uses **synthetic network geometries**, not observed crime forecasting data.

## Methods

- Compare regular ring, square-lattice, and triangular-lattice topologies with added long-range shortcuts.
- Compute the graph Laplacian under the convention **L = A − D**, its eigenvalues and eigenvectors.
- Use linear-stability/dispersion analysis to identify the modes with positive growth rates.
- Characterize eigenmode localization with the inverse participation ratio (IPR).
- Analyze associations with node degree and local clustering.
- Compare representative spectral predictions with nonlinear model integration.

## Main findings

1. Shortcuts modify spectral density while networks retain traces of their original geometry.
2. Eigenvector localization varies strongly across the Laplacian spectrum; increasing shortcuts does **not** guarantee a more localized unstable mode.
3. Different spectral regions have different topological supports. Modal degree and clustering associations depend on spectral position and base geometry.
4. Pattern selection depends on an overlap between the dynamics' unstable spectral window and the eigenmodes actually supported by the graph.

These findings describe spectral and linearized mechanisms. A critical eigenmode does not by itself establish the final nonlinear stationary pattern.

## Where to start

- [Technical research summary (English)](research_sample/RESEARCH_SUMMARY.md) — research question, equations, findings, limitations, and relevance.
- [Minimal numerical example](research_sample/src/spectral_demo.py) — executable graph/Laplacian/IPR analysis with documented scope.
- [Demo dependencies](research_sample/requirements_demo.txt)
- [Original exploratory notebook: eigenvector vs final pattern](critical_eigenvector_vs_final_pattern_validation.ipynb)
- [Original experimental pipeline](pipeline.ipynb)
- [Original research functions](functions.py)

The original notebooks are **research artifacts**, not a fully audited reproducibility package. In particular, a legacy dynamics function uses a hardcoded 16-node index range. The minimal example is intentionally separated from that code and does not claim to reproduce the manuscript's figures.

Run the standalone example from the repository root:

```bash
pip install -r research_sample/requirements_demo.txt
python research_sample/src/spectral_demo.py --topology triangular --p 0.1 --seed 17 --output spectral_demo.csv
```

The example exports spectral metrics for each graph eigenmode; no manuscript plots are recomputed.

## Research context

This is mathematical and computational research on emergent phenomena and collective dynamics, **not** a completed AI-safety, percolation, or mechanistic-interpretability study. Its relevance to Principles of Intelligence is methodological: spectral graph theory, critical-instability reasoning, numerical experimentation, and the study of how structure constrains emergent behavior.

**Figures:** A separate three-page English PDF with original data and translated plot labels is available in the accompanying project package. No figures or datasets have been recomputed to make the translation.
