# Turing patterns on networks

Reconstructs the figure pipeline from the ring, square, and triangular network notebooks. Computes Laplacian spectra, inverse participation ratios, and mode-weighted network measures.

- `src/turing_thesis/`: graph generation, spectral analysis, simulations, and plots
- `tests/`: reproduction checks
- `paper/Christian_Flores_Turing_Reproduced_Report.pdf`: report

Install and run:

```bash
python -m pip install -e '.[test]'
python -m turing_thesis --preset quick
```

Use `--preset manuscript` for the full parameter sweep.