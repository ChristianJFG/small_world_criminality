# Spectral localization and Turing instabilities in shortcut-enhanced networks

**Christian Jesús Flores Gómez** — Graduate research sample, complexity science / nonlinear dynamics.

## Research question

How do network topology and long-range shortcuts influence the onset and spatial localization of patterns in coupled nonlinear dynamics? This project analyzes a synthetic, crime-hotspot-inspired reaction–diffusion model. It does **not** infer that observed crime hotspots are necessarily Turing patterns.

## Mathematical formulation

Let $A$ be the adjacency matrix, $D$ the diagonal degree matrix, and $L=A-D$ the negative-semidefinite graph Laplacian. Its eigenpairs satisfy

$$
L\phi_\alpha=\lambda_\alpha\phi_\alpha,\qquad \lambda_\alpha\le 0.
$$

By linearizing the two-variable dynamics about a homogeneous state and expanding perturbations into Laplacian eigenmodes, the evolution of the modal coefficients satisfies

$$
\frac{d}{dt}
\begin{pmatrix}a_\alpha\\b_\alpha\end{pmatrix}
=M(\lambda_\alpha)
\begin{pmatrix}a_\alpha\\b_\alpha\end{pmatrix}.
$$

A mode can grow when the maximum real part of an eigenvalue of $M(\lambda_\alpha)$ is positive. Thus the network supplies the *discrete modes*, while the kinetic and diffusion parameters determine the *unstable spectral window*.

For each eigenvector, I evaluated the inverse participation ratio,

$$
\mathrm{IPR}(\phi)=
\frac{\sum_i|\phi_i|^4}{\left(\sum_i|\phi_i|^2\right)^2}.
$$

For a normalized mode distributed evenly over $N$ nodes, IPR is approximately $1/N$; a mode concentrated on few nodes has higher IPR.

## Computational approach

I studied three initially regular topologies: a ring, a square lattice, and a triangular lattice. Long-range shortcuts were added in a Newman–Watts-style procedure, preserving existing local edges. Across network realizations, I compared spectral density, eigenmode localization, eigenvector-weighted node degree and clustering, and representative nonlinear dynamics.

## Principal observations

**1. Shortcuts change the spectral repertoire.** Regular geometries generate distinctive spectral features; adding shortcuts changes and smooths the finite-network eigenvalue distributions. In the small-world regime, traces of the original geometry can remain.

**2. Localization is spectral and non-monotonic.** High-IPR modes arise in particular spectral regions, especially toward spectral edges. Increasing shortcut probability alone does not guarantee that the dynamically selected mode becomes more localized.

**3. Modes have distinct structural supports.** Modal weight on high- or low-degree nodes varies across the Laplacian spectrum, and its association with local clustering depends on the underlying geometry.

The central lesson is that structure and dynamics must be analyzed together: an unstable spectral window must overlap the graph's available modes. Neither graph topology nor IPR alone determines the eventual nonlinear steady-state pattern.

## Scope, limitations, and provenance

These results are summarized from my Spanish-language LaTeX thesis draft, *Efectos de la infraestructura urbana de transporte en los patrones espaciales de criminalidad*. The underlying experiments are synthetic-network comparisons. The original plots in the accompanying three-page PDF were relabeled in English **without recomputing the data**.

The full nonlinear model has been integrated in representative experiments, but broad parameter sweeps are primarily spectral/linearized analyses. The critical linear mode need not reproduce the sign, amplitude, or persistent localization of the final nonlinear pattern. Original notebooks are exploratory and not advertised as a fully validated one-command replication package.

## Relevance to Principles of Intelligence

The link is methodological rather than a claim of prior interpretability research: I used mathematical models, network spectral analysis, and numerical experiments to investigate how underlying structural constraints shape possible collective behaviors. I am interested in applying similar reasoning to hierarchical data, multiscale structure, and learned representations.
