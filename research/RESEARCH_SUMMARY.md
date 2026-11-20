# Spectral localization and Turing instabilities in shortcut-enhanced networks

**Research question.** How do topology and shortcuts change the available Laplacian modes, and which of those modes can grow under coupled reaction–diffusion dynamics?

**Model.** For a graph with adjacency matrix \(A\), degree matrix \(D\), and Laplacian \(L=A-D\), a perturbation along eigenvector \(\phi_\alpha\) with \(L\phi_\alpha=\lambda_\alpha\phi_\alpha\) evolves, after linearization, according to a two-variable matrix \(M(\lambda_\alpha)\). The mode grows when an eigenvalue of this matrix has positive real part. The inverse participation ratio, \(\mathrm{IPR}(\phi)=\sum_i|\phi_i|^4/(\sum_i|\phi_i|^2)^2\), quantifies spatial localization. The numerical code computes this matrix directly from the original model and verifies it against a finite-difference linearization of the full nonlinear equations.

**Approach.** Generate ring, square and triangular networks; add long-range shortcut edges at controlled probabilities; compute graph spectra, IPR, eigenvector-weighted node degree and local clustering; and integrate the original two-variable model in selected cases. The configuration and random seeds are recorded with every run.

**Research observations from my manuscript.** (1) Shortcuts modify spectral density while some geometric signatures of regular lattices remain. (2) Eigenmode localization varies by spectral region: more shortcuts do not imply that every mode becomes more localized. (3) The spectral position of a mode correlates with the node degree and clustering on which the mode has support. This means that the geometry supplies a *repertoire of modes*, while the nonlinear model's dispersion relation determines which are capable of growth.

**Limitations.** Spectral instability predicts initial modal growth, not the eventual sign, amplitude, or persistence of the nonlinear state. Eigenvector IPR is ambiguous within exactly degenerate eigenspaces. The clean source is a reimplementation of the mathematical methods and generates fresh demonstration figures; it is **not an exact reconstruction of the archived thesis figure datasets**. The original notebooks remain available on `master` for provenance.

**Research relevance.** The project demonstrates theoretical analysis, reproducible scientific programming, and measurement of how network structure shapes potential collective dynamics. These techniques can transfer to other questions in complex networks and representation learning without implying that Turing instability itself explains AI-model behavior.
