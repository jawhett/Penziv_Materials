# Fundamental Framework for First-Principles Materials Property Prediction

Predicting material properties for novel, exotic, or standard materials without empirical shortcuts requires a hierarchical multiscale continuum rooted strictly in quantum electrodynamics, density functional and many-body perturbation theory, statistical thermodynamics, and non-linear continuum mechanics.

---

## 🏛️ Multiscale First-Principles Pyramid

```
Macroscale / Component (FEA, Process Modeling)
  Governing Equations: ∇·Σ + b = ρ ü | Path-dependent Internal State Variables: S(t) = S₀ + ∫ f(S, σ, T, ε̇) dt
  Outputs: Residual stress, fatigue life, macroscopic yielding, dimensional tolerances
      ▲
      │
Continuum Micromechanics (CPFEM, Phase-Field Fracture, Homogenization)
  Governing Equations: F = Fᵉ·Fᵖ | Lᵖ = ∑ γ̇ᵃ(s₀ᵃ ⊗ m₀ᵃ) | E(u,d) = ∫ ψ(ε,d)dV + G_c Γ_l(d) | ⟨σ : δε⟩ = Σ : δE
      ▲
      │
Mesoscale Kinetics & Microstructure (Phase Field, DDD, CALPHAD/DICTRA)
  Governing Equations: ∂c/∂t = ∇·(M ∇(δF/δc)) | ∂η/∂t = -L(δF/δη) | f_PK = (σ·b) × ξ
  Scale Bridge: Interfacial stiffness, mobility tensors, dislocation interaction matrices
      ▲
      │
Atomistic Energetics & Defects (AIMD, Equivariant MLIPs, kMC, NEB)
  Governing Equations: M_I R̈_I = -∇_R V_BO(R) | Γ = ν₀ exp(-ΔE^‡/k_B T) | E = ∑ E_i(ρ_i)
  Scale Bridge: γ-surfaces, defect formation energies ΔG_f(q), migration barriers ΔE_m
      ▲
      │
Quantum / Electronic Structure (DFT, GW, BSE, DMFT, DFPT)
  Governing Equations: (-ħ²/2m ∇² + v_eff)ψ_i = ε_i ψ_i | G = G₀ + G₀ Σ G | H_BSE | Δ_nk(iω)
  Primary Inputs: Atomic numbers Z_I, positions R_I, fundamental constants (ħ, e, m_e, c)
```

---

## 1. Electronic Structure, Optical, Dielectric, & Quantum Transport

### 1.1 Relativistic Electronic Ground State & Non-Empirical Exchange-Correlation
The foundational equation for condensed matter is the relativistic Dirac-Coulomb-Breit Hamiltonian:
$$H_{\text{DCB}} = \sum_i \left[ c \boldsymbol{\alpha}\cdot\mathbf{p}_i + \beta m_0 c^2 + v_{\text{ext}}(\mathbf{r}_i) \right] + \frac{1}{2} \sum_{i \neq j} \left[ \frac{e^2}{4\pi\varepsilon_0 r_{ij}} - \frac{e^2}{8\pi\varepsilon_0 r_{ij}} \left(\boldsymbol{\alpha}_i\cdot\boldsymbol{\alpha}_j + \frac{(\boldsymbol{\alpha}_i\cdot\mathbf{r}_{ij})(\boldsymbol{\alpha}_j\cdot\mathbf{r}_{ij})}{r_{ij}^2}\right) \right]$$

Applying the Foldy-Wouthuysen transformation decouples the 4-spinor into scalar-relativistic and spin-orbit coupling (SOC) operators:
$$H_{\text{SOC}} = \frac{\hbar}{4 m_0^2 c^2} (\boldsymbol{\sigma} \times \nabla V(\mathbf{r})) \cdot \mathbf{p}$$

The exact ground-state electron density $n(\mathbf{r})$ is obtained via self-consistent Kohn-Sham equations:
$$\left[-\frac{\hbar^2}{2m_e}\nabla^2 + v_{\text{ext}}(\mathbf{r}) + \frac{e^2}{4\pi\varepsilon_0}\int \frac{n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}d^3r' + v_{xc}[n]\right] \psi_i(\mathbf{r}) = \varepsilon_i \psi_i(\mathbf{r})$$

#### Non-empirical exchange-correlation hierarchy:
- **Semilocal limit:** SCAN meta-GGA satisfying all 17 known exact mathematical constraints without empirical fitting.
- **Exact correlation:** Random Phase Approximation (RPA) via the Adiabatic-Connection Fluctuation-Dissipation Theorem (ACFDT):
$$E_c^{\text{RPA}} = \frac{1}{2\pi} \int_0^\infty du \, \text{Tr}\left[\ln(\mathbf{I} - \mathbf{v}\boldsymbol{\chi}_0(iu)) + \mathbf{v}\boldsymbol{\chi}_0(iu)\right]$$

### 1.2 Quasiparticle Spectra, Band Gaps, & Optical Excitons
Fundamental band gaps $E_g = I - A$ are computed by solving Hedin's $GW$ self-consistent equations:
$$\Sigma(\mathbf{r}, \mathbf{r}', \omega) = \frac{i}{2\pi} \int d\omega' e^{i\omega'\eta} G_0(\mathbf{r}, \mathbf{r}', \omega+\omega') W(\mathbf{r}, \mathbf{r}', \omega')$$
$$W(\mathbf{r}, \mathbf{r}', \omega) = \int d^3r'' \varepsilon^{-1}(\mathbf{r}, \mathbf{r}'', \omega) v(\mathbf{r}''-\mathbf{r}'), \quad \varepsilon = \mathbf{I} - v P_0, \quad P_0 = -i G_0 G_0$$

Optical absorption spectra and exciton binding energies are computed via the Bethe-Salpeter Equation (BSE):
$$(E_{ck}^{\text{QP}} - E_{vk}^{\text{QP}}) A_{vck}^S + \sum_{v'c'k'} \left(2 \bar{V}_{vck,v'c'k'}^{\text{exchange}} - W_{vck,v'c'k'}^{\text{screened}}\right) A_{v'c'k'}^S = \Omega_S A_{vck}^S$$

Macroscopic complex dielectric function:
$$\varepsilon_2(\omega) = \frac{8\pi^2 e^2}{\omega^2 \Omega} \sum_S \left| \sum_{vck} \langle vk | \hat{\mathbf{v}}\cdot\hat{\mathbf{e}} | ck \rangle A_{vck}^S \right|^2 \delta(\hbar\omega - \Omega_S)$$

### 1.3 Strongly Correlated Materials: Parameter-Free DFT + DMFT
For narrow $d$- and $f$-band systems, the four-index Coulomb tensor $U_{ijkl}(\omega)$ is computed from first principles via Constrained RPA (cRPA):
$$W_r(\omega) = [\mathbf{I} - v P_{\text{rest}}(\omega)]^{-1} v$$
$$U_{ijkl}(\omega) = \iint w_i^*(\mathbf{r}) w_j^*(\mathbf{r}') W_r(\mathbf{r}, \mathbf{r}', \omega) w_k(\mathbf{r}) w_l(\mathbf{r}') d^3r \, d^3r'$$

The Anderson Impurity Model is solved self-consistently using continuous-time quantum Monte Carlo (CT-HYB):
$$G_{\text{lat}}(\mathbf{k}, i\omega_n) = \left[(i\omega_n + \mu)\mathbf{I} - H_0(\mathbf{k}) - \Sigma_{\text{imp}}(i\omega_n) + V_{dc}\right]^{-1}$$
$$G_{\text{loc}}(i\omega_n) = \frac{1}{\Omega_{\text{BZ}}} \int_{\text{BZ}} d^3k \, G_{\text{lat}}(\mathbf{k}, i\omega_n)$$

### 1.4 Superconductivity: Anisotropic Migdal-Eliashberg Theory
Superconducting transition temperature ($T_c$) and anisotropic gap functions $\Delta_{n\mathbf{k}}(i\omega_m)$ are calculated without empirical pseudo-potentials using the Eliashberg spectral function $\alpha^2F(\omega)$:
$$\alpha^2F(\omega) = \frac{1}{2\pi N(\varepsilon_F)} \sum_{\mathbf{q}\nu} \delta(\omega - \omega_{\mathbf{q}\nu}) \sum_{mn\mathbf{k}} |g_{mn\nu}(\mathbf{k}, \mathbf{q})|^2 \delta(\varepsilon_{n\mathbf{k}} - \varepsilon_F) \delta(\varepsilon_{m,\mathbf{k}+\mathbf{q}} - \varepsilon_F)$$

Electron-phonon matrix elements $g_{mn\nu}(\mathbf{k},\mathbf{q})$ and phonon dynamical matrices are computed via Density Functional Perturbation Theory (DFPT).

### 1.5 Macroscopic Polarization, Dielectric Tensor, & Topological Invariants
- **Modern Theory of Polarization (Berry Phase):**
$$\mathbf{P} = \mathbf{P}_{\text{ion}} - \frac{2|e|i}{(2\pi)^3} \sum_{n=1}^{N_{\text{occ}}} \int_{\text{BZ}} d^3k \, \langle u_{n\mathbf{k}} | \nabla_{\mathbf{k}} | u_{n\mathbf{k}} \rangle$$
- **Born Effective Charge Tensor:**
$$Z_{\kappa,\alpha\beta}^* = \frac{\Omega}{|e|} \frac{\partial P_\alpha}{\partial u_{\kappa,\beta}}$$
- **Static Dielectric Tensor:**
$$\varepsilon_{\alpha\beta}(0) = \varepsilon_{\alpha\beta}^\infty + \frac{4\pi}{\Omega} \sum_\nu \frac{S_{\alpha\beta,\nu}}{\omega_\nu^2}$$

### 1.6 Carrier & Phonon Transport (Boltzmann Transport Equation)
Linearized Boltzmann Transport Equation (BTE) with ab initio electron-phonon and 3-phonon scattering rates:
$$\sigma_{\alpha\beta}(T; \mu) = e^2 \int \Sigma_{\alpha\beta}(\varepsilon) \left(-\frac{\partial f_0}{\partial \varepsilon}\right) d\varepsilon, \quad \Sigma_{\alpha\beta}(\varepsilon) = \frac{1}{\Omega}\sum_{n\mathbf{k}} v_{n\mathbf{k}}^\alpha v_{n\mathbf{k}}^\beta \tau_{n\mathbf{k}} \delta(\varepsilon - \varepsilon_{n\mathbf{k}})$$
$$\kappa_{\alpha\beta}^{\text{phonon}}(T) = \frac{1}{N_q \Omega}\sum_{\mathbf{q}\nu} C_{\mathbf{q}\nu}(T) v_{\mathbf{q}\nu}^\alpha v_{\mathbf{q}\nu}^\beta \tau_{\mathbf{q}\nu}$$

---

## 2. Atomistic Scale: Defects, Kinetics, & Interfacial Thermodynamics

### 2.1 Equivariant Machine-Learned Interatomic Potentials (MLIP)
Enforces strict $E(3)$ rotational, translational, and inversion equivariance using irreducible representations and Clebsch-Gordan tensor products:
$$E_{\text{tot}} = \sum_i E_i, \quad \mathbf{F}_i = -\nabla_{\mathbf{R}_i} E_{\text{tot}}, \quad \boldsymbol{\sigma} = -\frac{1}{\Omega}\left[\sum_i M_i \mathbf{v}_i \otimes \mathbf{v}_i + \sum_{i < j} \mathbf{r}_{ij} \otimes \mathbf{f}_{ij}\right]$$

### 2.2 Point Defect Formation Free Energy & Corrections
$$\Delta G_f(X^q, E_F, \{\mu_i\}) = E_{\text{tot}}(X^q) - E_{\text{tot}}(\text{bulk}) - \sum_i \Delta n_i \mu_i + q(E_{\text{VBM}} + E_F + \Delta v_{0/b}) + E_{\text{corr}}^{\text{FNV}}(q) - T \Delta S_{\text{vib}} + P \Delta V$$
Freysoldt-Neugebauer-Van de Walle (FNV) finite-size electrostatic supercell correction removes periodic charge image artifacts.

### 2.3 Defect Kinetics & Transition State Theory
Harmonic Transition State Theory (Vineyard):
$$\Gamma_{\text{HTST}} = \frac{1}{2\pi} \sqrt{\frac{\det \mathbf{H}(\mathbf{R}_{\min})}{|\det' \mathbf{H}(\mathbf{R}^\ddagger)|}} \exp\left(-\frac{E(\mathbf{R}^\ddagger) - E(\mathbf{R}_{\min})}{k_B T}\right) \left[1 + \frac{1}{24}\left(\frac{\hbar |\omega_{\text{unstable}}^\ddagger|}{k_B T}\right)^2\right]$$

### 2.4 Planar Defects, Dislocation Cores, & Grain Boundary Energies
- **Generalized Stacking Fault Energy ($\gamma$-surface):** $\gamma(\mathbf{u}) = \frac{E_{\text{slab}}(\mathbf{u}) - E_{\text{slab}}(\mathbf{0})}{A_{\text{slip}}}$
- **Semidiscrete Variational Peierls-Nabarro (SVPN):** Dislocation core width and Peierls stress $\tau_P$.
- **Universal Binding Energy Relation (UBER):** $\sigma(\delta) = \left(\frac{W_{\text{sep}}}{\lambda}\right) \left(\frac{\delta}{\lambda}\right) \exp\left(-\frac{\delta}{\lambda}\right)$

---

## 3. Mesoscale Microstructure, Kinetics, & Dislocation Dynamics

### 3.1 Phase Field Modeling (Thermodynamic & Kinetic Field Theories)
- **Conserved Order Parameter (Cahn-Hilliard):** $\frac{\partial c_k}{\partial t} = \nabla \cdot \left[\sum_l M_{kl} \nabla \frac{\delta F}{\delta c_l}\right] + \xi_c$
- **Non-Conserved Order Parameter (Allen-Cahn):** $\frac{\partial \eta_i}{\partial t} = -L_i \frac{\delta F}{\delta \eta_i} + \xi_\eta$
- **Total Free Energy Functional:**
$$F = \int \left[ f_{\text{chem}}(\{c_k\}, \{\eta_i\}, T) + \frac{1}{2}\kappa_c |\nabla c|^2 + \frac{1}{2}\sum \kappa_\eta |\nabla\eta_i|^2 + f_{\text{elast}}(c, \eta, \boldsymbol{\varepsilon}) \right] dV$$
- **Khachaturyan-Shtremel Microelasticity:** Exact Fourier-space elastic interaction kernel $B(\mathbf{n})$.

### 3.2 Discrete Dislocation Dynamics (DDD)
Peach-Koehler Force:
$$\mathbf{f}_{\text{PK}} = (\boldsymbol{\sigma}_{\text{total}} \cdot \mathbf{b}) \times \boldsymbol{\xi}$$
Nodal velocity: $\mathbf{v}_{\text{node}} = \mathbf{B}_{\text{node}}^{-1}(T, P) \cdot \mathbf{F}_{\text{node}}$.

---

## 4. Continuum Mechanics, Crystal Plasticity, Fracture, & Creep

### 4.1 Finite-Strain Crystal Plasticity (CPFEM / CPFFT)
- **Multiplicative Kinematics:** $\mathbf{F} = \mathbf{F}^e \mathbf{F}^p, \quad \mathbf{L}^p = \dot{\mathbf{F}}^p (\mathbf{F}^p)^{-1} = \sum_\alpha \dot{\gamma}^\alpha (\mathbf{s}_0^\alpha \otimes \mathbf{m}_0^\alpha)$
- **Mandel Stress:** $\bar{\mathbf{M}} = \bar{\mathbf{C}}^e \bar{\mathbf{S}} = \bar{\mathbf{C}}^e \left(\mathbb{C} : \frac{1}{2}(\bar{\mathbf{C}}^e - \mathbf{I})\right), \quad \tau^\alpha = \bar{\mathbf{M}} : (\mathbf{s}_0^\alpha \otimes \mathbf{m}_0^\alpha)$
- **Dislocation Density Accumulation:** $\boldsymbol{\alpha}_{\text{Nye}} = \nabla \times \mathbf{F}^p \implies \rho_{\text{GND}} = \frac{1}{b}\|\boldsymbol{\alpha}_{\text{Nye}}\|_1$

### 4.2 High-Temperature Diffusional & Dislocation Creep
$$\dot{\varepsilon}_{\text{creep}} = A_{\text{disl}}\left(\frac{\sigma_{\text{eq}}}{G}\right)^n \exp\left(-\frac{Q_{\text{core}}}{RT}\right) + A_{\text{diff}}\left(\frac{\sigma_{\text{eq}}}{d_{\text{grain}}^p}\right)\frac{\Omega D_{\text{eff}}}{k_B T}$$

### 4.3 Energy-Conserving Non-Local Fracture & Weibull Scaling
$$Y - \nabla \cdot (c_g \nabla Y) = \bar{Y}, \quad D = \mathcal{G}(Y)$$
$$P_f(\sigma) = 1 - \exp\left[-\int_V \left(\frac{\sigma}{\sigma_0}\right)^{m_{\text{weibull}}} \frac{dV}{V_0}\right]$$
