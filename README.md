# Penziv Materials (AetherMat v3.2.0-PROD)

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-0891B2.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-0A2540.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-30%2F30%20Passed-1E7065.svg)](#complete-verification-suite)
[![Physics Gates](https://img.shields.io/badge/Physics_Validation-Verified-1E7065.svg)](#bidirectional-scale-handshake-gates)
[![Architecture](https://img.shields.io/badge/Architecture-Multiscale_PSPP-0A2540.svg)](#multiscale-physics-architecture)
[![Design System](https://img.shields.io/badge/Design-Serene_Zenith-0891B2.svg)](https://github.com/jawhett/Penziv_Materials)

**Autonomous Multiscale First-Principles Materials Property Prediction & Inverse Discovery Framework**

*Zero-parameter scale bridging from relativistic quantum electrodynamics down to process synthesizability.*

</div>

---

## 🏛️ Multiscale Physics Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │            Meta-Orchestrator Discovery Agent           │
                    │ (Pareto Optimization, Scale Routing, UQ Error Bounding)│
                    └───────────┬────────────────────────────────┬───────────┘
                                │ ▲                            │ ▲
                                ▼ │ Forward & Backward Loops   ▼ │
                ┌───────────────┴───────────────┐ ┌──────────────┴───────────────┐
                │ Quantum & Electronic Agent    │ │ Atomistic & Defect Agent     │
                │ (Mermin-DFT, SRO-Planar Faults│ │ (Polar-MLIPs, GMM-OOD, kMC,  │
                │  TDEP, DMFT, Delta-Learning)  │ │  SVPN, Nix-Gao Indentation)  │
                └───────────────┬───────────────┘ └──────────────┬───────────────┘
                                │ ▲                            │ ▲
                                ▼ │                            ▼ │
                ┌───────────────┴───────────────┐ ┌──────────────┴───────────────┐
                │ Mesoscale Kinetics Agent      │ │ Continuum Micromechanics     │
                │ (Phase-Field, Level-Set RVE,  │ │ (CPFEM/CPFFT, Non-Local Dmg, │
                │  Dynamic DDD, CGM Partition)  │ │  Dislocation & Diff. Creep)  │
                └───────────────┬───────────────┘ └──────────────┬───────────────┘
                                │ ▲                            │ ▲
                                ▼ │ Thermal History & Texture  ▼ │ Yield Surfaces
                                └───────────────┬────────────────┘
                                                │ ▲
                                                ▼ │
                                ┌───────────────┴───────────────┐
                                │ Process & Synthesizability    │
                                │ (Stefan, Solute-Trap, Exergy, │
                                │  Sub-Grid Boundary Layer)     │
                                └───────────────┬───────────────┘
                                                │ ▲
                                                ▼ │
                ┌───────────────────────────────┴─┴──────────────────────────────┐
                │ Cross-Scale Uncertainty & Experimental Data Assimilation Bridge │
                │ (Hierarchical Bayesian PCE, SO(3)-PINO, Multi-Objective MCMC)  │
                └────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Physics Hierarchy & Scale Breakdown

| Scale | Layer | Physics Engine & Governing Equations | Scale-Bridging Ingests / Emits |
| :--- | :--- | :--- | :--- |
| **Scale 5** | **Quantum & Electronic (`Q-ELEC`)** | Relativistic Dirac-Coulomb-Breit, Mermin Finite-$T_e$ DFT, TDEP phonons, SCAN Meta-GGA, RPA correlation, $GW$-BSE, cRPA+DMFT, DLM paramagnetism, SRO GSFE $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$. | $Z_I, \mathbf{R}_I, \boldsymbol{\alpha}_{\text{SRO}} \longrightarrow \Delta H_f^0(T), \mathbb{C}_{ijkl}(T), \alpha(T), F(V,T), \Delta\text{-ML}$ |
| **Scale 4** | **Atomistic & Defects (`ATOM-DYN`)** | $E(3)$-Equivariant MLIPs with GMM-OOD detection, Charged Defect FNV corrections, Harmonic TST / CI-NEB $\Delta E_m \pm \sigma$, SVPN Peierls core width $\tau_P$, 5-parameter GB energy $\gamma_{\text{GB}}(\theta, \hat{\mathbf{n}})$. | Potential Energy Surface $\longrightarrow \mathbf{D}(T), \Delta E_m \pm \sigma, \tau_P, W_{\text{cs}}(\boldsymbol{\sigma}), W_{\text{sep}}$ |
| **Scale 3** | **Mesoscale Kinetics (`MESO-KINETIC`)** | Coupled Cahn-Hilliard & Allen-Cahn + Khachaturyan microelasticity, Dynamic Drag DDD ($\mathbf{f}_{\text{PK}} = (\boldsymbol{\sigma}\cdot\mathbf{b})\times\boldsymbol{\xi}$), CGM solute trapping with sub-grid boundary layers, Level-Set conformed RVEs & ODF. | $\mathbf{D}(T), \dot{T}, G, V \longrightarrow \text{Conformed RVE}, \text{ODF Texture}, \tau_{\text{CRSS}}^\alpha, h_{\alpha\beta}$ |
| **Scale 2** | **Continuum Micromechanics (`CONT-MICRO`)** | Multiplicative finite-strain CPFEM/CPFFT ($\mathbf{F} = \mathbf{F}^e\mathbf{F}^p$), Nye dislocation tensor ($\boldsymbol{\alpha}_{\text{Nye}} = \nabla\times\mathbf{F}^p$), High-$T$ dislocation/diffusional creep, Non-local gradient damage, Weibull modulus $m$. | $\text{RVE}, \tau_{\text{CRSS}}^\alpha, h_{\alpha\beta} \longrightarrow \boldsymbol{\sigma}(\boldsymbol{\varepsilon}), \Sigma(\boldsymbol{\sigma}), \dot{\varepsilon}_{\text{ss}}(T,\boldsymbol{\sigma}), K_{Ic}$ |
| **Scale 1** | **Process & Synthesizability (`PROC-MFG`)** | Stefan solidification with Marangoni thermofluids, Transient oxidation kinetics ($x^2=k_pt$), Stress-assisted interstitial diffusion, Minimum crustal exergy extraction work $\text{Ex}_{\min}$. | Composition, Processing $\longrightarrow T(t), G, \dot{T}, V, \boldsymbol{\sigma}_{\text{res}}(\mathbf{x}), \text{Ex}_{\min}$ |
| **Meta-Scale** | **Uncertainty & Sim-to-Real (`UQ-BRIDGE`)** | Hierarchical Bayesian PCE, $\text{SO}(3)$-PINO with hard frame indifference and arc-length continuation, Nix-Gao depth-corrected nanoindentation assimilation. | XRD, EBSD, APT, Nanoindentation $\longrightarrow \text{Calibrated Posteriors } P(\boldsymbol{\theta}\mid\mathbf{y})$ |

---

## 🛡️ Bidirectional Scale Handshake Gates

The framework enforces zero-compromise validation contracts across every scale interface:
1. **Force Residual Gate:** $\max_I \|\mathbf{F}_I + \nabla_{\mathbf{R}} E_{\text{tot}}\|_2 < 10^{-4}\text{ eV/\AA}$
2. **OOD Density Gate:** $\mathcal{L}_{\text{OOD}}(\mathbf{z}_i) \le \zeta_{\text{threshold}}$ (routes outliers to Q-ELEC)
3. **Stacking Fault Positivity:** $\min \gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}}) > 0 \quad \forall \mathbf{u} \neq \mathbf{0}$
4. **Log-Normal Rate Variance Gate:** $\sigma_{\ln \Gamma}^2 < 0.25$
5. **RVE Homogenization Convergence Gate:** $\|\langle\boldsymbol{\sigma}_{2L}\rangle - \langle\boldsymbol{\sigma}_L\rangle\| < 0.015$
6. **Plastic & Clausius-Duhem Dissipation Positivity:** $dW_p = \sum_\alpha \tau^\alpha d\gamma^\alpha \ge 0$, $\mathcal{D}_{\text{int}} = \boldsymbol{\sigma} : \dot{\boldsymbol{\varepsilon}}^p - \dot{\psi}_{\text{ISV}} \ge 0$
7. **Compound Scale Uncertainty Bound:** $\sigma_{\text{tot}}^2 / \mu^2 < 0.15$
8. **Born Mechanical Stability Criteria:** Full minor positivity and positive definiteness of the Voigt elasticity matrix ($\lambda_{\min}(\mathbb{C}_{\text{Voigt}}) > 0$).

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jawhett/Penziv_Materials.git
cd Penziv_Materials

# Install in editable mode
pip install -e .
```

### Master CLI Command Suite

```bash
# 1. Inspect architecture, scale solvers, and physical validation gates
penziv-mat status

# 2. Run Autonomous Pareto Alloy Discovery Search (Inverse design across 8 elements)
penziv-mat discover-alloy --samples 30 --elements "Ni,Cr,Al,Ti,Nb,Mo,W,B" --min-yield 1000 --max-exergy 85 --temp-k 1123.15

# 3. Execute Phase 4 Production High-Temperature Benchmark (T > 850°C)
penziv-mat benchmark --candidates 20

# 4. Run full forward multiscale prediction on a specific alloy
penziv-mat predict-forward --material "Penziv-Superalloy-718X" --temp-k 1123.15

# 5. Run Spectral Phase-Field simulation with Khachaturyan microelasticity
penziv-mat run-phase-field --steps 15

# 6. Execute Spectral CPFFT crystal plasticity strain increment with Nye dislocation tensor
penziv-mat run-cpfft --strain-rate 0.001

# 7. Evaluate Born mechanical stability for an elastic tensor
penziv-mat validate-born --c11 260.0 --c12 160.0 --c44 110.0 --system cubic

# 8. Mint provenance BibTeX citation and solver dependency tree
penziv-mat cite --title "Penziv Materials Discovery" --author "jawhett"
```

---

## 🧪 Complete Verification Suite

Run the full multiscale test suite (30 unit tests across all 4 phases):

```bash
python -m unittest discover tests
```

---

## 📜 Theoretical Foundations & Specifications

Detailed architectural specifications and full derivations are provided in [`docs/`](file:///C:/Users/jawhe/Penziv_Materials/docs):
- [`docs/FRAMEWORK_FIRST_PRINCIPLES.md`](file:///C:/Users/jawhe/Penziv_Materials/docs/FRAMEWORK_FIRST_PRINCIPLES.md): Complete mathematical derivations from relativistic Dirac-Coulomb-Breit to non-linear continuum plasticity.
- [`docs/PDRD_AETHERMAT_V3.2.0.md`](file:///C:/Users/jawhe/Penziv_Materials/docs/PDRD_AETHERMAT_V3.2.0.md): Product Development & Requirements Document v3.2.0-PROD.

---

## ⚖️ License & Provenance

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](file:///C:/Users/jawhe/Penziv_Materials/LICENSE) and [`CITATION.cff`](file:///C:/Users/jawhe/Penziv_Materials/CITATION.cff) for citations.
