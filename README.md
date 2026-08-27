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

## 🔬 Physics Hierarchy & Scale-Bridging Ingests / Emits

| Scale | Layer | Physics Engine & Governing Equations | Scale-Bridging Ingests (Inputs) | Scale-Bridging Emits (Outputs) |
| :--- | :--- | :--- | :--- | :--- |
| **Scale 5** | **Quantum & Electronic (`Q-ELEC`)** | Relativistic Dirac-Coulomb-Breit, Mermin Finite-$T_e$ DFT, TDEP phonons, SCAN Meta-GGA, RPA correlation via ACFDT, $GW$-BSE, cRPA+DMFT, DLM paramagnetism, SRO GSFE $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$. | Atomic numbers $Z_I$, positions $\mathbf{R}_I$, space group, temperatures ($T_e, T_{\text{ion}}$), magnetic moments $\mathbf{m}_I$, SRO tensors $\boldsymbol{\alpha}_{\text{SRO}}$. | $\Delta H_f^0(T)$, $\mathbb{C}_{ijkl}(T)$, thermal expansion $\boldsymbol{\alpha}(T)$, $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, $F(V,T)$, $\boldsymbol{\varepsilon}_{\alpha\beta}(\omega)$, $\Delta\text{-ML}$ offset operators. |
| **Scale 4** | **Atomistic & Defects (`ATOM-DYN`)** | $E(3)$-Equivariant MLIPs with GMM-OOD detection, Charged Defect FNV corrections, Harmonic TST / CI-NEB $\Delta E_m \pm \sigma$, SVPN Peierls core width $\tau_P$, 5-parameter GB energy $\gamma_{\text{GB}}(\theta, \hat{\mathbf{n}})$. | Potential Energy Surface, $\mathbb{C}_{ijkl}(T)$, $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, active learning DFT single-point retrain data. | Defect migration distributions $\Delta E_m \pm \sigma$, diffusion tensors $\mathbf{D}(T)$, $\gamma_{\text{GB}}(\theta, \hat{\mathbf{n}})$, $\Delta E_b^{\text{seg}}$, $\tau_P$, $W_{\text{cs}}(\boldsymbol{\sigma}) \pm \sigma$, traction laws ($W_{\text{sep}}, \lambda$). |
| **Scale 3** | **Mesoscale Kinetics (`MESO-KINETIC`)** | Coupled Cahn-Hilliard & Allen-Cahn + Khachaturyan microelasticity, Dynamic Drag DDD ($\mathbf{f}_{\text{PK}} = (\boldsymbol{\sigma}\cdot\mathbf{b})\times\boldsymbol{\xi}$), CGM solute trapping with sub-grid boundary layers, Level-Set conformed RVEs & ODF. | Diffusion tensors $\mathbf{D}(T)$, $\gamma_{\text{GB}}$ manifolds, $\Delta E_b^{\text{seg}}$, single-crystal $\mathbb{C}_{ijkl}(T)$, thermal history from Scale 1 ($T(t), G, \dot{T}, V$). | Smoothed conformed RVE topologies, ODF crystallographic texture maps $(\phi_1, \Phi, \phi_2)$, CRSS $\tau_{\text{CRSS}}^\alpha$, hardening matrix $h_{\alpha\beta}$, void swelling rates $dC_{\text{void}}/dt$. |
| **Scale 2** | **Continuum Micromechanics (`CONT-MICRO`)** | Multiplicative finite-strain CPFEM/CPFFT ($\mathbf{F} = \mathbf{F}^e\mathbf{F}^p$), Nye dislocation tensor ($\boldsymbol{\alpha}_{\text{Nye}} = \nabla\times\mathbf{F}^p$), High-$T$ dislocation/diffusional creep, Non-local gradient damage, Weibull modulus $m$. | Conformed RVE meshes, ODF textures, $\tau_{\text{CRSS}}^\alpha$, $h_{\alpha\beta}$, $\mathbb{C}_{ijkl}(T)$, intrinsic fracture energy $G_c$, length scale $l_c$, residual stresses $\boldsymbol{\sigma}_{\text{res}}(\mathbf{x})$ from Scale 1. | True stress-strain curves $\boldsymbol{\sigma}(\boldsymbol{\varepsilon})$, homogenized yield surfaces $\Sigma(\boldsymbol{\sigma})$, steady-state creep rate laws $\dot{\varepsilon}_{\text{ss}}(T, \boldsymbol{\sigma})$, fracture toughness $K_{Ic}$, Paris law ($C, m$), Weibull modulus $m_{\text{weibull}}$. |
| **Scale 1** | **Process & Synthesizability (`PROC-MFG`)** | Stefan solidification with Marangoni thermofluids, Transient oxidation kinetics ($x^2=k_pt$), Stress-assisted interstitial diffusion, Minimum crustal exergy extraction work $\text{Ex}_{\min}$. | Alloy composition, manufacturing parameters (laser power $P$, scan speed $v$, beam radius $r_b$, gas pressure $p_{\text{gas}}$), macroscopic yield surface $\Sigma(\boldsymbol{\sigma})$ from Scale 2. | Space-time thermal profiles ($T(\mathbf{x}, t), G, \dot{T}, V$), process residual stresses $\boldsymbol{\sigma}_{\text{res}}(\mathbf{x})$, parabolic oxidation constant $k_p$, synthesizability confidence index, minimum crustal exergy $\text{Ex}_{\min}$. |
| **Meta-Scale** | **Uncertainty & Sim-to-Real (`UQ-BRIDGE`)** | Hierarchical Bayesian PCE, $\text{SO}(3)$-PINO with hard frame indifference and arc-length continuation, Nix-Gao depth-corrected nanoindentation assimilation. | Synchrotron XRD phase fractions, EBSD orientation maps, Atom Probe Tomography (APT) cluster distributions, Nanoindentation depth-hardness curves. | Calibrated joint Bayesian posteriors $P(\boldsymbol{\theta}\mid\mathbf{y})$, verified $\text{SO}(3)$-PINO surrogates with pseudo-arc-length continuation bounds. |

---

## 🛡️ Bidirectional Scale Handshake Gates

The framework enforces zero-compromise validation contracts across every scale interface:
1. **Scale 5 $\longleftrightarrow$ Scale 4:** Force Residual Gate ($\max_I \|\mathbf{F}_I + \nabla_{\mathbf{R}} E_{\text{tot}}\|_2 < 10^{-4}\text{ eV/\AA}$); OOD Density Gate ($\mathcal{L}_{\text{OOD}}(\mathbf{z}_i) \le \zeta_{\text{threshold}}$).
2. **Scale 4 $\longleftrightarrow$ Scale 3:** Stacking Fault Positivity ($\min \gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}}) > 0 \quad \forall \mathbf{u} \neq \mathbf{0}$); Log-Normal Rate Variance Gate ($\sigma_{\ln \Gamma}^2 < 0.25$).
3. **Scale 3 $\longleftrightarrow$ Scale 2:** RVE Homogenization Convergence Gate ($\|\langle\boldsymbol{\sigma}_{2L}\rangle - \langle\boldsymbol{\sigma}_L\rangle\| < 0.015$); Plastic Dissipation Positivity ($dW_p = \sum_\alpha \tau^\alpha d\gamma^\alpha \ge 0$).
4. **Scale 2 $\longleftrightarrow$ Scale 1:** Clausius-Duhem Dissipation Positivity ($\mathcal{D}_{\text{int}} = \boldsymbol{\sigma} : \dot{\boldsymbol{\varepsilon}}^p - \dot{\psi}_{\text{ISV}} \ge 0$); Born Mechanical Stability Criteria ($\lambda_{\min}(\mathbb{C}_{\text{Voigt}}) > 0$).
5. **Meta-Scale:** Compound Scale Uncertainty Error Bounding Gate ($\sigma_{\text{tot}}^2 / \mu^2 < 0.15$); Bayesian Posterior Convergence ($D_{\text{KL}} < \varepsilon_{\text{bayes}}$).

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
- [`docs/PDRD_AETHERMAT_V3.2.0.md`](file:///C:/Users/jawhe/Penziv_Materials/docs/PDRD_AETHERMAT_V3.2.0.md): Product Development & Requirements Document v3.2.0-PROD with complete I/O specifications for all agents.

---

## ⚖️ License & Provenance

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](file:///C:/Users/jawhe/Penziv_Materials/LICENSE) and [`CITATION.cff`](file:///C:/Users/jawhe/Penziv_Materials/CITATION.cff) for citations.
