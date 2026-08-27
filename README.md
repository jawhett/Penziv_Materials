# Penziv Materials (AetherMat v3.2.0-PROD)

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-0891B2.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-0A2540.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-47%2F47%20Passed-1E7065.svg)](#complete-verification-suite)
[![Physics Gates](https://img.shields.io/badge/Physics_Validation-Verified-1E7065.svg)](#bidirectional-scale-handshake-gates)
[![Solid Electrolytes](https://img.shields.io/badge/Electrochem-Superionic_Fast_Ions-0891B2.svg)](#solid-electrolyte--heterogeneous-discovery-suite)
[![Techno-Economics](https://img.shields.io/badge/Economics_TEA-LCOS_%26_HHI_Risk-1E7065.svg)](#techno-economics-supply-chain-risk--ehs-suite)
[![Design System](https://img.shields.io/badge/Design-Serene_Zenith-0891B2.svg)](https://github.com/jawhett/Penziv_Materials)

**Autonomous Multiscale First-Principles Materials & Solid-State Electrolyte Discovery Framework**

*Zero-parameter scale bridging from relativistic quantum electrodynamics down to process synthesizability, complex multiphase architectures, superionic conductors, and techno-economic risk.*

</div>

---

## 🏛️ Universal Multiscale Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │            Meta-Orchestrator Discovery Agent           │
                    │ (Quality-Diversity MAP-Elites, Holistic Co-Design UQ)  │
                    └───────────┬────────────────────────────────┬───────────┘
                                │ ▲                            │ ▲
                                ▼ │ Forward & Backward Loops   ▼ │
                ┌───────────────┴───────────────┐ ┌──────────────┴───────────────┐
                │ Quantum & Electronic Agent    │ │ Atomistic & Defect Agent     │
                │ (Mermin-DFT, SRO-Planar Faults│ │ (Polar-MLIPs, CI-NEB MEP,    │
                │  TDEP, FNV Defect Energies)   │ │  AIMD MSD, Nernst-Einstein)  │
                └───────────────┬───────────────┘ └──────────────┬───────────────┘
                                │ ▲                            │ ▲
                                ▼ │                            ▼ │
                ┌───────────────┴───────────────┐ ┌──────────────┴───────────────┐
                │ Mesoscale Kinetics Agent      │ │ Continuum Micromechanics     │
                │ (Coupled PNP Space-Charge,    │ │ (CPFEM/CPFFT, Non-Local Dmg, │
                │  TPMS Gyroid, Dynamic DDD)    │ │  Butler-Volmer Interfacial)  │
                └───────────────┬───────────────┘ └──────────────┬───────────────┘
                                │ ▲                            │ ▲
                                ▼ │ Thermal & Field History    ▼ │ Yield & Overpotentials
                                └───────────────┬────────────────┘
                                                │ ▲
                                                ▼ │
                                ┌───────────────┴───────────────┐
                                │ Process & Synthesizability    │
                                │ (Stefan, Poro-Mechanics FSI,  │
                                │  Cold Sintering Retrosynth)   │
                                └───────────────┬───────────────┘
                                                │ ▲
                                                ▼ │
                ┌───────────────────────────────┴─┴──────────────────────────────┐
                │ Cross-Scale Uncertainty, TEA & Sim-to-Real Assimilation Bridge │
                │ (Commodity Pricing, HHI Risk, EPA CompTox, Multi-Modal MCMC)   │
                └────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Physics Hierarchy & Scale-Bridging Ingests / Emits

| Scale | Layer | Physics Engine & Governing Equations | Scale-Bridging Ingests (Inputs) | Scale-Bridging Emits (Outputs) |
| :--- | :--- | :--- | :--- | :--- |
| **Scale 5** | **Quantum & Electronic (`Q-ELEC`)** | Relativistic Dirac-Coulomb-Breit, Mermin Finite-$T_e$ DFT, TDEP phonons, SCAN Meta-GGA, RPA correlation via ACFDT, $GW$-BSE, cRPA+DMFT, DLM paramagnetism, SRO GSFE $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, FNV charged defect formation $\Delta H_f(D, q)$. | Atomic numbers $Z_I$, positions $\mathbf{R}_I$, space group, temperatures ($T_e, T_{\text{ion}}$), magnetic moments $\mathbf{m}_I$, SRO tensors $\boldsymbol{\alpha}_{\text{SRO}}$. | $\Delta H_f^0(T)$, $\mathbb{C}_{ijkl}(T)$, thermal expansion $\boldsymbol{\alpha}(T)$, $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, $F(V,T)$, $\boldsymbol{\varepsilon}_{\alpha\beta}(\omega)$, $\Delta\text{-ML}$ offset operators, FNV defect energies. |
| **Scale 4** | **Atomistic & Defects (`ATOM-DYN`)** | $E(3)$-Equivariant MLIPs with GMM-OOD detection, CI-NEB ion migration paths ($\Delta E_a$), multivalent polarization screening, AIMD Mean-Squared Displacement (MSD), Nernst-Einstein ionic conductivity ($\sigma_i, t_{\text{ion}}$), SVPN Peierls core width $\tau_P$. | Potential Energy Surface, $\mathbb{C}_{ijkl}(T)$, $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, active learning DFT retrain data, mobile cation radius $r_{\text{ion}}$. | Defect migration distributions $\Delta E_a \pm \sigma$, diffusion tensors $\mathbf{D}(T)$, $\sigma_{\text{ion}}(T)$, transference number $t_{\text{ion}}$, 5-parameter GB energy $\gamma_{\text{GB}}$, $\tau_P$, traction laws ($W_{\text{sep}}, \lambda$). |
| **Scale 3** | **Mesoscale Kinetics (`MESO-KINETIC`)** | Coupled Cahn-Hilliard & Allen-Cahn + Khachaturyan microelasticity, Coupled Poisson-Nernst-Planck (PNP) space-charge layers, Dynamic Drag DDD ($\mathbf{f}_{\text{PK}} = (\boldsymbol{\sigma}\cdot\mathbf{b})\times\boldsymbol{\xi}$), Triply Periodic Minimal Surfaces (TPMS Gyroid/Diamond). | Diffusion tensors $\mathbf{D}(T)$, GB energy manifolds $\gamma_{\text{GB}}$, single-crystal $\mathbb{C}_{ijkl}(T)$, thermal/electric history from Scale 1 ($T(t), \phi(t), G, \dot{T}, V$). | 3D TPMS multi-phase topologies (ceramic + pressurized channel + polymer skin), ODF texture maps, space-charge potential $\phi(x)$, Debye length $\lambda_D$, CRSS $\tau_{\text{CRSS}}^\alpha$, hardening matrix $h_{\alpha\beta}$. |
| **Scale 2** | **Continuum Micromechanics (`CONT-MICRO`)** | Multiplicative finite-strain CPFEM/CPFFT ($\mathbf{F} = \mathbf{F}^e\mathbf{F}^p$), Butler-Volmer interfacial charge transfer ($J_{\text{BV}}$), Nye dislocation tensor ($\boldsymbol{\alpha}_{\text{Nye}} = \nabla\times\mathbf{F}^p$), Chemo-mechanical stress coupling, Non-local gradient damage, Weibull modulus $m$. | Conformed RVE meshes, TPMS phase maps, $\tau_{\text{CRSS}}^\alpha$, $h_{\alpha\beta}$, $\mathbb{C}_{ijkl}(T)$, overpotential $\eta$, intrinsic fracture energy $G_c$, residual stresses $\boldsymbol{\sigma}_{\text{res}}(\mathbf{x})$ from Scale 1. | True stress-strain curves $\boldsymbol{\sigma}(\boldsymbol{\varepsilon})$, homogenized yield surfaces $\Sigma(\boldsymbol{\sigma})$, Butler-Volmer flux $J_{\text{BV}}(\eta)$, steady-state creep rate laws $\dot{\varepsilon}_{\text{ss}}(T, \boldsymbol{\sigma})$, fracture toughness $K_{Ic}$, Weibull modulus $m_{\text{weibull}}$. |
| **Scale 1** | **Process & Synthesizability (`PROC-MFG`)** | Stefan solidification with Marangoni thermofluids, Poro-elastic Darcy-Stokes FSI, Knudsen microchannel gas dynamics ($Kn > 0.1$), Transient oxidation ($x^2=k_pt$), Minimum crustal exergy extraction work $\text{Ex}_{\min}$, Retrosynthesis assembly planner. | Target alloy/electrolyte composition, processing parameters (laser power $P$, scan speed $v$, gas pressure $p_{\text{gas}}$), macroscopic yield surface $\Sigma(\boldsymbol{\sigma})$ and mechanical limit states from Scale 2. | Space-time thermal profiles ($T(\mathbf{x}, t), G, \dot{T}, V$), channel wall hydrostatic support, process residual stresses $\boldsymbol{\sigma}_{\text{res}}(\mathbf{x})$, synthesizability confidence index, minimum crustal exergy $\text{Ex}_{\min}$, causal retrosynthesis route. |
| **Meta-Scale** | **Uncertainty & Sim-to-Real (`UQ-BRIDGE`)** | Quality-Diversity (QD) MAP-Elites illumination swarm, Holistic multi-domain constraint relaxation, $\text{SO}(3)$-PINO with hard frame indifference, Grand Canonical stability windows $[V_{\text{red}}, V_{\text{ox}}]$, Commodity spot pricing, HHI geopolitical risk, EPA CompTox. | Synchrotron XRD phase fractions, EBSD orientation maps, Atom Probe Tomography cluster distributions, Nanoindentation depth curves, Commodity market spot rates. | Calibrated joint Bayesian posteriors $P(\boldsymbol{\theta}\mid\mathbf{y})$, MAP-Elites behavioral archive, holistic composite stability receipts, Levelized Cost of Storage (LCOS $/kWh), Embodied $\text{CO}_2$. |

---

## 📊 Techno-Economics, Supply Chain Risk & EHS Suite

The framework provides programmatic tool interfaces and optimization penalties for real-world viability:

1. **Commodity Spot Pricing & Precursors (`commodity_pricing.py`):** Live market pricing ($/kg) across elemental feedstocks and battery-grade precursors ($99.9\%$ purity multiplier).
2. **Geopolitical Concentration & USGS Critical Minerals (`supply_chain_risk.py`):** Evaluates Herfindahl-Hirschman Indices ($\text{HHI} = \sum s_i^2$) for mining and refining, tagging DOE/USGS critical minerals and byproduct dependence.
3. **Regulatory Toxicity & CompTox EHS (`toxicity_ehs.py`):** Pre-compute rejection of banned heavy metals ($\text{Tl, Cd, As, Hg, Pb, Be}$), EPA CompTox scores, REACH SVHC flags, and cradle-to-gate embodied carbon footprints ($\text{kg CO}_2\text{-eq}/\text{kg}$).
4. **Techno-Economic Analysis & LCOS Floor (`techno_economics.py`):** Calculates normalized Levelized Cost of Storage raw-material floor ($/kWh) and thermal sintering energy budgets ($\text{kWh}/\text{kg}$).

---

## 🛡️ Bidirectional Scale Handshake Gates

The framework enforces zero-compromise validation contracts across every scale interface:
1. **Pre-Compute EHS Gate:** Zero tolerance for banned toxic heavy metals ($\text{Tl, Cd, As, Hg, Pb, Be}$) and EPA CompTox hazard score $< 4.5$.
2. **Economic Resilience Gate:** Flags extreme geopolitical refining concentration ($\text{HHI}_{\text{refining}} > 6000$).
3. **Scale 5 $\longleftrightarrow$ Scale 4:** Force Residual Gate ($\max_I \|\mathbf{F}_I + \nabla_{\mathbf{R}} E_{\text{tot}}\|_2 < 10^{-4}\text{ eV/\AA}$); OOD Density Gate ($\mathcal{L}_{\text{OOD}}(\mathbf{z}_i) \le \zeta_{\text{threshold}}$).
4. **Scale 4 $\longleftrightarrow$ Scale 3:** Stacking Fault Positivity ($\min \gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}}) > 0 \quad \forall \mathbf{u} \neq \mathbf{0}$); Log-Normal Rate Variance Gate ($\sigma_{\ln \Gamma}^2 < 0.25$).
5. **Scale 3 $\longleftrightarrow$ Scale 2:** RVE Homogenization Convergence Gate ($\|\langle\boldsymbol{\sigma}_{2L}\rangle - \langle\boldsymbol{\sigma}_L\rangle\| < 0.015$); Plastic Dissipation Positivity ($dW_p = \sum_\alpha \tau^\alpha d\gamma^\alpha \ge 0$).
6. **Scale 2 $\longleftrightarrow$ Scale 1:** Clausius-Duhem Dissipation Positivity ($\mathcal{D}_{\text{int}} = \boldsymbol{\sigma} : \dot{\boldsymbol{\varepsilon}}^p - \dot{\psi}_{\text{ISV}} \ge 0$); Born Mechanical Stability Criteria ($\lambda_{\min}(\mathbb{C}_{\text{Voigt}}) > 0$).
7. **Meta-Scale:** Compound Scale Uncertainty Error Bounding Gate ($\sigma_{\text{tot}}^2 / \mu^2 < 0.15$); Holistic Composite Stability Relaxation ($\mathcal{F}_{\text{total}} < \mathcal{F}_{\text{threshold}}$).

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

# 2. Run instant Techno-Economic (TEA), Supply Chain HHI, and Toxicity EHS audit
penziv-mat evaluate-tea MgSc0.2Zr1.8S6 --purity battery_grade_99_9 --sinter-temp 850.0

# 3. Discover novel solid electrolytes & hybrid architectures (Mg2+, Na+, Li+) via QD MAP-Elites
penziv-mat discover-solid-electrolyte --carrier Mg --candidates 15 --min-sigma 1.0

# 4. Generate 3D Triply Periodic Minimal Surface (TPMS Gyroid/Diamond) multi-phase geometry
penziv-mat generate-tpms --surface gyroid --resolution 32

# 5. Solve Coupled Poisson-Nernst-Planck (PNP) space-charge layer & Butler-Volmer kinetics
penziv-mat solve-pnp --overpotential 0.05 --points 100

# 6. Run Autonomous Pareto Structural Alloy Discovery Search
penziv-mat discover-alloy --samples 30 --elements "Ni,Cr,Al,Ti,Nb,Mo,W,B" --min-yield 1000 --max-exergy 85 --temp-k 1123.15

# 7. Execute Phase 4 Production High-Temperature Benchmark (T > 850°C)
penziv-mat benchmark --candidates 20

# 8. Run full forward multiscale prediction on a specific alloy
penziv-mat predict-forward --material "Penziv-Superalloy-718X" --temp-k 1123.15

# 9. Run Spectral Phase-Field simulation with Khachaturyan microelasticity
penziv-mat run-phase-field --steps 15

# 10. Execute Spectral CPFFT crystal plasticity strain increment with Nye dislocation tensor
penziv-mat run-cpfft --strain-rate 0.001

# 11. Evaluate Born mechanical stability for an elastic tensor
penziv-mat validate-born --c11 260.0 --c12 160.0 --c44 110.0 --system cubic

# 12. Mint provenance BibTeX citation and solver dependency tree
penziv-mat cite --title "Penziv Materials Discovery" --author "jawhett"
```

---

## 🧪 Complete Verification Suite

Run the full multiscale test suite (47 unit tests across structural, electrochemical, multiphysics, TEA, and EHS domains):

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
