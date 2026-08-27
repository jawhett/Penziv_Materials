# Penziv Materials (AetherMat v3.2.0-PROD)

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-0891B2.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-0A2540.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-62%2F62%20Passed-1E7065.svg)](#complete-verification-suite)
[![Physics Gates](https://img.shields.io/badge/Physics_Validation-Verified-1E7065.svg)](#bidirectional-scale-handshake-gates)
[![Convex Hull](https://img.shields.io/badge/Thermodynamics-Grand_Canonical_Hull-0891B2.svg)](#grand-canonical-convex-hull--materials-project-stability)
[![Active Learning](https://img.shields.io/badge/Active_Learning-HPC_Slurm_Auto--Retrain-1E7065.svg)](#asynchronous-active-learning--hpc-dispatch-engine)
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
                │ (Mermin-DFT, SRO-Planar Faults│ │ (Universal MLIPs, CI-NEB MEP,│
                │  TDEP, FNV Defect Energies)   │ │  BFGS Relax, Nernst-Einstein)│
                └───────────────┬───────────────┘ └──────────────┬───────────────┘
                                │ ▲                            │ ▲
                                ▼ │                            ▼ │
                ┌───────────────┴───────────────┐ ┌──────────────┴───────────────┐
                │ Mesoscale Kinetics Agent      │ │ Continuum Micromechanics     │
                │ (Coupled PNP Space-Charge,    │ │ (CPFFT Polycrystal Plasticity│
                │  TPMS Gyroid, Dynamic DDD)    │ │  Hall-Petch, Non-Local Dmg)  │
                └───────────────┬───────────────┘ └──────────────┬───────────────┘
                                │ ▲                            │ ▲
                                ▼ │ Thermal & Field History    ▼ │ Yield & Overpotentials
                                └───────────────┬────────────────┘
                                                │ ▲
                                                ▼ │
                ┌───────────────┴───────────────┐
                │ Process & Synthesizability    │
                │ (Stefan, Poro-Mechanics FSI,  │
                │  Robotic Opentrons/A-Lab LIMS)│
                └───────────────┬───────────────┘
                                                │ ▲
                                                ▼ │
                ┌───────────────────────────────┴─┴──────────────────────────────┐
                │ Cross-Scale Uncertainty, TEA & Sim-to-Real Assimilation Bridge │
                │ (Grand Canonical Hull, HHI Risk, EPA CompTox, Rietveld XRD)    │
                └────────────────────────────────────────────────────────────────┘
```

---

## 💎 Advanced Capabilities & Upgrades

### 1. Crystallographic Structure Container & Universal MLIP Inference
- **Crystallographic Representation (`crystal_structure.py`):** True periodic lattice tensor $\mathbf{A}$, fractional coordinates $\mathbf{s}_i$, Wyckoff site symmetries, Voronoi bottleneck apertures, and CIF import/export.
- **Universal MLIP & Relaxation (`equivariant_mlip.py`):** MACE-MP-0 / SevenNet / CHGNet execution runtime with automated BFGS local geometry relaxation, strain-energy finite difference elastic tensor calculation $C_{ij}$, and CI-NEB minimum energy pathway calculation.

### 2. Grand Canonical Convex Hull & Phase Stability
- **Grand Canonical Solver (`convex_hull.py`):** Quickhull Delaunay triangulation calculating distance-to-hull ($\Delta E_{\text{hull}} \le 35\text{ meV/atom}$), grand potential $\Phi(V) = G - \sum \mu_i N_i$, competing phase decomposition reactions, and authentic $[V_{\text{red}}, V_{\text{ox}}]$ stability windows.

### 3. Asynchronous Active Learning & HPC Dispatch
- **Closed-Loop Epistemic Retraining (`active_learning_loop.py`):** Automatically detects OOD configurations ($NLL > 12.0$ or force variance $\sigma_F > \tau$), generates Quantum ESPRESSO `pw.x` / VASP input decks, submits batch Slurm jobs, ingests converged energies/forces, and fine-tunes MLIP weights on the fly.

### 4. Experimental Characterization Assimilation (Sim-to-Real)
- **Multi-Modal Assimilation (`bayesian_assimilation.py`):** Pseudo-Voigt XRD profile generation and weighted Rietveld profile matching ($R_{\text{wp}}$), EBSD orientation distribution functions, and Continuous Stiffness Measurement (CSM) nanoindentation load-displacement inversion.

### 5. Automated Robotic LIMS Protocol Export
- **Robotic Lab Automation (`retrosynthesis_planner.py`):** Exports machine-readable Python execution scripts for **Opentrons OT-2**, **Chemspeed**, and **A-Lab** platforms with solid/liquid dispensing volumes, CIP pressure ramp envelopes, and GHS safety handling.

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
penziv-mat evaluate-tea "Mg1.10Sc0.20Zr1.80(PS4)3" --purity battery_grade_99_9 --sinter-temp 850.0

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

Run the full multiscale test suite (62 unit tests across structural, electrochemical, multiphysics, TEA, crystallographic, active learning, and EHS domains):

```bash
python -m unittest discover tests
```

---

## ⚖️ License & Provenance

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](file:///C:/Users/jawhe/Penziv_Materials/LICENSE) and [`CITATION.cff`](file:///C:/Users/jawhe/Penziv_Materials/CITATION.cff) for citations.
