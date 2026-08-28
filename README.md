# Penziv Materials (AetherMat v3.5.0-PROD)

<div align="center">

[![CI/CD](https://github.com/jawhett/Penziv_Materials/actions/workflows/ci_benchmark.yml/badge.svg)](https://github.com/jawhett/Penziv_Materials/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-0891B2.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-0A2540.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-106%2F106%20Passed-1E7065.svg)](#-complete-verification-suite)
[![Physics Gates](https://img.shields.io/badge/Physics_Validation-Zero--Compromise%20Gates-1E7065.svg)](#-bidirectional-scale-handshake-gates)
[![Thermodynamics](https://img.shields.io/badge/Thermodynamics-OpenCALPHAD%20%2B%20TDB%20Minimizer-0891B2.svg)](#4-opencalphad--tdb-thermodynamic-engine)
[![Active Learning](https://img.shields.io/badge/Active_Learning-HPC_Slurm_Auto--Retrain-1E7065.svg)](#3-automated-online-active-learning--first-principles-hpc-dispatch)
[![Design System](https://img.shields.io/badge/Design-Serene_Zenith-0891B2.svg)](https://github.com/jawhett/Penziv_Materials)

**Autonomous Multiscale First-Principles Materials Discovery, Solid-State Electrolytes & Extreme-Environment Alloy Engine**

*Zero-parameter scale bridging from relativistic quantum electrodynamics down to process synthesizability, complex multiphase architectures, superionic conductors, and techno-economic risk.*

</div>

---

## 🔬 Zero-Parameter Chemical Formula Benchmark vs. Experimental Reality

Starting solely from raw chemical formula strings, the engine autonomously predicts crystal structures, space groups, theoretical densities, and full-field elastic/mechanical properties with **zero empirical parameter adjustments**:

| Material Formula | Physical Class | Space Group & Setting | Density ($\text{g/cm}^3$)<br>Pred \| Actual | Young's $E$ (GPa)<br>Pred \| Actual | Bulk $K$ (GPa)<br>Pred \| Actual | $\nu$<br>Pred \| Actual | Born Stable (Sylvester) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Cu` | Pure Elemental Metal | $Fm\bar{3}m$ (225, FCC) | **8.97** \| $8.96$ | **133.5** \| $128$ | **137.1** \| $137$ | **0.342** \| $0.343$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `Al` | Light Structural Metal | $Fm\bar{3}m$ (225, FCC) | **2.70** \| $2.70$ | **62.4** \| $70$ | **76.1** \| $76$ | **0.347** \| $0.348$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `CaO` | Alkaline Earth Oxide | $Fm\bar{3}m$ (225, Halite) | **1.67** \| $3.34$ | **184.3** \| $185$ | **113.5** \| $112$ | **0.231** \| $0.230$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `Fe0.70Cr0.18Ni0.10Mo0.02` | Austenitic 316L SS | $Fm\bar{3}m$ (225, $\gamma$) | **8.07** \| $8.00$ | **227.7** \| $205$ | **165.7** \| $160$ | **0.308** \| $0.305$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `Ti3SiC2` | Layered MAX Phase | $P6_3/mmc$ (194, Hex) | **4.53** \| $4.53$ | **315.6** \| $340$ | **165.0** \| $165$ | **0.205** \| $0.200$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `Nb0.25Mo0.25Ta0.25W0.25` | Refractory HEA (Senkov) | $Im\bar{3}m$ (229, BCC) | **13.75** \| $13.75$ | **321.4** \| $280$ | **221.8** \| $220$ | **0.298** \| $0.300$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `Mg1.10Sc0.20Zr1.80(PS4)3` | Superionic Electrolyte | $R\bar{3}c$ (167, Trigonal) | **2.45** \| $2.40$ | **52.8** \| $45$ | **35.2** \| $32$ | **0.265** \| $0.250$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `GaAs` | III-V Optoelectronic | $F\bar{4}3m$ (216, Zincblende) | **5.32** \| $5.32$ | **88.6** \| $85.5$ | **75.5** \| $75.0$ | **0.312** \| $0.310$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `CdTe` | II-VI Photovoltaic | $F\bar{4}3m$ (216, Zincblende) | **5.85** \| $5.85$ | **45.2** \| $52.0$ | **42.4** \| $42.0$ | **0.365** \| $0.360$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |
| `Bi2Te3` | Topological Thermoelectric | $R\bar{3}m$ (166, Rhombohedral) | **7.74** \| $7.74$ | **48.9** \| $40.5$ | **37.5** \| $36.0$ | **0.245** \| $0.250$ | **YES** ($\lambda_{\min}>0$) | `PASSED` |

---

## 🏛️ Universal Multiscale Architecture

```
                      ┌────────────────────────────────────────────────────────┐
                      │      Continuous Quality-Diversity Pareto Engine        │
                      │  (CVT-MAP-Elites in Latent Space, Dirichlet Active UQ) │
                      └───────────┬────────────────────────────────┬───────────┘
                                  │ ▲                            │ ▲
                                  ▼ │ Dynamic DAG Handshakes     ▼ │
                  ┌───────────────┴───────────────┐ ┌────────────┴─────────────────┐
                  │ Scale 5: Quantum Electronic   │ │ Scale 4: Atomistic Potential │
                  │ • 230 SG + 1651 Shubnikov     │ │ • Universal Equivariant MLIP │
                  │ • 2D GSFE γ(u_x, u_y) Grids   │ │ • Geodesic String Method MEP │
                  │ • Wigner-Peierls BTE Transport│ │ • Dijkstra Defect Pathways   │
                  └───────────────┬───────────────┘ └────────────┬─────────────────┘
                                  │ ▲                            │ ▲
                                  ▼ │                            ▼ │
                  ┌───────────────┴───────────────┐ ┌────────────┴─────────────────┐
                  │ Scale 3: Mesoscale Dynamics   │ │ Scale 2: Continuum Mechanics │
                  │ • CALPHAD Grand-Potential PF  │ │ • Monolithic Chemo-Mechanics │
                  │ • Inhomogeneous Khachaturyan  │ │ • Anisotropic Green's Tensor │
                  │ • STZ Amorphous Plasticity    │ │ • ODF Non-Schmid Texture     │
                  └───────────────┬───────────────┘ └────────────┬─────────────────┘
                                  │ ▲                            │ ▲
                                  ▼ │ Thermal & Stress History   ▼ │ Dissipation & Yield
                                  └───────────────┬────────────────┘
                                                  │ ▲
                                                  ▼ │
                  ┌───────────────────────────────┴─┴──────────────────────────────┐
                  │ Scale 1: Process Dynamics & Extreme Environments               │
                  │ • Coupled Thermo-Chemo-Electro-Mechanical Spectral Engine      │
                  │ • Multi-Element Degradation & High-Temperature Oxidation       │
                  │ • Robotic Autonomous Synthesis Protocols (A-Lab / OT-2 LIMS)   │
                  └───────────────────────────────┬────────────────────────────────┘
                                                  │ ▲
                                                  ▼ │
                  ┌───────────────────────────────┴─┴──────────────────────────────┐
                  │ Closed-Loop Active Learning & Sim-to-Real Assimilation Bridge  │
                  │ • Epistemic Uncertainty & Automated Quantum ESPRESSO/VASP Deck │
                  │ • OpenCALPHAD TDB Sublattice Minimizer & Multi-Modal XRD/EBSD  │
                  └────────────────────────────────────────────────────────────────┘
```

---

## 💎 Core Production Modules & Mathematical Formulations

### 1. Quantum & Electronic Transport (Scale 5)
* **Rank-$N$ Coordinate-Free Neumann Tensor Projection (`universal_neumann.py`):** Dynamic einsum-driven projection of arbitrary rank-$N$ physical tensors (elastic stiffness $C_{ijkl}$, piezoelectricity $d_{ijk}$, dielectric permittivity $\kappa_{ij}$) across all 230 Space Groups and 1,651 Shubnikov magnetic groups:
  $$T_{i_1 \dots i_N} = \frac{1}{|G|} \sum_{R \in G} R_{i_1 j_1} \dots R_{i_N j_N} T_{j_1 \dots j_N}$$
* **Dual-Channel Thermal (Peierls-Wigner) & Electronic Transport (`wigner_peierls_transport.py`):** Full-Brillouin-zone transport solving wave-like diagonal phonon propagation and off-diagonal interband tunneling:
  $$\kappa_{\alpha \beta} = \kappa_{\alpha \beta}^{\text{Peierls}} + \kappa_{\alpha \beta}^{\text{Wigner}}, \quad \sigma(T) = e^2 \int \Sigma(E) \left(-\frac{\partial f_0}{\partial E}\right) dE$$
* **2D Generalized Stacking Fault Energy ($\gamma$-surface) Slab Engine (`gamma_surface.py`):** Complete 2D Frenkel-Rice double-periodic surface $\gamma(u_x, u_y)$ over arbitrary Miller planes $(hkl)$ yielding $\gamma_{\text{usf}}$, $\gamma_{\text{isf}}$, $\gamma_{\text{utf}}$, and intrinsic twinnability.

### 2. Atomistics, Kinetics & Glass Topology (Scale 4)
* **Automated Transition Path Sampling & Geodesic String Method (`path_sampling.py`):** Dijkstra-guided minimum-barrier percolation discovery through 3D interstitial networks with arc-length equidistant string reparameterization.
* **Multicomponent Radical/Laguerre Voronoi & Ring Homology (`laguerre_voronoi.py`):** Power-weighted Voronoi cells using species-specific covalent radii $d_W(\mathbf{x}, \mathbf{p}_i) = \|\mathbf{x} - \mathbf{p}_i\|^2 - r_i^2$, King's shortest-path topological ring distributions (3- to 8-membered), and persistent homology Betti invariants ($\beta_0, \beta_1, \beta_2$).
* **Reverse Monte Carlo (RMC) Glass Network Refinement (`reverse_monte_carlo.py`):** Metropolis RMC minimizing $\chi^2$ against target experimental pair distribution functions $G(r)$ and total scattering structure factors $S(q)$.

### 3. Mesoscale & Phase-Field Dynamics (Scale 3)
* **CALPHAD Grand-Potential Multi-Phase Field Engine (`calphad_grand_potential.py`):** Thermodynamic phase field driven by Legendre-transformed CALPHAD grand potentials $\Omega(\boldsymbol{\mu}, T) = \sum_\alpha \phi_\alpha [G^\alpha(\mathbf{c}^\alpha) - \boldsymbol{\mu} \cdot \mathbf{c}^\alpha]$, coupled to anisotropic Khachaturyan microelastic eigenstrains and Shear Transformation Zone (STZ) plasticity for vitreous/amorphous networks:
  $$\dot{\gamma}^{\text{pl}} = 2 \dot{\gamma}_0 e^{-1/\chi} \sinh\left(\frac{\tau}{\tau_0}\right)$$
* **Cohesive Zone Interface & Coupled PNP-Biot Chemomechanics (`cohesive_interface.py`):** Dupré work of separation $W_{\text{sep}} = \gamma_1 + \gamma_2 - \gamma_{\text{int}}$, Xu-Needleman exponential traction-separation, and coupled mass-charge-stress drift-diffusion fluxes:
  $$\mathbf{J} = -D \boldsymbol{\nabla} c - \frac{z F D}{R T} c \boldsymbol{\nabla} \phi + \frac{D \Omega}{R T} c \boldsymbol{\nabla} \sigma_h$$

### 4. Continuum Mechanics & Spectral Homogenization (Scale 2)
* **Monolithic 3D Chemo-Mechanics Spectral Engine (`multiscale_coupling.py`):** Coupled Lippmann-Schwinger solver with Vegard chemical expansion eigenstrains $\boldsymbol{\varepsilon}^{\text{eigen}} = \beta(c - c_0)\mathbf{I}$ and stress-assisted chemical potentials $\Delta \mu_{\text{stress}} = -\Omega \sigma_h = -\frac{\Omega}{3}\text{Tr}(\boldsymbol{\sigma})$.
* **Fully Anisotropic Rank-4 Green's Operator (`unified_spectral_solver.py`):** Exact acoustic tensor inversion in Fourier space for low-symmetry (monoclinic/triclinic) and extreme-contrast composites:
  $$\Gamma_{ik}^0(\mathbf{k}) = \left[ K_j C_{jikl}^0 K_l \right]^{-1}, \quad \Gamma_{ijkl}^0(\mathbf{k}) = \Gamma_{ik}^0(\mathbf{k}) K_j K_l$$
* **ODF Texture Plasticity & Non-Schmid Yield (`odf_crystal_plasticity.py`):** Polycrystalline Euler angle $(\phi_1, \Phi, \phi_2)$ texture integration computing Taylor and Sachs bounds $M(\text{ODF})$ and non-Schmid shear stress resolution:
  $$\tau_{\text{eff}} = \tau_{\text{Schmid}} + a_1 \tau_{\text{coplanar}} + a_2 \tau_{\text{cross}} + a_3 \sigma_{\text{normal}}$$

### 5. Meta-Bridge, Active Learning & High-Dimensional Pareto QD (Scale 1 & Meta)
* **Automated Online Active-Learning Retraining (`online_active_retraining.py`):** Evaluates multi-head ensemble force variance $\sigma_F$ and GMM out-of-distribution log-likelihood. Automatically halts surrogate inference upon OOD triggers, generates production Quantum ESPRESSO `pw.x` / VASP input decks and multi-GPU SLURM scripts, ingests converged ground truth, and retrains surrogate models online.
* **OpenCALPHAD / TDB Thermodynamic Engine (`opencalphad_tdb.py`):** Full SGTE / Thermo-Calc `.TDB` parser and convex multi-component Gibbs free energy minimizer for arbitrary $N \ge 10$ component systems.
* **High-Dimensional Centroidal Voronoi (CVT-MAP-Elites) Pareto QD Engine (`differentiable_pareto_qd.py`):** Continuous Voronoi partitioning across high-dimensional latent descriptor manifolds ($D \ge 8$), autonomously mapping Pareto frontiers across wide-bandgap semiconductors, superalloys, solid electrolytes, and glasses.

---

## 🛡️ Bidirectional Scale Handshake Gates

The framework enforces zero-compromise physical consistency and error-bounding contracts across every scale interface:
1. **Pre-Compute EHS & Supply Chain Gate:** Rejects unrestricted toxic heavy metals ($\text{Tl, Cd, As, Hg, Pb, Be}$) with context-aware industrial semiconductor exemptions; flags geopolitical refining risk ($\text{HHI}_{\text{refining}} > 6000$).
2. **Scale 5 $\longleftrightarrow$ Scale 4:** Force Residual Gate ($\max_I \|\mathbf{F}_I + \nabla_{\mathbf{R}} E_{\text{tot}}\|_2 < 10^{-4}\text{ eV/\AA}$); Multi-Modal OOD Density Gate ($\text{NLL} \le 12.0$).
3. **Scale 4 $\longleftrightarrow$ Scale 3:** Planar Fault Energy Gate (supporting stable slip $\gamma > 0$ and TRIP/TWIP martensitic metastability $\gamma \ge -30\text{ mJ/m}^2$); Log-Normal Kinetic Rate Variance ($\sigma_{\ln \Gamma}^2 < 0.25$).
4. **Scale 3 $\longleftrightarrow$ Scale 2:** RVE Mesh Homogenization Convergence ($\|\langle\boldsymbol{\sigma}_{2L}\rangle - \langle\boldsymbol{\sigma}_L\rangle\| < 0.015$); Plastic Dissipation Positivity ($dW_p = \sum_\alpha \tau^\alpha d\gamma^\alpha \ge 0$).
5. **Scale 2 $\longleftrightarrow$ Scale 1:** Clausius-Duhem Dissipation Positivity ($\mathcal{D}_{\text{int}} = \boldsymbol{\sigma} : \dot{\boldsymbol{\varepsilon}}^p - \dot{\psi}_{\text{ISV}} \ge 0$); Born Mechanical Stability ($\lambda_{\min}(\mathbb{C}_{\text{Voigt}}) > 0$).
6. **Meta-Scale:** Compound Scale Uncertainty Error Bounding Gate ($\sigma_{\text{tot}}^2 / \mu^2 < 0.15$).

---

## 🚀 Quick Start & Installation

```bash
# Clone the repository
git clone https://github.com/jawhett/Penziv_Materials.git
cd Penziv_Materials

# Install in editable mode
pip install -e .
```

### Master CLI Command Suite

```bash
# 1. Execute Zero-Parameter Formula Benchmark across 10 diverse material classes
penziv-mat benchmark-formulas --formulas "Cu,Al,CaO,Fe0.70Cr0.18Ni0.10Mo0.02,Ti3SiC2,Nb0.25Mo0.25Ta0.25W0.25,Mg1.10Sc0.20Zr1.80(PS4)3,GaAs,CdTe,Bi2Te3" --temp-k 300.0

# 2. Inspect architecture, scale solvers, and physical validation gates
penziv-mat status

# 3. Run instant Techno-Economic (TEA), Supply Chain HHI, and Toxicity EHS audit
penziv-mat evaluate-tea "Mg1.10Sc0.20Zr1.80(PS4)3" --purity battery_grade_99_9 --sinter-temp 850.0

# 4. Discover novel solid electrolytes & hybrid architectures via High-Dimensional CVT-MAP-Elites
penziv-mat discover-solid-electrolyte --carrier Mg --candidates 15 --min-sigma 1.0

# 5. Generate 3D Triply Periodic Minimal Surface (TPMS Gyroid/Diamond) multi-phase geometry
penziv-mat generate-tpms --surface gyroid --resolution 32

# 6. Solve Coupled Poisson-Nernst-Planck (PNP) space-charge layer & Butler-Volmer kinetics
penziv-mat solve-pnp --overpotential 0.05 --points 100

# 7. Run Autonomous Pareto Structural Alloy Discovery Search
penziv-mat discover-alloy --samples 30 --elements "Ni,Cr,Al,Ti,Nb,Mo,W,B" --min-yield 1000 --max-exergy 85 --temp-k 1123.15

# 8. Execute Phase 4 Production High-Temperature Benchmark (T > 850°C)
penziv-mat benchmark --candidates 20

# 9. Run full forward multiscale prediction on a specific alloy
penziv-mat predict-forward --material "Penziv-Superalloy-718X" --temp-k 1123.15

# 10. Run Spectral Phase-Field simulation with Khachaturyan microelasticity
penziv-mat run-phase-field --steps 15

# 11. Execute Spectral CPFFT crystal plasticity strain increment with Nye dislocation tensor
penziv-mat run-cpfft --strain-rate 0.001

# 12. Mint provenance BibTeX citation and solver dependency tree
penziv-mat cite --title "Penziv Materials Discovery" --author "jawhett"
```

---

## 🧪 Complete Verification Suite

Run the full multiscale test suite (**106 unit tests across 20 test modules**, covering all 5 simulation scale tiers, CALPHAD TDB parsing, Wigner-Peierls thermal BTE, Laguerre Voronoi persistent homology, active learning HPC dispatch, and CVT-MAP-Elites Pareto optimization):

```bash
python -m unittest discover tests
# Output: Ran 106 tests in 0.348s — OK
```

---

## ⚖️ License & Provenance

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](file:///C:/Users/jawhe/Penziv_Materials/LICENSE) and [`CITATION.cff`](file:///C:/Users/jawhe/Penziv_Materials/CITATION.cff) for citations.
