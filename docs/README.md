# Penziv Materials Documentation Hub

Welcome to the technical documentation and architectural reference for **Penziv Materials (v3.5.0-PROD)**.

---

## 📚 Technical Documentation Index

1. **[Interactive Web Portal (index.html)](file:///C:/Users/jawhe/Penziv_Materials/docs/index.html)**: Live interactive single-page overview of the 5-scale simulation hierarchy, mathematical solvers, active learning loops, and CLI suite.
2. **[First-Principles Theoretical Framework](file:///C:/Users/jawhe/Penziv_Materials/docs/FRAMEWORK_FIRST_PRINCIPLES.md)**: Derivations spanning relativistic Dirac-Coulomb-Breit quantum electrodynamics, SCAN meta-GGA, BSE optical absorption, Wigner-Peierls thermal BTE, Legendre CALPHAD grand potentials, and monolithic chemo-mechanics.
3. **[Product Development & Engineering Specification](file:///C:/Users/jawhe/Penziv_Materials/docs/PDRD_AETHERMAT_V3.2.0.md)**: Zero-compromise verification gates, TEA commodity pricing indices, supply chain Herfindahl-Hirschman (HHI) concentration models, and automated robotic A-Lab/OT-2 LIMS synthesis planning.

---

## 🏛️ Simulation Scale Matrix

| Scale Tier | Core Engine File | Physical Formulation |
| :--- | :--- | :--- |
| **Scale 5: Quantum** | `universal_neumann.py`, `gamma_surface.py`, `wigner_peierls_transport.py` | Rank-$N$ Neumann tensor projection across all 230 SG + 1651 Shubnikov groups; 2D $\gamma$-surface grids; dual-channel Peierls-Wigner thermal BTE. |
| **Scale 4: Atomistics** | `path_sampling.py`, `laguerre_voronoi.py`, `reverse_monte_carlo.py` | Equivariant MLIPs; Dijkstra percolation with Geodesic String Method; species-weighted Laguerre Voronoi & King's ring homology; Metropolis RMC glass refinement. |
| **Scale 3: Mesoscale** | `calphad_grand_potential.py`, `cohesive_interface.py` | Multi-phase CALPHAD grand potentials $\Omega(\boldsymbol{\mu}, T)$ with Khachaturyan microelasticity & STZ glass plasticity; cohesive traction-separation with coupled PNP-Biot drift-diffusion. |
| **Scale 2: Continuum** | `multiscale_coupling.py`, `unified_spectral_solver.py`, `odf_crystal_plasticity.py` | Monolithic 3D chemo-mechanics ($\Delta \mu = -\Omega \sigma_h$); fully anisotropic rank-4 Green's tensor $\mathbf{\Gamma}^0(\mathbf{k})$; ODF texture plasticity with non-Schmid shear resolution. |
| **Scale 1: Process** | `retrosynthesis_planner.py`, `degradation_engine.py` | Multi-element environmental degradation; automated robotic Python execution scripts for Opentrons OT-2, Chemspeed, and Berkeley A-Lab platforms. |
| **Meta-Bridge** | `online_active_retraining.py`, `opencalphad_tdb.py`, `differentiable_pareto_qd.py` | Closed-loop active learning with automated Quantum ESPRESSO/VASP deck generation; OpenCALPHAD TDB parser; high-dimensional continuous CVT-MAP-Elites Pareto discovery. |

---

## 🧪 Verification & Testing

```bash
# Execute the comprehensive test suite (95 tests across 19 modules)
python -m unittest discover tests
```
