# Product Development & Requirements Document (PDRD)

**Project Codename:** Penziv Materials / AetherMat (Multiscale Autonomous Materials Discovery Framework - MAMDF)  
**Document Version:** 3.2.0-PROD (Rigorous Physics Verification & Scale-Coupling Revision)  
**Target Release Model:** Open Core (Dual Apache 2.0 / Commercial Proprietary Licensing)

---

## 1. Executive Summary & Enterprise Architecture

Penziv Materials (AetherMat v3.2.0) is an open-source, modular, agentic artificial intelligence framework engineered for **zero-parameter, multiscale materials discovery**. Version 3.2.0 resolves kinetic error propagation, multi-modal out-of-distribution (OOD) failures, numerical gradient corruption from lossy compression, and ill-posed inverse calibrations across the Bidirectional Process-Structure-Property-Performance (PSPP) loop.

---

## 2. Specialized Modular Agent Specifications & Scale-Bridging I/O

### 2.1 Quantum & Electronic Structure Agent (`Q-ELEC`)
- **Finite-Temperature Free Energy:** Total Helmholtz free energy:
  $$F(V, T) = E_{\text{DFT}}(V) + F_{\text{elec}}(V, T_e) + F_{\text{vib}}(V, T_{\text{ion}})$$
  evaluated via Temperature-Dependent Effective Potential (TDEP) methods.
- **SRO-Dependent Planar Fault Manifolds:** Computes Generalized Stacking Fault Energy $\gamma_{\text{GSFE}}(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, Anti-Phase Boundary energy $\gamma_{\text{APB}}(\boldsymbol{\alpha}_{\text{SRO}})$, and Superlattice Intrinsic Stacking Fault energy $\gamma_{\text{SISF}}(\boldsymbol{\alpha}_{\text{SRO}})$ parameterized by Warren-Cowley SRO tensors $\alpha_{ij}^m = 1 - P_{ij}^m / c_j$.
- **Relativistic Ground State & Many-Body Screening:** SCAN meta-GGA, RPA correlation via ACFDT, Foldy-Wouthuysen SOC, $G_0W_0$ quasiparticle states, and BSE excitonic dielectric tensors.
- **$\Delta$-Learning Solver Alignment:** Eliminates systematic energy/stress offsets when falling back between commercial and open-source solvers:
  $$\Delta E_{\text{correction}} = \mathcal{M}_\Delta(Z_I, \mathbf{R}_I, \text{functional})$$
- **Scale-Bridging Ingests (Inputs):**
  - Atomic numbers $Z_I$, coordinates $\mathbf{R}_I$, space group, temperatures ($T_e, T_{\text{ion}}$), magnetic moments $\mathbf{m}_I$, SRO tensors $\boldsymbol{\alpha}_{\text{SRO}}$.
- **Scale-Bridging Emits (Outputs):**
  - Formation enthalpy $\Delta H_f^0(T)$, 4th-order elastic tensor $\mathbb{C}_{ijkl}(T)$, thermal expansion tensor $\boldsymbol{\alpha}(T)$, SRO-dependent fault energies $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, Helmholtz free energy $F(V, T)$, excitonic dielectric tensor $\boldsymbol{\varepsilon}_{\alpha\beta}(\omega)$, $\Delta\text{-ML}$ offset operators $\mathcal{M}_\Delta$.

---

### 2.2 Atomistic Dynamics & Extended Defect Agent (`ATOM-DYN`)
- **Active Learning $E(3)$-Equivariant MLIPs with Multi-Modal OOD Detection:** Deploys message-passing models (MACE, Allegro) with Gaussian Mixture Model (GMM) and deep ensemble variance across latent feature spaces $\mathbf{z}_i$:
  $$\mathcal{L}_{\text{OOD}}(\mathbf{z}_i) = -\ln \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{z}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) > \zeta_{\text{threshold}} \quad \lor \quad \sigma_{\text{ensemble}}^2(\mathbf{F}_i) > \epsilon_F$$
  Outliers route autonomously to `Q-ELEC` for single-point quantum retraining.
- **Kinetic Activation Uncertainty Bounds:** Computes migration barriers $\Delta E_m$ and cross-slip activation energies $W_{\text{cs}}(\boldsymbol{\sigma})$ via CI-NEB, propagating log-normal variance on kinetic rate constants:
  $$\sigma_{\ln \Gamma}^2 = \left(\frac{\sigma_{\Delta E}}{k_B T}\right)^2 + \sigma_{\ln \nu_0}^2$$
- **Grain Boundary Manifolds & Core Structure:** 5-parameter GB energy $\gamma_{\text{GB}}(\theta, \hat{\mathbf{n}})$, solute segregation free energies $\Delta E_b^{\text{seg}}$, and Peierls stress $\tau_P$ via Semidiscrete Variational Peierls-Nabarro (SVPN).
- **Scale-Bridging Ingests (Inputs):**
  - Potential Energy Surface, $\mathbb{C}_{ijkl}(T)$, $\gamma(\mathbf{u}, \boldsymbol{\alpha}_{\text{SRO}})$, active learning DFT retrain data.
- **Scale-Bridging Emits (Outputs):**
  - Defect migration barrier distributions $\Delta E_m \pm \sigma$, diffusion tensors $\mathbf{D}(T)$, 5-parameter GB energy manifold $\gamma_{\text{GB}}(\theta, \hat{\mathbf{n}})$, solute segregation free energies $\Delta E_b^{\text{seg}}$, SVPN Peierls core stresses $\tau_P$, cross-slip activation energies $W_{\text{cs}}(\boldsymbol{\sigma}) \pm \sigma$, cohesive traction-separation parameters ($W_{\text{sep}}, \lambda$).

---

### 2.3 Mesoscale Kinetics & Microstructure Agent (`MESO-KINETIC`)
- **Coupled Field Kinetics & Microelasticity:** Cahn-Hilliard and Allen-Cahn equations coupled with Khachaturyan-Shtremel microelasticity kernels $B(\mathbf{n})$ and Continuous Growth Model (CGM) velocity-dependent solute trapping with sub-grid boundary layer formulations:
  $$k_i(V) = \frac{k_i^e + (a_0 V / D_i)}{1 + (a_0 V / D_i)}$$
- **Solute-Coupled Discrete Dislocation Dynamics (DDD):** Nodal kinematics driven by Peach-Koehler forces, dynamic Cottrell atmosphere solute drag $\mathbf{B}_{\text{drag}}(c_{\text{solute}}, T)$, and precipitate shearing/Orowan looping interactions parameterized by $\gamma_{\text{APB}}(\boldsymbol{\alpha}_{\text{SRO}})$.
- **Singularity-Free Conformed RVE Reconstruction:** Level-set interface regularizations and marching-tetrahedra surface smoothing converting continuous order parameters into finite element meshes without step-discontinuity stress concentrations. Orientation Distribution Function (ODF) extraction.
- **Scale-Bridging Ingests (Inputs):**
  - Diffusion tensors $\mathbf{D}(T)$, $\gamma_{\text{GB}}$ manifolds, $\Delta E_b^{\text{seg}}$, single-crystal stiffness $\mathbb{C}_{ijkl}(T)$, thermal history from Scale 1 ($T(t), G, \dot{T}, V$).
- **Scale-Bridging Emits (Outputs):**
  - Singularity-free conformed RVE topologies, ODF crystallographic texture maps $(\phi_1, \Phi, \phi_2)$, critical resolved shear stresses $\tau_{\text{CRSS}}^\alpha$, asymmetric dislocation hardening matrix $h_{\alpha\beta}$, void swelling rates $dC_{\text{void}}/dt$.

---

### 2.4 Continuum Micromechanics & Mechanics Agent (`CONT-MICRO`)
- **Finite-Strain Multiplicative Crystal Plasticity (CPFEM/CPFFT):** $\mathbf{F} = \mathbf{F}^e \mathbf{F}^p$, $\dot{\gamma}^\alpha = \dot{\gamma}_0 |\tau^\alpha / g^\alpha|^{1/m} \text{sgn}(\tau^\alpha)$, asymmetric dislocation hardening matrices $h_{\alpha\beta}$, and Nye dislocation density tensor $\boldsymbol{\alpha}_{\text{Nye}} = \nabla \times \mathbf{F}^p$.
- **High-Temperature Diffusional & Dislocation Creep:** Coupled power-law dislocation climb-assisted glide and grain boundary Coble/Nabarro-Herring diffusional creep:
  $$\dot{\varepsilon}_{\text{creep}} = A_{\text{disl}}\left(\frac{\sigma_{\text{eq}}}{G}\right)^n \exp\left(-\frac{Q_{\text{core}}}{RT}\right) + A_{\text{diff}}\left(\frac{\sigma_{\text{eq}}}{d_{\text{grain}}^p}\right)\frac{\Omega D_{\text{eff}}}{k_B T}$$
- **Energy-Conserving Non-Local Fracture & Weibull Scaling:** Non-local implicit gradient damage formulations preserving intrinsic fracture energy $G_c$, mapped to stochastic weakest-link Weibull failure distributions.
- **Scale-Bridging Ingests (Inputs):**
  - Conformed RVE meshes, ODF crystallographic textures, critical resolved shear stresses $\tau_{\text{CRSS}}^\alpha$, hardening matrix $h_{\alpha\beta}$, single-crystal $\mathbb{C}_{ijkl}(T)$, intrinsic fracture energy $G_c$, characteristic regularization length $l_c$, initial process residual stresses $\boldsymbol{\sigma}_{\text{res}}(\mathbf{x})$ from Scale 1.
- **Scale-Bridging Emits (Outputs):**
  - True stress-strain curves $\boldsymbol{\sigma}(\boldsymbol{\varepsilon})$, homogenized polycrystal yield surfaces $\Sigma(\boldsymbol{\sigma})$, steady-state creep rate laws $\dot{\varepsilon}_{\text{ss}}(T, \boldsymbol{\sigma})$, plane-strain fracture toughness $K_{Ic}$, Paris fatigue crack growth parameters ($C, m$), Weibull statistical failure modulus $m_{\text{weibull}}$.

---

### 2.5 Process Dynamics & Synthesizability Agent (`PROC-MFG`)
- **Melt-Pool Hydrodynamics & Thermal Field Generation:** Stefan solidification problem coupled with Navier-Stokes flows, buoyancy, and Marangoni thermocapillary shear:
  $$\boldsymbol{\tau}_s = \left(\frac{\partial\gamma}{\partial T}\right)\nabla_s T, \quad \rho C_p \left(\frac{\partial T}{\partial t} + \mathbf{u}\cdot\nabla T\right) = \nabla \cdot (k \nabla T) - \rho L_f \frac{\partial f_L}{\partial t}$$
- **Transient Environmental Degradation & Oxidation:** Dynamic oxide scale growth ($x^2 = k_p t$) and stress-assisted interstitial embrittlement along grain boundaries.
- **Synthesizability Screening & Exergy Limits:** Classical Nucleation Theory barriers $\Delta G^*$, equilibrium vapor pressures $p_i^{\text{vap}}(T)$, and minimum mineral ore extraction exergy work:
  $$\text{Ex}_{\min} = \Delta G_{\text{reduction}}^0 - RT \ln(\gamma c_{\text{ore}})$$
- **Scale-Bridging Ingests (Inputs):**
  - Target alloy composition, manufacturing parameters (laser power $P$, scan speed $v$, beam radius $r_b$, gas pressure $p_{\text{gas}}$), macroscopic yield surface $\Sigma(\boldsymbol{\sigma})$ and mechanical limit states from Scale 2.
- **Scale-Bridging Emits (Outputs):**
  - Space-time thermal history fields ($T(\mathbf{x}, t), G, \dot{T}, V$), process-induced residual stress fields $\boldsymbol{\sigma}_{\text{res}}(\mathbf{x})$, multi-element parabolic oxidation rate constants $k_p$, synthesizability confidence index, minimum crustal ore extraction exergy $\text{Ex}_{\min}$.

---

### 2.6 Cross-Scale Uncertainty Bridge & Experimental Assimilation Agent (`UQ-BRIDGE`)
- **Frame-Indifferent & Thermodynamic PINOs:** Fourier Neural Operators (FNO) enforcing hard physical symmetries and Principle of Material Frame Indifference under all rigid rotations $\mathbf{Q} \in \text{SO}(3)$:
  $$\mathcal{N}(\mathbf{Q}\mathbf{F}\mathbf{Q}^T) = \mathbf{Q}\mathcal{N}(\mathbf{F})\mathbf{Q}^T \quad \forall \mathbf{Q} \in \text{SO}(3)$$
- **Arc-Length Bifurcation Tracking:** Pseudo-arc-length continuation constraints capturing snap-back and crack localization paths.
- **Multi-Objective Depth-Corrected Sim-to-Real Assimilation:** Calibrates parameters against Synchrotron XRD, EBSD, APT cluster distributions, and Nix-Gao depth-corrected nanoindentation hardness maps:
  $$H^2 = H_0^2 \left(1 + \frac{h^*}{h}\right)$$
- **Scale-Bridging Ingests (Inputs):**
  - Orthogonal experimental datasets (Synchrotron XRD phase fractions, EBSD orientation maps, Atom Probe Tomography cluster distributions, Nanoindentation depth curves).
- **Scale-Bridging Emits (Outputs):**
  - Calibrated joint Bayesian parameter posteriors $P(\boldsymbol{\theta} \mid \mathbf{y})$, verified $\text{SO}(3)$-equivariant PINO surrogates with pseudo-arc-length continuation bounds.

---

## 3. High-Throughput I/O, Storage & Invariant Preservation

- **Invariant-Preserving Compression:**
  - Primitive scalar fields ($T, c_k$): Loss-bounded compression (SZ3/ZFP) with absolute bound $\varepsilon \le 10^{-6}$.
  - Differential invariant fields and tensors ($\mathbf{F}^p, \boldsymbol{\alpha}_{\text{Nye}}, \boldsymbol{\sigma}$): Lossless entropy encoding (Zstandard/Blosc) to prevent high-frequency gradient noise from corrupting $\nabla \times \mathbf{F}^p$ and $\nabla \cdot \boldsymbol{\sigma} = \mathbf{0}$.
- **Non-Blocking Tiered State Checkpointing:** Multi-tiered ring buffers (NVMe $\to$ parallel Zarr/HDF5 object stores).

---

## 4. Bidirectional Handshake Validation Gates

1. **Scale 5 $\longleftrightarrow$ Scale 4:** Force residual $< 10^{-4}\text{ eV/\AA}$; OOD gate $\mathcal{L}_{\text{OOD}}(\mathbf{z}_i) \le \zeta_{\text{threshold}}$.
2. **Scale 4 $\longleftrightarrow$ Scale 3:** Stacking fault positivity $\min \gamma(\mathbf{u}) > 0$; Log-normal kinetic rate variance $\sigma_{\ln \Gamma}^2 < 0.25$.
3. **Scale 3 $\longleftrightarrow$ Scale 2:** RVE convergence $\|\langle\boldsymbol{\sigma}_{2L}\rangle - \langle\boldsymbol{\sigma}_L\rangle\| < 0.015$; Plastic dissipation positivity $dW_p \ge 0$.
4. **Scale 2 $\longleftrightarrow$ Scale 1:** Clausius-Duhem dissipation $\mathcal{D}_{\text{int}} \ge 0$; Full Born Mechanical Stability $\lambda_{\min}(\mathbb{C}_{\text{Voigt}}) > 0$.
5. **Meta-Scale:** Compound scale variance bound $\sigma_{\text{tot}}^2 / \mu^2 < 0.15$; Bayesian posterior convergence $D_{\text{KL}} < \varepsilon_{\text{bayes}}$.
