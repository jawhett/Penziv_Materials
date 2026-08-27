"""Physical constants, unit conversions, and standard scientific thresholds."""

import numpy as np

# Fundamental Physical Constants (CODATA 2018/2022)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
M_ELECTRON = 9.1093837015e-31  # Electron rest mass (kg)
SPEED_OF_LIGHT = 299792458.0  # Speed of light in vacuum (m/s)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
KB = 1.380649e-23  # Boltzmann constant (J/K)
KB_EV = KB / E_CHARGE  # Boltzmann constant (eV/K) ~ 8.617333262e-5 eV/K
R_GAS = 8.314462618  # Universal gas constant (J/(mol·K))
AVOGADRO = 6.02214076e23  # Avogadro constant (mol^-1)

# Atomic & Length Scales
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_METERS = 1.0e-10
RYDBERG_TO_EV = 13.605693122994
HARTREE_TO_EV = 27.211386245988
EV_TO_JOULE = E_CHARGE
GPA_TO_PA = 1.0e9

# Framework Validation Thresholds
TOL_FORCE_RESIDUAL_EV_ANG = 1.0e-4  # eV/Å
TOL_OOD_GMM_NLL_DEFAULT = 12.0  # Negative log likelihood threshold
TOL_LOGNORMAL_RATE_VAR = 0.25  # Max allowable log-normal kinetic rate variance
TOL_RVE_STRESS_CONVERGENCE = 0.015  # Relative L2 stress difference
TOL_COMPOUND_VARIANCE_BOUND = 0.15  # σ_tot^2 / μ^2 meta-bridge bound
