"""Physical and Crystallographic Constants for the Penziv Materials Engine."""

# Physical Constants (SI and standard electronic units)
BOLTZMANN_EV_K = 8.617333262e-5      # eV / K
KB_EV = 8.617333262e-5               # eV / K
BOLTZMANN_J_K = 1.380649e-23         # J / K
GAS_CONSTANT_J_MOL_K = 8.314462618   # J / (mol * K)
R_GAS = 8.314462618                  # J / (mol * K)
AVOGADRO_NUMBER = 6.02214076e23      # mol^-1
AVOGADRO_N_A = 6.02214076e23         # mol^-1
FARADAY_C_MOL = 96485.33212          # C / mol
FARADAY_CONSTANT = 96485.33212       # C / mol
PLANCK_CONSTANT = 6.62607015e-34     # J * s
PLANCK_EV_S = 4.135667696e-15        # eV * s
HBAR = 1.054571817e-34               # J * s
HBAR_EV_S = 6.582119569e-16          # eV * s
E_CHARGE = 1.602176634e-19           # Coulombs
ELEMENTARY_CHARGE_C = 1.602176634e-19# Coulombs
M_ELECTRON = 9.1093837015e-31        # kg
SPEED_OF_LIGHT = 299792458.0         # m / s
EPSILON_0 = 8.8541878128e-12         # F / m

# Unit Conversions
HARTREE_TO_EV = 27.211386245988
RYDBERG_TO_EV = 13.605693122994
EV_TO_JOULES = 1.602176634e-19
EV_TO_JOULE = 1.602176634e-19
GPA_TO_PA = 1.0e9
ANGSTROM_TO_METERS = 1.0e-10

# Handshake Gatekeeper Tolerance Bounds
TOL_FORCE_RESIDUAL_EV_ANG = 1.0e-3
TOL_OOD_GMM_NLL_DEFAULT = 12.0
TOL_LOGNORMAL_RATE_VAR = 0.50
TOL_RVE_STRESS_CONVERGENCE = 0.015
TOL_COMPOUND_VARIANCE_BOUND = 0.35
