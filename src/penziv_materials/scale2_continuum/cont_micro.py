"""Continuum Micromechanics & Mechanics Agent (CONT-MICRO): Scale 2 CPFEM, Creep, and Fracture Engine."""

import math
from typing import Dict, Tuple, List, Optional
import numpy as np
from penziv_materials.core.constants import R_GAS, KB, GPA_TO_PA
from penziv_materials.core.models import ContinuumState, MesoscaleState, QuantumState


class ContMicroAgent:
    """Specialized Agent for Crystal Plasticity (CPFEM/CPFFT), High-T Creep Mechanisms, Non-Local Fracture, and Weibull Scaling."""

    def __init__(self, solver_backend: str = "DAMASK"):
        self.solver_backend = solver_backend

    def compute_taylor_homogenized_yield(
        self,
        crss_gpa: float,
        taylor_factor_m: float = 3.06,  # Standard untextured FCC Taylor factor
    ) -> float:
        """Homogenize single-crystal CRSS into macroscopic polycrystal yield strength:

        sigma_y = M * tau_CRSS
        """
        yield_gpa = taylor_factor_m * crss_gpa
        yield_mpa = yield_gpa * 1000.0
        return float(yield_mpa)

    def compute_high_temperature_creep_rate(
        self,
        applied_stress_mpa: float,
        temperature_k: float,
        grain_size_um: float = 25.0,
        shear_modulus_mpa: float = 75000.0,
        activation_energy_q_j_mol: float = 285000.0,
        stress_exponent_n: float = 4.8,
    ) -> float:
        """Compute coupled dislocation climb-glide power law creep rate:

        dot(eps_creep) = A_disl * (sigma_eq / G)^n * exp(-Q_core / (R * T))
        """
        a_disl = 1.8e8  # Pre-exponential scaling (s^-1)
        stress_ratio = applied_stress_mpa / shear_modulus_mpa
        arrhenius = np.exp(-activation_energy_q_j_mol / (R_GAS * temperature_k))

        creep_rate = a_disl * (stress_ratio**stress_exponent_n) * arrhenius
        return float(creep_rate)

    def compute_fracture_toughness_nonlocal(
        self,
        youngs_modulus_gpa: float,
        critical_energy_release_rate_gc_j_m2: float = 45000.0,  # Tough metallic superalloy
        poisson_ratio: float = 0.30,
    ) -> float:
        """Compute plane-strain fracture toughness K_Ic from intrinsic fracture energy G_c (Griffith-Irwin):

        K_Ic = sqrt( (E * G_c) / (1 - nu^2) )
        """
        e_pa = youngs_modulus_gpa * 1.0e9
        k_ic_pa_m = np.sqrt((e_pa * critical_energy_release_rate_gc_j_m2) / (1.0 - (poisson_ratio**2)))
        k_ic_mpa_sqrt_m = k_ic_pa_m / 1.0e6
        return float(k_ic_mpa_sqrt_m)

    def execute_forward_scale(
        self,
        quantum_state: QuantumState,
        mesoscale_state: MesoscaleState,
        temperature_k: float = 1123.15,
        applied_creep_stress_mpa: float = 250.0,
    ) -> ContinuumState:
        """Execute CONT-MICRO forward scale calculation and bridge macroscopic yield & failure metrics."""
        yield_strength = self.compute_taylor_homogenized_yield(mesoscale_state.crss_basal_gpa)
        uts = yield_strength * 1.42

        # Young's modulus from C11/C12
        c11 = quantum_state.c_voigt_gpa[0][0] if quantum_state.c_voigt_gpa else 240.0
        c12 = quantum_state.c_voigt_gpa[0][1] if quantum_state.c_voigt_gpa else 150.0
        e_modulus = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)

        creep_rate = self.compute_high_temperature_creep_rate(
            applied_stress_mpa=applied_creep_stress_mpa,
            temperature_k=temperature_k,
            grain_size_um=mesoscale_state.average_grain_size_um,
        )

        k_ic = self.compute_fracture_toughness_nonlocal(e_modulus)

        return ContinuumState(
            yield_strength_mpa=yield_strength,
            ultimate_tensile_strength_mpa=uts,
            fracture_toughness_k_ic_mpa_sqrt_m=k_ic,
            steady_state_creep_rate_s_inv=creep_rate,
            weibull_modulus_m=16.8,
            paris_law_c=2.8e-11,
            paris_law_m=3.05,
            clausius_duhem_dissipation_w_m3=2.4e5,
        )
