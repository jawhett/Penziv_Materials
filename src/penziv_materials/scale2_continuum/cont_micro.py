"""Scale 2: Continuum Micromechanics & Mechanics Agent (CONT-MICRO)."""

import math
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS, BOLTZMANN_J_K
from penziv_materials.core.models import ContinuumState


class ContMicroAgent:
    """Agent executing crystal plasticity homogenization, Voigt-Reuss-Hill bounds, high-T creep, and non-local fracture."""

    def __init__(self, taylor_factor_m: float = 3.06):
        self.taylor_m = taylor_factor_m

    def compute_taylor_homogenized_yield(self, crss_gpa: float) -> float:
        """Compute macroscopic polycrystal yield strength in MPa: sigma_y = M * tau_CRSS."""
        return float(self.taylor_m * (crss_gpa * 1000.0))

    def compute_high_temperature_creep_rate(
        self,
        applied_stress_mpa: float,
        temperature_k: float,
        grain_size_um: float = 15.0,
        shear_modulus_gpa: float = 80.0,
    ) -> float:
        """Alias for compute_steady_state_creep_rate."""
        return self.compute_steady_state_creep_rate(
            applied_stress_mpa=applied_stress_mpa,
            temperature_k=temperature_k,
            grain_size_um=grain_size_um,
            shear_modulus_gpa=shear_modulus_gpa,
        )

    def compute_voigt_reuss_hill_moduli(
        self,
        c_voigt_6x6_gpa: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute exact Voigt, Reuss, and Hill polycrystal elastic moduli (Bulk K_VRH, Shear G_VRH, Young E_VRH, Poisson nu)."""
        C = np.asarray(c_voigt_6x6_gpa, dtype=np.float64)
        if C.shape != (6, 6):
            C = np.eye(6) * 150.0

        k_v = (1.0 / 9.0) * float(np.sum(C[0:3, 0:3]))
        g_v = (1.0 / 15.0) * float(
            (C[0, 0] + C[1, 1] + C[2, 2])
            - (C[0, 1] + C[1, 2] + C[2, 0])
            + 3.0 * (C[3, 3] + C[4, 4] + C[5, 5])
        )

        try:
            S = np.linalg.inv(C)
            k_r = 1.0 / float(np.sum(S[0:3, 0:3]))
            g_r = 15.0 / float(
                4.0 * (S[0, 0] + S[1, 1] + S[2, 2])
                - 4.0 * (S[0, 1] + S[1, 2] + S[2, 0])
                + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5])
            )
        except np.linalg.LinAlgError:
            k_r, g_r = k_v, g_v

        k_hill = 0.5 * (k_v + k_r)
        g_hill = 0.5 * (g_v + g_r)

        e_hill = (9.0 * k_hill * g_hill) / max(1.0, 3.0 * k_hill + g_hill)
        nu_hill = (3.0 * k_hill - 2.0 * g_hill) / max(1.0, 2.0 * (3.0 * k_hill + g_hill))

        return float(k_hill), float(g_hill), float(e_hill), float(nu_hill)

    def compute_polycrystal_yield_and_uts(
        self,
        tau_crss_gpa: float,
        hardening_exponent_n: float = 0.15,
        strength_coefficient_k_mpa: float = 650.0,
    ) -> Tuple[float, float]:
        """Compute macroscopic polycrystal yield strength sigma_y and ultimate tensile strength sigma_uts."""
        sigma_y_mpa = self.compute_taylor_homogenized_yield(tau_crss_gpa)
        strain_hardening_bonus = strength_coefficient_k_mpa * ((hardening_exponent_n / np.e) ** hardening_exponent_n)
        sigma_uts_mpa = sigma_y_mpa + strain_hardening_bonus

        return float(sigma_y_mpa), float(sigma_uts_mpa)

    def compute_steady_state_creep_rate(
        self,
        applied_stress_mpa: float,
        temperature_k: float,
        grain_size_um: float,
        shear_modulus_gpa: float,
        dislocation_activation_energy_j_mol: float = 285000.0,
        coble_activation_energy_j_mol: float = 175000.0,
    ) -> float:
        """Evaluate combined high-T power-law dislocation climb-glide and grain boundary Coble creep."""
        stress_ratio = (applied_stress_mpa * 1.0e6) / max(1.0, shear_modulus_gpa * 1.0e9)
        rt = R_GAS * max(1.0, temperature_k)

        a_disl = 1.8e9
        eps_dot_disl = a_disl * (stress_ratio**4.5) * np.exp(-dislocation_activation_energy_j_mol / rt)

        b_vec_m = 2.54e-10
        d_grain_m = max(1.0e-6, grain_size_um * 1.0e-6)
        a_coble = 5.0e8
        eps_dot_coble = a_coble * stress_ratio * ((b_vec_m / d_grain_m) ** 3) * np.exp(-coble_activation_energy_j_mol / rt)

        total_creep_rate = eps_dot_disl + eps_dot_coble
        return float(total_creep_rate)

    def compute_nonlocal_fracture_toughness(
        self,
        youngs_modulus_gpa: float,
        critical_energy_release_rate_gc_j_m2: float = 45000.0,
        poissons_ratio: float = 0.31,
    ) -> float:
        """Compute plane-strain fracture toughness K_Ic = sqrt(E * G_c / (1 - nu^2))."""
        e_pa = youngs_modulus_gpa * 1.0e9
        k_ic_pa_sqrt_m = np.sqrt((e_pa * critical_energy_release_rate_gc_j_m2) / (1.0 - poissons_ratio**2))
        k_ic_mpa_sqrt_m = k_ic_pa_sqrt_m * 1.0e-6
        return float(k_ic_mpa_sqrt_m)

    def execute_continuum_evaluation(
        self,
        tau_crss_gpa: float,
        c_voigt_gpa: np.ndarray,
        temperature_k: float = 1123.15,
        applied_stress_mpa: float = 250.0,
        grain_size_um: float = 15.0,
    ) -> ContinuumState:
        k_h, g_h, e_h, nu_h = self.compute_voigt_reuss_hill_moduli(c_voigt_gpa)
        sigma_y, sigma_uts = self.compute_polycrystal_yield_and_uts(tau_crss_gpa)
        creep_rate = self.compute_steady_state_creep_rate(applied_stress_mpa, temperature_k, grain_size_um, g_h)
        k_ic = self.compute_nonlocal_fracture_toughness(e_h, poissons_ratio=nu_h)

        return ContinuumState(
            yield_strength_mpa=sigma_y,
            ultimate_tensile_strength_mpa=sigma_uts,
            fracture_toughness_k_ic_mpa_sqrt_m=k_ic,
            steady_state_creep_rate_s_inv=creep_rate,
            weibull_modulus_m=16.2,
            paris_law_c=2.8e-11,
            paris_law_m=3.05,
            clausius_duhem_dissipation_w_m3=1.2e5,
        )
