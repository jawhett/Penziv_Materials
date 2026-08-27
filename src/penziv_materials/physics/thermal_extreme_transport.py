"""Lattice Thermal Conductivity (BTE / Slack), Wiedemann-Franz Electronic Transport & Radiation Resistance."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, HBAR, AVOGADRO, E_CHARGE


class ThermalExtremeTransportEngine:
    """Evaluates anisotropic phonon/electron thermal conductivity, radiation threshold displacement, and extreme environment metrics."""

    LORENZ_NUMBER_W_OHM_K2 = 2.44e-8  # Standard Sommerfeld Lorenz number L_0

    def __init__(self, temperature_k: float = 300.0):
        self.T = temperature_k

    def compute_lattice_thermal_conductivity_slack(
        self,
        average_atomic_mass_amu: float,
        debye_temperature_k: float,
        volume_per_atom_ang3: float,
        gruneisen_parameter_gamma: float = 1.65,
        num_atoms_per_primitive_cell: int = 2,
    ) -> Dict[str, float]:
        """Evaluate intrinsic acoustic phonon-limited lattice thermal conductivity kappa_L(T) via Slack's BTE relation:

        kappa_L = (3.04e-6 * M_bar_g_mol * Theta_D^3 * delta_angstrom) / (gamma^2 * n^(2/3) * T)
        """
        m_bar = max(1.0, average_atomic_mass_amu)
        delta_ang = max(1.0, (volume_per_atom_ang3) ** (1.0 / 3.0))

        theta_d = max(50.0, debye_temperature_k)
        gamma = max(0.5, gruneisen_parameter_gamma)
        n_atoms = max(1, num_atoms_per_primitive_cell)

        numerator = 3.04e-6 * m_bar * (theta_d**3) * delta_ang
        denominator = (gamma**2) * (n_atoms ** (2.0 / 3.0)) * max(1.0, self.T)

        kappa_l_w_m_k = numerator / max(1e-12, denominator)
        kappa_clamped = float(np.clip(kappa_l_w_m_k, 0.1, 2500.0))

        return {
            "lattice_thermal_conductivity_w_m_k": kappa_clamped,
            "debye_temperature_k": float(theta_d),
            "gruneisen_parameter_gamma": float(gamma),
            "phonon_mean_free_path_nm": float((3.0 * kappa_clamped / (2.5e6 * 4500.0)) * 1.0e9),
        }

    def compute_electronic_thermal_conductivity(
        self,
        electrical_conductivity_s_m: float,
    ) -> float:
        """Compute electronic thermal conductivity kappa_e via Wiedemann-Franz Law:

        kappa_e = L_0 * sigma * T
        """
        kappa_e = self.LORENZ_NUMBER_W_OHM_K2 * electrical_conductivity_s_m * self.T
        return float(kappa_e)

    def compute_thermal_shock_resistance_parameter(
        self,
        fracture_strength_mpa: float,
        youngs_modulus_gpa: float,
        poisson_ratio: float,
        thermal_conductivity_w_m_k: float,
        thermal_expansion_coeff_1_k: float,
    ) -> Dict[str, float]:
        """Compute Kingery thermal shock resistance parameters R and R':

        R = (sigma_f * (1 - nu)) / (E * alpha)   [K]
        R' = R * kappa                          [W/m]
        """
        sigma_f_pa = fracture_strength_mpa * 1.0e6
        e_pa = youngs_modulus_gpa * 1.0e9
        alpha = max(1e-7, thermal_expansion_coeff_1_k)
        nu = poisson_ratio

        r_crit_k = (sigma_f_pa * (1.0 - nu)) / (e_pa * alpha)
        r_prime_w_m = r_crit_k * thermal_conductivity_w_m_k

        return {
            "thermal_shock_critical_delta_t_k": float(r_crit_k),
            "steady_state_heat_flux_resistance_w_m": float(r_prime_w_m),
            "is_thermal_shock_resilient": bool(r_crit_k >= 250.0),
        }

    def compute_radiation_displacement_threshold(
        self,
        cohesive_energy_ev_atom: float,
        shear_modulus_gpa: float,
    ) -> Dict[str, float]:
        """Evaluate radiation threshold displacement energy E_d (Kinchin-Pease radiation damage threshold):

        E_d = 2.5 * E_coh + 0.12 * G
        """
        e_d_ev = 2.5 * abs(cohesive_energy_ev_atom) + 0.12 * shear_modulus_gpa
        e_d_clamped = float(np.clip(e_d_ev, 15.0, 95.0))

        return {
            "threshold_displacement_energy_ed_ev": e_d_clamped,
            "frenkel_pair_formation_threshold_ev": float(e_d_clamped * 1.8),
            "is_radiation_hardened": bool(e_d_clamped >= 40.0),
        }
