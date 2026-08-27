"""Lattice Thermal Conductivity (BTE + Allen-Feldman Diffusons), Space Vacuum Outgassing & Radiation."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, HBAR, AVOGADRO, E_CHARGE, R_GAS


class ThermalExtremeTransportEngine:
    """Evaluates unified Propagon-Diffuson thermal transport, space vacuum outgassing (HKL), and 3D radiation threshold displacement surfaces."""

    LORENZ_NUMBER_W_OHM_K2 = 2.44e-8  # Sommerfeld Lorenz number L_0

    def __init__(self, temperature_k: float = 300.0):
        self.T = temperature_k

    def compute_unified_propagon_diffuson_thermal_conductivity(
        self,
        average_atomic_mass_amu: float,
        debye_temperature_k: float,
        volume_per_atom_ang3: float,
        fraction_amorphous_or_cage: float = 0.0,
        gruneisen_parameter_gamma: float = 1.65,
        num_atoms_per_primitive_cell: int = 2,
    ) -> Dict[str, float]:
        """Unified Propagon (BTE) + Diffuson (Allen-Feldman) thermal transport:

        kappa_total = (1 - f_disorder) * kappa_propagons + f_disorder * kappa_diffusons
        """
        # 1. Propagon contribution via Slack BTE
        m_bar = max(1.0, average_atomic_mass_amu)
        delta_ang = max(1.0, (volume_per_atom_ang3) ** (1.0 / 3.0))
        theta_d = max(50.0, debye_temperature_k)
        gamma = max(0.5, gruneisen_parameter_gamma)
        n_atoms = max(1, num_atoms_per_primitive_cell)

        num_slack = 3.04e-6 * m_bar * (theta_d**3) * delta_ang
        den_slack = (gamma**2) * (n_atoms ** (2.0 / 3.0)) * max(1.0, self.T)
        kappa_propagon = float(np.clip(num_slack / max(1e-12, den_slack), 0.1, 2500.0))

        # 2. Allen-Feldman diffuson-locon minimum thermal conductivity (Cahill-Pohl limit)
        n_density = 1.0 / (volume_per_atom_ang3 * 1.0e-30)  # atoms / m^3
        v_sound = (theta_d * BOLTZMANN_J_K / HBAR) * (6.0 * np.pi**2 * n_density) ** (-1.0 / 3.0)
        kappa_diffuson = float(0.5 * (np.pi / 6.0) ** (1.0 / 3.0) * BOLTZMANN_J_K * (n_density ** (2.0 / 3.0)) * v_sound)
        kappa_diffuson = float(np.clip(kappa_diffuson, 0.4, 3.5))

        f_dis = np.clip(fraction_amorphous_or_cage, 0.0, 1.0)
        kappa_total = (1.0 - f_dis) * kappa_propagon + f_dis * kappa_diffuson

        return {
            "total_thermal_conductivity_w_m_k": float(kappa_total),
            "propagon_bte_conductivity_w_m_k": kappa_propagon,
            "diffuson_af_conductivity_w_m_k": kappa_diffuson,
            "fraction_diffuson_contribution": float(f_dis),
        }

    def compute_lattice_thermal_conductivity_slack(
        self,
        average_atomic_mass_amu: float,
        debye_temperature_k: float,
        volume_per_atom_ang3: float,
        gruneisen_parameter_gamma: float = 1.65,
        num_atoms_per_primitive_cell: int = 2,
    ) -> Dict[str, float]:
        """Slack BTE lattice thermal conductivity."""
        res = self.compute_unified_propagon_diffuson_thermal_conductivity(
            average_atomic_mass_amu=average_atomic_mass_amu,
            debye_temperature_k=debye_temperature_k,
            volume_per_atom_ang3=volume_per_atom_ang3,
            fraction_amorphous_or_cage=0.0,
            gruneisen_parameter_gamma=gruneisen_parameter_gamma,
            num_atoms_per_primitive_cell=num_atoms_per_primitive_cell,
        )
        return {
            "lattice_thermal_conductivity_w_m_k": res["propagon_bte_conductivity_w_m_k"],
            "debye_temperature_k": float(debye_temperature_k),
            "gruneisen_parameter_gamma": float(gruneisen_parameter_gamma),
            "phonon_mean_free_path_nm": float((3.0 * res["propagon_bte_conductivity_w_m_k"] / (2.5e6 * 4500.0)) * 1.0e9),
        }

    def compute_space_vacuum_outgassing_rate_hkl(
        self,
        molecular_weight_g_mol: float,
        vapor_pressure_pa: float,
        ambient_pressure_pa: float = 1.0e-7,
        condensation_coeff: float = 1.0,
    ) -> Dict[str, float]:
        """Hertz-Knudsen-Langmuir (HKL) sublimative vacuum mass-loss rate:

        J_evap = (alpha_v * (P_sat - P_amb)) / sqrt(2 * pi * M * R * T)  [kg / (m^2 * s)]
        """
        m_kg_mol = molecular_weight_g_mol * 1.0e-3
        delta_p = max(0.0, vapor_pressure_pa - ambient_pressure_pa)

        denominator = np.sqrt(2.0 * np.pi * m_kg_mol * R_GAS * max(1.0, self.T))
        j_evap_kg_m2_s = (condensation_coeff * delta_p) / max(1e-12, denominator)

        # Annualized mass loss (g / cm^2 / year)
        mass_loss_annual = j_evap_kg_m2_s * 3.1536e7 * 10.0  # kg/m2/s -> g/cm2/year

        return {
            "sublimation_mass_flux_kg_m2_s": float(j_evap_kg_m2_s),
            "annual_vacuum_mass_loss_g_cm2_yr": float(mass_loss_annual),
            "is_space_vacuum_stable": bool(mass_loss_annual <= 0.01),
        }

    def compute_electronic_thermal_conductivity(
        self,
        electrical_conductivity_s_m: float,
    ) -> float:
        """Compute electronic thermal conductivity kappa_e via Wiedemann-Franz Law."""
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
        """Compute Kingery thermal shock resistance parameters R and R'."""
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
        """Evaluate radiation threshold displacement energy E_d."""
        e_d_ev = 2.5 * abs(cohesive_energy_ev_atom) + 0.12 * shear_modulus_gpa
        e_d_clamped = float(np.clip(e_d_ev, 15.0, 95.0))

        return {
            "threshold_displacement_energy_ed_ev": e_d_clamped,
            "frenkel_pair_formation_threshold_ev": float(e_d_clamped * 1.8),
            "is_radiation_hardened": bool(e_d_clamped >= 40.0),
        }
