"""Poro-Elastic Fluid-Structure Interaction (FSI) & Knudsen Microchannel Gas Dynamics."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS, BOLTZMANN_J_K


class PoroMechanicsFSIEngine:
    """Solves coupled Darcy-Stokes poro-elastic stress, internal gas channel pressurization, and Knudsen permeation."""

    def __init__(self, gas_molar_mass_g_mol: float = 4.0, gas_pressure_mpa: float = 2.5):
        self.m_gas = gas_molar_mass_g_mol * 1.0e-3  # kg/mol (e.g. He = 4 g/mol, Ar = 40 g/mol)
        self.p_gas_pa = gas_pressure_mpa * 1.0e6

    def compute_knudsen_diffusion_coefficient(
        self,
        pore_diameter_nm: float = 50.0,
        temperature_k: float = 300.0,
    ) -> Dict[str, float]:
        """Compute Knudsen number Kn and Knudsen gas diffusivity D_Kn:

        Kn = lambda_mean_free_path / d_pore
        D_Kn = (d_pore / 3) * sqrt(8 * R * T / (pi * M))
        """
        d_m = pore_diameter_nm * 1.0e-9
        # Mean free path lambda = (k_B * T) / (sqrt(2) * pi * d_mol^2 * P)
        d_mol = 0.26e-9  # m
        lambda_mfp = (BOLTZMANN_J_K * temperature_k) / (np.sqrt(2.0) * np.pi * (d_mol**2) * max(1e3, self.p_gas_pa))

        knudsen_number = lambda_mfp / d_m
        d_kn_m2_s = (d_m / 3.0) * np.sqrt((8.0 * R_GAS * temperature_k) / (np.pi * self.m_gas))

        return {
            "knudsen_number": float(knudsen_number),
            "knudsen_diffusivity_m2_s": float(d_kn_m2_s),
            "is_non_continuum_regime": bool(knudsen_number > 0.1),
        }

    def evaluate_channel_wall_hydrostatic_support(
        self,
        applied_external_compressive_stress_mpa: float,
        matrix_shear_modulus_gpa: float = 25.0,
        internal_gas_pressure_mpa: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate how internal pressurized gas pockets mechanically stabilize microchannel walls against collapse:

        Effective stress sigma_eff = sigma_ext - alpha_Biot * P_fluid
        """
        p_fluid = internal_gas_pressure_mpa if internal_gas_pressure_mpa is not None else (self.p_gas_pa * 1.0e-6)
        alpha_biot = 0.75  # Biot effective stress coefficient

        effective_stress_mpa = applied_external_compressive_stress_mpa - (alpha_biot * p_fluid)

        # Critical collapse stress sigma_crit ~ G * (t/R)^2
        sigma_crit_collapse_mpa = (matrix_shear_modulus_gpa * 1000.0) * 0.04
        is_stabilized = effective_stress_mpa < sigma_crit_collapse_mpa

        return {
            "effective_wall_stress_mpa": float(effective_stress_mpa),
            "internal_fluid_counter_pressure_mpa": float(p_fluid),
            "is_mechanically_stabilized": bool(is_stabilized),
            "delamination_safety_factor": float(sigma_crit_collapse_mpa / max(1.0, effective_stress_mpa)),
        }
