"""Wagner High-Temperature Parabolic Oxidation & Environmental Fatigue Degradation Engine."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS


class WagnerOxidationEngine:
    """Evaluates parabolic oxide scale growth, subsurface depletion profiles, and environmental notch fatigue reduction."""

    def __init__(self, temperature_k: float = 1000.0):
        self.T = max(300.0, float(temperature_k))

    def compute_parabolic_oxidation_kinetics(
        self,
        exposure_time_hours: float = 100.0,
        pre_exponential_kp0_m2_s: float = 1.2e-4,
        activation_energy_q_j_mol: float = 240000.0,
        solute_diffusivity_d0_m2_s: float = 1.0e-5,
        solute_activation_energy_q_j_mol: float = 260000.0,
        base_fatigue_endurance_mpa: float = 400.0,
        surface_roughness_root_radius_um: float = 2.0,
    ) -> Dict[str, Any]:
        """Evaluate Wagner parabolic oxidation scale thickness and resultant environmental fatigue reduction:

        x_ox^2 = 2 * k_p(T) * t
        """
        t_seconds = max(1.0, exposure_time_hours * 3600.0)
        r_t = R_GAS * self.T

        # Parabolic rate constant k_p in m^2 / s
        k_p = pre_exponential_kp0_m2_s * np.exp(-activation_energy_q_j_mol / r_t)

        # Oxide scale thickness x_ox in meters and micrometers
        x_ox_m = np.sqrt(2.0 * k_p * t_seconds)
        x_ox_um = float(x_ox_m * 1.0e6)

        # Subsurface solute depletion depth (e.g. Cr2O3 or Al2O3 protective layer formation)
        d_solute = solute_diffusivity_d0_m2_s * np.exp(-solute_activation_energy_q_j_mol / r_t)
        depletion_depth_m = 2.0 * np.sqrt(d_solute * t_seconds)
        depletion_depth_um = float(depletion_depth_m * 1.0e6)

        # Surface oxide notch stress concentration factor (Inglis-Neuber model)
        rho_root_m = max(0.1, surface_roughness_root_radius_um) * 1.0e-6
        k_t_notch = float(1.0 + 2.0 * np.sqrt(x_ox_m / rho_root_m))

        # Degraded surface fatigue endurance limit
        sigma_e_degraded = float(max(50.0, base_fatigue_endurance_mpa / k_t_notch))

        return {
            "temperature_k": self.T,
            "exposure_time_hours": float(exposure_time_hours),
            "parabolic_rate_constant_kp_m2_s": float(k_p),
            "oxide_scale_thickness_um": x_ox_um,
            "subsurface_solute_depletion_depth_um": depletion_depth_um,
            "oxide_notch_stress_concentration_factor_kt": k_t_notch,
            "base_fatigue_endurance_mpa": float(base_fatigue_endurance_mpa),
            "environmentally_degraded_fatigue_endurance_mpa": sigma_e_degraded,
            "fatigue_retention_percent": float((sigma_e_degraded / max(1.0, base_fatigue_endurance_mpa)) * 100.0),
        }
