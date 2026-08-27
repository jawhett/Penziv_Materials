"""Transient Environmental Degradation, Stress-Assisted Diffusion & Oxidation Engine."""

from typing import Dict, Tuple, List, Optional
import numpy as np
from penziv_materials.core.constants import R_GAS


class EnvironmentalDegradationEngine:
    """Simulates dynamic oxide scale growth and stress-assisted interstitial embrittlement diffusion."""

    def __init__(
        self,
        partial_molar_volume_m3_mol: float = 2.0e-6,  # ~2 cm^3/mol for interstitial H/O in Ni
    ):
        self.v_bar = partial_molar_volume_m3_mol

    def compute_stress_assisted_interstitial_flux(
        self,
        concentration: float,
        concentration_gradient: float,
        hydrostatic_stress_gradient_pa_m: float,
        temperature_k: float,
        diffusivity_m2_s: float = 1.0e-12,
    ) -> float:
        """Evaluate coupled interstitial flux driven by Fickian chemical and hydrostatic stress gradients:

        J_interstitial = -D_eff * grad(C) + (D_eff * C * V_bar / (R * T)) * grad(sigma_h)
        """
        fickian_flux = -diffusivity_m2_s * concentration_gradient
        stress_drift_flux = (
            diffusivity_m2_s * concentration * self.v_bar / (R_GAS * temperature_k)
        ) * hydrostatic_stress_gradient_pa_m

        total_flux = fickian_flux + stress_drift_flux
        return float(total_flux)

    def evaluate_multi_element_oxidation_life(
        self,
        composition: Dict[str, float],
        temperature_k: float,
        exposure_time_hours: float = 1000.0,
    ) -> Dict[str, float]:
        """Compute oxide scale thickness and life limit before breakaway oxidation."""
        cr_fraction = composition.get("Cr", 0.0)
        al_fraction = composition.get("Al", 0.0)

        # Wagner parabolic rate constant k_p (m^2/s)
        if al_fraction >= 0.045:
            # Continuous Al2O3 scale formation
            q_ox = 260000.0  # J/mol
            k_p0 = 8.5e-7
        elif cr_fraction >= 0.12:
            # Cr2O3 scale formation
            q_ox = 210000.0
            k_p0 = 1.4e-6
        else:
            # Non-passivating NiO scale
            q_ox = 145000.0
            k_p0 = 3.2e-5

        kp = k_p0 * np.exp(-q_ox / (R_GAS * temperature_k))
        time_sec = exposure_time_hours * 3600.0
        oxide_thickness_m = np.sqrt(kp * time_sec)
        oxide_thickness_um = oxide_thickness_m * 1.0e6

        return {
            "parabolic_rate_kp": float(kp),
            "oxide_thickness_um": float(oxide_thickness_um),
            "is_protective_alumina": bool(al_fraction >= 0.045),
        }
