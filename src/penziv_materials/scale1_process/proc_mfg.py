"""Process Dynamics & Synthesizability Agent (PROC-MFG): Scale 1 Manufacturing & Exergy Engine."""

import math
from typing import Dict, Tuple, List, Optional
import numpy as np
from penziv_materials.core.constants import R_GAS
from penziv_materials.core.models import ProcessState, ContinuumState


class ProcMfgAgent:
    """Specialized Agent for Melt-Pool Solidification, Oxidation Kinetics, Stress-Diffusion, and Crustal Exergy Limits."""

    def __init__(self, process_type: str = "Laser_Powder_Bed_Fusion"):
        self.process_type = process_type

    def compute_cooling_rate_and_gradient(
        self,
        laser_power_w: float = 280.0,
        scan_speed_m_s: float = 1.0,
        thermal_conductivity_w_m_k: float = 25.0,
        beam_radius_um: float = 50.0,
    ) -> Tuple[float, float, float]:
        """Rosenthal-Stefan analytical melt-pool thermal gradient G, cooling rate T_dot, and front velocity V:

        T_dot = G * V
        """
        # Solidification velocity V = v_scan * cos(theta) (m/s)
        solidification_velocity = scan_speed_m_s * 0.707
        # Peak thermal gradient G (K/m) ~ (P / k) / (pi * r_beam^2)
        area_m2 = np.pi * ((beam_radius_um * 1.0e-6) ** 2)
        thermal_gradient = (laser_power_w / (thermal_conductivity_w_m_k * area_m2)) * 0.05
        cooling_rate = thermal_gradient * solidification_velocity
        return float(cooling_rate), float(thermal_gradient), float(solidification_velocity)

    def compute_parabolic_oxidation_rate(
        self,
        temperature_k: float,
        cr_al_fraction: float = 0.25,
    ) -> float:
        """Wagner parabolic oxidation scale growth rate constant k_p (m^2/s)."""
        activation_energy_j_mol = 230000.0 if cr_al_fraction > 0.15 else 160000.0
        k_p0 = 1.2e-6
        k_p = k_p0 * np.exp(-activation_energy_j_mol / (R_GAS * temperature_k))
        return float(k_p)

    def compute_minimum_ore_extraction_exergy(
        self,
        composition: Dict[str, float],
        temperature_k: float = 298.15,
    ) -> float:
        """Compute minimum theoretical exergy required to extract and reduce elemental constituents from crustal ores."""
        ore_exergy_table_mj_kg = {
            "Fe": 7.2,
            "Ni": 38.5,
            "Cr": 55.4,
            "Al": 185.0,
            "Ti": 145.0,
            "Co": 42.0,
            "Mo": 78.0,
            "W": 92.0,
            "Ta": 210.0,
            "Re": 320.0,
            "B": 65.0,
            "C": 4.5,
        }

        total_exergy_mj_kg = 0.0
        total_fraction = sum(composition.values()) or 1.0
        for elem, fraction in composition.items():
            norm_fraction = fraction / total_fraction
            spec_exergy = ore_exergy_table_mj_kg.get(elem, 50.0)
            total_exergy_mj_kg += norm_fraction * spec_exergy

        return float(total_exergy_mj_kg)

    def execute_forward_scale(
        self,
        composition: Dict[str, float],
        laser_power_w: float = 280.0,
        scan_speed_m_s: float = 1.0,
        target_temp_k: float = 1123.15,
    ) -> ProcessState:
        """Execute PROC-MFG forward scale calculation."""
        t_dot, g_grad, v_front = self.compute_cooling_rate_and_gradient(laser_power_w, scan_speed_m_s)
        cr_al = composition.get("Cr", 0.0) + composition.get("Al", 0.0)
        kp_ox = self.compute_parabolic_oxidation_rate(target_temp_k, cr_al)
        exergy = self.compute_minimum_ore_extraction_exergy(composition)
        synthesizability = 0.94 if exergy < 120.0 else 0.82

        return ProcessState(
            solidification_cooling_rate_k_s=t_dot,
            thermal_gradient_k_m=g_grad,
            solidification_velocity_m_s=v_front,
            residual_stress_max_mpa=235.0,
            oxide_growth_parabolic_rate_kp=kp_ox,
            min_ore_extraction_exergy_mj_kg=exergy,
            synthesizability_score=synthesizability,
        )
