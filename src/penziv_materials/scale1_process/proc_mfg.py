"""Scale 1: Process Dynamics & Synthesizability Agent (PROC-MFG)."""

import math
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS
from penziv_materials.core.models import ProcessState


class ProcMfgAgent:
    """Agent executing moving laser meltpool thermodynamics, thermal cracking susceptibility, and exergy work limits."""

    def __init__(self, laser_power_w: float = 285.0, scan_speed_m_s: float = 1.0):
        self.power = laser_power_w
        self.v_scan = scan_speed_m_s

    def compute_minimum_ore_extraction_exergy(self, composition: Dict[str, float]) -> float:
        """Alias for compute_mineral_ore_reduction_exergy."""
        return self.compute_mineral_ore_reduction_exergy(composition)

    def compute_rosenthal_solidification_kinetics(
        self,
        thermal_conductivity_w_m_k: float = 26.0,
        volumetric_heat_capacity_j_m3_k: float = 3.6e6,
        liquidus_temperature_k: float = 1620.0,
        preheat_temperature_k: float = 353.15,
    ) -> Tuple[float, float, float]:
        """Compute moving heat source thermal gradient G, cooling rate T_dot, and solidification velocity V_s."""
        p_absorbed = self.power * 0.42
        delta_t = liquidus_temperature_k - preheat_temperature_k

        t_dot_k_s = (2.0 * np.pi * thermal_conductivity_w_m_k * (delta_t**2) * self.v_scan) / max(1.0, p_absorbed)
        v_s_m_s = self.v_scan * 0.707
        g_k_m = t_dot_k_s / max(1e-4, v_s_m_s)

        return float(t_dot_k_s), float(g_k_m), float(v_s_m_s)

    def evaluate_thermal_stress_cracking_susceptibility(
        self,
        youngs_modulus_gpa: float,
        thermal_expansion_coeff: float,
        yield_strength_mpa: float,
        delta_t_k: float = 1200.0,
        poissons_ratio: float = 0.31,
    ) -> Dict[str, float]:
        """Compute thermal shock / solidification cracking susceptibility index:

        chi_crack = (E * alpha * Delta T) / [ sigma_y * (1 - nu) ]
        """
        thermal_stress_mpa = (youngs_modulus_gpa * 1000.0 * thermal_expansion_coeff * delta_t_k) / (1.0 - poissons_ratio)
        chi_crack = thermal_stress_mpa / max(1.0, yield_strength_mpa)
        synthesizability_score = 1.0 / (1.0 + np.exp(chi_crack - 2.0))

        return {
            "thermal_mismatch_stress_mpa": float(thermal_stress_mpa),
            "cracking_susceptibility_index": float(chi_crack),
            "synthesizability_score": float(np.clip(synthesizability_score, 0.10, 0.98)),
        }

    def compute_mineral_ore_reduction_exergy(
        self,
        composition: Dict[str, float],
        temperature_k: float = 298.15,
    ) -> float:
        """Compute minimum thermodynamic work required to extract and reduce metals from crustal oxide ores."""
        element_reduction_exergy_mj_kg = {
            "Fe": 7.5, "Al": 32.0, "Cu": 12.0, "Ni": 28.0, "Cr": 35.0, "Ti": 65.0,
            "Mo": 45.0, "W": 55.0, "Nb": 85.0, "Sc": 320.0, "Zr": 78.0, "Mg": 42.0,
            "Na": 24.0, "Li": 68.0, "Si": 22.0, "P": 18.0, "S": 2.5, "B": 45.0,
        }

        total_exergy_mj_kg = sum(w * element_reduction_exergy_mj_kg.get(elem, 40.0) for elem, w in composition.items())
        return float(total_exergy_mj_kg)

    def execute_process_evaluation(
        self,
        composition: Dict[str, float],
        youngs_modulus_gpa: float = 210.0,
        yield_strength_mpa: float = 950.0,
        thermal_expansion_coeff: float = 1.45e-5,
    ) -> ProcessState:
        t_dot, g_k_m, v_s = self.compute_rosenthal_solidification_kinetics()
        crack_res = self.evaluate_thermal_stress_cracking_susceptibility(
            youngs_modulus_gpa=youngs_modulus_gpa,
            thermal_expansion_coeff=thermal_expansion_coeff,
            yield_strength_mpa=yield_strength_mpa,
        )
        exergy_work = self.compute_mineral_ore_reduction_exergy(composition)

        return ProcessState(
            solidification_cooling_rate_k_s=t_dot,
            thermal_gradient_k_m=g_k_m,
            solidification_velocity_m_s=v_s,
            residual_stress_max_mpa=crack_res["thermal_mismatch_stress_mpa"] * 0.35,
            oxide_growth_parabolic_rate_kp=1.8e-14,
            min_ore_extraction_exergy_mj_kg=exergy_work,
            synthesizability_score=crack_res["synthesizability_score"],
        )
