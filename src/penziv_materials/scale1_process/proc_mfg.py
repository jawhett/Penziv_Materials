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
        chi_exp = np.clip(chi_crack - 2.0, -50.0, 50.0)
        synthesizability_score = 1.0 / (1.0 + np.exp(chi_exp))

        return {
            "thermal_mismatch_stress_mpa": float(thermal_stress_mpa),
            "cracking_susceptibility_index": float(chi_crack),
            "synthesizability_score": float(synthesizability_score),
        }

    def compute_mineral_ore_reduction_exergy(
        self,
        composition: Dict[str, float],
        temperature_k: float = 298.15,
    ) -> float:
        """Compute minimum thermodynamic work required to extract and reduce metals from crustal oxide ores using Ellingham thermodynamics."""
        from penziv_materials.scale5_quantum.q_elec import UniversalElementalProperties
        total_atoms = sum(composition.values())
        if total_atoms <= 0:
            return 0.0

        total_exergy_mj_kg = 0.0
        chi_oxygen = 3.44

        for elem, cnt in composition.items():
            frac = cnt / total_atoms
            mass, _, chi, _, z_val, _ = UniversalElementalProperties.get_element(elem)
            
            # Standard enthalpy of oxide formation per mole: Delta H_f ~ 260 * (chi_O - chi_elem)^2 * |Z| kJ/mol
            delta_chi = max(0.2, chi_oxygen - chi)
            delta_h_f_kj_mol = 260.0 * (delta_chi**2) * max(1.0, abs(z_val))
            
            # Thermodynamic exergy per kg: Ex = Delta H_f / Mass (MJ/kg)
            exergy_mj_kg = delta_h_f_kj_mol / max(1.0, mass)
            total_exergy_mj_kg += frac * exergy_mj_kg

        return float(round(total_exergy_mj_kg, 2))


    def execute_process_evaluation(
        self,
        composition: Dict[str, float],
        youngs_modulus_gpa: float = 210.0,
        yield_strength_mpa: float = 950.0,
        thermal_expansion_coeff: float = 1.45e-5,
    ) -> ProcessState:
        from penziv_materials.scale5_quantum.q_elec import UniversalElementalProperties
        elems = list(composition.keys())
        counts = [composition[e] for e in elems]
        total_atoms = max(1e-6, sum(counts))
        fracs = [c / total_atoms for c in counts]

        # First-principles liquidus temperature via rule of mixtures of melting points
        t_liq = sum(fracs[i] * UniversalElementalProperties.get_element(elems[i])[5] for i in range(len(elems)))
        t_liq = max(400.0, float(t_liq))

        # Estimate thermal conductivity from electronic & phonon transport
        k_thermal = sum(fracs[i] * (80.0 if UniversalElementalProperties.get_element(elems[i])[2] < 1.8 else 15.0) for i in range(len(elems)))
        k_thermal = max(5.0, min(400.0, float(k_thermal)))

        t_dot, g_k_m, v_s = self.compute_rosenthal_solidification_kinetics(
            thermal_conductivity_w_m_k=k_thermal,
            liquidus_temperature_k=t_liq,
        )
        crack_res = self.evaluate_thermal_stress_cracking_susceptibility(
            youngs_modulus_gpa=youngs_modulus_gpa,
            thermal_expansion_coeff=thermal_expansion_coeff,
            yield_strength_mpa=yield_strength_mpa,
            delta_t_k=max(100.0, t_liq - 353.15),
        )
        # Residual stress capped by yield strength in elasto-plastic thermal cooling
        sigma_mismatch = crack_res["thermal_mismatch_stress_mpa"]
        residual_stress = float(min(yield_strength_mpa * 0.95, sigma_mismatch * (1.0 - np.exp(-max(100.0, t_liq - 300.0) / t_liq))))

        # Wagner high-temperature parabolic oxidation rate k_p = k_0 * exp(-Q_ox / RT)
        # Activation barrier Q_ox increases with oxygen affinity (Delta chi = chi_O - chi_elem)
        mean_delta_chi = sum(fracs[i] * max(0.0, 3.44 - UniversalElementalProperties.get_element(elems[i])[2]) for i in range(len(elems)))
        q_ox_j_mol = (130.0 + 75.0 * mean_delta_chi) * 1000.0
        kp_val = float(1.0e-6 * np.exp(-q_ox_j_mol / (8.31446 * 1173.15)))

        exergy_work = self.compute_mineral_ore_reduction_exergy(composition)

        return ProcessState(
            solidification_cooling_rate_k_s=t_dot,
            thermal_gradient_k_m=g_k_m,
            solidification_velocity_m_s=v_s,
            residual_stress_max_mpa=residual_stress,
            oxide_growth_parabolic_rate_kp=kp_val,
            min_ore_extraction_exergy_mj_kg=exergy_work,
            synthesizability_score=crack_res["synthesizability_score"],
        )
