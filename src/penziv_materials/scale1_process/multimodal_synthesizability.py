"""Multi-Modal Synthesizability Engine (CVD/PVD, SPS Sintering, Sol-Gel, Melt Spinning)."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS, BOLTZMANN_J_K


class MultiModalSynthesizabilityEngine:
    """Evaluates process feasibility, kinetics, and densification across diverse manufacturing modalities."""

    def __init__(self, target_temperature_k: float = 1273.15):
        self.T = target_temperature_k

    def evaluate_chemical_vapor_deposition(
        self,
        precursor_flow_rate_sccm: float = 50.0,
        chamber_pressure_torr: float = 10.0,
        substrate_temperature_c: float = 850.0,
        activation_energy_kj_mol: float = 100.0,
    ) -> Dict[str, Any]:
        """CVD surface reaction-limited vs mass transport-limited growth rate:

        R_growth = k_0 * exp(-E_a / RT) * P_precursor
        """
        t_sub_k = substrate_temperature_c + 273.15
        p_atm = chamber_pressure_torr / 760.0
        rt = R_GAS * t_sub_k

        k_rxn = 2.5e7 * np.exp(-(activation_energy_kj_mol * 1000.0) / rt)
        growth_rate_nm_min = k_rxn * p_atm * (precursor_flow_rate_sccm / 50.0)

        is_reaction_limited = bool(substrate_temperature_c < 900.0)

        return {
            "growth_rate_nm_min": float(np.clip(growth_rate_nm_min, 0.05, 500.0)),
            "is_reaction_limited": is_reaction_limited,
            "regime": "SURFACE_REACTION_LIMITED" if is_reaction_limited else "MASS_TRANSPORT_LIMITED",
            "is_synthetically_feasible": bool(growth_rate_nm_min >= 0.10),
        }

    def evaluate_spark_plasma_sintering(
        self,
        applied_pressure_mpa: float = 50.0,
        heating_rate_c_min: float = 100.0,
        dwell_temperature_c: float = 1100.0,
        initial_relative_density: float = 0.55,
        diffusion_coeff_m2_s: float = 1.0e-13,
    ) -> Dict[str, Any]:
        """SPS pressure-assisted densification via grain boundary diffusion and plastic yielding."""
        densification_rate = 1.2e-4 * (applied_pressure_mpa / 50.0) * (diffusion_coeff_m2_s / 1e-13)
        final_density = np.clip(initial_relative_density + densification_rate * 600.0, 0.55, 0.995)

        return {
            "final_relative_density": float(final_density),
            "is_fully_densified": bool(final_density >= 0.98),
            "densification_mechanism": "DISLOCATION_CREEP_AND_ELECTROMIGRATION",
            "dwell_time_minutes": 10.0,
        }

    def evaluate_melt_spinning_glass_formation(
        self,
        wheel_speed_m_s: float = 35.0,
        liquidus_temperature_k: float = 1450.0,
        glass_transition_temp_k: float = 720.0,
    ) -> Dict[str, Any]:
        """Rapid solidification quenching rate and Inoue criteria for bulk metallic glass vitrification."""
        cooling_rate_k_s = 2.5e4 * wheel_speed_m_s
        reduced_tg = glass_transition_temp_k / max(1.0, liquidus_temperature_k)

        r_crit_k_s = 1.0e6 * np.exp(-15.0 * (reduced_tg - 0.5))
        is_amorphous = cooling_rate_k_s >= r_crit_k_s

        return {
            "cooling_rate_k_s": float(cooling_rate_k_s),
            "critical_cooling_rate_rc_k_s": float(r_crit_k_s),
            "reduced_glass_transition_trg": float(reduced_tg),
            "is_vitrified_amorphous_ribbon": bool(is_amorphous),
        }
