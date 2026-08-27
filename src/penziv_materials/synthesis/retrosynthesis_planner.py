"""Retrosynthetic Reaction Network Thermodynamics & Robotic Automated Lab Protocol Export."""

import datetime
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS


class RetrosynthesisAssemblyPlanner:
    """Evaluates multi-step precursor reaction networks Delta G_rxn(T) and exports robotic synthesis recipes (A-Lab / Chemspeed)."""

    # Precursor standard formation enthalpies (kJ/mol) and entropies (J/(mol*K))
    PRECURSOR_THERMO_DATA: Dict[str, Tuple[float, float]] = {
        "MgS": (-345.0, 50.3),
        "Sc2S3": (-1240.0, 142.0),
        "ZrS2": (-578.0, 78.2),
        "P2S5": (-255.0, 168.0),
        "Na2S": (-365.0, 83.7),
        "SiO2": (-910.9, 41.5),
        "ZrO2": (-1100.6, 50.4),
        "MgCO3": (-1095.8, 65.7),
        "TiO2": (-944.0, 50.6),
        "Al2O3": (-1675.7, 50.9),
    }

    def compute_solid_state_reaction_free_energy(
        self,
        reactants: Dict[str, float],  # e.g. {"MgS": 1.0, "ZrS2": 2.0, "P2S5": 1.5}
        product_formation_enthalpy_kj_mol: float,
        product_entropy_j_mol_k: float,
        temperature_k: float = 873.15,
    ) -> Dict[str, float]:
        """Evaluate Gibbs free energy change of solid-state synthesis reaction Delta G_rxn(T):

        Delta G_rxn(T) = Delta H_rxn - T * Delta S_rxn
        """
        h_reactants = sum(coeff * self.PRECURSOR_THERMO_DATA.get(chem, (-500.0, 60.0))[0] for chem, coeff in reactants.items())
        s_reactants = sum(coeff * self.PRECURSOR_THERMO_DATA.get(chem, (-500.0, 60.0))[1] for chem, coeff in reactants.items())

        delta_h_rxn_kj = product_formation_enthalpy_kj_mol - h_reactants
        delta_s_rxn_j_k = product_entropy_j_mol_k - s_reactants
        delta_g_rxn_kj = delta_h_rxn_kj - (temperature_k * delta_s_rxn_j_k * 1.0e-3)

        return {
            "delta_h_reaction_kj_mol": float(delta_h_rxn_kj),
            "delta_s_reaction_j_mol_k": float(delta_s_rxn_j_k),
            "delta_g_reaction_kj_mol": float(delta_g_rxn_kj),
            "is_thermodynamically_spontaneous": bool(delta_g_rxn_kj < 0.0),
        }

    def evaluate_hybrid_manufacturing_route(
        self,
        ceramic_sintering_temp_c: float,
        polymer_degradation_temp_c: float = 240.0,
        channel_fluid_injection_pressure_mpa: float = 2.5,
    ) -> Dict[str, Any]:
        """Synthesize a causal manufacturing execution graph for heterogeneous multi-material systems."""
        has_thermal_clash = ceramic_sintering_temp_c > polymer_degradation_temp_c

        recommended_route = []
        if has_thermal_clash:
            recommended_route.append("Step 1: Solid-state precursor ball-milling in Ar atmosphere glovebox (350 rpm, 12h).")
            recommended_route.append("Step 2: Low-temperature Cold Sintering Process (CSP at 180°C / 300 MPa) to form 3D Gyroid ceramic framework.")
            recommended_route.append("Step 3: Vacuum-assisted Sol-Gel infiltration of conformal polymeric electrolyte membrane (< 150°C).")
            recommended_route.append(f"Step 4: Pressurize internal microchannels to {channel_fluid_injection_pressure_mpa:.1f} MPa.")
            recommended_route.append("Step 5: Hermetic boundary ALD Al2O3 nanoseal deposition (120°C).")
            feasible = True
            primary_process = "SEQUENTIAL_COLD_SINTERING_AND_INFILTRATION"
        else:
            recommended_route.append("Step 1: Co-sinter multi-material green body.")
            feasible = True
            primary_process = "DIRECT_CO_SINTERING"

        return {
            "has_thermal_processing_clash": has_thermal_clash,
            "is_synthetically_feasible": feasible,
            "primary_recommended_process": primary_process,
            "synthesis_route_graph": recommended_route,
            "max_tolerated_processing_temp_c": min(ceramic_sintering_temp_c, polymer_degradation_temp_c),
        }

    def export_robotic_synthesis_protocol(
        self,
        candidate_formula: str,
        precursor_masses_g: Dict[str, float],
        sintering_temp_c: float = 180.0,
        dwell_time_minutes: int = 120,
    ) -> Dict[str, Any]:
        """Generate machine-readable JSON protocol schema compatible with automated synthesis platforms (A-Lab / Chemspeed)."""
        protocol = {
            "protocol_schema": "A-Lab-Chemspeed-Autonomous-Synthesis-v1.2",
            "target_material": candidate_formula,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dispensing_steps": [
                {"action": "solid_dispense", "reagent": chem, "mass_g": mass, "tolerance_mg": 0.5}
                for chem, mass in precursor_masses_g.items()
            ],
            "processing_sequence": [
                {"step": 1, "operation": "planetary_ball_milling", "speed_rpm": 400, "duration_min": 60, "atmosphere": "Ar"},
                {"step": 2, "operation": "die_pressing", "pressure_mpa": 300.0, "dwell_sec": 30},
                {"step": 3, "operation": "cold_sintering", "temperature_c": sintering_temp_c, "duration_min": dwell_time_minutes},
                {"step": 4, "operation": "characterization_dispatch", "modalities": ["XRD", "EIS_conductivity", "SEM_cross_section"]},
            ],
        }
        return protocol
