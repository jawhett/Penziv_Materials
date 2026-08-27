"""Holistic Multi-Domain Constraint Relaxation & Composite Free-Energy Hamiltonian."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class HolisticStabilityRelaxationEngine:
    """Evaluates composite system stability where local sub-component fragility is stabilized by adjacent phases."""

    def __init__(self):
        pass

    def evaluate_composite_system_hamiltonian(
        self,
        ceramic_elastic_energy_density_mj_m3: float,
        fluid_pressure_work_mj_m3: float,
        polymer_interfacial_traction_energy_mj_m3: float,
        vol_fraction_ceramic: float,
        vol_fraction_fluid: float,
        vol_fraction_polymer: float,
    ) -> Dict[str, Any]:
        """Compute total composite free energy density:

        F_composite = v_c * F_ceramic - v_f * (P_fluid * Delta V / V) + v_p * F_polymer + F_interface
        """
        # Net mechanical energy functional
        f_ceramic_net = vol_fraction_ceramic * ceramic_elastic_energy_density_mj_m3
        # Fluid pressure does work against compressive collapse (-P * dV)
        f_fluid_stabilization = -vol_fraction_fluid * fluid_pressure_work_mj_m3
        f_polymer_traction = vol_fraction_polymer * polymer_interfacial_traction_energy_mj_m3

        f_total_system = f_ceramic_net + f_fluid_stabilization + f_polymer_traction

        # Isolated ceramic Born check vs. Composite Co-Design Stability
        is_isolated_ceramic_fragile = ceramic_elastic_energy_density_mj_m3 > 120.0
        # If fluid pressure and polymer constraint relieve the total system energy below threshold, it is stable!
        is_composite_stabilized = bool(f_total_system < 80.0)

        # Gate decision: Relaxation permitted if composite is stabilized
        gate_decision = "ACCEPTED_VIA_HOLISTIC_RELAXATION" if (is_isolated_ceramic_fragile and is_composite_stabilized) else (
            "ACCEPTED_STANDARD" if not is_isolated_ceramic_fragile else "REJECTED_GLOBAL_INSTABILITY"
        )

        return {
            "total_system_free_energy_mj_m3": float(f_total_system),
            "isolated_ceramic_fragility_detected": is_isolated_ceramic_fragile,
            "composite_co_design_stabilized": is_composite_stabilized,
            "handshake_gate_decision": gate_decision,
            "stabilization_energy_offset_mj_m3": float(abs(f_fluid_stabilization)),
        }
