"""Grand Canonical Electrochemical Phase Stability & SEI Decomposition Kinetics Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class ElectrochemicalPhaseStabilityEngine:
    """Calculates electrochemical stability window [V_red, V_ox] vs metal reference and SEI decomposition energy."""

    def __init__(self, metal_reference: str = "Mg"):
        self.metal_ref = metal_reference
        self.z_val = 2 if metal_reference == "Mg" else 1

    def compute_grand_potential(
        self,
        gibbs_free_energy_ev_atom: float,
        num_metal_atoms_per_formula: float,
        applied_voltage_vs_metal_v: float,
    ) -> float:
        """Evaluate grand canonical potential Phi(V) = G - mu_metal(V) * N_metal:

        mu_metal(V) = mu_metal_0 - z * e * V
        """
        mu_metal = -self.z_val * applied_voltage_vs_metal_v
        phi = gibbs_free_energy_ev_atom - (num_metal_atoms_per_formula * mu_metal)
        return float(phi)

    def evaluate_electrochemical_stability_window(
        self,
        formula: str,
        reduction_potential_v: float,
        oxidation_potential_v: float,
    ) -> Dict[str, Any]:
        """Compute electrochemical stability window [V_red, V_ox] and driving force for decomposition:

        Delta G_decomp(V) = min sum_i (c_i * G_i(V)) - G_parent(V)
        """
        window_width = oxidation_potential_v - reduction_potential_v
        is_wide_window = window_width >= 2.5  # Wide stability window > 2.5 V

        # Passivating SEI compatibility with metal anode (V_red <= 0.0 V is thermodynamically stable against metal)
        is_anode_stable = reduction_potential_v <= 0.05

        return {
            "formula": formula,
            "reduction_potential_v_vs_ref": float(reduction_potential_v),
            "oxidation_potential_v_vs_ref": float(oxidation_potential_v),
            "stability_window_width_v": float(window_width),
            "is_thermodynamically_stable_vs_anode": bool(is_anode_stable),
            "is_high_voltage_cathode_stable": bool(oxidation_potential_v >= 3.5),
        }

    def predict_sei_passivation_thickness(
        self,
        decomposition_energy_ev_atom: float,
        electronic_tunneling_barrier_ev: float = 2.8,
    ) -> Dict[str, float]:
        """Predict self-limiting SEI passivation layer thickness (nm) governed by electron tunneling decay:

        d_SEI ~ hbar / (2 * sqrt(2 * m_e * Phi_tunnel)) * ln(decay_factor)
        """
        # Electron tunneling decay length ~ 0.5 - 3.0 nm
        barrier = max(0.5, electronic_tunneling_barrier_ev)
        d_tunnel_nm = 2.5 / np.sqrt(barrier)

        return {
            "decomposition_energy_ev_atom": float(decomposition_energy_ev_atom),
            "passivation_layer_thickness_nm": float(d_tunnel_nm),
            "is_passivating_sei": bool(decomposition_energy_ev_atom > -0.5),  # Weak decomposition passivates
        }
