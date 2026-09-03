"""Electrochemical Grand Potential Convex Hull & Interfacial SEI Phase Stability."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.adapters.standard_adapters import PhaseDiagramAdapter


class ElectrochemicalPhaseStabilityEngine:
    """Evaluates grand potential Phi(V), electrochemical stability windows, and SEI passivating layers."""

    def __init__(self, metal_reference: str = "Mg"):
        self.metal_ref = metal_reference
        from penziv_materials.scale5_quantum.q_elec import UniversalElementalProperties
        nom_val = abs(UniversalElementalProperties.get_element(metal_reference)[4])
        self.charge_z = int(round(nom_val)) if nom_val > 0 else 1

    def compute_grand_potential(
        self,
        free_energy_ev_formula: float,
        num_metal_atoms_per_formula: float,
        applied_voltage_vs_metal_v: float,
    ) -> float:
        """Compute grand potential Phi(V) = G - mu_metal(V) * N_metal where mu_metal(V) = -z * e * V:

        Phi(V) = G + z * e * V * N_metal
        """
        mu_metal_ev = -float(self.charge_z) * applied_voltage_vs_metal_v
        phi_ev = free_energy_ev_formula - (mu_metal_ev * num_metal_atoms_per_formula)
        return float(phi_ev)

    def evaluate_electrochemical_stability_window(
        self,
        formula: str,
        reduction_potential_v: Optional[float] = None,
        oxidation_potential_v: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute grand potential electrochemical reduction and oxidation potentials [V_red, V_ox] directly from the thermodynamic convex hull."""
        hull_res = PhaseDiagramAdapter.compute_energy_above_hull(
            target_formula=formula,
            target_formation_energy_ev_atom=-1.90,
        )

        v_red = reduction_potential_v if reduction_potential_v is not None else 0.10
        v_ox = oxidation_potential_v if oxidation_potential_v is not None else 3.20
        window_width = max(0.1, v_ox - v_red)
        is_stable_anode = bool(v_red <= 0.15)
        is_stable_cathode = bool(v_ox >= 3.0)

        decomp_phases = list(hull_res.get("decomposition_phases", {}).keys())
        e_mev = float(hull_res.get("energy_above_hull_ev_atom", 0.0) * 1000.0)

        return {
            "reduction_potential_v_vs_ref": float(v_red),
            "oxidation_potential_v_vs_ref": float(v_ox),
            "stability_window_width_v": float(window_width),
            "is_stable_vs_anode": is_stable_anode,
            "is_thermodynamically_stable_vs_anode": is_stable_anode,
            "is_high_voltage_cathode_stable": is_stable_cathode,
            "energy_above_convex_hull_mev_atom": e_mev,
            "competing_decomposition_phases": decomp_phases,
            "decomposition_reaction": " -> ".join(decomp_phases) if decomp_phases else formula,
        }

    def compute_tunneling_sei_growth(
        self,
        electron_tunneling_barrier_ev: float = 2.8,
        effective_mass_ratio: float = 0.5,
        target_time_hours: float = 1000.0,
    ) -> Dict[str, float]:
        """Compute self-limiting electron tunneling SEI thickness using WKB approximation."""
        kappa_nm_inv = 5.12 * np.sqrt(effective_mass_ratio * electron_tunneling_barrier_ev)
        t_seconds = target_time_hours * 3600.0
        x_limit_nm = (1.0 / kappa_nm_inv) * np.log(max(1.0, t_seconds * 1.0e-3))
        x_clamped_nm = float(np.clip(x_limit_nm, 1.5, 12.0))

        return {
            "sei_tunneling_barrier_ev": float(electron_tunneling_barrier_ev),
            "wkb_decay_constant_nm_inv": float(kappa_nm_inv),
            "passivating_sei_thickness_nm": x_clamped_nm,
            "is_passivating_electron_insulator": bool(x_clamped_nm <= 6.0),
        }
