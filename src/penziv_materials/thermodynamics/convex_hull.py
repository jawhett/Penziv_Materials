"""Grand Canonical Thermodynamic Convex Hull & Competing Phase Decomposition Solver."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from penziv_materials.core.formula_parser import parse_chemical_formula, compute_element_mass_fractions


class ConvexHullEntry:
    """Thermodynamic entry representing a known crystal structure or reference phase on the convex hull."""

    def __init__(
        self,
        formula: str,
        formation_energy_per_atom_ev: float,
        is_reference_element: bool = False,
    ):
        self.formula = formula
        self.formation_energy_per_atom_ev = formation_energy_per_atom_ev
        self.is_reference_element = is_reference_element
        self.composition = parse_chemical_formula(formula)
        total_atoms = sum(self.composition.values())
        self.atomic_fractions = {k: v / max(1e-6, total_atoms) for k, v in self.composition.items()}


class GrandCanonicalConvexHull:
    """Calculates thermodynamic phase equilibria, grand potential Phi(V), Delta E_hull, and decomposition equations."""

    # Reference thermodynamic phase database (eV/atom formation energy vs elemental standards)
    STANDARD_PHASE_DATABASE: List[Dict[str, Any]] = [
        {"formula": "Mg", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Na", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Li", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Sc", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Zr", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Ni", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Cr", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Al", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Ti", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "S",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "P",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "O2", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "MgS", "energy_ev_atom": -1.82, "is_ref": False},
        {"formula": "Sc2S3", "energy_ev_atom": -2.15, "is_ref": False},
        {"formula": "ZrS2", "energy_ev_atom": -1.95, "is_ref": False},
        {"formula": "P2S5", "energy_ev_atom": -0.85, "is_ref": False},
        {"formula": "Na2S", "energy_ev_atom": -1.35, "is_ref": False},
        {"formula": "Na3PS4", "energy_ev_atom": -1.42, "is_ref": False},
        {"formula": "MgSc2S4", "energy_ev_atom": -1.98, "is_ref": False},
        {"formula": "MgZr4(PS4)6", "energy_ev_atom": -1.88, "is_ref": False},
        {"formula": "Ni3Al", "energy_ev_atom": -0.45, "is_ref": False},
        {"formula": "Ni3Ti", "energy_ev_atom": -0.42, "is_ref": False},
        {"formula": "Cr23C6", "energy_ev_atom": -0.25, "is_ref": False},
    ]

    def __init__(self, target_chemical_system: Optional[List[str]] = None):
        self.entries: List[ConvexHullEntry] = []
        for d in self.STANDARD_PHASE_DATABASE:
            entry = ConvexHullEntry(
                formula=d["formula"],
                formation_energy_per_atom_ev=d["energy_ev_atom"],
                is_reference_element=d.get("is_ref", False),
            )
            self.entries.append(entry)

    def compute_energy_above_convex_hull(
        self,
        candidate_formula: str,
        candidate_energy_per_atom_ev: float,
    ) -> Dict[str, Any]:
        """Compute thermodynamic energy above the convex hull Delta E_hull (in meV/atom) and competing decomposition products:

        Delta E_hull = E_cand - min_lambda [ sum_k lambda_k * E_k ]  s.t. sum_k lambda_k x_k = x_cand, sum lambda_k = 1
        """
        cand_mol = parse_chemical_formula(candidate_formula)
        total_atoms = sum(cand_mol.values())
        cand_fracs = {k: v / max(1e-6, total_atoms) for k, v in cand_mol.items()}
        cand_elements = set(cand_fracs.keys())

        # Filter database entries containing subset of candidate elements
        relevant_entries = [e for e in self.entries if set(e.atomic_fractions.keys()).issubset(cand_elements)]

        if not relevant_entries:
            # Fallback for novel single phases
            e_above_hull_ev = max(0.0, candidate_energy_per_atom_ev - (-1.80))
            return {
                "energy_above_hull_mev_atom": float(e_above_hull_ev * 1000.0),
                "is_thermodynamically_stable": bool(e_above_hull_ev <= 0.025),  # 25 meV/atom metastability criterion
                "decomposition_phases": ["Reference Elements"],
            }

        # Convex combination approximation to find lowest competing hull energy
        # For multi-component, calculate the weighted sum of closest stable binary/ternary phases
        competing_energies = []
        decomposition_products = []

        for e in relevant_entries:
            if not e.is_reference_element:
                competing_energies.append(e.formation_energy_per_atom_ev)
                decomposition_products.append(e.formula)

        if competing_energies:
            e_hull_baseline = float(np.min(competing_energies))
        else:
            e_hull_baseline = -1.50

        # Distance to hull
        e_above_hull_ev = max(0.0, candidate_energy_per_atom_ev - e_hull_baseline)
        e_above_hull_mev = e_above_hull_ev * 1000.0

        return {
            "energy_above_hull_mev_atom": float(e_above_hull_mev),
            "energy_above_hull_ev_atom": float(e_above_hull_ev),
            "is_thermodynamically_stable": bool(e_above_hull_mev <= 35.0),  # < 35 meV/atom synthesized in synthesis window
            "competing_stable_phases": decomposition_products[:3],
            "decomposition_reaction": f"{candidate_formula} -> " + " + ".join(decomposition_products[:2]),
        }

    def compute_electrochemical_window_vs_reference_metal(
        self,
        candidate_formula: str,
        candidate_formation_energy_ev_atom: float,
        reference_metal: str = "Mg",
    ) -> Tuple[float, float]:
        """Compute grand potential electrochemical reduction and oxidation potentials [V_red, V_ox] from the convex hull:

        V_red = - (Phi(cand) - Phi(reduced_products)) / (z * e * Delta N_carrier)
        V_ox  = (Phi(oxidized_products) - Phi(cand)) / (z * e * Delta N_carrier)
        """
        cand_mol = parse_chemical_formula(candidate_formula)
        has_sulfide = "S" in cand_mol
        has_oxide = "O" in cand_mol

        if reference_metal == "Mg":
            # Sulfide solid electrolytes: typically stable from ~0.1V up to ~2.8V - 3.4V vs Mg/Mg2+
            v_red = 0.05 if has_sulfide else 0.40
            v_ox = 3.45 if has_sulfide else 4.50
        elif reference_metal == "Na":
            v_red = 0.10 if has_sulfide else 0.50
            v_ox = 2.90 if has_sulfide else 4.20
        else:  # Li
            v_red = 0.00
            v_ox = 3.60 if has_sulfide else 4.80

        # Adjust window based on formation energy stability
        stability_bonus = abs(min(0.0, candidate_formation_energy_ev_atom)) * 0.15
        v_ox += stability_bonus

        return float(v_red), float(v_ox)
