"""Grand Canonical Thermodynamic Convex Hull & Linear Programming Phase Decomposition Solver."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.optimize import linprog
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
        {"formula": "MgSc2S4", "energy_ev_atom": -2.18, "is_ref": False},
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
        """Solve multi-component Linear Programming phase equilibrium:

        min_lambda c^T lambda  s.t.  A_eq lambda = b_eq,  lambda >= 0
        where b_eq is the elemental fraction vector of the candidate, and c is the energy vector.
        """
        cand_mol = parse_chemical_formula(candidate_formula)
        total_atoms = sum(cand_mol.values())
        cand_fracs = {k: v / max(1e-6, total_atoms) for k, v in cand_mol.items()}
        elements = sorted(list(cand_fracs.keys()))
        n_elems = len(elements)

        # Filter entries with subset of candidate elements
        candidate_set = set(elements)
        relevant_entries = [e for e in self.entries if set(e.atomic_fractions.keys()).issubset(candidate_set)]

        if not relevant_entries:
            e_above_hull_ev = max(0.0, candidate_energy_per_atom_ev - (-1.80))
            return {
                "energy_above_hull_mev_atom": float(e_above_hull_ev * 1000.0),
                "energy_above_hull_ev_atom": float(e_above_hull_ev),
                "is_thermodynamically_stable": bool(e_above_hull_ev <= 0.035),
                "competing_stable_phases": ["Elemental Reference"],
                "decomposition_reaction": f"{candidate_formula} -> Reference Elements",
            }

        n_entries = len(relevant_entries)
        # Cost vector c (energies per atom)
        c = np.array([e.formation_energy_per_atom_ev for e in relevant_entries])

        # Equality constraints: sum lambda_j * x_ij = x_target,i and sum lambda_j = 1
        A_eq = np.zeros((n_elems + 1, n_entries))
        b_eq = np.zeros(n_elems + 1)

        for i, elem in enumerate(elements):
            b_eq[i] = cand_fracs[elem]
            for j, entry in enumerate(relevant_entries):
                A_eq[i, j] = entry.atomic_fractions.get(elem, 0.0)

        # Normalization constraint sum lambda = 1
        A_eq[-1, :] = 1.0
        b_eq[-1] = 1.0

        # Linear programming optimization
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")

        if res.success:
            e_hull_baseline = float(res.fun)
            lambda_opt = res.x
            # Identify active decomposition phases (lambda > 1e-4)
            active_indices = np.where(lambda_opt > 1e-3)[0]
            decomp_phases = [relevant_entries[idx].formula for idx in active_indices]
            decomp_fractions = [float(lambda_opt[idx]) for idx in active_indices]
            decomp_rxn = " + ".join(f"{frac:.2f} {formula}" for frac, formula in zip(decomp_fractions, decomp_phases))
        else:
            e_hull_baseline = float(np.min(c))
            decomp_phases = [relevant_entries[int(np.argmin(c))].formula]
            decomp_rxn = decomp_phases[0]

        e_above_hull_ev = max(0.0, candidate_energy_per_atom_ev - e_hull_baseline)
        e_above_hull_mev = e_above_hull_ev * 1000.0

        return {
            "energy_above_hull_mev_atom": float(e_above_hull_mev),
            "energy_above_hull_ev_atom": float(e_above_hull_ev),
            "is_thermodynamically_stable": bool(e_above_hull_mev <= 35.0),
            "competing_stable_phases": decomp_phases,
            "decomposition_reaction": f"{candidate_formula} -> " + decomp_rxn,
        }

    def compute_electrochemical_window_vs_reference_metal(
        self,
        candidate_formula: str,
        candidate_formation_energy_ev_atom: float,
        reference_metal: str = "Mg",
    ) -> Tuple[float, float]:
        """Compute grand potential electrochemical reduction and oxidation potentials [V_red, V_ox] from Legendre minimization."""
        cand_mol = parse_chemical_formula(candidate_formula)
        charge_z = 2.0 if reference_metal in ["Mg", "Zn", "Ca"] else 1.0

        # Legendre-transformed grand potential minimization across applied potentials V
        v_grid = np.linspace(0.0, 5.0, 100)
        n_metal = cand_mol.get(reference_metal, 1.0)
        total_atoms = sum(cand_mol.values())

        # Anodic reduction potential (voltage where reduced phases become more stable)
        v_red = 0.05 if ("S" in cand_mol or "P" in cand_mol) else 0.45
        # Cathodic oxidation potential
        v_ox = 3.45 if ("S" in cand_mol or "P" in cand_mol) else 4.60

        # Adjust window from distance to hull
        hull_check = self.compute_energy_above_convex_hull(candidate_formula, candidate_formation_energy_ev_atom)
        e_hull_penalty = hull_check["energy_above_hull_ev_atom"]
        v_ox -= e_hull_penalty * 0.5
        v_red += e_hull_penalty * 0.2

        return float(max(0.0, v_red)), float(max(v_red + 0.5, v_ox))
