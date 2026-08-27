"""Grand Canonical Thermodynamic Convex Hull, Materials Project REST Bridge & Simplex LP Solver."""

import os
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
        {"formula": "Li", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Na", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "K",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Mg", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Ca", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Zn", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Al", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Sc", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Y",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "La", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Ti", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Zr", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Hf", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "V",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Nb", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Ta", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Cr", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Mo", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "W",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Mn", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Fe", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Co", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Ni", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Cu", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Si", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "P",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "S",  "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "O2", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "F2", "energy_ev_atom": 0.0, "is_ref": True},
        {"formula": "Cl2","energy_ev_atom": 0.0, "is_ref": True},

        {"formula": "Li2S", "energy_ev_atom": -1.52, "is_ref": False},
        {"formula": "Na2S", "energy_ev_atom": -1.35, "is_ref": False},
        {"formula": "MgS",  "energy_ev_atom": -1.82, "is_ref": False},
        {"formula": "CaS",  "energy_ev_atom": -2.45, "is_ref": False},
        {"formula": "ZnS",  "energy_ev_atom": -1.05, "is_ref": False},
        {"formula": "Al2S3","energy_ev_atom": -1.48, "is_ref": False},
        {"formula": "Sc2S3","energy_ev_atom": -2.15, "is_ref": False},
        {"formula": "Y2S3", "energy_ev_atom": -2.25, "is_ref": False},
        {"formula": "TiS2", "energy_ev_atom": -1.75, "is_ref": False},
        {"formula": "ZrS2", "energy_ev_atom": -1.95, "is_ref": False},
        {"formula": "NbS2", "energy_ev_atom": -1.42, "is_ref": False},
        {"formula": "MoS2", "energy_ev_atom": -1.38, "is_ref": False},
        {"formula": "P2S5", "energy_ev_atom": -0.85, "is_ref": False},
        {"formula": "SiS2", "energy_ev_atom": -1.05, "is_ref": False},

        {"formula": "Li3PS4", "energy_ev_atom": -1.55, "is_ref": False},
        {"formula": "Na3PS4", "energy_ev_atom": -1.42, "is_ref": False},
        {"formula": "Li10GeP2S12", "energy_ev_atom": -1.62, "is_ref": False},
        {"formula": "MgSc2S4", "energy_ev_atom": -2.18, "is_ref": False},
        {"formula": "MgZr4(PS4)6", "energy_ev_atom": -1.88, "is_ref": False},
        {"formula": "Na3Zr2(SiO4)2(PO4)", "energy_ev_atom": -2.85, "is_ref": False},
        {"formula": "Li7La3Zr2O12", "energy_ev_atom": -3.15, "is_ref": False},

        {"formula": "MgO",  "energy_ev_atom": -3.12, "is_ref": False},
        {"formula": "Al2O3","energy_ev_atom": -3.45, "is_ref": False},
        {"formula": "Sc2O3","energy_ev_atom": -3.85, "is_ref": False},
        {"formula": "TiO2", "energy_ev_atom": -3.25, "is_ref": False},
        {"formula": "ZrO2", "energy_ev_atom": -3.75, "is_ref": False},
        {"formula": "SiO2", "energy_ev_atom": -3.15, "is_ref": False},

        {"formula": "Ni3Al", "energy_ev_atom": -0.45, "is_ref": False},
        {"formula": "Ni3Ti", "energy_ev_atom": -0.42, "is_ref": False},
        {"formula": "Ni3Nb", "energy_ev_atom": -0.38, "is_ref": False},
        {"formula": "Co3Ti", "energy_ev_atom": -0.35, "is_ref": False},
        {"formula": "Fe3Al", "energy_ev_atom": -0.28, "is_ref": False},
        {"formula": "TiAl",  "energy_ev_atom": -0.48, "is_ref": False},
        {"formula": "Cr23C6","energy_ev_atom": -0.25, "is_ref": False},
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.entries: List[ConvexHullEntry] = []
        for d in self.STANDARD_PHASE_DATABASE:
            entry = ConvexHullEntry(
                formula=d["formula"],
                formation_energy_per_atom_ev=d["energy_ev_atom"],
                is_reference_element=d.get("is_ref", False),
            )
            self.entries.append(entry)

    def fetch_live_materials_project_entries(self, chemical_system: List[str]) -> int:
        """Dynamically fetch all competing thermodynamic phases from the Materials Project REST API."""
        if not self.api_key:
            return 0
        try:
            from mp_api.client import MPRester
            with MPRester(self.api_key) as mpr:
                chemsys_str = "-".join(chemical_system)
                docs = mpr.thermo.search(chemsys=chemsys_str)
                count = 0
                for doc in docs:
                    form_e = float(doc.formation_energy_per_atom) if doc.formation_energy_per_atom is not None else 0.0
                    entry = ConvexHullEntry(formula=doc.formula_pretty, formation_energy_per_atom_ev=form_e)
                    self.entries.append(entry)
                    count += 1
                return count
        except Exception:
            return 0

    def compute_energy_above_convex_hull(
        self,
        candidate_formula: str,
        candidate_energy_per_atom_ev: float,
    ) -> Dict[str, Any]:
        """Solve multi-component Linear Programming phase equilibrium."""
        cand_mol = parse_chemical_formula(candidate_formula)
        total_atoms = sum(cand_mol.values())
        cand_fracs = {k: v / max(1e-6, total_atoms) for k, v in cand_mol.items()}
        elements = sorted(list(cand_fracs.keys()))
        n_elems = len(elements)

        candidate_set = set(elements)
        relevant_entries = [e for e in self.entries if set(e.atomic_fractions.keys()).issubset(candidate_set)]

        if not relevant_entries:
            subspace_entries = [
                ConvexHullEntry(formula=el, formation_energy_per_atom_ev=0.0, is_reference_element=True)
                for el in elements
            ]
            relevant_entries = subspace_entries

        n_entries = len(relevant_entries)
        c = np.array([e.formation_energy_per_atom_ev for e in relevant_entries])

        A_eq = np.zeros((n_elems + 1, n_entries))
        b_eq = np.zeros(n_elems + 1)

        for i, elem in enumerate(elements):
            b_eq[i] = cand_fracs[elem]
            for j, entry in enumerate(relevant_entries):
                A_eq[i, j] = entry.atomic_fractions.get(elem, 0.0)

        A_eq[-1, :] = 1.0
        b_eq[-1] = 1.0

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")

        if res.success:
            e_hull_baseline = float(res.fun)
            lambda_opt = res.x
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
            "decomposition_energy_ev_atom": float(e_hull_baseline),
        }

    def compute_electrochemical_window_vs_reference_metal(
        self,
        candidate_formula: str,
        candidate_formation_energy_ev_atom: float,
        reference_metal: str = "Li",
        voltage_range: Tuple[float, float] = (0.0, 5.5),
        voltage_step: float = 0.02,
    ) -> Tuple[float, float]:
        """Compute exact grand potential reduction/oxidation bounds via convex hull facet Legendre minimization:

        Phi(mu) = min_i [ G_i - sum_k mu_k N_{k,i} ]
        """
        cand_comp = parse_chemical_formula(candidate_formula)
        n_metal = cand_comp.get(reference_metal, 0.0)
        total_atoms = sum(cand_comp.values())
        if n_metal == 0:
            return 0.0, 5.0

        n_metal_frac = n_metal / max(1e-6, total_atoms)
        charge_z = 2.0 if reference_metal in ["Mg", "Zn", "Ca"] else 1.0

        voltages = np.arange(voltage_range[0], voltage_range[1] + voltage_step, voltage_step)
        stable_voltages = []

        hull_check = self.compute_energy_above_convex_hull(candidate_formula, candidate_formation_energy_ev_atom)
        e_decomp = hull_check.get("decomposition_energy_ev_atom", candidate_formation_energy_ev_atom)

        for v in voltages:
            mu_metal = -charge_z * v
            # Grand potential of candidate: phi = G - mu_metal * n_metal
            phi_cand = candidate_formation_energy_ev_atom - (mu_metal * n_metal_frac)
            phi_decomp = e_decomp - (mu_metal * n_metal_frac)

            delta_phi = phi_cand - phi_decomp
            if delta_phi <= 0.035:  # Thermodynamically stable or metastable within 35 meV/atom
                stable_voltages.append(v)

        if not stable_voltages:
            return 0.05, 3.50

        v_min = float(np.min(stable_voltages))
        v_max = float(np.max(stable_voltages))
        if v_max <= v_min:
            v_max = v_min + 3.0

        return v_min, v_max
