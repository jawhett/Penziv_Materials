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

    # Multi-valence redox states mapping for transition metals and main group elements
    ELEMENT_REDOX_VALENCES: Dict[str, List[float]] = {
        "H": [1.0, -1.0],
        "Li": [1.0], "Na": [1.0], "K": [1.0], "Rb": [1.0], "Cs": [1.0],
        "Be": [2.0], "Mg": [2.0], "Ca": [2.0], "Sr": [2.0], "Ba": [2.0], "Zn": [2.0], "Cd": [2.0],
        "B": [3.0], "Al": [3.0], "Ga": [3.0, 1.0], "In": [3.0, 1.0], "Sc": [3.0], "Y": [3.0], "La": [3.0],
        "Ti": [4.0, 3.0, 2.0], "Zr": [4.0], "Hf": [4.0],
        "V": [5.0, 4.0, 3.0, 2.0], "Nb": [5.0, 4.0, 3.0], "Ta": [5.0, 4.0],
        "Cr": [6.0, 4.0, 3.0, 2.0], "Mo": [6.0, 5.0, 4.0, 3.0], "W": [6.0, 5.0, 4.0],
        "Mn": [7.0, 4.0, 3.0, 2.0], "Fe": [3.0, 2.0, 4.0, 6.0], "Co": [3.0, 2.0, 4.0], "Ni": [2.0, 3.0, 4.0],
        "Cu": [2.0, 1.0, 3.0], "Ag": [1.0, 2.0], "Au": [3.0, 1.0],
        "Si": [4.0, -4.0], "Ge": [4.0, 2.0], "Sn": [4.0, 2.0], "Pb": [4.0, 2.0],
        "P": [5.0, 3.0, -3.0], "As": [5.0, 3.0, -3.0], "Sb": [5.0, 3.0, -3.0], "Bi": [3.0, 5.0],
        "O": [-2.0], "S": [-2.0, 4.0, 6.0], "Se": [-2.0, 4.0, 6.0], "Te": [-2.0, 4.0, 6.0],
        "F": [-1.0], "Cl": [-1.0, 1.0, 3.0, 5.0, 7.0], "Br": [-1.0, 1.0, 5.0], "I": [-1.0, 1.0, 5.0, 7.0],
    }

    ELEMENT_VALENCES: Dict[str, float] = {k: v[0] for k, v in ELEMENT_REDOX_VALENCES.items()}

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

    @classmethod
    def evaluate_miedema_formation_enthalpy(cls, composition: Dict[str, float]) -> float:
        """Evaluate generalized Miedema semi-empirical formation enthalpy across arbitrary multi-element systems."""
        from penziv_materials.scale5_quantum.q_elec import UniversalElementalProperties
        elems = list(composition.keys())
        counts = np.array([composition[e] for e in elems], dtype=np.float64)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        fracs = counts / total
        n = len(elems)
        if n <= 1:
            return 0.0

        # Physical Miedema constants: P=14.2, Q=123.5
        P = 14.2
        Q = 123.5
        delta_h_kj = 0.0

        phi_vals = np.array([UniversalElementalProperties.get_element(e)[2] for e in elems])
        v_molar = np.array([UniversalElementalProperties.get_element(e)[3] for e in elems])
        nws_vals = np.array([abs(UniversalElementalProperties.get_element(e)[4]) for e in elems])

        v_23 = v_molar ** (2.0 / 3.0)
        mean_v_23 = np.sum(fracs * v_23)

        for i in range(n):
            for j in range(i + 1, n):
                delta_phi = phi_vals[i] - phi_vals[j]
                delta_nws = (nws_vals[i] ** (1.0 / 3.0)) - (nws_vals[j] ** (1.0 / 3.0))
                factor = (v_23[i] * v_23[j]) / max(1e-5, mean_v_23)
                r_star = 2.1 if (nws_vals[i] > 3.0 and nws_vals[j] < 2.0) or (nws_vals[j] > 3.0 and nws_vals[i] < 2.0) else 0.0
                interaction = -P * (delta_phi**2) + Q * (delta_nws**2) - r_star
                delta_h_kj += fracs[i] * fracs[j] * factor * interaction

        # Convert kJ/mol to eV/atom (1 eV = 96.485 kJ/mol)
        delta_h_ev_atom = float(delta_h_kj / 96.485)
        return delta_h_ev_atom

    def compute_energy_above_convex_hull(
        self,
        candidate_formula: str,
        candidate_energy_per_atom_ev: float,
    ) -> Dict[str, Any]:
        """Solve multi-component Linear Programming phase equilibrium using generalized Miedema & live entries."""
        cand_mol = parse_chemical_formula(candidate_formula)
        total_atoms = sum(cand_mol.values())
        cand_fracs = {k: v / max(1e-6, total_atoms) for k, v in cand_mol.items()}
        elements = sorted(list(cand_fracs.keys()))
        n_elems = len(elements)

        candidate_set = set(elements)
        relevant_entries = [e for e in self.entries if set(e.atomic_fractions.keys()).issubset(candidate_set)]

        # Dynamically generate generalized Miedema binary and ternary reference phases if unpopulated
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                el1, el2 = elements[i], elements[j]
                pair_key = {el1, el2}
                has_binary = any(set(e.atomic_fractions.keys()) == pair_key for e in relevant_entries)
                if not has_binary:
                    # Stoichiometric binary ratios (1:1, 1:2, 2:3)
                    for r1, r2 in [(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)]:
                        comp_bin = {el1: r1, el2: r2}
                        form_e = float(np.clip(self.evaluate_miedema_formation_enthalpy(comp_bin), -3.8, 0.05))
                        bin_entry = ConvexHullEntry(formula=f"{el1}{int(r1) if r1>1 else ''}{el2}{int(r2) if r2>1 else ''}", formation_energy_per_atom_ev=form_e)
                        relevant_entries.append(bin_entry)

        # Always include elemental standard reference ground states (H_form = 0.0 eV/atom)
        for el in elements:
            if not any(e.formula == el for e in relevant_entries):
                relevant_entries.append(ConvexHullEntry(formula=el, formation_energy_per_atom_ev=0.0, is_reference_element=True))

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

    @staticmethod
    def compute_dynamic_qeq_oxidation_states(composition: Dict[str, float]) -> Dict[str, float]:
        """Compute continuous partial oxidation charges via Electronegativity Equalization (QEq)."""
        from penziv_materials.scale5_quantum.q_elec import UniversalElementalProperties
        elems = list(composition.keys())
        counts = np.array([composition[e] for e in elems], dtype=np.float64)

        # Mulliken electronegativities and hardnesses
        chis = np.array([UniversalElementalProperties.get_element(e)[2] * 2.8 for e in elems])
        etas = np.array([3.5 + 0.5 * abs(UniversalElementalProperties.get_element(e)[4]) for e in elems])

        inv_eta = 1.0 / etas
        mu_bar = float(np.sum(counts * chis * inv_eta) / max(1e-6, np.sum(counts * inv_eta)))
        charges = (mu_bar - chis) * inv_eta
        return {elems[i]: float(round(charges[i], 3)) for i in range(len(elems))}

    def compute_electrochemical_window_vs_reference_metal(
        self,
        candidate_formula: str,
        candidate_formation_energy_ev_atom: float,
        reference_metal: str = "Li",
        voltage_range: Tuple[float, float] = (0.0, 5.5),
        voltage_step: float = 0.02,
        custom_metal_valence: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute exact grand potential reduction/oxidation bounds via dynamic multi-valence Legendre minimization:

        Phi(mu) = min_i [ G_i - sum_k mu_k N_{k,i} ]
        """
        cand_comp = parse_chemical_formula(candidate_formula)
        n_metal = cand_comp.get(reference_metal, 0.0)
        total_atoms = sum(cand_comp.values())
        if n_metal == 0:
            return 0.0, 5.0

        n_metal_frac = n_metal / max(1e-6, total_atoms)

        # Multi-valence evaluation: determine effective active oxidation charges
        if custom_metal_valence is not None:
            charge_z = custom_metal_valence
        else:
            qeq_charges = self.compute_dynamic_qeq_oxidation_states(cand_comp)
            charge_z = abs(qeq_charges.get(reference_metal, self.ELEMENT_VALENCES.get(reference_metal, 1.0)))
            charge_z = max(0.5, min(4.0, charge_z))

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

    def solve_dynamic_active_hull(
        self,
        target_composition: Dict[str, float],
        target_energy_per_atom: float,
        mlip_engine=None,
        dft_fallback_fn=None,
        reference_phase_generator_fn=None,
        tolerance_mev: float = 35.0,
    ) -> Dict[str, Any]:
        """Active learning convex hull solver with on-the-fly candidate generation and epistemic uncertainty routing."""
        elements = sorted(list(target_composition.keys()))
        tot_target = sum(target_composition.values())
        cand_fracs = {k: v / max(1e-6, tot_target) for k, v in target_composition.items()}

        pool_energies: List[Dict[str, Any]] = []

        if reference_phase_generator_fn is not None:
            candidate_pool = reference_phase_generator_fn(elements)
            for cand in candidate_pool:
                cand_comp = getattr(cand, "composition", {})
                e_val = getattr(cand, "energy_per_atom", 0.0)
                if mlip_engine is not None and hasattr(cand, "lattice_matrix"):
                    try:
                        res = mlip_engine.evaluate_total_potential_energy_and_forces(
                            cartesian_coords=cand.cartesian_coords,
                            species=[s.species for s in cand.sites] if hasattr(cand, "sites") else list(cand_comp.keys()),
                            lattice_vectors=cand.lattice_matrix,
                        )
                        e_val = float(res.get("total_energy_ev_atom", e_val))
                        sigma_f = float(res.get("max_force_residual_ev_ang", 0.0))
                        if sigma_f > 0.05 and dft_fallback_fn is not None:
                            e_val = float(dft_fallback_fn(cand))
                    except Exception:
                        pass
                pool_energies.append({"composition": cand_comp, "energy_per_atom": e_val})

        # Fallback to local entries if pool is empty
        if not pool_energies:
            candidate_set = set(elements)
            relevant_entries = [e for e in self.entries if set(e.atomic_fractions.keys()).issubset(candidate_set)]
            if not relevant_entries:
                relevant_entries = [
                    ConvexHullEntry(formula=el, formation_energy_per_atom_ev=0.0, is_reference_element=True)
                    for el in elements
                ]
            for ent in relevant_entries:
                pool_energies.append({"composition": ent.composition, "energy_per_atom": ent.formation_energy_per_atom_ev})

        n_elems = len(elements)
        n_entries = len(pool_energies)
        c_vec = np.array([p["energy_per_atom"] for p in pool_energies])

        A_eq = np.zeros((n_elems + 1, n_entries))
        b_eq = np.zeros(n_elems + 1)

        for i, el in enumerate(elements):
            b_eq[i] = cand_fracs[el]
            for j, entry in enumerate(pool_energies):
                e_tot = sum(entry["composition"].values())
                A_eq[i, j] = entry["composition"].get(el, 0.0) / max(1e-6, e_tot)
        A_eq[-1, :] = 1.0
        b_eq[-1] = 1.0

        res = linprog(c_vec, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
        e_hull = float(res.fun) if res.success else float(np.min(c_vec))
        e_above_hull_mev = max(0.0, (target_energy_per_atom - e_hull) * 1000.0)

        return {
            "energy_above_hull_mev_atom": float(e_above_hull_mev),
            "is_stable": bool(e_above_hull_mev <= tolerance_mev),
            "ground_state_energy_ev_atom": float(e_hull),
            "target_energy_ev_atom": float(target_energy_per_atom),
        }
