"""Standard Library Adapter Layer for Crystallography, CALPHAD, Phase Diagrams, and Topology.

Strictly delegates all operations to established C-backed and standard scientific libraries:
- spglib: Space group symmetry determination, Seitz operators, Wyckoff orbits
- pymatgen: Phase diagram convex hulls, E_hull, and elastic tensor Voigt-Reuss-Hill homogenization
- pycalphad: Multi-component CALPHAD Gibbs energy minimization
- gudhi: Simplicial persistent homology Betti invariants
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Standard C-backed & verified scientific libraries (Strict Mandatory Requirements)
import spglib
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.phase_diagram import PhaseDiagram as PmgPhaseDiagram, PDEntry
from pymatgen.analysis.elasticity import ElasticTensor as PmgElasticTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as PmgSpacegroupAnalyzer
from pycalphad import Database as PycalphadDB, equilibrium as pycalphad_equilibrium, variables as v
import gudhi
import ase
from penziv_materials.core.formula_parser import parse_chemical_formula


class SymmetryAdapter:
    """Delegates space group detection, Seitz matrix generation, and Wyckoff expansion directly to spglib and pymatgen."""

    @classmethod
    def get_space_group_info(
        cls,
        lattice_matrix_3x3: np.ndarray,
        scaled_positions: np.ndarray,
        atomic_numbers: List[int],
        symprec: float = 1e-4,
    ) -> Dict[str, Any]:
        """Determine international space group number, Hermann-Mauguin symbol, and Hall symbol using spglib."""
        cell = (
            np.asarray(lattice_matrix_3x3, dtype=np.float64),
            np.asarray(scaled_positions, dtype=np.float64),
            list(atomic_numbers),
        )

        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            raise ValueError(
                f"spglib failed to resolve space group symmetry for cell with {len(atomic_numbers)} atoms at symprec={symprec}"
            )

        return {
            "space_group_number": int(dataset.number),
            "international_symbol": str(dataset.international),
            "hall_symbol": str(dataset.hall),
            "wyckoffs": list(dataset.wyckoffs),
            "equivalent_atoms": list(dataset.equivalent_atoms),
            "backend": "spglib",
        }

    @classmethod
    def get_symmetry_operations(cls, space_group_number: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve Seitz affine operators [R | t] directly from spglib symmetry database."""
        if not (1 <= space_group_number <= 230):
            raise ValueError(f"Space group number must be between 1 and 230, got {space_group_number}")

        ops_dict = spglib.get_symmetry_from_database(space_group_number)
        rotations = ops_dict["rotations"]
        translations = ops_dict["translations"]

        return [
            (np.asarray(rotations[i], dtype=np.float64), np.asarray(translations[i], dtype=np.float64))
            for i in range(len(rotations))
        ]

    @classmethod
    def expand_wyckoff_orbit(
        cls,
        space_group_number: int,
        wyckoff_letter: str,
        coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[np.ndarray]:
        """Expand a Wyckoff coordinate into its unique crystallographic orbit using spglib operators."""
        ops = cls.get_symmetry_operations(space_group_number)
        x0 = np.asarray(coordinates, dtype=np.float64)

        orbit: List[np.ndarray] = []
        for rot, trans in ops:
            cand = np.mod(rot @ x0 + trans, 1.0)
            # Standard boundary canonicalization: map ~1.0 back to 0.0
            cand = np.where(np.isclose(cand, 1.0, atol=1e-5), 0.0, cand)
            cand = np.where(np.isclose(cand, 0.0, atol=1e-5), 0.0, cand)

            # Check if point already exists in orbit
            already_in = False
            for existing in orbit:
                diff = np.abs(cand - existing)
                diff = np.minimum(diff, 1.0 - diff)
                if np.all(diff < 1e-4):
                    already_in = True
                    break

            if not already_in:
                orbit.append(cand)

        return orbit


class PhaseDiagramAdapter:
    """Delegates thermodynamic convex hull and energy above hull calculations directly to pymatgen."""

    # Standard reference formation energies across key elemental and binary compounds
    STANDARD_REFERENCE_DATABASE: List[Dict[str, Any]] = [
        {"formula": "Fe", "energy_ev_atom": -8.31},
        {"formula": "Cr", "energy_ev_atom": -9.50},
        {"formula": "Ni", "energy_ev_atom": -4.45},
        {"formula": "Ti", "energy_ev_atom": -7.70},
        {"formula": "Al", "energy_ev_atom": -3.36},
        {"formula": "V", "energy_ev_atom": -8.90},
        {"formula": "Cu", "energy_ev_atom": -3.49},
        {"formula": "Si", "energy_ev_atom": -5.40},
        {"formula": "C", "energy_ev_atom": -9.10},
        {"formula": "O2", "energy_ev_atom": -4.95},
        {"formula": "N2", "energy_ev_atom": -8.30},
        {"formula": "TiO2", "energy_ev_atom": -8.70},
        {"formula": "Al2O3", "energy_ev_atom": -6.80},
        {"formula": "SiO2", "energy_ev_atom": -6.50},
        {"formula": "Fe3C", "energy_ev_atom": -7.90},
        {"formula": "Ni3Al", "energy_ev_atom": -4.80},
        {"formula": "TiC", "energy_ev_atom": -8.50},
    ]

    @classmethod
    def compute_energy_above_hull(
        cls,
        target_formula: str,
        target_formation_energy_ev_atom: float,
        reference_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculate thermodynamic stability Delta E_hull (eV/atom) and equilibrium decomposition using pymatgen."""
        pmg_entries: List[PDEntry] = []
        ref_list = reference_entries or cls.STANDARD_REFERENCE_DATABASE

        for r in ref_list:
            f = r["formula"]
            e = float(r["energy_ev_atom"])
            comp_dict = parse_chemical_formula(f)
            total_atoms = sum(comp_dict.values())
            pmg_entries.append(PDEntry(f, e * total_atoms))

        target_comp = parse_chemical_formula(target_formula)
        target_total = sum(target_comp.values())
        target_pd_entry = PDEntry(target_formula, target_formation_energy_ev_atom * target_total)
        pmg_entries.append(target_pd_entry)

        pd = PmgPhaseDiagram(pmg_entries)
        e_above_hull = float(pd.get_e_above_hull(target_pd_entry))
        decomp = pd.get_decomposition(target_pd_entry.composition)

        decomp_dict = {entry.composition.reduced_formula: float(fraction) for entry, fraction in decomp.items()}
        is_stable = bool(e_above_hull <= 1e-4)

        return {
            "energy_above_hull_ev_atom": e_above_hull,
            "is_thermodynamically_stable": is_stable,
            "decomposition_phases": decomp_dict,
            "backend": "pymatgen",
        }


class CalphadAdapter:
    """Delegates CALPHAD Gibbs energy polynomial evaluations directly to pycalphad."""

    @classmethod
    def evaluate_gibbs_equilibrium(
        cls,
        elements: List[str],
        phases: List[str],
        temperature_k: float,
        pressure_pa: float = 101325.0,
        tdb_file_content: Optional[str] = None,
        composition_fractions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute thermodynamic equilibrium phase fractions and Gibbs energy using pycalphad."""
        if not tdb_file_content:
            raise ValueError("tdb_file_content is required for pycalphad equilibrium evaluation")

        db = PycalphadDB(tdb_file_content)
        conditions = {v.N: 1.0, v.P: pressure_pa, v.T: temperature_k}

        # Set degrees of freedom for multicomponent systems
        if len(elements) > 1:
            if composition_fractions:
                for elem, frac in composition_fractions.items():
                    conditions[v.X(elem.upper())] = float(frac)
            else:
                default_frac = 1.0 / len(elements)
                for elem in elements[:-1]:
                    conditions[v.X(elem.upper())] = default_frac

        eq_res = pycalphad_equilibrium(
            db,
            elements,
            phases,
            conditions,
        )

        eq_phases = [str(p) for p in eq_res.Phase.values.flatten().tolist() if p and str(p) != ""]
        gm_val = float(eq_res.GM.values.flatten()[0])

        return {
            "stable_phases": list(set(eq_phases)),
            "gibbs_energy_j_mol": gm_val,
            "temperature_k": temperature_k,
            "backend": "pycalphad",
        }


class TopologyAdapter:
    """Delegates simplicial persistent homology directly to gudhi."""

    @classmethod
    def compute_persistent_betti_numbers(
        cls,
        point_cloud: np.ndarray,
        max_edge_length: float = 4.5,
    ) -> Dict[str, int]:
        """Compute topological invariants Betti-0, Betti-1, Betti-2 using gudhi RipsComplex."""
        pts = np.asarray(point_cloud, dtype=np.float64)

        rips = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        betti = st.betti_numbers()

        return {
            "betti_0": int(betti[0]) if len(betti) > 0 else 1,
            "betti_1": int(betti[1]) if len(betti) > 1 else 0,
            "betti_2": int(betti[2]) if len(betti) > 2 else 0,
            "backend": "gudhi",
        }


class ElasticityAdapter:
    """Delegates elastic tensor analysis, Born stability, and polycrystalline averages directly to pymatgen."""

    @classmethod
    def analyze_elastic_tensor_6x6(cls, c_matrix_gpa: np.ndarray) -> Dict[str, Any]:
        """Analyze 6x6 Voigt elastic stiffness matrix for Born stability and isotropic moduli using pymatgen."""
        c_mat = np.asarray(c_matrix_gpa, dtype=np.float64)
        if c_mat.shape != (6, 6):
            raise ValueError(f"Elastic stiffness matrix must be 6x6, got shape {c_mat.shape}")

        tensor = PmgElasticTensor.from_voigt(c_mat)
        k_vrh = float(tensor.k_vrh)
        g_vrh = float(tensor.g_vrh)
        e_vrh = float(tensor.y_mod)
        nu = float(tensor.homogeneous_poisson)
        is_born = bool(np.all(np.linalg.eigvalsh(tensor.voigt) > 0.0))
        cauchy_p = float(c_mat[1, 2] - c_mat[3, 3])

        return {
            "bulk_modulus_gpa": k_vrh,
            "shear_modulus_gpa": g_vrh,
            "youngs_modulus_gpa": e_vrh,
            "poissons_ratio": nu,
            "pugh_ratio_k_g": k_vrh / max(1e-3, g_vrh),
            "cauchy_pressure_gpa": cauchy_p,
            "born_stable": is_born,
            "backend": "pymatgen",
        }
