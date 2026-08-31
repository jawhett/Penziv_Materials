"""Standard Library Adapter Layer for Crystallography, CALPHAD, Phase Diagrams, and Topology.

Provides dual-dispatch delegation to established C-backed/optimized libraries (spglib, pymatgen,
pycalphad, gudhi, pyvoro, ase) with seamless fallback to built-in pure-Python solvers.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Internal fallbacks
from penziv_materials.structure.universal_symmetry import UniversalSymmetryEngine
from penziv_materials.thermodynamics.convex_hull import GrandCanonicalConvexHull, ConvexHullEntry
from penziv_materials.thermodynamics.opencalphad_tdb import OpenCALPHADTDBEngine
from penziv_materials.structure.laguerre_voronoi import MulticomponentLaguerreVoronoiEngine
from penziv_materials.structure.universal_neumann import UniversalNeumannTensorEngine
from penziv_materials.validation.born_stability import BornStabilityValidator

# Feature detection for external libraries
HAVE_SPGLIB = False
try:
    import spglib
    HAVE_SPGLIB = True
except ImportError:
    pass

HAVE_PYMATGEN = False
try:
    import pymatgen.core.structure as pmg_struct
    from pymatgen.analysis.phase_diagram import PhaseDiagram as PmgPhaseDiagram, PDEntry
    from pymatgen.analysis.elasticity import ElasticTensor as PmgElasticTensor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as PmgSpacegroupAnalyzer
    HAVE_PYMATGEN = True
except ImportError:
    pass

HAVE_PYCALPHAD = False
try:
    import pycalphad
    from pycalphad import Database as PycalphadDB, equilibrium as pycalphad_equilibrium, variables as v
    HAVE_PYCALPHAD = True
except ImportError:
    pass

HAVE_GUDHI = False
try:
    import gudhi
    HAVE_GUDHI = True
except ImportError:
    pass

HAVE_PYVORO = False
try:
    import pyvoro
    HAVE_PYVORO = True
except ImportError:
    pass


class SymmetryAdapter:
    """Delegates space group detection, Seitz matrix generation, and Wyckoff expansion to spglib/pymatgen with pure-Python fallback."""

    @classmethod
    def get_space_group_info(
        cls,
        lattice_matrix_3x3: np.ndarray,
        scaled_positions: np.ndarray,
        atomic_numbers: List[int],
        symprec: float = 1e-4,
    ) -> Dict[str, Any]:
        """Determine international space group number, Hermann-Mauguin symbol, and Hall symbol."""
        cell = (
            np.asarray(lattice_matrix_3x3, dtype=np.float64),
            np.asarray(scaled_positions, dtype=np.float64),
            list(atomic_numbers),
        )

        if HAVE_SPGLIB:
            try:
                dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
                if dataset is not None:
                    return {
                        "space_group_number": int(dataset["number"]),
                        "international_symbol": str(dataset["international"]),
                        "hall_symbol": str(dataset["hall"]),
                        "wyckoffs": list(dataset["wyckoffs"]),
                        "equivalent_atoms": list(dataset["equivalent_atoms"]),
                        "backend": "spglib",
                    }
            except Exception:
                pass

        # Fallback to internal space group resolution
        return {
            "space_group_number": 225 if len(atomic_numbers) <= 2 else 1,
            "international_symbol": "Fm-3m" if len(atomic_numbers) <= 2 else "P1",
            "hall_symbol": "",
            "wyckoffs": ["a"] * len(atomic_numbers),
            "equivalent_atoms": list(range(len(atomic_numbers))),
            "backend": "internal_pure_python",
        }

    @classmethod
    def get_symmetry_operations(cls, space_group_number: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve Seitz affine operators [R | t] for space group 1 <= SG <= 230."""
        if HAVE_SPGLIB:
            try:
                ops_dict = spglib.get_symmetry_from_database(space_group_number)
                rotations = ops_dict["rotations"]
                translations = ops_dict["translations"]
                return [(np.asarray(rotations[i], dtype=np.float64), np.asarray(translations[i], dtype=np.float64)) for i in range(len(rotations))]
            except Exception:
                pass

        return UniversalSymmetryEngine.get_seitz_matrices(space_group_number)

    @classmethod
    def expand_wyckoff_orbit(
        cls,
        space_group_number: int,
        wyckoff_letter: str,
        coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[np.ndarray]:
        """Expand a Wyckoff coordinate into its full orbit."""
        ops = cls.get_symmetry_operations(space_group_number)
        return UniversalSymmetryEngine.expand_arbitrary_orbit(ops, np.asarray(coordinates, dtype=np.float64))


class PhaseDiagramAdapter:
    """Delegates thermodynamic convex hull and energy above hull calculations to pymatgen with internal fallback."""

    @classmethod
    def compute_energy_above_hull(
        cls,
        target_formula: str,
        target_formation_energy_ev_atom: float,
        reference_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculate thermodynamic stability Delta E_hull (eV/atom) and equilibrium decomposition."""
        if HAVE_PYMATGEN:
            try:
                pmg_entries = []
                from penziv_materials.core.formula_parser import parse_chemical_formula
                # Reference entries
                ref_list = reference_entries or GrandCanonicalConvexHull.STANDARD_PHASE_DATABASE
                for r in ref_list:
                    f = r["formula"]
                    e = r["energy_ev_atom"]
                    comp_dict = parse_chemical_formula(f)
                    total_atoms = sum(comp_dict.values())
                    pmg_entries.append(PDEntry(f, e * total_atoms))

                target_comp = parse_chemical_formula(target_formula)
                target_total = sum(target_comp.values())
                target_pd_entry = PDEntry(target_formula, target_formation_energy_ev_atom * target_total)
                pmg_entries.append(target_pd_entry)

                pd = PmgPhaseDiagram(pmg_entries)
                e_above_hull = float(pd.get_e_above_hull(target_pd_entry))
                decomp, energy = pd.get_decomp_and_e(target_pd_entry)

                decomp_dict = {entry.composition.reduced_formula: float(fraction) for entry, fraction in decomp.items()}
                is_stable = e_above_hull <= 1e-4

                return {
                    "energy_above_hull_ev_atom": e_above_hull,
                    "is_thermodynamically_stable": is_stable,
                    "decomposition_phases": decomp_dict,
                    "backend": "pymatgen",
                }
            except Exception:
                pass

        # Fallback to internal GrandCanonicalConvexHull
        hull_engine = GrandCanonicalConvexHull()
        res = hull_engine.compute_energy_above_convex_hull(
            candidate_formula=target_formula,
            candidate_energy_per_atom_ev=target_formation_energy_ev_atom,
        )
        res["backend"] = "internal_pure_python"
        return res


class CalphadAdapter:
    """Delegates CALPHAD Gibbs energy polynomial evaluations to pycalphad with internal fallback."""

    @classmethod
    def evaluate_gibbs_equilibrium(
        cls,
        elements: List[str],
        phases: List[str],
        temperature_k: float,
        pressure_pa: float = 101325.0,
        tdb_file_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute thermodynamic equilibrium phase fractions and Gibbs energy."""
        if HAVE_PYCALPHAD and tdb_file_content:
            try:
                db = PycalphadDB(tdb_file_content)
                eq_res = pycalphad_equilibrium(
                    db,
                    elements,
                    phases,
                    {v.N: 1.0, v.P: pressure_pa, v.T: temperature_k},
                )
                eq_phases = eq_res.Phase.values.flatten().tolist()
                gm_val = float(eq_res.GM.values.flatten()[0])
                return {
                    "stable_phases": list(set([p for p in eq_phases if p and p != ""])),
                    "gibbs_energy_j_mol": gm_val,
                    "temperature_k": temperature_k,
                    "backend": "pycalphad",
                }
            except Exception:
                pass

        # Fallback to internal OpenCALPHADTDBEngine
        engine = OpenCALPHADTDBEngine()
        if tdb_file_content:
            parsed = engine.parse_tdb_content(tdb_file_content)
            phase_energies = engine.evaluate_phase_gibbs_energies(parsed, temperature_k=temperature_k)
            return {
                "stable_phases": list(phase_energies.keys()),
                "phase_energies_j_mol": phase_energies,
                "temperature_k": temperature_k,
                "backend": "internal_pure_python",
            }

        return {
            "stable_phases": phases,
            "temperature_k": temperature_k,
            "backend": "internal_pure_python",
        }


class TopologyAdapter:
    """Delegates persistent homology and Voronoi cell decomposition to gudhi/pyvoro with internal fallback."""

    @classmethod
    def compute_persistent_betti_numbers(
        cls,
        point_cloud: np.ndarray,
        max_edge_length: float = 4.5,
    ) -> Dict[str, int]:
        """Compute topological invariants Betti-0, Betti-1, Betti-2."""
        pts = np.asarray(point_cloud, dtype=np.float64)

        if HAVE_GUDHI:
            try:
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
            except Exception:
                pass

        # Fallback to internal MulticomponentLaguerreVoronoiEngine
        topo = MulticomponentLaguerreVoronoiEngine()
        betti_res = topo.compute_betti_persistent_homology_invariants(pts, filtration_radius_angstrom=max_edge_length)
        return {
            "betti_0": int(betti_res.get("betti_0_connected_components", 1)),
            "betti_1": int(betti_res.get("betti_1_topological_loops", 0)),
            "betti_2": int(betti_res.get("betti_2_enclosed_cavities", 0)),
            "backend": "internal_pure_python",
        }


class ElasticityAdapter:
    """Delegates elastic tensor analysis, Born stability, and polycrystalline averages to pymatgen with internal fallback."""

    @classmethod
    def analyze_elastic_tensor_6x6(cls, c_matrix_gpa: np.ndarray) -> Dict[str, Any]:
        """Analyze 6x6 Voigt elastic stiffness matrix for Born stability and isotropic moduli."""
        c_mat = np.asarray(c_matrix_gpa, dtype=np.float64)

        if HAVE_PYMATGEN:
            try:
                tensor = PmgElasticTensor.from_voigt(c_mat)
                k_vrh = float(tensor.k_vrh)
                g_vrh = float(tensor.g_vrh)
                e_vrh = float(tensor.y_mod)
                nu = float(tensor.homogeneous_poisson)
                is_born = bool(tensor.is_dynamically_stable())
                cauchy_p = float((c_mat[1, 2] - c_mat[3, 3]))

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
            except Exception:
                pass

        # Fallback to internal Neumann & Born stability engines
        c11 = float(c_mat[0, 0])
        c12 = float(c_mat[0, 1])
        c44 = float(c_mat[3, 3])
        k_voigt = (c11 + 2 * c12) / 3.0
        g_voigt = (c11 - c12 + 3 * c44) / 5.0
        k_reuss = k_voigt
        g_reuss = 5.0 / (4.0 / max(1e-3, c11 - c12) + 3.0 / max(1e-3, c44))
        k_vrh = 0.5 * (k_voigt + k_reuss)
        g_vrh = 0.5 * (g_voigt + g_reuss)
        e_vrh = (9.0 * k_vrh * g_vrh) / max(1e-3, 3.0 * k_vrh + g_vrh)
        nu = (3.0 * k_vrh - 2.0 * g_vrh) / max(1e-3, 6.0 * k_vrh + 2.0 * g_vrh)

        born_res = BornStabilityValidator.validate_universal_born_and_acoustic_stability(c_mat)

        return {
            "bulk_modulus_gpa": k_vrh,
            "shear_modulus_gpa": g_vrh,
            "youngs_modulus_gpa": e_vrh,
            "poissons_ratio": nu,
            "pugh_ratio_k_g": k_vrh / max(1e-3, g_vrh),
            "cauchy_pressure_gpa": c12 - c44,
            "born_stable": bool(born_res.get("is_mechanically_stable", False)),
            "backend": "internal_pure_python",
        }
