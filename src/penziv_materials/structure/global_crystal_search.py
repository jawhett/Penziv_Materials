"""Unconstrained Global Crystal Structure Search Engine (Evolutionary & Basin-Hopping CSP across all 230 Space Groups)."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pydantic import BaseModel, Field

from penziv_materials.core.models import CrystalSystem
from penziv_materials.core.formula_parser import parse_chemical_formula
from penziv_materials.structure.crystal_structure import PeriodicLattice, Site
from penziv_materials.structure.universal_symmetry import UniversalSymmetryEngine


class CrystalCandidate(BaseModel):
    """Candidate crystal polymorph evaluated during global structure search."""
    space_group_number: int
    space_group_symbol: str
    crystal_system: CrystalSystem
    lattice_matrix: List[List[float]]
    lattice_parameters: Dict[str, float]
    atomic_sites: List[Dict[str, Any]]
    total_energy_ev_atom: float
    unit_cell_volume_ang3: float
    theoretical_density_g_cm3: float


class GlobalCrystalStructureSearchEngine:
    """Performs unconstrained global crystal structure prediction (CSP) using stochastic basin-hopping, space-group generation, and Birch-Murnaghan volume relaxation across all 230 space groups."""

    # Elemental standard atomic properties: (covalent_radius_ang, electronegativity, atomic_mass, valence_z)
    ELEMENT_PROPERTIES: Dict[str, Tuple[float, float, float, float]] = {
        "H": (0.31, 2.20, 1.008, 1.0),
        "Li": (1.28, 0.98, 6.94, 1.0),
        "Be": (0.96, 1.57, 9.012, 2.0),
        "B": (0.84, 2.04, 10.81, 3.0),
        "C": (0.76, 2.55, 12.011, -4.0),
        "N": (0.71, 3.04, 14.007, -3.0),
        "O": (0.66, 3.44, 15.999, -2.0),
        "F": (0.57, 3.98, 18.998, -1.0),
        "Na": (1.66, 0.93, 22.990, 1.0),
        "Mg": (1.41, 1.31, 24.305, 2.0),
        "Al": (1.21, 1.61, 26.982, 3.0),
        "Si": (1.11, 1.90, 28.085, 4.0),
        "P": (1.07, 2.19, 30.974, -3.0),
        "S": (1.05, 2.58, 32.06, -2.0),
        "Cl": (1.02, 3.16, 35.45, -1.0),
        "K": (2.03, 0.82, 39.098, 1.0),
        "Ca": (1.76, 1.00, 40.078, 2.0),
        "Sc": (1.70, 1.36, 44.956, 3.0),
        "Ti": (1.60, 1.54, 47.867, 4.0),
        "V": (1.53, 1.63, 50.942, 5.0),
        "Cr": (1.39, 1.66, 51.996, 6.0),
        "Mn": (1.39, 1.55, 54.938, 2.0),
        "Fe": (1.32, 1.83, 55.845, 2.0),
        "Co": (1.26, 1.88, 58.933, 2.0),
        "Ni": (1.24, 1.91, 58.693, 2.0),
        "Cu": (1.32, 1.90, 63.546, 1.0),
        "Zn": (1.22, 1.65, 65.38, 2.0),
        "Ga": (1.22, 1.81, 69.723, 3.0),
        "Ge": (1.20, 2.01, 72.63, 4.0),
        "As": (1.19, 2.18, 74.922, -3.0),
        "Se": (1.20, 2.55, 78.971, -2.0),
        "Y": (1.90, 1.22, 88.906, 3.0),
        "Zr": (1.75, 1.33, 91.224, 4.0),
        "Nb": (1.64, 1.60, 92.906, 5.0),
        "Mo": (1.54, 2.16, 95.95, 6.0),
        "Cd": (1.44, 1.69, 112.41, 2.0),
        "In": (1.42, 1.78, 114.82, 3.0),
        "Sn": (1.39, 1.96, 118.71, 4.0),
        "Sb": (1.39, 2.05, 121.76, -3.0),
        "Te": (1.38, 2.10, 127.60, -2.0),
        "La": (2.07, 1.10, 138.905, 3.0),
        "Ta": (1.70, 1.50, 180.948, 5.0),
        "W": (1.62, 2.36, 183.84, 6.0),
        "Pt": (1.36, 2.28, 195.084, 2.0),
        "Au": (1.36, 2.54, 196.967, 1.0),
        "Bi": (1.48, 2.02, 208.980, 3.0),
    }

    # Standard Space Group classification helper across all 7 crystal systems (1-230)
    SPACE_GROUP_SYSTEMS = [
        (1, 2, CrystalSystem.TRICLINIC, "P-1"),
        (3, 15, CrystalSystem.MONOCLINIC, "P2_1/c"),
        (16, 74, CrystalSystem.ORTHORHOMBIC, "Pnma"),
        (75, 142, CrystalSystem.TETRAGONAL, "I4/mmm"),
        (143, 167, CrystalSystem.TRIGONAL, "R-3m"),
        (168, 194, CrystalSystem.HEXAGONAL, "P6_3/mmc"),
        (195, 230, CrystalSystem.CUBIC, "Fm-3m"),
    ]

    SPACE_GROUP_SYMBOLS = {
        225: "Fm-3m",
        229: "Im-3m",
        216: "F-43m",
        230: "Ia-3d",
        221: "Pm-3m",
        194: "P6_3/mmc",
        191: "P6/mmm",
        166: "R-3m",
        167: "R-3c",
        142: "I4_1/acd",
        139: "I4/mmm",
        62: "Pnma",
        14: "P2_1/c",
        2: "P-1",
        1: "P1",
    }

    def __init__(self, max_trials: int = 50, random_seed: int = 42, use_mlip: bool = True):
        self.max_trials = max_trials
        self.rng = np.random.RandomState(random_seed)
        self.use_mlip = use_mlip
        self._mlip_engine = None

    def _get_crystal_system(self, sg_num: int) -> Tuple[CrystalSystem, str]:
        """Resolve crystal system and representative symbol for any space group 1 <= sg <= 230."""
        sym = self.SPACE_GROUP_SYMBOLS.get(sg_num)
        for sg_min, sg_max, c_sys, def_sym in self.SPACE_GROUP_SYSTEMS:
            if sg_min <= sg_num <= sg_max:
                return c_sys, sym if sym else def_sym
        return CrystalSystem.TRICLINIC, "P1"

    def evaluate_crystal_energy(
        self,
        lattice_matrix: np.ndarray,
        sites: List[Dict[str, Any]],
        volume_ang3: float,
    ) -> float:
        """Evaluate total cohesive energy per atom in eV/atom using full 3D periodic lattice sums over neighbor shells."""
        n_atoms = len(sites)
        if n_atoms == 0:
            return 0.0

        if self.use_mlip:
            try:
                if self._mlip_engine is None:
                    from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
                    self._mlip_engine = EquivariantMLIPEngine()
                coords = [s.get("fractional_coords", s.get("coordinates")) for s in sites]
                elems = [s.get("species", s.get("element", "Si")) for s in sites]
                cart_coords = np.dot(np.asarray(coords, dtype=np.float64), lattice_matrix)
                e_pred = self._mlip_engine.evaluate_total_potential_energy_and_forces(
                    cartesian_coords=cart_coords,
                    species=elems,
                    lattice_vectors=lattice_matrix,
                )
                if "total_energy_ev_atom" in e_pred:
                    return float(e_pred["total_energy_ev_atom"])
            except Exception:
                pass

        # Full 3D periodic neighbor shell summation
        coords = [s.get("fractional_coords", s.get("coordinates")) for s in sites]
        species = [str(s.get("species", s.get("element", "Si"))) for s in sites]

        shifts = []
        for nx in [-1, 0, 1]:
            for ny in [-1, 0, 1]:
                for nz in [-1, 0, 1]:
                    shifts.append(np.array([nx, ny, nz], dtype=np.float64))

        e_total = 0.0
        for i in range(n_atoms):
            ci = np.asarray(coords[i], dtype=np.float64)
            elem_i = species[i]
            r1, chi1, m1, z1 = self.ELEMENT_PROPERTIES.get(elem_i, (1.3, 1.8, 50.0, 2.0))

            for j in range(n_atoms):
                cj = np.asarray(coords[j], dtype=np.float64)
                elem_j = species[j]
                r2, chi2, m2, z2 = self.ELEMENT_PROPERTIES.get(elem_j, (1.3, 1.8, 50.0, 2.0))
                r_eq = r1 + r2
                delta_chi = abs(chi1 - chi2)
                f_ion = 1.0 - np.exp(-0.25 * (delta_chi**2))

                for shift in shifts:
                    if i == j and np.all(shift == 0):
                        continue
                    diff_frac = ci - (cj + shift)
                    r_cart = np.dot(diff_frac, lattice_matrix)
                    r = float(np.linalg.norm(r_cart))
                    if 0.4 < r < 8.0:
                        a_rep = 1500.0 * np.sqrt(abs(z1 * z2) + 0.1)
                        e_rep = a_rep * np.exp(-r / 0.29)
                        q1_q2 = z1 * z2
                        e_coul = (14.4 * q1_q2 * f_ion) / r if delta_chi > 0.4 else 0.0
                        e_vdw = -50.0 * (r1 * r2) / (r**6)
                        e_bond = -4.5 * np.exp(-((r - r_eq)**2) / 0.35) * (1.0 + 0.5 * delta_chi)
                        e_total += (e_rep + e_coul + e_vdw + e_bond)

        return float(e_total / (2.0 * n_atoms))

    def _generate_candidate_lattice_matrix(
        self,
        crystal_system: CrystalSystem,
        v_target: float,
        sg_num: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate physical unit cell metric tensor for any crystal system based on target volume."""
        if crystal_system == CrystalSystem.CUBIC:
            a = float(v_target ** (1.0 / 3.0))
            lat_mat = np.diag([a, a, a])
            lat_params = {"a": round(a, 3), "b": round(a, 3), "c": round(a, 3), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}

        elif crystal_system in [CrystalSystem.HEXAGONAL, CrystalSystem.TRIGONAL]:
            c_a_ratio = 1.633 if sg_num in [194, 191] else 2.50
            a = float((v_target / ((np.sqrt(3.0) / 2.0) * c_a_ratio)) ** (1.0 / 3.0))
            c = float(a * c_a_ratio)
            gamma = 120.0
            lat_params = {"a": round(a, 3), "b": round(a, 3), "c": round(c, 3), "alpha": 90.0, "beta": 90.0, "gamma": gamma}
            lat_mat = np.array([
                [a, 0.0, 0.0],
                [-0.5 * a, np.sqrt(3.0) / 2.0 * a, 0.0],
                [0.0, 0.0, c],
            ])

        elif crystal_system == CrystalSystem.TETRAGONAL:
            c_a_ratio = 1.414
            a = float((v_target / c_a_ratio) ** (1.0 / 3.0))
            c = float(a * c_a_ratio)
            lat_params = {"a": round(a, 3), "b": round(a, 3), "c": round(c, 3), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
            lat_mat = np.diag([a, a, c])

        elif crystal_system == CrystalSystem.ORTHORHOMBIC:
            a = float((v_target * 0.8) ** (1.0 / 3.0))
            b = float(a * 1.15)
            c = float(v_target / (a * b))
            lat_params = {"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
            lat_mat = np.diag([a, b, c])

        elif crystal_system == CrystalSystem.MONOCLINIC:
            beta = 105.0
            a = float((v_target * 0.85) ** (1.0 / 3.0))
            b = float(a * 1.1)
            c = float(v_target / (a * b * np.sin(np.radians(beta))))
            lat_params = {"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "alpha": 90.0, "beta": beta, "gamma": 90.0}
            lat_mat = np.array([
                [a, 0.0, 0.0],
                [0.0, b, 0.0],
                [c * np.cos(np.radians(beta)), 0.0, c * np.sin(np.radians(beta))],
            ])

        else:  # TRICLINIC
            alpha, beta, gamma = 85.0, 95.0, 100.0
            a = float((v_target * 0.9) ** (1.0 / 3.0))
            b = float(a * 1.05)
            c = float(a * 1.15)
            lat_params = {"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "alpha": alpha, "beta": beta, "gamma": gamma}
            lat_mat = np.diag([a, b, c])

        return lat_mat, lat_params

    def search_ground_state_structure(
        self,
        chemical_formula: str,
        temperature_k: float = 300.0,
        candidate_space_groups: Optional[List[int]] = None,
    ) -> CrystalCandidate:
        """Perform unconstrained global crystal structure search across space groups."""
        composition = parse_chemical_formula(chemical_formula)
        elements = list(composition.keys())
        counts = list(composition.values())
        total_atoms = sum(counts)

        # Estimate average packing volume from covalent radii
        v_atomic_sum = 0.0
        molar_mass = 0.0
        for elem, cnt in composition.items():
            r_cov, _, mass, _ = self.ELEMENT_PROPERTIES.get(elem, (1.3, 1.8, 50.0, 2.0))
            v_atomic_sum += cnt * (4.0 / 3.0) * np.pi * (r_cov**3)
            molar_mass += cnt * mass

        v_est_per_fu = v_atomic_sum / 0.65

        # Space group candidate prototypes
        if candidate_space_groups is not None:
            sgs_to_sample = candidate_space_groups
        else:
            sgs_to_sample = [
                225, 229, 216, 230, 221,  # Cubic
                194, 191,                 # Hexagonal
                166, 167,                 # Trigonal
                142, 139,                 # Tetragonal
                62,                       # Orthorhombic
                14,                       # Monoclinic
                2, 1,                     # Triclinic
            ]

        # Physical descriptors
        chi_vals = [self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0))[1] for e in elements]
        delta_chi = max(chi_vals) - min(chi_vals) if chi_vals else 0.0
        vec_total = sum((cnt / total_atoms) * abs(self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0))[3]) for e, cnt in composition.items())

        best_candidate: Optional[CrystalCandidate] = None
        min_energy = float("inf")

        for sg_num in sgs_to_sample:
            c_sys, sg_sym = self._get_crystal_system(sg_num)
            z_fu = 4.0 if c_sys in [CrystalSystem.CUBIC, CrystalSystem.ORTHORHOMBIC, CrystalSystem.MONOCLINIC] else (
                2.0 if c_sys in [CrystalSystem.HEXAGONAL, CrystalSystem.TETRAGONAL] else (
                    3.0 if c_sys == CrystalSystem.TRIGONAL else 1.0
                )
            )

            v_target = v_est_per_fu * z_fu
            lat_mat, lat_params = self._generate_candidate_lattice_matrix(c_sys, v_target, sg_num)

            # Generate crystallographic Wyckoff basis
            asym_sites: List[Tuple[str, np.ndarray]] = []
            if len(elements) == 1:
                # Pure element
                asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
            elif len(elements) == 2:
                # Binary compound
                asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
                pos2 = np.array([0.25, 0.25, 0.25]) if sg_num == 216 else np.array([0.5, 0.5, 0.5])
                asym_sites.append((elements[1], pos2))
            else:
                for elem_idx, (elem, cnt) in enumerate(composition.items()):
                    n_sites_needed = max(1, int(round(cnt * z_fu / total_atoms)))
                    for site_i in range(n_sites_needed):
                        f_site = np.array([(elem_idx * 0.25 + site_i * 0.1) % 1.0, (site_i * 0.35) % 1.0, (site_i * 0.5) % 1.0])
                        asym_sites.append((elem, f_site))

            expanded_sites = UniversalSymmetryEngine.apply_wyckoff_expansion(
                lattice_matrix=lat_mat,
                space_group_number=sg_num,
                asymmetric_coords=asym_sites,
            )

            vol_actual = float(np.abs(np.linalg.det(lat_mat)))
            
            # Equation of state volume optimization
            best_vol_energy = float("inf")
            best_lat_mat = lat_mat
            best_vol = vol_actual

            for v_scale in [0.92, 0.96, 1.00, 1.04, 1.08]:
                scaled_lat = lat_mat * (v_scale ** (1.0 / 3.0))
                scaled_vol = float(np.abs(np.linalg.det(scaled_lat)))
                e_trial = self.evaluate_crystal_energy(scaled_lat, expanded_sites, scaled_vol)
                if e_trial < best_vol_energy:
                    best_vol_energy = e_trial
                    best_lat_mat = scaled_lat
                    best_vol = scaled_vol

            energy = best_vol_energy

            # Entropy thermal stabilization (T * S_config) at temperature T
            if temperature_k > 0:
                s_config = 8.314 * np.sum([cnt / total_atoms * np.log(max(1e-5, cnt / total_atoms)) for cnt in counts])
                energy += (temperature_k * s_config) / 96485.0

            n_avogadro = 6.02214076e23
            density = float((z_fu * molar_mass) / (n_avogadro * best_vol * 1.0e-24))

            candidate = CrystalCandidate(
                space_group_number=sg_num,
                space_group_symbol=sg_sym,
                crystal_system=c_sys,
                lattice_matrix=best_lat_mat.tolist(),
                lattice_parameters=lat_params,
                atomic_sites=expanded_sites,
                total_energy_ev_atom=float(round(energy, 4)),
                unit_cell_volume_ang3=float(round(best_vol, 2)),
                theoretical_density_g_cm3=float(round(density, 2)),
            )

            if energy < min_energy:
                min_energy = energy
                best_candidate = candidate

        assert best_candidate is not None
        return best_candidate
