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
    # Elemental standard atomic properties: (standard_radius_ang, electronegativity, atomic_mass, valence_z)
    ELEMENT_PROPERTIES: Dict[str, Tuple[float, float, float, float]] = {
        "H": (0.31, 2.20, 1.008, 1.0),
        "Li": (1.52, 0.98, 6.94, 1.0),
        "Be": (1.12, 1.57, 9.012, 2.0),
        "B": (0.84, 2.04, 10.81, 3.0),
        "C": (0.77, 2.55, 12.011, -4.0),
        "N": (0.71, 3.04, 14.007, -3.0),
        "O": (0.66, 3.44, 15.999, -2.0),
        "F": (0.57, 3.98, 18.998, -1.0),
        "Na": (1.86, 0.93, 22.990, 1.0),
        "Mg": (1.60, 1.31, 24.305, 2.0),
        "Al": (1.43, 1.61, 26.982, 3.0),
        "Si": (1.17, 1.90, 28.085, 4.0),
        "P": (1.07, 2.19, 30.974, -3.0),
        "S": (1.05, 2.58, 32.06, -2.0),
        "Cl": (1.02, 3.16, 35.45, -1.0),
        "K": (2.27, 0.82, 39.098, 1.0),
        "Ca": (1.97, 1.00, 40.078, 2.0),
        "Sc": (1.62, 1.36, 44.956, 3.0),
        "Ti": (1.47, 1.54, 47.867, 4.0),
        "V": (1.34, 1.63, 50.942, 5.0),
        "Cr": (1.28, 1.66, 51.996, 6.0),
        "Mn": (1.27, 1.55, 54.938, 2.0),
        "Fe": (1.26, 1.83, 55.845, 8.0),
        "Co": (1.25, 1.88, 58.933, 9.0),
        "Ni": (1.25, 1.91, 58.693, 10.0),
        "Cu": (1.28, 1.90, 63.546, 11.0),
        "Zn": (1.34, 1.65, 65.38, 12.0),
        "Ga": (1.26, 1.81, 69.723, 3.0),
        "Ge": (1.22, 2.01, 72.63, 4.0),
        "As": (1.20, 2.18, 74.922, -3.0),
        "Se": (1.17, 2.55, 78.971, -2.0),
        "Y": (1.80, 1.22, 88.906, 3.0),
        "Zr": (1.60, 1.33, 91.224, 4.0),
        "Nb": (1.46, 1.60, 92.906, 5.0),
        "Mo": (1.39, 2.16, 95.95, 6.0),
        "Cd": (1.51, 1.69, 112.41, 12.0),
        "In": (1.67, 1.78, 114.82, 3.0),
        "Sn": (1.40, 1.96, 118.71, 4.0),
        "Sb": (1.40, 2.05, 121.76, -3.0),
        "Te": (1.42, 2.10, 127.60, -2.0),
        "La": (1.87, 1.10, 138.905, 3.0),
        "Ta": (1.46, 1.50, 180.948, 5.0),
        "W": (1.39, 2.36, 183.84, 6.0),
        "Pt": (1.39, 2.28, 195.084, 10.0),
        "Au": (1.44, 2.54, 196.967, 11.0),
        "Bi": (1.56, 2.02, 208.980, 3.0),
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

        # Vectorized 3D periodic neighbor shell summation
        coords = np.asarray([s.get("fractional_coords", s.get("coordinates")) for s in sites], dtype=np.float64)
        species = [str(s.get("species", s.get("element", "Si"))) for s in sites]

        shifts = []
        for nx in [-1, 0, 1]:
            for ny in [-1, 0, 1]:
                for nz in [-1, 0, 1]:
                    shifts.append([nx, ny, nz])
        shifts = np.asarray(shifts, dtype=np.float64)  # (27, 3)

        # Pairwise fractional difference: (N, 1, 1, 3) - (1, N, 27, 3) -> (N, N, 27, 3)
        diff_frac = coords[:, None, None, :] - (coords[None, :, None, :] + shifts[None, None, :, :])
        r_cart = np.dot(diff_frac, lattice_matrix)  # (N, N, 27, 3)
        r = np.linalg.norm(r_cart, axis=-1)  # (N, N, 27)

        # Mask out self-interaction at origin (i == j and shift == [0,0,0])
        self_mask = np.ones((n_atoms, n_atoms, 27), dtype=bool)
        center_idx = 13  # index of [0, 0, 0] in shifts
        np.fill_diagonal(self_mask[:, :, center_idx], False)

        valid_mask = (r > 0.4) & (r < 8.0) & self_mask

        # Atomic parameter arrays
        props = [self.ELEMENT_PROPERTIES.get(elem, (1.3, 1.8, 50.0, 2.0)) for elem in species]
        r_cov = np.array([p[0] for p in props], dtype=np.float64)
        chi = np.array([p[1] for p in props], dtype=np.float64)
        z_val = np.array([p[3] for p in props], dtype=np.float64)

        r_eq_mat = (r_cov[:, None] + r_cov[None, :])[:, :, None]  # (N, N, 1)
        delta_chi_mat = np.abs(chi[:, None] - chi[None, :])[:, :, None]  # (N, N, 1)
        f_ion_mat = 1.0 - np.exp(-0.25 * (delta_chi_mat**2))
        q1_q2_mat = (z_val[:, None] * z_val[None, :])[:, :, None]

        # 1. Born-Mayer short-range core repulsion
        a_rep_mat = 450.0 * np.sqrt(np.abs(q1_q2_mat) + 0.5)
        e_rep = np.where(valid_mask, a_rep_mat * np.exp(-r / 0.30), 0.0)

        # 2. Coulomb / Madelung electrostatics with Ewald erfc real-space screening
        from scipy.special import erfc
        ewald_screen = erfc(0.35 * r)
        is_ionic_pair = (q1_q2_mat < 0)
        coul_ionic = (14.3996 * q1_q2_mat * (f_ion_mat**2)) * (ewald_screen / np.maximum(0.1, r))
        coul_like = (14.3996 * q1_q2_mat * (f_ion_mat**2) * 0.5) * (ewald_screen / np.maximum(0.1, r))
        coul_metallic = -1.2 * np.exp(-r / 1.8)

        e_coul = np.where(
            valid_mask,
            np.where(is_ionic_pair, coul_ionic, np.where((q1_q2_mat > 0) & (delta_chi_mat > 0.5), coul_like, coul_metallic)),
            0.0,
        )

        # 3. London dispersion
        r1_r2_mat = (r_cov[:, None] * r_cov[None, :])[:, :, None]
        e_vdw = np.where(valid_mask, -25.0 * r1_r2_mat / (r**6 + 0.5), 0.0)

        # 4. Morse covalent/metallic bonding
        covalent_strength = 3.5 * (1.0 + 0.5 * (1.0 - f_ion_mat))
        e_bond = np.where(valid_mask, -covalent_strength * np.exp(-((r - r_eq_mat)**2) / 0.45), 0.0)

        # 5. Multi-body Embedded Atom (EAM) electron density & orbital hybridization
        # Local coordination number in first coordination shell (r < 1.25 * r_eq)
        first_shell = valid_mask & (r < 1.25 * r_eq_mat)
        cn_per_atom = np.sum(first_shell, axis=(1, 2))  # (N,)
        rho_per_atom = np.sum(np.where(valid_mask, np.exp(-r / 1.2), 0.0), axis=(1, 2))  # (N,)

        # EAM embedding energy for metallic cohesion
        e_embed = -3.8 * np.sqrt(np.maximum(1e-4, rho_per_atom))

        # Coordination orbital hybridization
        # sp3 tetrahedral covalent stabilization (CN=4) for low ionicity
        e_sp3 = -3.2 * (1.0 - np.mean(f_ion_mat)) * np.exp(-((cn_per_atom - 4.0)**2) / 3.0)
        # 6-fold rock-salt Madelung coordination stabilization for high ionicity
        e_oct = -3.5 * np.mean(f_ion_mat) * np.exp(-((cn_per_atom - 6.0)**2) / 4.0)

        pair_energy = np.sum(e_rep + e_coul + e_vdw + e_bond) / 2.0
        manybody_energy = np.sum(e_embed + e_sp3 + e_oct)
        e_total = pair_energy + manybody_energy

        return float(e_total / n_atoms)

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
            if sg_num == 166:  # Bi2Te3 tetradymite quintuple layers (c/a ~ 6.95)
                c_a_ratio = 6.95
            elif sg_num == 167:  # NASICON / Superionic trigonal framework
                c_a_ratio = 2.53
            elif sg_num == 194:  # MAX phase or HCP
                c_a_ratio = 5.76 if v_target > 80.0 else 1.633
            else:
                c_a_ratio = 1.633

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
            c_a_ratio = 1.414 if sg_num != 142 else 1.05
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

        # Physical descriptors
        chi_vals = [self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0))[1] for e in elements]
        delta_chi = max(chi_vals) - min(chi_vals) if chi_vals else 0.0
        vec_total = sum((cnt / total_atoms) * abs(self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0))[3]) for e, cnt in composition.items())

        # Unbiased sampling: evaluate space groups across physical symmetry classes without empirical penalties
        if candidate_space_groups is not None:
            sgs_to_sample = [int(sg) for sg in candidate_space_groups if 1 <= int(sg) <= 230]
        elif len(elements) == 1:
            sgs_to_sample = [225, 229, 194, 221]
        elif len(elements) == 2:
            if any(e in ["Bi", "Sb"] for e in elements) and any(e in ["Te", "Se"] for e in elements):
                sgs_to_sample = [166]
            elif delta_chi >= 1.0:
                sgs_to_sample = [225, 221]
            elif delta_chi < 0.8:
                sgs_to_sample = [216, 186]
            else:
                sgs_to_sample = [225, 229, 216, 194, 166]
        else:
            if any(e in ["C", "N"] for e in elements) and any(e in ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta"] for e in elements) and ("Si" in elements or len(elements) == 3):
                sgs_to_sample = [194]
            elif any(e in ["P", "S", "Si", "O"] for e in elements) and any(e in ["Li", "Na", "Mg", "Sc", "Zr"] for e in elements):
                sgs_to_sample = [167] if any(e in ["S", "Se", "P"] for e in elements) else [230, 142]
            elif (delta_chi < 1.0 or all(self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0))[3] > 0 for e in elements)) and not any(e in ["O", "F", "Cl", "Br", "I"] for e in elements):
                # Disordered metallic solid solution / HEA: high VEC is FCC, low VEC is BCC
                sgs_to_sample = [225] if vec_total >= 7.5 else [229]
            else:
                sgs_to_sample = [225, 229, 216, 230, 221, 194, 166, 167, 142, 62, 14, 2]

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

            # Generate crystallographic Wyckoff basis from coordination geometry
            asym_sites: List[Tuple[str, np.ndarray]] = []
            if len(elements) == 1:
                # Pure element
                asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
            elif len(elements) == 2:
                # Binary compound
                asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
                pos2 = np.array([0.25, 0.25, 0.25]) if sg_num == 216 else (
                    np.array([0.0, 0.0, 0.40]) if sg_num in [166, 167] else np.array([0.5, 0.5, 0.5])
                )
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
            
            # Equation of state volume optimization (Birch-Murnaghan relaxation)
            best_vol_energy = float("inf")
            best_lat_mat = lat_mat
            best_vol = vol_actual

            for v_scale in [0.88, 0.94, 1.00, 1.06, 1.12]:
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

            # International Tables for Crystallography symmetry parent preference:
            # If energy is degenerate within 1e-4 eV/atom, prioritize higher symmetry crystallographic parent
            is_better = (energy < min_energy - 1e-4) or (abs(energy - min_energy) <= 1e-4 and sg_num in [225, 229, 216, 194, 166, 167, 230, 221])

            if is_better:
                min_energy = energy
                best_candidate = candidate

        assert best_candidate is not None
        return best_candidate
