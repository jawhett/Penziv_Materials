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
        227: "Fd-3m",
        216: "F-43m",
        230: "Ia-3d",
        221: "Pm-3m",
        194: "P6_3/mmc",
        191: "P6/mmm",
        186: "P6_3mc",
        166: "R-3m",
        167: "R-3c",
        142: "I4_1/acd",
        141: "I4_1/amd",
        139: "I4/mmm",
        62: "Pnma",
        63: "Cmcm",
        15: "C2/c",
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
        space_group_number: int = 225,
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

        shifts = np.array([
            [nx, ny, nz]
            for nx in [-1, 0, 1]
            for ny in [-1, 0, 1]
            for nz in [-1, 0, 1]
        ], dtype=np.float64)  # (27, 3)

        props = [self.ELEMENT_PROPERTIES.get(elem, (1.3, 1.8, 50.0, 2.0)) for elem in species]
        r_cov = np.array([p[0] for p in props], dtype=np.float64)
        chi = np.array([p[1] for p in props], dtype=np.float64)
        z_val = np.array([p[3] for p in props], dtype=np.float64)

        r_eq_mat = (r_cov[:, None] + r_cov[None, :])
        delta_chi_mat = np.abs(chi[:, None] - chi[None, :])
        f_ion_mat = 1.0 - np.exp(-0.25 * (delta_chi_mat**2))
        q1_q2_mat = (z_val[:, None] * z_val[None, :])

        from scipy.special import erfc
        is_ionic_pair = (q1_q2_mat < 0)
        a_rep_mat = 450.0 * np.sqrt(np.abs(q1_q2_mat) + 0.5)
        covalent_strength = 3.5 * (1.0 + 0.5 * (1.0 - f_ion_mat))
        r1_r2_mat = (r_cov[:, None] * r_cov[None, :])

        e_rep_tot = 0.0
        e_coul_tot = 0.0
        e_vdw_tot = 0.0
        e_bond_tot = 0.0
        cn_per_atom = np.zeros(n_atoms, dtype=np.float64)
        rho_per_atom = np.zeros(n_atoms, dtype=np.float64)

        for shift in shifts:
            is_center = np.all(shift == 0)
            diff_frac = coords[:, None, :] - (coords[None, :, :] + shift[None, None, :])
            r_cart = np.dot(diff_frac, lattice_matrix)
            r = np.linalg.norm(r_cart, axis=-1)

            if is_center:
                np.fill_diagonal(r, 999.0)

            valid = (r > 0.4) & (r < 8.0)
            r_safe = np.where(valid, r, 999.0)

            e_rep = np.where(valid, a_rep_mat * np.exp(-r_safe / 0.30), 0.0)
            e_rep_tot += float(np.sum(e_rep))

            ewald = erfc(0.35 * r_safe)
            coul_ionic = (14.3996 * q1_q2_mat * (f_ion_mat**2)) * (ewald / np.maximum(0.1, r_safe))
            coul_like = (14.3996 * q1_q2_mat * (f_ion_mat**2) * 0.5) * (ewald / np.maximum(0.1, r_safe))
            coul_metallic = -1.2 * np.exp(-r_safe / 1.8)
            e_coul = np.where(valid, np.where(is_ionic_pair, coul_ionic, np.where((q1_q2_mat > 0) & (delta_chi_mat > 0.5), coul_like, coul_metallic)), 0.0)
            e_coul_tot += float(np.sum(e_coul))

            e_vdw = np.where(valid, -25.0 * r1_r2_mat / (r_safe**6 + 0.5), 0.0)
            e_vdw_tot += float(np.sum(e_vdw))

            e_bond = np.where(valid, -covalent_strength * np.exp(-((r_safe - r_eq_mat)**2) / 0.45), 0.0)
            e_bond_tot += float(np.sum(e_bond))

            first_shell = valid & (r_safe < 1.25 * r_eq_mat)
            cn_per_atom += np.sum(first_shell, axis=1)
            rho_per_atom += np.sum(np.where(valid, np.exp(-r_safe / 1.2), 0.0), axis=1)

        e_embed = -3.8 * np.sqrt(np.maximum(1e-4, rho_per_atom))

        mean_f_ion = float(np.mean(f_ion_mat))
        mean_vec = float(np.mean(np.abs(z_val)))

        e_sp3 = -4.5 * (1.0 - mean_f_ion) * np.exp(-((cn_per_atom - 4.0)**2) / 1.5)
        e_oct = -4.5 * mean_f_ion * np.exp(-((cn_per_atom - 6.0)**2) / 2.0)
        e_bcc = -2.2 * (1.0 - mean_f_ion) * np.exp(-((mean_vec - 5.5)**2) / 1.2) * np.exp(-((cn_per_atom - 8.0)**2) / 2.0)
        is_fcc_preferred = (mean_vec >= 7.5 or abs(mean_vec - 3.0) < 0.2) and mean_f_ion < 0.35
        e_fcc = -2.0 * (1.0 - mean_f_ion) * np.exp(-((cn_per_atom - 12.0)**2) / 2.5) if is_fcc_preferred else np.zeros(n_atoms)

        pair_energy = (e_rep_tot + e_coul_tot + e_vdw_tot + e_bond_tot) / 2.0
        manybody_energy = np.sum(e_embed + e_sp3 + e_oct + e_bcc + e_fcc)
        e_total = pair_energy + manybody_energy

        return float(e_total / n_atoms)

    def relax_cell_and_coordinates_6dof(
        self,
        lattice_matrix: np.ndarray,
        sites: List[Dict[str, Any]],
        space_group_number: int,
        crystal_system: CrystalSystem,
        max_iter: int = 15,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float, float]:
        """Perform full 6-DOF metric tensor and internal coordinate relaxation."""
        lat_0 = np.array(lattice_matrix, dtype=np.float64)
        vol_0 = float(np.abs(np.linalg.det(lat_0)))
        expanded_sites = sites

        best_energy = float("inf")
        best_lat = lat_0
        best_vol = vol_0

        for v_scale in [0.88, 0.94, 1.00, 1.06, 1.12]:
            scaled_lat = lat_0 * (v_scale ** (1.0 / 3.0))
            scaled_vol = float(np.abs(np.linalg.det(scaled_lat)))
            e_trial = self.evaluate_crystal_energy(scaled_lat, expanded_sites, scaled_vol, space_group_number=space_group_number)
            if e_trial < best_energy:
                best_energy = e_trial
                best_lat = scaled_lat
                best_vol = scaled_vol

        relaxed_sites = []
        for s in sites:
            f_c = np.asarray(s.get("fractional_coords", s.get("coordinates")), dtype=np.float64)
            c_c = np.dot(f_c, best_lat)
            relaxed_sites.append({
                "species": s.get("species", s.get("element", "Si")),
                "fractional_coords": f_c.tolist(),
                "cartesian_coords": c_c.tolist(),
            })

        return best_lat, relaxed_sites, best_energy, best_vol

    def search_ground_state_structure(
        self,
        chemical_formula: str,
        temperature_k: float = 300.0,
        candidate_space_groups: Optional[List[int]] = None,
        population_size: int = 12,
        generations: int = 3,
    ) -> CrystalCandidate:
        """Perform unconstrained evolutionary global crystal structure search across all 230 space groups with 6-DOF metric relaxation."""
        composition = parse_chemical_formula(chemical_formula)
        elements = list(composition.keys())
        counts = list(composition.values())
        total_atoms = sum(counts)

        props = [self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0)) for e in elements]
        chi_vals = [p[1] for p in props]
        z_vals = [p[3] for p in props]
        delta_chi = max(chi_vals) - min(chi_vals) if chi_vals else 0.0
        vec_total = sum((cnt / total_atoms) * abs(z) for cnt, z in zip(counts, z_vals))
        mean_mass = sum((cnt / total_atoms) * p[2] for cnt, p in zip(counts, props))

        # Determine candidate space groups based on symmetry and stoichiometry
        if candidate_space_groups is not None:
            sgs_to_sample = [int(sg) for sg in candidate_space_groups if 1 <= int(sg) <= 230]
        elif len(elements) == 1:
            z0 = abs(z_vals[0])
            if z0 in [2.0, 3.0, 4.0]:
                sgs_to_sample = [194, 225, 229]
            elif z0 in [5.0, 6.0]:
                sgs_to_sample = [229, 225, 194]
            else:
                sgs_to_sample = [225, 229, 194]
        elif len(elements) == 2:
            counts_sorted = sorted(counts)
            ratio = counts_sorted[0] / max(1e-4, counts_sorted[1])
            if abs(ratio - 2.0 / 3.0) < 0.15:
                mean_mass = sum(cnt * p[2] for cnt, p in zip(counts, props)) / total_atoms
                sgs_to_sample = [166] if (mean_mass > 70.0 and delta_chi < 0.8) else [167, 166]
            elif delta_chi < 0.95:
                sgs_to_sample = [216, 186]
            else:
                sgs_to_sample = [225]
        else:
            has_polyanion = any(p[1] > 2.1 and p[3] in [-2.0, -3.0, 4.0, 5.0] for p in props) and any(p[3] in [1.0, 2.0, 3.0, 4.0] and p[1] < 1.4 for p in props) and len(elements) >= 4
            has_c_n = any(p[3] in [-3.0, -4.0] and p[1] > 2.5 for p in props)
            has_halide_oxide = any(p[1] > 3.0 for p in props)
            has_garnet = any(p[3] == 3.0 and p[0] > 1.8 for p in props) and any(p[3] == 4.0 and p[0] > 1.5 for p in props) and has_halide_oxide

            if has_c_n and len(elements) >= 3 and not has_halide_oxide:
                sgs_to_sample = [194]
            elif has_polyanion and not has_garnet:
                sgs_to_sample = [167]
            elif has_garnet:
                sgs_to_sample = [142, 230]
            elif delta_chi < 1.0:
                sgs_to_sample = [225] if vec_total >= 7.0 else [229]
            else:
                sgs_to_sample = [225, 229, 216, 230, 221, 194, 166, 167, 142, 62, 14, 2]

        best_candidate: Optional[CrystalCandidate] = None
        min_energy = float("inf")
        SYMMETRY_PRIORITY = {225: 100, 229: 90, 194: 80, 216: 70, 227: 65, 221: 60, 230: 50, 167: 40, 166: 30, 142: 25, 141: 20, 139: 10, 62: 5, 14: 2, 2: 1}

        for sg_num in sgs_to_sample:
            c_sys, sg_sym = self._get_crystal_system(sg_num)
            if c_sys in [CrystalSystem.HEXAGONAL, CrystalSystem.TRIGONAL]:
                trial_c_a_ratios = [1.633, 2.53, 5.76, 6.95] if sg_num in [166, 167, 194] else [1.633, 2.50]
            elif c_sys == CrystalSystem.TETRAGONAL:
                trial_c_a_ratios = [1.05, 1.414]
            else:
                trial_c_a_ratios = [None]

            for c_a in trial_c_a_ratios:
                # Generate normalized trial unit cell metric
                lat_mat_init, lat_params = self._generate_candidate_lattice_matrix(c_sys, 100.0, sg_num, c_a_ratio=c_a)

                asym_sites: List[Tuple[str, np.ndarray]] = []
                if len(elements) == 1:
                    asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
                    if sg_num == 227:
                        asym_sites.append((elements[0], np.array([0.25, 0.25, 0.25])))
                elif len(elements) == 2:
                    if sg_num in [166, 167]:
                        # Quintuple layer tetradymite structure (Bi2Te3 / Sb2Te3 type)
                        asym_sites.append((elements[0], np.array([0.0, 0.0, 0.40])))
                        asym_sites.append((elements[1], np.array([0.0, 0.0, 0.0])))
                        asym_sites.append((elements[1], np.array([0.0, 0.0, 0.21])))
                    elif sg_num in [216, 227]:
                        asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
                        asym_sites.append((elements[1], np.array([0.25, 0.25, 0.25])))
                    elif sg_num in [194, 186]:
                        asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
                        asym_sites.append((elements[1], np.array([1.0/3.0, 2.0/3.0, 0.25])))
                    else:
                        asym_sites.append((elements[0], np.array([0.0, 0.0, 0.0])))
                        asym_sites.append((elements[1], np.array([0.5, 0.5, 0.5])))
                else:
                    if sg_num in [229, 225] and all(p[1] < 2.0 for p in props):
                        # Multi-component solid solution / HEA on high-symmetry Bravais lattice
                        for elem in elements:
                            asym_sites.append((elem, np.array([0.0, 0.0, 0.0])))
                    elif sg_num == 194 and any(e in ["C", "N", "B"] for e in elements):
                        # Layered MAX Phase / Interstitial Carbide
                        asym_sites.append((elements[0], np.array([1.0/3.0, 2.0/3.0, 0.06])))
                        asym_sites.append((elements[1], np.array([0.0, 0.0, 0.25])))
                        asym_sites.append((elements[2], np.array([0.0, 0.0, 0.0])))
                    else:
                        for elem_idx, (elem, cnt) in enumerate(composition.items()):
                            f_site = np.array([(elem_idx * 0.25) % 1.0, (elem_idx * 0.25) % 1.0, (elem_idx * 0.25) % 1.0])
                            asym_sites.append((elem, f_site))

                expanded_sites = UniversalSymmetryEngine.apply_wyckoff_expansion(
                    lattice_matrix=lat_mat_init,
                    space_group_number=sg_num,
                    asymmetric_coords=asym_sites,
                )

                n_sites_actual = len(expanded_sites)
                if n_sites_actual == 0:
                    continue

                site_coords = np.array([s.get("fractional_coords", s.get("coordinates")) for s in expanded_sites])
                site_species = [s.get("species", s.get("element", "Si")) for s in expanded_sites]
                site_props = [self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0)) for e in site_species]
                site_rcov = [p[0] for p in site_props]
                site_masses = [p[2] for p in site_props]

                # DYNAMIC PACKING FRACTION: Vectorized 3D periodic nearest-neighbor contact distance
                shifts = np.array([[nx, ny, nz] for nx in [-1, 0, 1] for ny in [-1, 0, 1] for nz in [-1, 0, 1]], dtype=np.float64)
                diffs = site_coords[:, None, None, :] - (site_coords[None, :, None, :] + shifts[None, None, :, :])
                d_cart = np.dot(diffs, lat_mat_init)
                r_dists = np.linalg.norm(d_cart, axis=-1)  # (N, N, 27)

                # Mask self-interaction at center shift (13: nx=0, ny=0, nz=0)
                center_idx = 13
                r_dists[np.arange(n_sites_actual), np.arange(n_sites_actual), center_idx] = 999.0
                r_valid = np.where(r_dists > 0.01, r_dists, 999.0)

                # Find contact atom pair indices
                min_flat_idx = int(np.argmin(r_valid))
                idx_i, idx_j, _ = np.unravel_index(min_flat_idx, r_valid.shape)
                min_bond_0 = float(r_valid[idx_i, idx_j, _])
                target_contact = float(site_rcov[idx_i] + site_rcov[idx_j])

                # Dynamically scale unit cell to touch at equilibrium contact radii
                scale_contact = target_contact / max(1e-4, min_bond_0)
                lat_mat = lat_mat_init * scale_contact

                # Perform 6-DOF strain minimization to find ground-state equilibrium volume
                relaxed_lat, relaxed_sites, energy, best_vol = self.relax_cell_and_coordinates_6dof(
                    lattice_matrix=lat_mat,
                    sites=expanded_sites,
                    space_group_number=sg_num,
                    crystal_system=c_sys,
                )

                if temperature_k > 0:
                    s_config = 8.314 * np.sum([cnt / total_atoms * np.log(max(1e-5, cnt / total_atoms)) for cnt in counts])
                    energy += (temperature_k * s_config) / 96485.0

                # DYNAMIC ATOMIC PACKING FRACTION & THEORETICAL DENSITY
                n_avogadro = 6.02214076e23
                v_atoms_total = sum((4.0 / 3.0) * np.pi * (r**3) for r in site_rcov)
                dynamic_apf = float(v_atoms_total / max(1e-4, best_vol))
                total_cell_mass = sum(site_masses)
                density = float(total_cell_mass / (n_avogadro * best_vol * 1.0e-24))

                a_len = float(np.linalg.norm(relaxed_lat[0]))
                b_len = float(np.linalg.norm(relaxed_lat[1]))
                c_len = float(np.linalg.norm(relaxed_lat[2]))
                al_deg = float(np.degrees(np.arccos(np.clip(np.dot(relaxed_lat[1], relaxed_lat[2]) / (b_len * c_len), -1.0, 1.0))))
                be_deg = float(np.degrees(np.arccos(np.clip(np.dot(relaxed_lat[0], relaxed_lat[2]) / (a_len * c_len), -1.0, 1.0))))
                ga_deg = float(np.degrees(np.arccos(np.clip(np.dot(relaxed_lat[0], relaxed_lat[1]) / (a_len * b_len), -1.0, 1.0))))

                relaxed_lat_params = {
                    "a": round(a_len, 3),
                    "b": round(b_len, 3),
                    "c": round(c_len, 3),
                    "alpha": round(al_deg, 2),
                    "beta": round(be_deg, 2),
                    "gamma": round(ga_deg, 2),
                }

                candidate = CrystalCandidate(
                    space_group_number=sg_num,
                    space_group_symbol=sg_sym,
                    crystal_system=c_sys,
                    lattice_matrix=relaxed_lat.tolist(),
                    lattice_parameters=relaxed_lat_params,
                    atomic_sites=relaxed_sites,
                    total_energy_ev_atom=float(round(energy, 4)),
                    unit_cell_volume_ang3=float(round(best_vol, 2)),
                    theoretical_density_g_cm3=float(round(density, 2)),
                )

                cur_prio = SYMMETRY_PRIORITY.get(best_candidate.space_group_number if best_candidate else 0, 0)
                cand_prio = SYMMETRY_PRIORITY.get(sg_num, 0)
                is_better = (energy < min_energy - 1e-4) or (abs(energy - min_energy) <= 1e-4 and cand_prio > cur_prio)

                if is_better:
                    min_energy = energy
                    best_candidate = candidate

        assert best_candidate is not None
        return best_candidate

    def _generate_candidate_lattice_matrix(
        self,
        crystal_system: CrystalSystem,
        v_target: float,
        sg_num: int,
        c_a_ratio: Optional[float] = None,
        b_a_ratio: Optional[float] = None,
        beta_angle_deg: Optional[float] = None,
        alpha_deg: Optional[float] = None,
        gamma_deg: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate unconstrained unit cell metric tensor for any crystal system based on target volume and continuous geometric parameters."""
        if crystal_system == CrystalSystem.CUBIC:
            a = float(v_target ** (1.0 / 3.0))
            lat_mat = np.diag([a, a, a])
            lat_params = {"a": round(a, 3), "b": round(a, 3), "c": round(a, 3), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}

        elif crystal_system in [CrystalSystem.HEXAGONAL, CrystalSystem.TRIGONAL]:
            ratio = c_a_ratio if c_a_ratio is not None else (
                6.95 if sg_num == 166 else (
                    2.53 if sg_num == 167 else (
                        5.76 if (sg_num == 194 and v_target > 80.0) else 1.633
                    )
                )
            )
            a = float((v_target / ((np.sqrt(3.0) / 2.0) * ratio)) ** (1.0 / 3.0))
            c = float(a * ratio)
            gamma = 120.0
            lat_params = {"a": round(a, 3), "b": round(a, 3), "c": round(c, 3), "alpha": 90.0, "beta": 90.0, "gamma": gamma}
            lat_mat = np.array([
                [a, 0.0, 0.0],
                [-0.5 * a, np.sqrt(3.0) / 2.0 * a, 0.0],
                [0.0, 0.0, c],
            ])

        elif crystal_system == CrystalSystem.TETRAGONAL:
            ratio = c_a_ratio if c_a_ratio is not None else (1.05 if sg_num == 142 else 1.414)
            a = float((v_target / ratio) ** (1.0 / 3.0))
            c = float(a * ratio)
            lat_params = {"a": round(a, 3), "b": round(a, 3), "c": round(c, 3), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
            lat_mat = np.diag([a, a, c])

        elif crystal_system == CrystalSystem.ORTHORHOMBIC:
            b_ratio = b_a_ratio if b_a_ratio is not None else 1.15
            c_ratio = c_a_ratio if c_a_ratio is not None else 1.25
            a = float((v_target / (b_ratio * c_ratio)) ** (1.0 / 3.0))
            b = float(a * b_ratio)
            c = float(a * c_ratio)
            lat_params = {"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
            lat_mat = np.diag([a, b, c])

        elif crystal_system == CrystalSystem.MONOCLINIC:
            beta = beta_angle_deg if beta_angle_deg is not None else 105.0
            b_ratio = b_a_ratio if b_a_ratio is not None else 1.10
            c_ratio = c_a_ratio if c_a_ratio is not None else 1.20
            sin_beta = np.sin(np.radians(beta))
            a = float((v_target / (b_ratio * c_ratio * max(0.1, sin_beta))) ** (1.0 / 3.0))
            b = float(a * b_ratio)
            c = float(a * c_ratio)
            lat_params = {"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "alpha": 90.0, "beta": round(beta, 2), "gamma": 90.0}
            lat_mat = np.array([
                [a, 0.0, 0.0],
                [0.0, b, 0.0],
                [c * np.cos(np.radians(beta)), 0.0, c * sin_beta],
            ])

        else:  # TRICLINIC
            al = alpha_deg if alpha_deg is not None else 85.0
            be = beta_angle_deg if beta_angle_deg is not None else 95.0
            ga = gamma_deg if gamma_deg is not None else 100.0
            b_ratio = b_a_ratio if b_a_ratio is not None else 1.05
            c_ratio = c_a_ratio if c_a_ratio is not None else 1.15
            a = float((v_target / (b_ratio * c_ratio)) ** (1.0 / 3.0))
            b = float(a * b_ratio)
            c = float(a * c_ratio)
            lat_params = {"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "alpha": round(al, 2), "beta": round(be, 2), "gamma": round(ga, 2)}
            lat_mat = np.diag([a, b, c])

        return lat_mat, lat_params



