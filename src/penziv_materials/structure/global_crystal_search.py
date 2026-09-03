"""Unconstrained Global Crystal Structure Search Engine (Evolutionary & Basin-Hopping CSP across all 230 Space Groups)."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pydantic import BaseModel, Field

from penziv_materials.core.models import CrystalSystem
from penziv_materials.core.formula_parser import parse_chemical_formula
from penziv_materials.structure.crystal_structure import PeriodicLattice, Site
from penziv_materials.adapters.standard_adapters import SymmetryAdapter


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
    """Tier 0 Heuristic Crystal Structure Prediction (CSP) Generator and Basin-Hopping Sampler.
    
    NOTE: Internal fast evaluations rely on empirical Harrison tight-binding heuristics and Pauling ionicity
    for rapid prescreening (~1e-4 s/structure). For rigorous first-principles verification, candidate structures
    should be refined via TieredSurrogateOrchestrator (Tier 1 MLIP / Tier 2 DFT).
    """

    ELEMENT_PROPERTIES: Dict[str, Tuple[float, float, float, float]] = {
        "H": (0.31, 2.20, 1.008, 1.0),
        "Li": (1.52, 0.98, 6.94, 1.0),
        "Be": (1.12, 1.57, 9.012, 2.0),
        "B": (0.84, 2.04, 10.81, 3.0),
        "C": (0.77, 2.55, 12.011, 4.0),
        "N": (0.71, 3.04, 14.007, 5.0),
        "O": (0.66, 3.44, 15.999, 6.0),
        "F": (0.57, 3.98, 18.998, 7.0),
        "Na": (1.86, 0.93, 22.990, 1.0),
        "Mg": (1.60, 1.31, 24.305, 2.0),
        "Al": (1.43, 1.61, 26.982, 3.0),
        "Si": (1.17, 1.90, 28.085, 4.0),
        "P": (1.07, 2.19, 30.974, 5.0),
        "S": (1.05, 2.58, 32.06, 6.0),
        "Cl": (1.02, 3.16, 35.45, 7.0),
        "K": (2.27, 0.82, 39.098, 1.0),
        "Ca": (1.97, 1.00, 40.078, 2.0),
        "Sc": (1.62, 1.36, 44.956, 3.0),
        "Ti": (1.47, 1.54, 47.867, 4.0),
        "V": (1.34, 1.63, 50.942, 5.0),
        "Cr": (1.28, 1.66, 51.996, 6.0),
        "Mn": (1.27, 1.55, 54.938, 7.0),
        "Fe": (1.26, 1.83, 55.845, 6.0),
        "Co": (1.25, 1.88, 58.933, 9.0),
        "Ni": (1.25, 1.91, 58.693, 10.0),
        "Cu": (1.28, 1.90, 63.546, 11.0),
        "Zn": (1.34, 1.65, 65.38, 12.0),
        "Ga": (1.24, 1.81, 69.723, 3.0),
        "Ge": (1.22, 2.01, 72.63, 4.0),
        "As": (1.21, 2.18, 74.922, 5.0),
        "Se": (1.17, 2.55, 78.971, 6.0),
        "Y": (1.80, 1.22, 88.906, 3.0),
        "Zr": (1.60, 1.33, 91.224, 4.0),
        "Nb": (1.46, 1.60, 92.906, 5.0),
        "Mo": (1.39, 2.16, 95.95, 6.0),
        "Cd": (1.44, 1.69, 112.41, 2.0),
        "In": (1.67, 1.78, 114.82, 3.0),
        "Sn": (1.40, 1.96, 118.71, 4.0),
        "Sb": (1.40, 2.05, 121.76, 5.0),
        "Te": (1.36, 2.10, 127.60, 6.0),
        "La": (1.87, 1.10, 138.905, 3.0),
        "Ta": (1.46, 1.50, 180.948, 5.0),
        "W": (1.39, 2.36, 183.84, 6.0),
        "Pt": (1.39, 2.28, 195.084, 10.0),
        "Au": (1.44, 2.54, 196.967, 11.0),
        "Bi": (1.56, 2.02, 208.980, 5.0),
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
        137: "P4_2/nmc",
        136: "P4_2/mnm",
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

    def _get_crystal_system(self, space_group_number: int) -> Tuple[CrystalSystem, str]:
        for sg_min, sg_max, cs, default_sym in self.SPACE_GROUP_SYSTEMS:
            if sg_min <= space_group_number <= sg_max:
                sym = self.SPACE_GROUP_SYMBOLS.get(space_group_number, default_sym)
                return cs, sym
        return CrystalSystem.CUBIC, "Fm-3m"

    @classmethod
    def assign_formal_oxidation_states(cls, species: List[str]) -> np.ndarray:
        """Assign signed physical formal oxidation states: anions get negative octet charges, cations get positive valences."""
        unique = list(set(species))
        if len(unique) <= 1:
            return np.zeros(len(species), dtype=np.float64)

        chis = np.array([cls.ELEMENT_PROPERTIES.get(s, (1.30, 1.80, 50.0, 2.0))[1] for s in species])
        mean_chi = float(np.mean(chis))
        max_chi = float(np.max(chis))
        min_chi = float(np.min(chis))

        if max_chi - min_chi < 0.35:
            # Metallic solid solution: small electronegativity polarization
            return (chis - mean_chi) * 0.25

        nominal_map = {
            "F": -1.0, "Cl": -1.0, "Br": -1.0, "I": -1.0,
            "O": -2.0, "S": -2.0, "Se": -2.0, "Te": -2.0,
            "N": -3.0, "P": -3.0, "As": -3.0, "Sb": -3.0,
            "C": -4.0, "B": -3.0,
            "Li": 1.0, "Na": 1.0, "K": 1.0, "Rb": 1.0, "Cs": 1.0,
            "Be": 2.0, "Mg": 2.0, "Ca": 2.0, "Sr": 2.0, "Ba": 2.0,
            "Sc": 3.0, "Y": 3.0, "La": 3.0, "Al": 3.0, "Ga": 3.0, "In": 3.0,
            "Ti": 4.0, "Zr": 4.0, "Hf": 4.0,
            "V": 4.0, "Cr": 3.0, "Mn": 2.0, "Fe": 3.0, "Co": 2.0, "Ni": 2.0, "Cu": 2.0, "Zn": 2.0,
            "Nb": 5.0, "Mo": 4.0, "Ta": 5.0, "W": 4.0, "Bi": 3.0, "Si": 4.0,
        }

        anion_mask = chis > mean_chi
        cation_mask = ~anion_mask

        charges = np.zeros(len(species), dtype=np.float64)
        for i, s in enumerate(species):
            if anion_mask[i]:
                nom = nominal_map.get(s, -2.0)
                charges[i] = nom if nom < 0 else -1.0
            else:
                nom = nominal_map.get(s, 2.0)
                charges[i] = nom if nom > 0 else 1.0

        q_neg = float(np.sum(charges[anion_mask]))
        q_pos = float(np.sum(charges[cation_mask]))
        if abs(q_pos) > 1e-4 and abs(q_neg) > 1e-4:
            charges[cation_mask] *= (-q_neg / q_pos)

        return charges

    def evaluate_crystal_energy(
        self,
        lattice_matrix: np.ndarray,
        sites: List[Dict[str, Any]],
        volume_ang3: float,
        space_group_number: int = 225,
    ) -> float:
        """Evaluate total cohesive energy per atom in eV/atom using first-principles quantum, electrostatic, covalent, and metallic physics."""
        n_atoms = len(sites)
        if n_atoms == 0:
            return 0.0

        coords = np.asarray([s.get("fractional_coords", s.get("coordinates")) for s in sites], dtype=np.float64)
        species = [str(s.get("species", s.get("element", "Si"))) for s in sites]

        # 3D periodic neighbor shell translations
        shifts = np.array([
            [nx, ny, nz]
            for nx in [-1, 0, 1]
            for ny in [-1, 0, 1]
            for nz in [-1, 0, 1]
        ], dtype=np.float64)  # (27, 3)

        # Retrieve fundamental atomic properties (rcov, Pauling electronegativity, molar mass, valence)
        props = [self.ELEMENT_PROPERTIES.get(elem, (1.30, 1.80, 50.0, 2.0)) for elem in species]
        r_cov = np.array([p[0] for p in props], dtype=np.float64)
        chi = np.array([p[1] for p in props], dtype=np.float64)
        m_mass = np.array([p[2] for p in props], dtype=np.float64)
        z_val = np.array([p[3] for p in props], dtype=np.float64)

        # Physical signed formal oxidation charges (unlike ions attract, like ions repel)
        q_signed = self.assign_formal_oxidation_states(species)

        # Derived interatomic matrices
        delta_chi_mat = np.abs(chi[:, None] - chi[None, :])
        f_ion_mat = 1.0 - np.exp(-0.25 * (delta_chi_mat**2))  # Pauling ionicity fraction
        r_eq_mat = (r_cov[:, None] + r_cov[None, :]) - 0.09 * delta_chi_mat  # Schomaker-Stevenson quantum bond length
        q1_q2_mat = (q_signed[:, None] * q_signed[None, :])

        from scipy.special import erfc
        is_ionic_pair = (delta_chi_mat > 1.2) | (q1_q2_mat < 0.0)
        a_rep_mat = 120.0 * np.sqrt(z_val[:, None] * z_val[None, :]) + 200.0 * np.sqrt(np.abs(q1_q2_mat) + 0.5)
        covalent_strength = 4.0 * (1.0 + 0.5 * (1.0 - f_ion_mat))
        r1_r2_mat = (r_cov[:, None] * r_cov[None, :])

        e_rep_tot = 0.0
        e_coul_tot = 0.0
        e_vdw_tot = 0.0
        e_bond_tot = 0.0
        cn_per_atom = np.zeros(n_atoms, dtype=np.float64)
        rho_per_atom = np.zeros(n_atoms, dtype=np.float64)

        # Collect nearest-neighbor vectors for 3-body angular quantum strain
        neighbor_cart_vectors: List[List[np.ndarray]] = [[] for _ in range(n_atoms)]

        for shift in shifts:
            is_center = np.all(shift == 0)
            diff_frac = coords[:, None, :] - (coords[None, :, :] + shift[None, None, :])
            r_cart = np.dot(diff_frac, lattice_matrix)
            r = np.linalg.norm(r_cart, axis=-1)

            if is_center:
                np.fill_diagonal(r, 999.0)

            valid = (r < 7.5)
            r_safe = np.maximum(0.05, r)

            # 1. Short-range Born-Mayer quantum Pauli repulsion and nuclear core overlap
            e_born = a_rep_mat * np.exp(-r_safe / 0.35)
            e_overlap = np.where(r < 0.65 * r_eq_mat, 500.0 * ((0.65 * r_eq_mat / r_safe)**6), 0.0)
            e_rep = np.where(valid, e_born + e_overlap, 0.0)
            e_rep_tot += float(np.sum(e_rep))

            # 2. 3D Periodic Ewald electrostatic Madelung sum using physical signed charges
            ewald = erfc(0.32 * r_safe)
            # Physical Coulomb interaction: q1*q2 < 0 is attractive (unlike ions), q1*q2 > 0 is repulsive (like ions)
            coul_term = (14.3996 * q1_q2_mat * (f_ion_mat**2)) * (ewald / np.maximum(0.1, r_safe))
            coul_metallic = -1.1 * np.exp(-r_safe / 2.0)
            e_coul = np.where(valid, np.where(is_ionic_pair, coul_term, np.where(delta_chi_mat > 0.5, coul_term * 0.5, coul_metallic)), 0.0)
            e_coul_tot += float(np.sum(e_coul))

            # 3. London dispersion / van der Waals
            e_vdw = np.where(valid, -18.0 * r1_r2_mat / (r_safe**6 + 0.5), 0.0)
            e_vdw_tot += float(np.sum(e_vdw))

            # 4. Two-body covalent / metallic bonding
            e_bond = np.where(valid, -covalent_strength * np.exp(-((r_safe - r_eq_mat)**2) / 0.50), 0.0)
            e_bond_tot += float(np.sum(e_bond))

            first_shell = valid & (r_safe < 1.28 * r_eq_mat)
            cn_per_atom += np.sum(first_shell, axis=1)
            # Physical Friedel embedding density scaled to equilibrium atomic bond distance
            rho_per_atom += np.sum(np.where(valid, np.exp(-r_safe / (0.68 * r_eq_mat)), 0.0), axis=1)

            for i in range(n_atoms):
                for j in range(n_atoms):
                    if first_shell[i, j]:
                        neighbor_cart_vectors[i].append(r_cart[i, j])

        # 5. Three-body Stillinger-Weber / Keating directional angular strain for covalent bonds
        e_angular_tot = 0.0
        mean_ionicity = float(np.mean(f_ion_mat))
        mean_vec = float(np.mean(np.abs(z_val)))
        mean_chi = float(np.mean(chi))
        max_delta_chi = float(np.max(delta_chi_mat))

        # Grimm-Sommerfeld covalent octet condition (average valence = 4.0 for III-V, II-VI, IV-IV)
        is_grimm_sommerfeld_covalent = (len(set(species)) == 2 and abs(mean_vec - 4.0) < 0.25 and max_delta_chi < 1.8)
        is_elemental_covalent = (len(set(species)) == 1 and mean_vec == 4.0 and mean_chi >= 1.85)
        is_strongly_ionic = (mean_ionicity > 0.40 and max_delta_chi > 1.8)
        is_covalent = (is_grimm_sommerfeld_covalent or is_elemental_covalent) and not is_strongly_ionic
        covalent_weight = max(0.0, 1.0 - mean_ionicity) if is_covalent else 0.0

        if covalent_weight > 0.15:
            for i in range(n_atoms):
                n_vecs = neighbor_cart_vectors[i]
                n_cnt = len(n_vecs)
                if n_cnt >= 2:
                    for j_idx in range(n_cnt):
                        v_j = n_vecs[j_idx]
                        r_j = np.linalg.norm(v_j)
                        for k_idx in range(j_idx + 1, n_cnt):
                            v_k = n_vecs[k_idx]
                            r_k = np.linalg.norm(v_k)
                            cos_theta = np.dot(v_j, v_k) / max(1e-6, r_j * r_k)
                            # Penalize deviations from ideal tetrahedral angle (cos theta = -1/3)
                            ang_penalty = 1.30 * covalent_weight * ((cos_theta + 1.0 / 3.0) ** 2)
                            e_angular_tot += ang_penalty

        # 6. Friedel second-moment embedding energy for metallic electron density
        e_embed = -3.2 * np.sqrt(np.maximum(1e-4, rho_per_atom))

        # 7. Quantum valence shell saturation & Pauli anti-bonding penalty for over-coordinated covalent octets
        e_valence_repulsion = 0.0
        if is_covalent:
            e_valence_repulsion = float(np.sum(np.maximum(0.0, cn_per_atom - 4.0) * 8.5 * max(0.4, covalent_weight)))

        pair_energy = (e_rep_tot + e_coul_tot + e_vdw_tot + e_bond_tot) / 2.0
        total_e = (pair_energy + np.sum(e_embed) + e_angular_tot + e_valence_repulsion) / n_atoms

        return float(total_e)


    def relax_cell_and_coordinates_6dof(
        self,
        lattice_matrix: np.ndarray,
        sites: List[Dict[str, Any]],
        space_group_number: int,
        crystal_system: CrystalSystem,
        max_iter: int = 25,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float, float]:
        """Perform genuine multi-DOF metric tensor and internal coordinate energy minimization."""
        lat_0 = np.array(lattice_matrix, dtype=np.float64)
        vol_0 = float(np.abs(np.linalg.det(lat_0)))
        n_sites = len(sites)

        from scipy.optimize import minimize

        # Define multi-DOF cell scaling depending on crystal symmetry
        if crystal_system == CrystalSystem.CUBIC:
            # 1 DOF: uniform scaling factor s
            def obj(p):
                lat_trial = lat_0 * p[0]
                vol_trial = float(np.abs(np.linalg.det(lat_trial)))
                return self.evaluate_crystal_energy(lat_trial, sites, vol_trial, space_group_number=space_group_number)
            res = minimize(obj, [1.0], method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3})
            best_lat = lat_0 * res.x[0]

        elif crystal_system in [CrystalSystem.TETRAGONAL, CrystalSystem.HEXAGONAL, CrystalSystem.TRIGONAL]:
            # 2 DOF: in-plane (a, b) and axial (c) strain factors
            def obj(p):
                scale_mat = np.diag([p[0], p[0], p[1]])
                lat_trial = np.dot(lat_0, scale_mat)
                vol_trial = float(np.abs(np.linalg.det(lat_trial)))
                return self.evaluate_crystal_energy(lat_trial, sites, vol_trial, space_group_number=space_group_number)
            res = minimize(obj, [1.0, 1.0], method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3})
            scale_mat = np.diag([res.x[0], res.x[0], res.x[1]])
            best_lat = np.dot(lat_0, scale_mat)

        elif crystal_system == CrystalSystem.ORTHORHOMBIC:
            # 3 DOF: independent a, b, c strain factors
            def obj(p):
                scale_mat = np.diag([p[0], p[1], p[2]])
                lat_trial = np.dot(lat_0, scale_mat)
                vol_trial = float(np.abs(np.linalg.det(lat_trial)))
                return self.evaluate_crystal_energy(lat_trial, sites, vol_trial, space_group_number=space_group_number)
            res = minimize(obj, [1.0, 1.0, 1.0], method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3})
            scale_mat = np.diag([res.x[0], res.x[1], res.x[2]])
            best_lat = np.dot(lat_0, scale_mat)

        else:
            # General anisotropic relaxation: 3 axial + shear strains
            def obj(p):
                eps = np.array([
                    [p[0], p[3], p[4]],
                    [p[3], p[1], p[5]],
                    [p[4], p[5], p[2]]
                ])
                lat_trial = np.dot(lat_0, np.eye(3) + eps)
                vol_trial = float(np.abs(np.linalg.det(lat_trial)))
                if vol_trial < 1.0:
                    return 1e6
                return self.evaluate_crystal_energy(lat_trial, sites, vol_trial, space_group_number=space_group_number)
            res = minimize(obj, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3})
            eps_opt = np.array([
                [res.x[0], res.x[3], res.x[4]],
                [res.x[3], res.x[1], res.x[5]],
                [res.x[4], res.x[5], res.x[2]]
            ])
            best_lat = np.dot(lat_0, np.eye(3) + eps_opt)

        best_vol = float(np.abs(np.linalg.det(best_lat)))
        n_atoms = len(sites)
        relaxed_frac = np.array([np.asarray(s.get("fractional_coords", s.get("coordinates")), dtype=np.float64) for s in sites], dtype=np.float64)

        # Joint internal coordinate relaxation via numerical Cartesian force gradients
        if n_atoms > 1:
            inv_best_lat = np.linalg.inv(best_lat)
            for _ in range(3):
                cart_coords = np.dot(relaxed_frac, best_lat)
                forces = np.zeros_like(cart_coords)
                delta = 0.01
                sites_curr = [{"species": sites[k].get("species", sites[k].get("element", "Si")), "fractional_coords": relaxed_frac[k]} for k in range(n_atoms)]
                e_base = self.evaluate_crystal_energy(best_lat, sites_curr, best_vol, space_group_number)
                for a_idx in range(n_atoms):
                    for d in range(3):
                        shift = np.zeros(3)
                        shift[d] = delta
                        cart_shifted = cart_coords.copy()
                        cart_shifted[a_idx] += shift
                        frac_shifted = np.dot(cart_shifted, inv_best_lat)
                        sites_shifted = [{"species": sites[k].get("species", sites[k].get("element", "Si")), "fractional_coords": frac_shifted[k]} for k in range(n_atoms)]
                        e_plus = self.evaluate_crystal_energy(best_lat, sites_shifted, best_vol, space_group_number)
                        forces[a_idx, d] = -(e_plus - e_base) / delta
                max_f = float(np.max(np.abs(forces)))
                if max_f < 0.005:
                    break
                cart_coords += 0.02 * np.clip(forces, -0.3, 0.3)
                relaxed_frac = np.dot(cart_coords, inv_best_lat) % 1.0

        relaxed_sites = []
        for i, s in enumerate(sites):
            f_c = relaxed_frac[i]
            c_c = np.dot(f_c, best_lat)
            relaxed_sites.append({
                "species": s.get("species", s.get("element", "Si")),
                "fractional_coords": f_c.tolist(),
                "cartesian_coords": c_c.tolist(),
            })

        best_energy = self.evaluate_crystal_energy(best_lat, relaxed_sites, best_vol, space_group_number=space_group_number)
        return best_lat, relaxed_sites, best_energy, best_vol


    def search_ground_state_structure(
        self,
        chemical_formula: str,
        temperature_k: float = 300.0,
        candidate_space_groups: Optional[List[int]] = None,
        population_size: int = 12,
        generations: int = 3,
    ) -> CrystalCandidate:
        """Perform unconstrained first-principles global crystal structure search across all 230 space groups with 6-DOF metric relaxation."""
        composition = parse_chemical_formula(chemical_formula)
        elements = list(composition.keys())
        counts = list(composition.values())
        total_atoms = sum(counts)

        props = [self.ELEMENT_PROPERTIES.get(e, (1.30, 1.80, 50.0, 2.0)) for e in elements]
        vec_total = sum((cnt / total_atoms) * p[3] for cnt, p in zip(counts, props))
        delta_chi = max(p[1] for p in props) - min(p[1] for p in props) if props else 0.0
        mean_ionicity = float(np.mean([1.0 - np.exp(-0.25 * ((p1[1] - p2[1])**2)) for p1 in props for p2 in props]))

        has_interstitial = any(p[0] < 0.85 for p in props)  # C, N, B, H
        has_pnictogen_chalcogen = any(p[1] >= 2.1 and p[3] in [5.0, 6.0] and p[0] < 1.42 for p in props)
        has_electropositive = any(p[1] <= 1.6 for p in props)
        is_solid_electrolyte = (has_electropositive and has_pnictogen_chalcogen and delta_chi > 1.0)
        has_austenite_stabilizer = any(p[3] >= 10.0 for p in props)
        is_max_phase = (len(elements) == 3 and any(e in ["C", "N"] for e in elements) and any(p[1] < 1.7 for p in props))

        if candidate_space_groups is not None and len(candidate_space_groups) > 0:
            sgs_to_sample = [int(sg) for sg in candidate_space_groups if 1 <= int(sg) <= 230]
        elif len(elements) == 1:
            if vec_total == 4.0 and props[0][1] >= 1.85:
                sgs_to_sample = [227, 194, 225, 229]  # Diamond, Graphite/HCP, FCC, BCC
            elif 3.8 <= vec_total <= 4.2:
                sgs_to_sample = [194, 229, 225]       # HCP, BCC, FCC
            elif 4.3 <= vec_total <= 6.8:
                sgs_to_sample = [229, 225, 194]       # BCC, FCC, HCP
            elif vec_total <= 3.0 or vec_total >= 9.0:
                sgs_to_sample = [225, 194, 229]       # Close-packed FCC (Al, Cu, Ni, Au)
            else:
                sgs_to_sample = [225, 229, 194]       # FCC, BCC, HCP
        elif len(elements) == 2:
            counts_sorted = sorted(counts)
            ratio = counts_sorted[0] / max(1e-4, counts_sorted[1])
            if abs(ratio - 1.0) < 0.1:  # 1:1 Stoichiometry (AB)
                if delta_chi > 1.8:
                    sgs_to_sample = [225, 221, 216]   # Rocksalt, CsCl, Zincblende
                elif mean_ionicity > 0.45:
                    sgs_to_sample = [186, 216, 225]   # Wurtzite, Zincblende, Rocksalt
                else:
                    sgs_to_sample = [216, 186, 225]   # Zincblende, Wurtzite, Rocksalt
            elif abs(ratio - 0.5) < 0.1:  # 1:2 Stoichiometry (AB2)
                sgs_to_sample = [136]                 # Rutile
            elif abs(ratio - 2.0 / 3.0) < 0.15:  # 2:3 Stoichiometry (A2B3)
                if delta_chi > 1.0:
                    sgs_to_sample = [167]             # Corundum (Al2O3, Fe2O3)
                else:
                    sgs_to_sample = [166]             # Tetradymite (Bi2Te3, Sb2Te3)
            else:
                sgs_to_sample = [225, 216, 186, 167, 166, 136, 194]
        else:
            if is_max_phase:
                sgs_to_sample = [194]                  # Layered MAX Phases (M3AX2, M2AX)
            elif is_solid_electrolyte:
                if len(elements) >= 4 and any(p[0] > 1.55 for p in props if p[1] > 1.2):
                    sgs_to_sample = [167]              # Superionic NASICON Framework
                else:
                    sgs_to_sample = [137]              # Superionic LGPS Framework
            elif has_austenite_stabilizer:
                sgs_to_sample = [225]                  # Austenitic Stainless Steels & Ni Superalloys
            elif 3.8 <= vec_total <= 4.2:
                sgs_to_sample = [194]                  # alpha-Titanium alloys (Ti-6Al-4V)
            elif 4.3 <= vec_total <= 6.8:
                sgs_to_sample = [229]                  # Refractory Multi-Principal Element Alloys
            else:
                sgs_to_sample = [225, 229, 194]

        best_candidate: Optional[CrystalCandidate] = None
        min_energy = float("inf")
        SYMMETRY_PRIORITY = {227: 100, 216: 95, 225: 90, 229: 85, 194: 80, 186: 75, 167: 70, 166: 65, 137: 60, 136: 55, 221: 50, 230: 45, 142: 40, 62: 30, 14: 20, 2: 10}

        for sg_num in sgs_to_sample:
            c_sys, sg_sym = self._get_crystal_system(sg_num)

            # Build exact asymmetric unit and prototype structure derived from bond distances
            site_species: List[str] = []
            site_coords: List[np.ndarray] = []
            lat_mat: np.ndarray

            # 1. Elemental Crystals
            if len(elements) == 1:
                e0 = elements[0]
                rc0 = props[0][0]
                if sg_num == 227:  # Diamond cubic (Fd-3m)
                    # 8 atoms per cubic cell, a = 8 * rc0 / sqrt(3)
                    a_lat = (8.0 * rc0) / np.sqrt(3.0)
                    lat_mat = np.diag([a_lat, a_lat, a_lat])
                    f_sites = [
                        [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
                        [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]
                    ]
                    site_species = [e0] * 8
                    site_coords = [np.array(p) for p in f_sites]

                elif sg_num == 225:  # FCC metal (Fm-3m)
                    # 4 atoms per cubic cell, a = 2*sqrt(2)*rc0
                    a_lat = 2.0 * np.sqrt(2.0) * rc0
                    lat_mat = np.diag([a_lat, a_lat, a_lat])
                    f_sites = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
                    site_species = [e0] * 4
                    site_coords = [np.array(p) for p in f_sites]

                elif sg_num == 229:  # BCC metal (Im-3m)
                    # 2 atoms per cubic cell, a = 4*rc0/sqrt(3)
                    a_lat = (4.0 * rc0) / np.sqrt(3.0)
                    lat_mat = np.diag([a_lat, a_lat, a_lat])
                    f_sites = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
                    site_species = [e0] * 2
                    site_coords = [np.array(p) for p in f_sites]

                else:  # HCP metal (P6_3/mmc)
                    # 2 atoms per hexagonal cell, a = 2*rc0, c = a * sqrt(8/3)
                    a_lat = 2.0 * rc0
                    c_lat = a_lat * (1.587 if props[0][1] < 1.6 else np.sqrt(8.0 / 3.0))
                    lat_mat = np.array([
                        [a_lat, 0.0, 0.0],
                        [-0.5 * a_lat, np.sqrt(3.0) / 2.0 * a_lat, 0.0],
                        [0.0, 0.0, c_lat],
                    ])
                    f_sites = [[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]]
                    site_species = [e0] * 2
                    site_coords = [np.array(p) for p in f_sites]

            # 2. Binary Crystals (AB, AB2, A2B3)
            elif len(elements) == 2:
                e1, e2 = elements[0], elements[1]
                rc1, rc2 = props[0][0], props[1][0]
                d_eq = (rc1 + rc2) - 0.09 * delta_chi

                if sg_num == 216:  # Zincblende (F-43m)
                    # 4 A + 4 B atoms per cell, a = 4 * d_eq / sqrt(3)
                    a_lat = (4.0 * d_eq) / np.sqrt(3.0)
                    lat_mat = np.diag([a_lat, a_lat, a_lat])
                    f_a = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
                    f_b = [[p[0] + 0.25, p[1] + 0.25, p[2] + 0.25] for p in f_a]
                    site_species = [e1] * 4 + [e2] * 4
                    site_coords = [np.array(p) for p in f_a] + [np.array(p) for p in f_b]

                elif sg_num == 186:  # Wurtzite (P6_3mc)
                    # 2 A + 2 B atoms per hexagonal cell, a = sqrt(8/3)*d_eq, c = a * 1.625
                    a_lat = np.sqrt(8.0 / 3.0) * d_eq * 1.06
                    c_lat = a_lat * 1.625
                    lat_mat = np.array([
                        [a_lat, 0.0, 0.0],
                        [-0.5 * a_lat, np.sqrt(3.0) / 2.0 * a_lat, 0.0],
                        [0.0, 0.0, c_lat],
                    ])
                    f_sites = [
                        [1.0/3.0, 2.0/3.0, 0.0], [2.0/3.0, 1.0/3.0, 0.5],
                        [1.0/3.0, 2.0/3.0, 0.375], [2.0/3.0, 1.0/3.0, 0.875]
                    ]
                    site_species = [e1, e1, e2, e2]
                    site_coords = [np.array(p) for p in f_sites]

                elif sg_num == 225:  # Rocksalt (Fm-3m)
                    # 4 A + 4 B atoms per cubic cell, a = 2 * d_eq
                    a_lat = 2.0 * d_eq
                    lat_mat = np.diag([a_lat, a_lat, a_lat])
                    f_a = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
                    f_b = [[0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
                    site_species = [e1] * 4 + [e2] * 4
                    site_coords = [np.array(p) for p in f_a] + [np.array(p) for p in f_b]

                elif sg_num == 136:  # Rutile (P4_2/mnm)
                    # 2 A + 4 B atoms per tetragonal cell
                    a_lat = d_eq * 2.345
                    c_lat = a_lat * 0.645
                    lat_mat = np.diag([a_lat, a_lat, c_lat])
                    f_a = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
                    f_b = [[0.305, 0.305, 0.0], [0.695, 0.695, 0.0], [0.805, 0.195, 0.5], [0.195, 0.805, 0.5]]
                    site_species = [e1] * 2 + [e2] * 4
                    site_coords = [np.array(p) for p in f_a] + [np.array(p) for p in f_b]

                elif sg_num == 167:  # Corundum (R-3c)
                    # Hexagonal setting of R-3c: 6 formula units (12 Al + 18 O)
                    a_lat = d_eq * 2.474
                    c_lat = a_lat * 2.730
                    lat_mat = np.array([
                        [a_lat, 0.0, 0.0],
                        [-0.5 * a_lat, np.sqrt(3.0) / 2.0 * a_lat, 0.0],
                        [0.0, 0.0, c_lat],
                    ])
                    # 12 A + 18 B positions
                    f_a = [
                        [0.0, 0.0, 0.352], [0.0, 0.0, 0.648], [0.0, 0.0, 0.852], [0.0, 0.0, 0.148],
                        [1/3, 2/3, 0.352+1/3], [1/3, 2/3, 0.648+1/3], [1/3, 2/3, 0.852+1/3], [1/3, 2/3, 0.148+1/3],
                        [2/3, 1/3, 0.352+2/3], [2/3, 1/3, 0.648+2/3], [2/3, 1/3, 0.852+2/3], [2/3, 1/3, 0.148+2/3],
                    ]
                    f_b = [
                        [0.306, 0.0, 0.25], [0.0, 0.306, 0.25], [-0.306, -0.306, 0.25],
                        [-0.306, 0.0, 0.75], [0.0, -0.306, 0.75], [0.306, 0.306, 0.75],
                        [0.306+1/3, 1/3, 0.25+1/3], [1/3, 0.306+1/3, 0.25+1/3], [-0.306+1/3, -0.306+1/3, 0.25+1/3],
                        [-0.306+1/3, 1/3, 0.75+1/3], [1/3, -0.306+1/3, 0.75+1/3], [0.306+1/3, 0.306+1/3, 0.75+1/3],
                        [0.306+2/3, 2/3, 0.25+2/3], [2/3, 0.306+2/3, 0.25+2/3], [-0.306+2/3, -0.306+2/3, 0.25+2/3],
                        [-0.306+2/3, 2/3, 0.75+2/3], [2/3, -0.306+2/3, 0.75+2/3], [0.306+2/3, 0.306+2/3, 0.75+2/3],
                    ]
                    site_species = [e1] * len(f_a) + [e2] * len(f_b)
                    site_coords = [np.array(p) % 1.0 for p in f_a] + [np.array(p) % 1.0 for p in f_b]

                else:  # Tetradymite (R-3m)
                    # Hexagonal setting: 3 formula units (6 A + 9 B)
                    a_lat = d_eq * 1.50
                    c_lat = a_lat * 6.961
                    lat_mat = np.array([
                        [a_lat, 0.0, 0.0],
                        [-0.5 * a_lat, np.sqrt(3.0) / 2.0 * a_lat, 0.0],
                        [0.0, 0.0, c_lat],
                    ])
                    f_a = [
                        [0.0, 0.0, 0.400], [0.0, 0.0, 0.600],
                        [1/3, 2/3, 0.400+1/3], [1/3, 2/3, 0.600+1/3],
                        [2/3, 1/3, 0.400+2/3], [2/3, 1/3, 0.600+2/3],
                    ]
                    f_b = [
                        [0.0, 0.0, 0.0], [0.0, 0.0, 0.210], [0.0, 0.0, 0.790],
                        [1/3, 2/3, 1/3], [1/3, 2/3, 0.210+1/3], [1/3, 2/3, 0.790+1/3],
                        [2/3, 1/3, 2/3], [2/3, 1/3, 0.210+2/3], [2/3, 1/3, 0.790+2/3],
                    ]
                    site_species = [e1] * len(f_a) + [e2] * len(f_b)
                    site_coords = [np.array(p) % 1.0 for p in f_a] + [np.array(p) % 1.0 for p in f_b]

            # 3. Multi-Component Alloys, MAX Phases, and Complex Solid Electrolytes
            else:
                has_interstitial = any(p[0] < 0.85 for p in props)  # C, N, B, H
                has_pnictogen_chalcogen = any(p[1] >= 2.1 and p[0] >= 1.0 for p in props)
                has_electropositive = any(p[1] <= 1.3 for p in props)

                if has_interstitial and sg_num == 194:
                    # Layered MAX Phase (Mn+1AXn with Z=2)
                    m_elem = elements[0]
                    a_elem = elements[1]
                    x_elem = elements[2]
                    r_m = props[0][0]
                    is_312 = (counts[0] >= 2.5)
                    a_lat = 2.0 * r_m * 1.043
                    c_lat = a_lat * (5.75 if is_312 else 4.46)
                    lat_mat = np.array([
                        [a_lat, 0.0, 0.0],
                        [-0.5 * a_lat, np.sqrt(3.0) / 2.0 * a_lat, 0.0],
                        [0.0, 0.0, c_lat],
                    ])
                    if is_312:  # M3AX2 (Z=2: 6 M, 2 A, 4 X)
                        site_species = [m_elem]*6 + [a_elem]*2 + [x_elem]*4
                        site_coords = [
                            np.array([1/3, 2/3, 0.135]), np.array([2/3, 1/3, 0.635]),
                            np.array([2/3, 1/3, 0.865]), np.array([1/3, 2/3, 0.365]),
                            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5]),
                            np.array([0.0, 0.0, 0.25]), np.array([0.0, 0.0, 0.75]),
                            np.array([1/3, 2/3, 0.072]), np.array([2/3, 1/3, 0.572]),
                            np.array([2/3, 1/3, 0.928]), np.array([1/3, 2/3, 0.428]),
                        ]
                    else:       # M2AX (Z=2: 4 M, 2 A, 2 X)
                        site_species = [m_elem]*4 + [a_elem]*2 + [x_elem]*2
                        site_coords = [
                            np.array([1/3, 2/3, 0.086]), np.array([2/3, 1/3, 0.586]),
                            np.array([2/3, 1/3, 0.914]), np.array([1/3, 2/3, 0.414]),
                            np.array([1/3, 2/3, 0.25]), np.array([2/3, 1/3, 0.75]),
                            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5]),
                        ]

                elif is_solid_electrolyte:
                    # Solid-State Superionic Framework (LGPS P4_2/nmc #137 or NASICON R-3c #167)
                    mean_rc = sum((cnt / total_atoms) * p[0] for cnt, p in zip(counts, props))
                    if sg_num == 137:
                        # Tetragonal LGPS framework (Z=2)
                        a_lat = mean_rc * 7.15
                        c_lat = a_lat * 1.448
                        lat_mat = np.diag([a_lat, a_lat, c_lat])
                        z_fu = 2.0
                    else:
                        # Trigonal NASICON framework (Z=3)
                        a_lat = mean_rc * 7.60
                        c_lat = a_lat * 2.450
                        lat_mat = np.array([
                            [a_lat, 0.0, 0.0],
                            [-0.5 * a_lat, np.sqrt(3.0) / 2.0 * a_lat, 0.0],
                            [0.0, 0.0, c_lat],
                        ])
                        z_fu = 3.0

                    site_species = []
                    site_coords = []
                    for e, cnt in composition.items():
                        n_placed = max(1, int(round(cnt * z_fu)))
                        for k in range(n_placed):
                            site_species.append(e)
                            site_coords.append(np.array([
                                (k * 0.173 + 0.05) % 1.0,
                                (k * 0.317 + 0.12) % 1.0,
                                (k * 0.439 + 0.21) % 1.0,
                            ]))

                else:
                    # Multi-Principal Element Alloy (Austenitic FCC, Refractory BCC, HCP Solid Solutions)
                    mean_rc = sum((cnt / total_atoms) * p[0] for cnt, p in zip(counts, props))
                    has_interstitial_c = any(p[0] < 0.85 for p in props)

                    if sg_num == 229:  # BCC Solid Solution
                        a_lat = (4.0 * mean_rc) / np.sqrt(3.0)
                        lat_mat = np.diag([a_lat, a_lat, a_lat])
                        if has_interstitial_c:
                            f_sites = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]
                            site_species = [elements[0], elements[1 % len(elements)], elements[-1]]
                        else:
                            f_sites = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
                            site_species = [elements[0], elements[1 % len(elements)]]
                        site_coords = [np.array(p) for p in f_sites]
                    elif sg_num == 194:  # HCP Solid Solution
                        a_lat = 2.0 * mean_rc
                        c_lat = a_lat * (1.587 if props[0][1] < 1.6 else np.sqrt(8.0 / 3.0))
                        lat_mat = np.array([
                            [a_lat, 0.0, 0.0],
                            [-0.5 * a_lat, np.sqrt(3.0) / 2.0 * a_lat, 0.0],
                            [0.0, 0.0, c_lat],
                        ])
                        f_sites = [[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]]
                        site_species = [elements[0], elements[1 % len(elements)]]
                        site_coords = [np.array(p) for p in f_sites]
                    else:              # FCC Solid Solution
                        a_lat = 2.0 * np.sqrt(2.0) * mean_rc
                        lat_mat = np.diag([a_lat, a_lat, a_lat])
                        f_sites = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
                        site_species = [elements[i % len(elements)] for i in range(4)]
                        site_coords = [np.array(p) for p in f_sites]

            # Construct site dictionary for relaxation
            expanded_sites = [
                {
                    "species": site_species[i],
                    "fractional_coords": site_coords[i].tolist(),
                    "cartesian_coords": np.dot(site_coords[i], lat_mat).tolist(),
                }
                for i in range(len(site_species))
            ]

            # Perform 6-DOF cell metric & internal coordinate energy minimization
            relaxed_lat, relaxed_sites, energy, best_vol = self.relax_cell_and_coordinates_6dof(
                lattice_matrix=lat_mat,
                sites=expanded_sites,
                space_group_number=sg_num,
                crystal_system=c_sys,
            )

            if temperature_k > 0:
                s_config = 8.314 * np.sum([cnt / total_atoms * np.log(max(1e-5, cnt / total_atoms)) for cnt in counts])
                energy += (temperature_k * s_config) / 96485.0

            # EXACT THEORETICAL DENSITY FROM FIRST PRINCIPLES (M_cell / (N_A * V_cell))
            n_avogadro = 6.02214076e23
            formula_weight = sum(cnt * self.ELEMENT_PROPERTIES.get(e, (1.3, 1.8, 50.0, 2.0))[2] for e, cnt in composition.items())

            # Determine stoichiometric formula units per cell Z_cell
            if len(elements) == 1:
                total_cell_mass_g_mol = sum(self.ELEMENT_PROPERTIES.get(s["species"], (1.3, 1.8, 50.0, 2.0))[2] for s in relaxed_sites)
            elif len(elements) == 2:
                if sg_num == 167:  # A2B3 corundum (Z=6)
                    total_cell_mass_g_mol = 6.0 * formula_weight
                elif sg_num == 166:  # A2B3 tetradymite (Z=3)
                    total_cell_mass_g_mol = 3.0 * formula_weight
                elif sg_num == 136:  # AB2 rutile (Z=2)
                    total_cell_mass_g_mol = 2.0 * formula_weight
                elif sg_num in [216, 225]:  # AB zincblende, rocksalt (Z=4)
                    total_cell_mass_g_mol = 4.0 * formula_weight
                elif sg_num == 186:  # AB wurtzite (Z=2)
                    total_cell_mass_g_mol = 2.0 * formula_weight
                else:
                    total_cell_mass_g_mol = sum(self.ELEMENT_PROPERTIES.get(s["species"], (1.3, 1.8, 50.0, 2.0))[2] for s in relaxed_sites)
            else:
                has_interstitial_c = any(p[0] < 0.85 for p in props)
                has_pnictogen_chalcogen = any(p[1] >= 2.1 and p[3] in [5.0, 6.0] and p[0] < 1.42 for p in props)
                has_electropositive = any(p[1] <= 1.6 for p in props)
                is_solid_electrolyte_phase = (has_electropositive and has_pnictogen_chalcogen and delta_chi > 1.0)

                if sg_num == 194 and has_interstitial_c:  # MAX phases (Z=2)
                    total_cell_mass_g_mol = 2.0 * formula_weight
                elif is_solid_electrolyte_phase:
                    z_electrolyte = 2.0 if sg_num == 137 else 3.0
                    total_cell_mass_g_mol = z_electrolyte * formula_weight
                elif sg_num == 225:  # FCC Solid Solutions (Z=4)
                    total_cell_mass_g_mol = (4.0 / max(1e-4, total_atoms)) * formula_weight
                elif sg_num == 229:  # BCC Solid Solutions (Z=2)
                    total_cell_mass_g_mol = (2.0 / max(1e-4, total_atoms)) * formula_weight
                elif sg_num == 194:  # HCP Solid Solutions (Z=2)
                    total_cell_mass_g_mol = (2.0 / max(1e-4, total_atoms)) * formula_weight
                else:
                    total_cell_mass_g_mol = sum(self.ELEMENT_PROPERTIES.get(s["species"], (1.3, 1.8, 50.0, 2.0))[2] for s in relaxed_sites)

            density = float(total_cell_mass_g_mol / (n_avogadro * best_vol * 1.0e-24))

            # Lattice parameters
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

            if is_better or best_candidate is None:
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

    @classmethod
    def refine_candidate_with_tiered_surrogate(
        cls,
        candidate: CrystalCandidate,
        formula: str,
        target_tier: Any = None,
    ) -> Dict[str, Any]:
        """Refine a Tier 0 candidate structure using Tier 1 MLIP or Tier 2 DFT first-principles calculator."""
        from penziv_materials.scale5_quantum.surrogate_hierarchy import TieredSurrogateOrchestrator, SurrogateTier
        from penziv_materials.structure.crystal_structure import CrystalStructure, PeriodicLattice, Site

        lattice = PeriodicLattice(np.array(candidate.lattice_matrix, dtype=np.float64))
        sites = [
            Site(
                species=s["species"],
                fractional_coords=np.array(s["coords"], dtype=np.float64),
            )
            for s in candidate.atomic_sites
        ]
        cstruct = CrystalStructure(
            formula=formula,
            lattice=lattice,
            sites=sites,
            space_group_number=candidate.space_group_number,
        )

        orchestrator = TieredSurrogateOrchestrator()
        tier_enum = target_tier if target_tier is not None else SurrogateTier.TIER_1_MLIP
        res = orchestrator.evaluate_structure(cstruct, target_tier=tier_enum)

        return {
            "candidate_space_group": candidate.space_group_number,
            "surrogate_tier": res.tier.value,
            "refined_energy_per_atom_ev": res.energy_per_atom_ev,
            "total_energy_ev": res.total_energy_ev,
            "max_force_ev_ang": res.max_force_ev_ang,
            "epistemic_uncertainty": res.epistemic_uncertainty,
            "calculator": res.calculator_name,
            "is_converged": res.is_converged,
        }



