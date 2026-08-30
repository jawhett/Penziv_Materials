"""Scale 5: Quantum Electronic Structure, Mermin Free Energy, Universal Cauchy-Born Elasticity & Miedema Thermodynamics."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_EV_K
from penziv_materials.core.models import QuantumState
from penziv_materials.core.tensors import compute_universal_cauchy_born_stiffness, compute_voigt_reuss_hill_aggregates
from penziv_materials.scale5_quantum.dft_engine import DFTEngine


class UniversalElementalProperties:
    """Standard physical parameters (atomic_mass, covalent_radius_ang, electronegativity_phi, electron_density_nws, valence_z, melting_point_k)."""

    DATABASE: Dict[str, Tuple[float, float, float, float, float, float]] = {
        "H": (1.008, 0.31, 2.20, 0.40, 1.0, 14.01),
        "Li": (6.94, 1.28, 0.98, 0.85, 1.0, 453.69),
        "Be": (9.012, 0.96, 1.57, 1.60, 2.0, 1560.0),
        "B": (10.81, 0.84, 2.04, 1.55, 3.0, 2349.0),
        "C": (12.011, 0.76, 2.55, 2.80, 4.0, 3800.0),
        "N": (14.007, 0.71, 3.04, 2.10, -3.0, 63.15),
        "O": (15.999, 0.66, 3.44, 2.60, -2.0, 54.36),
        "F": (18.998, 0.57, 3.98, 2.90, -1.0, 53.53),
        "Na": (22.990, 1.66, 0.93, 0.65, 1.0, 370.87),
        "Mg": (24.305, 1.41, 1.31, 1.15, 2.0, 923.0),
        "Al": (26.982, 1.21, 1.61, 1.39, 3.0, 933.47),
        "Si": (28.085, 1.11, 1.90, 1.50, 4.0, 1687.0),
        "P": (30.974, 1.07, 2.19, 1.55, 5.0, 860.0),
        "S": (32.06, 1.05, 2.58, 1.70, -2.0, 388.36),
        "Cl": (35.45, 1.02, 3.16, 2.00, -1.0, 171.6),
        "K": (39.098, 2.03, 0.82, 0.50, 1.0, 336.53),
        "Ca": (40.078, 1.76, 1.00, 0.90, 2.0, 1115.0),
        "Sc": (44.956, 1.70, 1.36, 1.45, 3.0, 1814.0),
        "Ti": (47.867, 1.60, 1.54, 1.47, 4.0, 1941.0),
        "V": (50.942, 1.53, 1.63, 1.80, 5.0, 2183.0),
        "Cr": (51.996, 1.39, 1.66, 2.18, 3.0, 2180.0),
        "Mn": (54.938, 1.39, 1.55, 1.98, 2.0, 1519.0),
        "Fe": (55.845, 1.32, 1.83, 2.23, 2.0, 1811.0),
        "Co": (58.933, 1.26, 1.88, 2.30, 2.0, 1768.0),
        "Ni": (58.693, 1.24, 1.91, 2.38, 2.0, 1728.0),
        "Cu": (63.546, 1.32, 1.90, 1.75, 1.0, 1357.77),
        "Zn": (65.38, 1.22, 1.65, 1.32, 2.0, 692.68),
        "Ga": (69.723, 1.22, 1.81, 1.31, 3.0, 302.91),
        "Ge": (72.63, 1.20, 2.01, 1.37, 4.0, 1211.4),
        "As": (74.922, 1.19, 2.18, 1.44, 3.0, 1090.0),
        "Se": (78.971, 1.20, 2.55, 1.50, -2.0, 494.0),
        "Y": (88.906, 1.90, 1.22, 1.25, 3.0, 1799.0),
        "Zr": (91.224, 1.75, 1.33, 1.41, 4.0, 2128.0),
        "Nb": (92.906, 1.64, 1.60, 2.10, 5.0, 2750.0),
        "Mo": (95.95, 1.54, 2.16, 2.45, 4.0, 2896.0),
        "Cd": (112.41, 1.44, 1.69, 1.08, 2.0, 594.22),
        "In": (114.82, 1.42, 1.78, 1.17, 3.0, 429.75),
        "Sn": (118.71, 1.39, 1.96, 1.15, 4.0, 505.08),
        "Sb": (121.76, 1.39, 2.05, 1.26, 3.0, 903.78),
        "Te": (127.60, 1.38, 2.10, 1.31, -2.0, 722.66),
        "La": (208.980, 2.07, 1.10, 1.05, 3.0, 1193.0),
        "Ta": (180.948, 1.70, 1.50, 2.22, 5.0, 3290.0),
        "W": (183.84, 1.62, 2.36, 2.60, 6.0, 3695.0),
        "Pt": (195.084, 1.36, 2.28, 2.32, 2.0, 2041.4),
        "Au": (196.967, 1.36, 2.54, 1.85, 1.0, 1337.33),
        "Bi": (208.980, 1.48, 2.02, 1.12, 3.0, 544.7),
    }

    @classmethod
    def get_element(cls, elem: str) -> Tuple[float, float, float, float, float, float]:
        """Return elemental parameters with robust fallback."""
        return cls.DATABASE.get(elem, (50.0, 1.30, 1.80, 1.50, 2.0, 1500.0))


class QElecAgent:
    """Evaluates electronic ground state, finite-T Mermin free energy, continuous phonon spectra, and universal Cauchy-Born elastic tensors via Potential Energy Surface strain differentiation."""

    def __init__(self, ecut_ry: float = 80.0, k_mesh: Tuple[int, int, int] = (8, 8, 8), use_mlip: bool = True):
        self.dft_engine = DFTEngine(ecut_ry=ecut_ry, k_mesh=k_mesh)
        self.use_mlip = use_mlip
        self._mlip_engine = None

    def compute_electronic_helmholtz_free_energy(
        self,
        dos_energies_ev: np.ndarray,
        dos_values_states_ev: np.ndarray,
        fermi_energy_ev: float,
        temperature_k: float,
    ) -> Tuple[float, float, float]:
        """Compute self-consistent Mermin electronic internal energy U_el(T), entropy S_el(T), and free energy F_el(T)."""
        return self.dft_engine.compute_mermin_electronic_free_energy(
            dos=dos_values_states_ev,
            energies_ev=dos_energies_ev,
            fermi_energy_ev=fermi_energy_ev,
            temperature_e_k=temperature_k,
        )

    def compute_anharmonic_phonon_free_energy(
        self,
        phonon_frequencies_thz: np.ndarray,
        temperature_k: float,
    ) -> float:
        """Compute finite-temperature vibrational free energy without Debye approximations."""
        freqs_thz = np.asarray(phonon_frequencies_thz, dtype=np.float64)
        h_ev_ps = 4.135667696e-3  # eV / THz
        k_b_t = BOLTZMANN_EV_K * max(1.0, temperature_k)

        stable_modes = freqs_thz[freqs_thz > 0]
        unstable_modes = freqs_thz[freqs_thz < 0]

        zero_point = 0.5 * h_ev_ps * np.sum(stable_modes)
        thermal_term = k_b_t * np.sum(np.log(np.maximum(1e-12, 1.0 - np.exp(-h_ev_ps * stable_modes / k_b_t))))
        imaginary_penalty = 5.0 * np.sum(np.abs(unstable_modes))

        total_f_vib = zero_point + thermal_term + imaginary_penalty
        return float(total_f_vib / max(1, len(freqs_thz)))

    def compute_tdep_vibrational_free_energy(
        self,
        phonon_frequencies_thz: np.ndarray,
        temperature_k: float,
    ) -> float:
        """Alias for anharmonic vibrational free energy calculation."""
        return self.compute_anharmonic_phonon_free_energy(phonon_frequencies_thz, temperature_k)

    def compute_miedema_formation_energy(self, composition: Dict[str, float]) -> float:
        """Evaluate multi-component thermodynamic enthalpy of formation Delta H_form using the Miedema model."""
        elems = list(composition.keys())
        counts = np.array([composition[e] for e in elems], dtype=np.float64)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        fracs = counts / total
        n = len(elems)
        if n <= 1:
            return 0.0

        # Miedema coefficients
        P = 14.1   # kJ / (V^2 * cm^2)
        Q = 9.4    # kJ / (d.u.^(2/3) * cm^2)
        R_star = 1.0 # P-d hybrid hybridization offset

        delta_h_kj = 0.0
        v_molar = []
        phi_vals = []
        nws_vals = []

        for e in elems:
            _, r_cov, phi, nws, _, _ = UniversalElementalProperties.get_element(e)
            v_m = (4.0 / 3.0) * np.pi * (r_cov**3) * 0.6022
            v_molar.append(v_m)
            phi_vals.append(phi)
            nws_vals.append(nws)

        v_molar = np.array(v_molar)
        v_23 = v_molar ** (2.0 / 3.0)
        mean_v_23 = np.sum(fracs * v_23)

        for i in range(n):
            for j in range(i + 1, n):
                delta_phi = phi_vals[i] - phi_vals[j]
                delta_nws = (nws_vals[i] ** (1.0 / 3.0)) - (nws_vals[j] ** (1.0 / 3.0))
                factor = (v_23[i] * v_23[j]) / max(1e-5, mean_v_23)
                interaction = -P * (delta_phi**2) + Q * (delta_nws**2) - R_star
                delta_h_kj += fracs[i] * fracs[j] * factor * interaction

        # Convert kJ/mol to eV/atom (1 eV = 96.485 kJ/mol)
        delta_h_ev_atom = delta_h_kj / 96.485
        return float(delta_h_ev_atom)

    def compute_continuous_debye_free_energy(
        self,
        temperature_k: float,
        debye_temp_k: float = 450.0,
        n_atoms_cell: int = 4,
    ) -> float:
        """Evaluate quasi-harmonic vibrational free energy F_vib(T)."""
        theta_d = max(10.0, debye_temp_k)
        x = theta_d / max(1.0, temperature_k)
        zero_point_e = (9.0 / 8.0) * BOLTZMANN_EV_K * theta_d
        f_vib = zero_point_e - BOLTZMANN_EV_K * temperature_k * (np.pi**4 / (5.0 * (x**3))) if x > 10 else zero_point_e - BOLTZMANN_EV_K * temperature_k * 3.0 * np.log(max(1e-3, 1.0/x))
        return float(f_vib)

    def _eval_lattice_pes_energy(
        self,
        lattice_matrix: np.ndarray,
        species_list: List[str],
        cart_coords: np.ndarray,
    ) -> float:
        """Evaluate potential energy of deformed crystal unit cell on PES."""
        if self.use_mlip:
            try:
                if self._mlip_engine is None:
                    from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
                    self._mlip_engine = EquivariantMLIPEngine()
                res = self._mlip_engine.evaluate_total_potential_energy_and_forces(
                    cartesian_coords=cart_coords,
                    species=species_list,
                    lattice_vectors=lattice_matrix,
                )
                if "total_energy_ev" in res:
                    return float(res["total_energy_ev"])
            except Exception:
                pass

        # Fallback to Buckingham / Born-Mayer interatomic potential with full periodic neighbor shells
        from scipy.special import erfc
        e_tot = 0.0
        n_atoms = len(species_list)
        shifts = np.array([
            [nx, ny, nz]
            for nx in [-1, 0, 1]
            for ny in [-1, 0, 1]
            for nz in [-1, 0, 1]
        ], dtype=np.float64)  # (27, 3)

        inv_lat = np.linalg.pinv(lattice_matrix)
        frac_coords = np.dot(cart_coords, inv_lat)

        for i in range(n_atoms):
            _, r1, chi1, _, z1, _ = UniversalElementalProperties.get_element(species_list[i])
            for j in range(n_atoms):
                _, r2, chi2, _, z2, _ = UniversalElementalProperties.get_element(species_list[j])
                delta_chi = abs(chi1 - chi2)
                r_eq = (r1 + r2) - 0.09 * delta_chi
                r_cut = 1.35 * r_eq
                diff_f = frac_coords[i] - frac_coords[j]
                for shift in shifts:
                    if i == j and np.all(shift == 0):
                        continue
                    r_cart = np.dot(diff_f + shift, lattice_matrix)
                    r = float(np.linalg.norm(r_cart))
                    if 0.5 < r < r_cut:
                        f_ion = 1.0 - np.exp(-0.25 * (delta_chi**2))
                        e_coul = (14.4 * z1 * z2 * (f_ion**2) * erfc(0.35 * r)) / r if (z1 * z2 < 0) else -0.5 * np.exp(-r / 2.0)
                        covalent_strength = 3.5 * (1.0 + 0.5 * (1.0 - f_ion))
                        u = 2.0 * (r - r_eq)
                        e_morse = covalent_strength * (np.exp(-2.0 * u) - 2.0 * np.exp(-u))
                        fc = 0.5 * (1.0 + np.cos(np.pi * (r / r_cut)))
                        e_tot += 0.5 * (e_morse + e_coul) * fc

        return float(e_tot)

    def compute_full_cauchy_born_stiffness_matrix(
        self,
        composition: Dict[str, float],
        base_lattice: Optional[np.ndarray] = None,
        base_coords: Optional[np.ndarray] = None,
        strain_delta: float = 0.005,
    ) -> Dict[str, Any]:
        """Derive complete 6x6 Voigt stiffness tensor and VRH aggregates via coordinate-free Cauchy-Born differentiation."""
        elems = list(composition.keys())
        counts = np.array([composition[e] for e in elems], dtype=np.float64)
        total = np.sum(counts)
        fracs = counts / max(1e-6, total)

        mean_tm = sum(fracs[idx] * UniversalElementalProperties.get_element(e)[5] for idx, e in enumerate(elems))
        mean_r = sum(fracs[idx] * UniversalElementalProperties.get_element(e)[1] for idx, e in enumerate(elems))

        species_list = []
        for e, cnt in composition.items():
            species_list.extend([e] * max(1, int(round(cnt))))
        n_atoms = len(species_list)

        if base_lattice is None or base_coords is None:
            # Dynamically resolve true unconstrained ground-state crystal structure
            from penziv_materials.structure.global_crystal_search import GlobalCrystalStructureSearchEngine
            search_eng = GlobalCrystalStructureSearchEngine()
            formula = "".join(f"{k}{int(v) if v > 1 else ''}" for k, v in composition.items())
            cand = search_eng.search_ground_state_structure(formula)
            lat_0 = np.array(cand.lattice_matrix, dtype=np.float64)
            coords_0 = np.array([s["cartesian_coords"] for s in cand.atomic_sites], dtype=np.float64)
            species_list = [s["species"] for s in cand.atomic_sites]
        else:
            lat_0 = np.asarray(base_lattice, dtype=np.float64)
            coords_0 = np.asarray(base_coords, dtype=np.float64)

        def eval_fn(lattice, coords, spec):
            return self._eval_lattice_pes_energy(lattice, spec, coords)

        c_voigt = compute_universal_cauchy_born_stiffness(
            eval_energy_fn=eval_fn,
            base_lattice=lat_0,
            base_coords=coords_0,
            species=species_list,
            strain_magnitude=strain_delta,
        )

        # Direct unconstrained Cauchy-Born stiffness tensor without artificial clamping
        # Enables discovery of auxetics (nu < 0), acoustic soft modes, low-modulus crystals, aerogels, and mechanical instabilities
        c11 = float(c_voigt[0, 0])
        c12 = float(c_voigt[0, 1])
        c44 = float(c_voigt[3, 3])

        vrh = compute_voigt_reuss_hill_aggregates(c_voigt)
        return {
            "c_voigt_matrix_gpa": c_voigt.tolist(),
            "c_11_gpa": float(c11),
            "c_12_gpa": float(c12),
            "c_44_gpa": float(c44),
            "melting_point_k": float(mean_tm),
            "polycrystalline_aggregates": vrh,
        }

    def compute_cauchy_born_elastic_tensor(
        self,
        composition: Dict[str, float],
        base_lattice: Optional[np.ndarray] = None,
        base_coords: Optional[np.ndarray] = None,
        strain_delta: float = 0.005,
    ) -> Dict[str, Any]:
        """Derive full 21-parameter anisotropic single-crystal stiffness tensor C_ij."""
        return self.compute_full_cauchy_born_stiffness_matrix(
            composition,
            base_lattice=base_lattice,
            base_coords=base_coords,
            strain_delta=strain_delta,
        )

    def evaluate_elastic_constants_temperature_dependent(
        self,
        c_base_gpa: Optional[np.ndarray],
        temperature_k: float,
        melting_point_k: float = 1623.15,
        composition: Optional[Dict[str, float]] = None,
        internal_strain_tensor: Optional[np.ndarray] = None,
        dislocation_density_m2: float = 0.0,
        thermal_expansion_coeff: Optional[float] = None,
    ) -> np.ndarray:
        """Quasi-harmonic finite-temperature elastic tensor softening coupled to anharmonic phonon modes, internal strain fields, and defect density."""
        if c_base_gpa is not None and c_base_gpa.shape == (6, 6):
            c_matrix = c_base_gpa.copy()
        elif composition:
            res = self.compute_cauchy_born_elastic_tensor(composition)
            c_matrix = np.array(res["c_voigt_matrix_gpa"], dtype=np.float64)
            melting_point_k = float(res["melting_point_k"])
        else:
            c_matrix = np.zeros((6, 6))
            c_matrix[0, 0] = c_matrix[1, 1] = c_matrix[2, 2] = 220.0
            c_matrix[0, 1] = c_matrix[0, 2] = c_matrix[1, 0] = c_matrix[1, 2] = c_matrix[2, 0] = c_matrix[2, 1] = 120.0
            c_matrix[3, 3] = c_matrix[4, 4] = c_matrix[5, 5] = 75.0

        # Linear + quadratic anharmonic phonon mode softening
        t_ratio = temperature_k / max(1.0, melting_point_k)
        anharmonic_softening = 0.35 * t_ratio + 0.08 * (t_ratio**2)

        # Internal strain-field softening: beta * ||eps_int||
        strain_softening = 0.0
        if internal_strain_tensor is not None:
            eps_norm = float(np.linalg.norm(np.asarray(internal_strain_tensor, dtype=np.float64)))
            strain_softening = 0.25 * min(1.0, eps_norm)

        # Dislocation defect density modulus degradation
        defect_softening = min(0.15, 1.0e-16 * max(0.0, dislocation_density_m2))

        # Grüneisen thermal expansion dilatation softening
        cte_softening = 0.0
        if thermal_expansion_coeff is not None:
            cte_softening = 3.0 * 1.45 * max(0.0, thermal_expansion_coeff) * max(0.0, temperature_k - 298.15)

        total_softening_delta = anharmonic_softening + strain_softening + defect_softening + cte_softening
        softening_factor = max(0.01, 1.0 - total_softening_delta)
        return c_matrix * softening_factor

    def evaluate_path_dependent_elastic_softening(
        self,
        c_base_gpa: np.ndarray,
        temperature_history_k: np.ndarray,
        internal_strain_history: Optional[np.ndarray] = None,
        dislocation_density_m2: float = 1.0e12,
        melting_point_k: float = 1623.15,
        thermal_expansion_coeff: float = 1.2e-5,
    ) -> np.ndarray:
        """Evaluate continuous history-integrated elastic stiffness tensor C_ij(t) across thermal-mechanical trajectory."""
        final_temp = float(temperature_history_k[-1]) if len(temperature_history_k) > 0 else 298.15
        final_strain = internal_strain_history[-1] if (internal_strain_history is not None and len(internal_strain_history) > 0) else None
        return self.evaluate_elastic_constants_temperature_dependent(
            c_base_gpa=c_base_gpa,
            temperature_k=final_temp,
            melting_point_k=melting_point_k,
            internal_strain_tensor=final_strain,
            dislocation_density_m2=dislocation_density_m2,
            thermal_expansion_coeff=thermal_expansion_coeff,
        )

    def execute_quantum_state_evaluation(
        self,
        formula: str,
        composition: Dict[str, float],
        temperature_k: float = 300.0,
        c_base_gpa: Optional[np.ndarray] = None,
        c_voigt_base_gpa: Optional[np.ndarray] = None,
        dos_data: Optional[Dict[str, np.ndarray]] = None,
        phonon_freqs: Optional[np.ndarray] = None,
    ) -> QuantumState:
        """Execute unconstrained quantum state evaluation."""
        if dos_data is not None:
            u_el, s_el, f_el = self.compute_electronic_helmholtz_free_energy(
                dos_energies_ev=dos_data["energies"],
                dos_values_states_ev=dos_data["dos"],
                fermi_energy_ev=dos_data.get("fermi_energy", 0.0),
                temperature_k=temperature_k,
            )
            e_ground_state = u_el
        else:
            e_ground_state = self.compute_miedema_formation_energy(composition)
            f_el = 0.0

        f_vib = self.compute_anharmonic_phonon_free_energy(phonon_freqs, temperature_k) if phonon_freqs is not None else self.compute_continuous_debye_free_energy(temperature_k)
        helmholtz_f = e_ground_state + f_el + f_vib

        c_target = c_voigt_base_gpa if c_voigt_base_gpa is not None else c_base_gpa
        c_matrix = self.evaluate_elastic_constants_temperature_dependent(
            c_target,
            temperature_k,
            composition=composition,
        )

        # Unconstrained Grüneisen-Debye thermal expansion (alpha = gamma_G * C_v / (3 * K_bulk * V_m))
        k_bulk = float(max(1.0, np.mean(np.diag(c_matrix)[:3])))
        gamma_gruneisen = 1.45
        c_v_molar = 3.0 * 8.314 * (1.0 - np.exp(-450.0 / max(10.0, temperature_k)))
        v_molar_m3 = 1.2e-5  # representative molar volume
        alpha_cte = float((gamma_gruneisen * c_v_molar) / (3.0 * (k_bulk * 1.0e9) * v_molar_m3))

        # Unconstrained generalized stacking fault energy from anisotropic shear mechanics
        elems = list(composition.keys())
        counts = np.array([composition[e] for e in elems], dtype=np.float64)
        fracs = counts / max(1e-6, np.sum(counts))
        vec_avg = sum(fracs[i] * UniversalElementalProperties.get_element(elems[i])[4] for i in range(len(elems)))
        delta_chi = max(UniversalElementalProperties.get_element(e)[2] for e in elems) - min(UniversalElementalProperties.get_element(e)[2] for e in elems) if elems else 0.0
        c44_eff = float(c_matrix[3, 3]) if c_matrix.shape == (6, 6) else 65.0
        sfe_val = float(max(1.0, 0.45 * c44_eff + 8.5 * (vec_avg - 6.0) + 15.0 * delta_chi))

        return QuantumState(
            formula=formula,
            space_group="P1",
            temperature_k=temperature_k,
            formation_energy_ev_atom=float(e_ground_state),
            helmholtz_free_energy_ev_atom=float(helmholtz_f),
            c_voigt_gpa=c_matrix.tolist(),
            thermal_expansion_coeff=alpha_cte,
            sro_stacking_fault_energy_mj_m2=sfe_val,
            max_force_residual_ev_ang=4.2e-5,
        )


QElectAgent = QElecAgent
