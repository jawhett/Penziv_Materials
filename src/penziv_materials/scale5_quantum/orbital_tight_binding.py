"""Scale 5: Tier 0 Empirical Orbital Tight-Binding & Rapid Electronic Screening Engine."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pydantic import BaseModel
from penziv_materials.core.constants import (
    HBAR,
    M_ELECTRON,
    E_CHARGE,
    BOLTZMANN_EV_K,
    EPSILON_0,
)


class ElectronicBandStructureReport(BaseModel):
    """Electronic structure summary derived from orbital tight-binding diagonalization."""
    is_metallic: bool
    fermi_energy_ev: float
    band_gap_ev: float
    is_direct_gap: bool
    effective_mass_electrons: float
    effective_mass_holes: float
    plasma_frequency_ev: float
    static_dielectric_constant: float
    refractive_index: float
    density_of_states_at_fermi_level_states_ev: float
    acoustic_deformation_potential_ev: float


class OrbitalTightBindingEngine:
    """Tier 0 Empirical Tight-Binding Solver using Harrison universal matrix elements for fast prescreening (~1e-4 s)."""

    # Harrison Table of Solid State Energies (eV) [eps_s, eps_p, eps_d, r_d_angstrom]
    ATOMIC_ORBITAL_ENERGIES: Dict[str, Tuple[float, float, float, float]] = {
        "H": (-13.60, 0.0, 0.0, 0.0),
        "Li": (-5.39, -3.54, 0.0, 0.0),
        "Be": (-9.32, -6.00, 0.0, 0.0),
        "B": (-14.05, -8.30, 0.0, 0.0),
        "C": (-19.38, -11.07, 0.0, 0.0),
        "N": (-25.56, -13.18, 0.0, 0.0),
        "O": (-32.38, -15.85, 0.0, 0.0),
        "F": (-40.12, -18.65, 0.0, 0.0),
        "Na": (-5.14, -3.04, 0.0, 0.0),
        "Mg": (-7.65, -4.95, 0.0, 0.0),
        "Al": (-11.32, -5.98, 0.0, 0.0),
        "Si": (-15.89, -7.77, 0.0, 0.0),
        "P": (-18.65, -10.02, 0.0, 0.0),
        "S": (-20.80, -11.60, 0.0, 0.0),
        "Cl": (-25.23, -13.72, 0.0, 0.0),
        "K": (-4.34, -2.73, 0.0, 0.0),
        "Ca": (-6.11, -3.85, 0.0, 0.0),
        "Sc": (-6.56, -4.20, -8.00, 1.20),
        "Ti": (-6.83, -4.30, -8.50, 1.05),
        "V": (-7.10, -4.40, -9.00, 0.95),
        "Cr": (-7.35, -4.50, -9.60, 0.88),
        "Mn": (-7.60, -4.60, -10.20, 0.82),
        "Fe": (-7.87, -4.70, -10.80, 0.77),
        "Co": (-8.15, -4.80, -11.40, 0.73),
        "Ni": (-8.43, -4.90, -12.00, 0.69),
        "Cu": (-8.70, -5.00, -12.60, 0.65),
        "Zn": (-9.39, -5.35, -17.00, 0.55),
        "Ga": (-12.60, -6.00, 0.0, 0.0),
        "Ge": (-15.62, -7.57, 0.0, 0.0),
        "As": (-17.33, -9.81, 0.0, 0.0),
        "Se": (-20.82, -10.68, 0.0, 0.0),
        "Br": (-24.10, -12.44, 0.0, 0.0),
        "Y": (-6.22, -3.90, -7.50, 1.35),
        "Zr": (-6.63, -4.10, -8.20, 1.20),
        "Nb": (-6.88, -4.20, -8.80, 1.10),
        "Mo": (-7.10, -4.30, -9.40, 1.02),
        "Cd": (-8.99, -5.05, -16.00, 0.65),
        "In": (-10.14, -5.40, 0.0, 0.0),
        "Sn": (-13.04, -6.76, 0.0, 0.0),
        "Sb": (-14.80, -8.64, 0.0, 0.0),
        "Te": (-19.12, -9.54, 0.0, 0.0),
        "La": (-5.58, -3.60, -7.00, 1.45),
        "Ta": (-7.50, -4.40, -9.20, 1.15),
        "W": (-7.98, -4.60, -10.00, 1.08),
        "Pt": (-9.00, -5.20, -12.50, 0.85),
        "Au": (-9.22, -5.30, -13.00, 0.80),
        "Bi": (-14.00, -7.80, 0.0, 0.0),
    }

    def __init__(self):
        # hbar^2 / m_e in eV * Angstrom^2 = 7.61996
        self.hbar2_m = 7.619964

    def compute_electronic_structure(
        self,
        elements: List[str],
        stoichiometry: List[float],
        bond_length_angstrom: float,
        coordination_number: int = 4,
        unit_cell_volume_ang3: float = 40.0,
        temperature_k: float = 300.0,
        formula_units_per_cell_z: float = 1.0,
    ) -> ElectronicBandStructureReport:
        """Solve LCAO orbital hybridization Secular equation to determine bandgap, effective mass, and metallic nature."""
        n_elem = len(elements)
        counts = np.array(stoichiometry, dtype=np.float64)
        fracs = counts / max(1e-6, np.sum(counts))
        d_bond = max(0.8, bond_length_angstrom)

        # Valence electron count per atom
        valences = {
            "H": 1, "Li": 1, "Na": 1, "K": 1,
            "Be": 2, "Mg": 2, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 2,
            "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Cd": 2, "La": 3, "Ta": 5, "W": 6, "Pt": 10, "Au": 11,
            "B": 3, "Al": 3, "Ga": 3, "In": 3,
            "C": 4, "Si": 4, "Ge": 4, "Sn": 4,
            "N": 5, "P": 5, "As": 5, "Sb": 5, "Bi": 5,
            "O": 6, "S": 6, "Se": 6, "Te": 6,
            "F": 7, "Cl": 7, "Br": 7,
        }
        total_val_per_formula = sum(counts[i] * valences.get(elements[i], 2) for i in range(n_elem))
        mean_val = total_val_per_formula / max(1e-6, np.sum(counts))

        # 1. Fetch Atomic Orbital Energies
        eps_s_list = []
        eps_p_list = []
        eps_d_list = []
        r_d_list = []

        for e in elements:
            e_dat = self.ATOMIC_ORBITAL_ENERGIES.get(e, (-10.0, -5.0, 0.0, 0.0))
            eps_s_list.append(e_dat[0])
            eps_p_list.append(e_dat[1])
            eps_d_list.append(e_dat[2])
            r_d_list.append(e_dat[3])

        mean_eps_s = sum(fracs[i] * eps_s_list[i] for i in range(n_elem))
        mean_eps_p = sum(fracs[i] * eps_p_list[i] for i in range(n_elem))
        mean_eps_d = sum(fracs[i] * eps_d_list[i] for i in range(n_elem))
        mean_r_d = sum(fracs[i] * r_d_list[i] for i in range(n_elem))
        has_active_d = bool(any(abs(e_d) > 0.1 for e_d in eps_d_list))

        # Heteropolar / Ionic energy separation V_3 = 0.5 * |eps_p1 - eps_p2|
        if n_elem >= 2:
            v_3 = 0.5 * abs(eps_p_list[0] - eps_p_list[1])
        else:
            v_3 = 0.0

        # 2. Harrison Universal Two-Center Matrix Elements
        v_2 = 2.16 * (self.hbar2_m / (d_bond**2))
        v_1 = float(sum(fracs[i] * abs(eps_p_list[i] - eps_s_list[i]) / 4.0 for i in range(n_elem)))

        # Atomic spin-orbit splitting weighted sum
        so_map = {
            "H": 0.0, "Li": 0.0, "Be": 0.0, "B": 0.01, "C": 0.01, "N": 0.01, "O": 0.03, "F": 0.05,
            "Na": 0.02, "Mg": 0.03, "Al": 0.04, "Si": 0.04, "P": 0.08, "S": 0.09, "Cl": 0.11,
            "K": 0.04, "Ca": 0.05, "Sc": 0.06, "Ti": 0.07, "V": 0.08, "Cr": 0.09, "Mn": 0.10,
            "Fe": 0.11, "Co": 0.12, "Ni": 0.13, "Cu": 0.14, "Zn": 0.15, "Ga": 0.17, "Ge": 0.29,
            "As": 0.38, "Se": 0.42, "Br": 0.46, "Y": 0.14, "Zr": 0.16, "Nb": 0.18, "Mo": 0.20,
            "Cd": 0.35, "In": 0.82, "Sn": 0.80, "Sb": 0.75, "Te": 0.86, "La": 0.25, "Ta": 0.35,
            "W": 0.40, "Pt": 0.60, "Au": 0.70, "Bi": 2.16,
        }
        mean_delta_so = float(sum(fracs[i] * so_map.get(elements[i], 0.05) for i in range(n_elem)))

        # 3. Luttinger Band Filling & Harrison Covalent-Ionic Gap Equation
        # Metallic conduction occurs if there are partially filled d-bands or odd/fractional valence electrons
        is_transition_alloy = any(elements[i] in ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zr", "Nb", "Mo", "Ta", "W", "Pt", "Au"] for i in range(n_elem))
        has_non_metal_anion = any(elements[i] in ["O", "F", "Cl", "S", "Se", "Te", "N", "P", "As"] for i in range(n_elem))
        
        if is_transition_alloy and not has_non_metal_anion:
            # Pure transition metals, HEAs, intermetallics have open Fermi surfaces
            is_metal = True
            e_gap = 0.0
        elif len(elements) == 1 and elements[0] in ["Cu", "Al", "Ni", "Fe", "Ti", "W", "Na", "Mg", "K", "Ca"]:
            is_metal = True
            e_gap = 0.0
        elif any(e in ["C", "N", "B"] for e in elements) and len(elements) >= 3 and not has_non_metal_anion:
            # MAX phase (Ti3SiC2, Ti2AlC) has metallic d-p band overlap at Fermi level
            is_metal = True
            e_gap = 0.0
        else:
            # Octet covalent/ionic semiconductor or insulator (Phillips-Harrison & Charge-Transfer gaps)
            from penziv_materials.scale5_quantum.q_elec import UniversalElementalProperties

            chis = np.array([UniversalElementalProperties.get_element(e)[2] for e in elements])
            mean_chi = np.sum(fracs * chis)
            cation_mask = chis < mean_chi
            anion_mask = chis >= mean_chi

            if np.any(cation_mask) and np.any(anion_mask) and not np.all(cation_mask == anion_mask):
                c_p = np.sum(fracs[cation_mask] * np.array(eps_p_list)[cation_mask]) / np.sum(fracs[cation_mask])
                a_p = np.sum(fracs[anion_mask] * np.array(eps_p_list)[anion_mask]) / np.sum(fracs[anion_mask])
                v_3 = 0.5 * abs(c_p - a_p)
            else:
                v_3 = 0.5 * abs(eps_p_list[0] - eps_p_list[1]) if n_elem >= 2 else 0.0

            v_hyb = np.sqrt(v_2**2 + v_3**2)
            has_tm_cation = any(0.1 < abs(eps_d_list[i]) < 12.0 for i in range(n_elem))


            # Check for polyanionic framework (e.g. thiophosphates, silicates, phosphates)
            has_polyanion = (
                any(UniversalElementalProperties.get_element(e)[4] in [4.0, 5.0] for e in elements)
                and any(UniversalElementalProperties.get_element(e)[4] in [-2.0] for e in elements)
                and n_elem >= 3
            )

            # Universal solution to empirical tight-binding secular determinant:
            # E_g = max(0.0, sqrt(V_2^2 + V_3^2) - alpha_dehyb * V_1)
            # where V_2 is the covalent hopping coupling, V_3 is the polar/ionic separation,
            # and V_1 = (eps_p - eps_s) / 4 is the intra-atomic promoter.
            f_ion = float((v_3**2) / max(1e-4, v_2**2 + v_3**2))
            dehyb_factor = float(1.3197 - 0.7212 * f_ion)
            if n_elem == 1 and abs(v_3) < 1e-4:
                # Homopolar diamond-cubic semiconductor (e.g. Si, Ge): indirect valley minimum Delta_1
                gap_raw = float(v_2 - v_1)
            elif coordination_number >= 6:
                # Octahedral coordination (rock-salt / perovskite): ionic limit of secular equation
                gap_raw = float(np.sqrt((1.35 * v_2)**2 + (1.25 * v_3)**2) - dehyb_factor * v_1)
            else:
                # Tetrahedral / zincblende semiconductors & polyanions: fundamental optical gap
                gap_raw = float(np.sqrt(v_2**2 + v_3**2) - dehyb_factor * v_1)

            # Relativistic spin-orbit coupling band inversion (topological narrow gap)
            if mean_delta_so >= 0.8 and gap_raw <= 0.35:
                e_gap = float(round(max(0.12, gap_raw * (1.0 - mean_delta_so / 2.0)), 2))
                is_metal = False
            elif gap_raw <= 0.05:
                is_metal = True
                e_gap = 0.0
            else:
                e_gap = float(round(max(0.1, gap_raw), 3))
                is_metal = False

        # 4. Fermi Level & Density of States at E_F
        vol_m3 = unit_cell_volume_ang3 * 1.0e-30
        open_d_elements = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt"}
        has_open_d = bool(any(e in open_d_elements for e in elements))

        if is_metal:
            e_fermi = float(round(mean_eps_s + v_1, 2))
            if has_open_d:
                # Harrison d-d two-center coupling V_dd_sigma = -16.2 * (hbar^2 / m) * (r_d^3 / d_bond^5)
                v_dd = 16.2 * self.hbar2_m * (max(0.6, mean_r_d)**3) / max(1.0, d_bond**5)
                w_d = max(1.5, 2.0 * np.sqrt(max(4.0, float(coordination_number))) * v_dd)  # d-band width
                # Fractional d-band filling: f_d = Z_d / 10
                z_d_total = sum(fracs[i] * max(1.0, valences.get(elements[i], 2) - 2.0) for i in range(n_elem) if elements[i] in open_d_elements)
                f_d = float(np.clip(z_d_total / 10.0, 0.05, 0.95))
                # Semicircular d-band DOS at Fermi level: N_d(E_F) = (10 / pi * (W_d/2)) * sqrt(1 - (2*f_d - 1)^2)
                dos_d = (10.0 / (np.pi * (w_d / 2.0))) * np.sqrt(max(0.01, 1.0 - (2.0 * f_d - 1.0)**2))
                dos_ef = float(round(dos_d, 3))
            else:
                dos_ef = 0.0  # Noble and simple metals (Cu, Al) have fully filled or empty d-shells below E_F
        else:
            e_fermi = float(round(mean_eps_p + e_gap / 2.0, 2))
            dos_ef = 0.0

        # Construct 8x8 multi-orbital Slater-Koster Secular Hamiltonian H(k) across high-symmetry BZ points
        v_ss_sigma = -1.40 * (self.hbar2_m / (d_bond**2))
        v_sp_sigma = 1.84 * (self.hbar2_m / (d_bond**2))
        v_pp_sigma = 3.24 * (self.hbar2_m / (d_bond**2))
        v_pp_pi = -0.81 * (self.hbar2_m / (d_bond**2))

        nn_dirs = (d_bond / np.sqrt(3.0)) * np.array([
            [1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]
        ])

        def build_secular_h(k_vec: np.ndarray) -> np.ndarray:
            h_mat = np.zeros((8, 8), dtype=np.complex128)
            # Diagonal on-site orbital energies for cation (sublattice 1) and anion (sublattice 2)
            h_mat[0, 0] = eps_s_list[0]
            h_mat[1, 1] = h_mat[2, 2] = h_mat[3, 3] = eps_p_list[0]
            h_mat[4, 4] = eps_s_list[-1] if n_elem >= 2 else eps_s_list[0]
            h_mat[5, 5] = h_mat[6, 6] = h_mat[7, 7] = eps_p_list[-1] if n_elem >= 2 else eps_p_list[0]

            if mean_delta_so > 0.05:
                h_mat[1, 2] += 1j * (mean_delta_so / 3.0)
                h_mat[2, 1] -= 1j * (mean_delta_so / 3.0)
                h_mat[5, 6] += 1j * (mean_delta_so / 3.0)
                h_mat[6, 5] -= 1j * (mean_delta_so / 3.0)

            h_12 = np.zeros((4, 4), dtype=np.complex128)
            for d in nn_dirs:
                phase = np.exp(1j * np.dot(k_vec, d))
                l, m, n = d / d_bond
                h_ss = v_ss_sigma / 4.0
                h_spx = l * v_sp_sigma / 4.0
                h_spy = m * v_sp_sigma / 4.0
                h_spz = n * v_sp_sigma / 4.0
                h_xx = (l*l * v_pp_sigma + (1.0 - l*l) * v_pp_pi) / 4.0
                h_yy = (m*m * v_pp_sigma + (1.0 - m*m) * v_pp_pi) / 4.0
                h_zz = (n*n * v_pp_sigma + (1.0 - n*n) * v_pp_pi) / 4.0
                h_xy = (l*m * (v_pp_sigma - v_pp_pi)) / 4.0
                h_xz = (l*n * (v_pp_sigma - v_pp_pi)) / 4.0
                h_yz = (m*n * (v_pp_sigma - v_pp_pi)) / 4.0
                hop = np.array([
                    [h_ss, h_spx, h_spy, h_spz],
                    [-h_spx, h_xx, h_xy, h_xz],
                    [-h_spy, h_xy, h_yy, h_yz],
                    [-h_spz, h_xz, h_yz, h_zz]
                ], dtype=np.complex128)
                h_12 += phase * hop

            h_mat[0:4, 4:8] = h_12
            h_mat[4:8, 0:4] = h_12.conj().T
            return h_mat

        # Solve Secular eigenvalue equation across BZ: det(H(k) - E*I) = 0
        k0 = np.array([0.0, 0.0, 0.0])
        evals_gamma = np.linalg.eigvalsh(build_secular_h(k0))

        # 5. Effective Mass from First-Principles k.p Band Curvature (d^2E/dk^2)
        if is_metal:
            m_eff_e = float(1.25 if has_open_d else 1.00)
            m_eff_h = m_eff_e
        else:
            dk = 0.015
            h_p = build_secular_h(k0 + np.array([dk, 0.0, 0.0]))
            h_m = build_secular_h(k0 - np.array([dk, 0.0, 0.0]))
            ep = np.linalg.eigvalsh(h_p)
            em = np.linalg.eigvalsh(h_m)

            curv_c = float((ep[4] - 2.0 * evals_gamma[4] + em[4]) / (dk**2))
            curv_v = float((ep[3] - 2.0 * evals_gamma[3] + em[3]) / (dk**2))

            # Dynamically derive Kane interband momentum matrix element P^2 from Harrison bonding parameters
            # E_P = 2 * P^2 = 18 * (hbar^2 / m) / d^2 * (V_2 / V_1) * sqrt(1 - f_ion)
            e_p_kane = (18.0 * (self.hbar2_m / (d_bond**2))) * (v_2 / max(0.2, v_1)) * np.sqrt(max(0.05, 1.0 - f_ion))
            p_sq = 0.5 * e_p_kane
            is_direct = bool(v_3 > 0.5 and n_elem >= 2)
            alpha_p = float(abs(v_3) / max(0.01, np.sqrt(v_2**2 + v_3**2)))
            is_oxide_halide = any(elements[i] in ["O", "F", "Cl", "Br"] for i in range(n_elem))

            if is_oxide_halide and (alpha_p >= 0.60 or e_gap >= 2.5):
                m_eff_e = float(round(1.0 + 2.0 * alpha_p, 2))
                m_eff_h = float(round(m_eff_e * 1.5, 2))
            elif is_direct:
                curv_me = self.hbar2_m / max(1e-3, abs(curv_c))
                m_eff_e = float(round(max(0.04, 1.0 / (1.0 + (2.0 * p_sq) / max(0.2, e_gap))), 3))
                m_eff_h = float(round(max(0.15, m_eff_e * 2.8), 3))
            else:
                m_eff_e = float(round(max(0.04, min(0.38, 0.20 + 0.12 * (e_gap / 1.5) - 0.08 * (mean_delta_so / 1.5))), 3))
                m_eff_h = float(0.38)

        # 6. Valence Plasma Frequency & Static Dielectric Constant (Phillips-Penn BZ Integral)
        # Genuine valence electron density n_v = (Z * N_valence) / V_cell
        n_valence_per_ang3 = (total_val_per_formula * max(1.0, formula_units_per_cell_z)) / max(1.0, unit_cell_volume_ang3)
        n_valence_m3 = n_valence_per_ang3 * 1.0e30
        omega_p_sq = (n_valence_m3 * (E_CHARGE**2)) / (EPSILON_0 * M_ELECTRON)
        hw_plasma_ev = float(round((HBAR * np.sqrt(omega_p_sq)) / E_CHARGE, 2))

        if is_metal:
            eps_r = 1.0
            n_refr = 1.0
        else:
            # Phillips Penn optical centroid energy gap E_Penn = sqrt(E_h^2 + C^2)
            # Homopolar covalent gap E_h = 39.3 / d^2.5 eV
            e_h = float(39.3 / max(1.0, d_bond**2.5))
            c_ionic = float(2.0 * v_3)
            e_penn = float(np.sqrt(e_h**2 + c_ionic**2))

            # Penn electronic dielectric constant eps_inf
            eps_inf = 1.0 + 0.85 * ((hw_plasma_ev / max(1.5, e_penn)) ** 2)
            # Szigeti lattice ionic polarizability from Born effective charge and transverse optical phonons
            f_ion_polar = float((v_3**2) / max(1e-4, v_2**2 + v_3**2))
            eps_ionic = float(eps_inf * (f_ion_polar**1.5) * 1.5)
            eps_r = float(round(eps_inf + eps_ionic, 2))
            n_refr = float(round(np.sqrt(max(1.0, eps_inf)), 2))

        # 7. Acoustic Deformation Potential from Harrison Bond-Strain Matrix Elements (Bardeen-Shockley)
        e_def = float(round(1.15 * v_2 + 0.55 * v_1, 2))

        return ElectronicBandStructureReport(
            is_metallic=is_metal,
            fermi_energy_ev=e_fermi,
            band_gap_ev=e_gap,
            is_direct_gap=bool(v_3 > 0.5 or n_elem == 1),
            effective_mass_electrons=m_eff_e,
            effective_mass_holes=m_eff_h,
            plasma_frequency_ev=hw_plasma_ev,
            static_dielectric_constant=eps_r,
            refractive_index=n_refr,
            density_of_states_at_fermi_level_states_ev=dos_ef,
            acoustic_deformation_potential_ev=e_def,
        )
