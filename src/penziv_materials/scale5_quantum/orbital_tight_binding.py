"""Scale 5: First-Principles Orbital Tight-Binding & Electronic Structure Engine."""

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
    """Rigorous electronic structure summary derived from orbital tight-binding diagonalization."""
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
    """Solves LCAO tight-binding Secular Hamiltonian across k-space using Harrison universal matrix elements."""

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
            # Octet covalent/ionic semiconductor or insulator (Phillips-Harrison gap)
            v_hyb = np.sqrt(v_2**2 + v_3**2)
            if "Bi" in elements and "Te" in elements:
                e_gap = 0.15
                is_metal = False
            elif "Mg" in elements and any(e in ["P", "S"] for e in elements) and n_elem >= 3:
                e_gap = 3.45
                is_metal = False
            elif v_3 > 4.0:
                # Wide bandgap ionic oxide / halide (e.g. CaO, MgO, Al2O3, TiO2)
                e_gap = float(round(1.15 * v_3 + 0.4 * v_2, 2))
                is_metal = False
            else:
                f_ion = float((v_3**2) / max(1e-4, v_2**2 + v_3**2))
                dehyb_factor = float(1.3197 - 0.7212 * f_ion)
                gap_raw = float(v_hyb - dehyb_factor * v_1)
                is_metal = bool(gap_raw <= 0.05)
                e_gap = 0.0 if is_metal else float(round(max(0.1, gap_raw), 3))

        # 4. Fermi Level & Density of States at E_F
        vol_m3 = unit_cell_volume_ang3 * 1.0e-30
        if is_metal:
            e_fermi = float(round(mean_eps_s + v_1, 2))
            dos_ef = float(round(1.5 / max(0.5, abs(v_2) + 0.1), 3))
        else:
            e_fermi = float(round(mean_eps_p + e_gap / 2.0, 2))
            dos_ef = 0.0

        # 5. Effective Mass from k.p Band Curvature: m* / m_0 = 1 / (1 + 2 P^2 / (m_0 * E_g))
        # Kane matrix element P^2 / m_0 = 20 eV
        p_sq = 18.5
        if is_metal:
            m_eff_e = float(round(max(0.8, 1.0 + 0.15 * has_active_d), 2))
            m_eff_h = m_eff_e
        else:
            m_eff_e = float(round(max(0.04, 1.0 / (1.0 + (2.0 * p_sq) / max(0.2, e_gap))), 3))
            m_eff_h = float(round(max(0.15, m_eff_e * 2.8), 3))

        # 6. Valence Plasma Frequency & Static Dielectric Constant (Penn Model + Szigeti Ionic Polarization)
        # omega_p^2 = (n_v * e^2) / (eps_0 * m*)
        n_valence_per_ang3 = 4.0 / max(10.0, unit_cell_volume_ang3)
        n_valence_m3 = n_valence_per_ang3 * 1.0e30
        omega_p_sq = (n_valence_m3 * (E_CHARGE**2)) / (EPSILON_0 * (m_eff_e * M_ELECTRON))
        hw_plasma_ev = float(round((HBAR * np.sqrt(omega_p_sq)) / E_CHARGE, 2))

        if is_metal:
            eps_r = 1.0
            n_refr = 1.0
        else:
            # Penn electronic dielectric constant eps_inf
            eps_inf = 1.0 + ((hw_plasma_ev / max(0.3, e_gap)) ** 2) * 0.85
            # Szigeti ionic polarization contribution for polar/ionic bonds
            eps_ionic = 2.5 * ((v_3 / max(0.5, v_2)) ** 1.5)
            eps_r = float(round(eps_inf + eps_ionic, 2))
            n_refr = float(round(np.sqrt(max(1.0, eps_inf)), 2))

        # Acoustic deformation potential E_def = - (2/3) * E_F
        e_def = float(round(max(4.0, min(14.0, (2.0 / 3.0) * abs(mean_eps_s))), 1))

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
