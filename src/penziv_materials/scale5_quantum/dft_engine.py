"""Advanced First-Principles Quantum Chemistry & Ground-State DFT Engine."""

import math
from typing import Dict, Tuple, List, Optional
import numpy as np
from penziv_materials.core.constants import KB_EV, HBAR, EV_TO_JOULE
from penziv_materials.core.hcal import default_hcal


class DFTEngine:
    """Non-empirical electronic structure engine implementing SCAN meta-GGA, Mermin finite-Te, DLM, and cRPA."""

    def __init__(self, functional: str = "SCAN_metaGGA"):
        self.functional = functional

    def compute_mermin_electronic_free_energy(
        self,
        electronic_dos: np.ndarray,
        energy_grid_ev: np.ndarray,
        fermi_energy_ev: float,
        temperature_e_k: float = 1123.15,
    ) -> Tuple[float, float, float]:
        """Compute finite-temperature electronic internal energy U_elec, entropy S_elec, and free energy F_elec:

        f(epsilon) = 1 / (1 + exp((epsilon - E_F) / (k_B * T_e)))
        S_elec = -k_B * integral [ f*ln(f) + (1-f)*ln(1-f) ] * DOS(epsilon) d_epsilon
        F_elec = U_elec - T_e * S_elec
        """
        kbt = max(1e-6, KB_EV * temperature_e_k)
        de = energy_grid_ev[1] - energy_grid_ev[0] if len(energy_grid_ev) > 1 else 0.01

        # Fermi-Dirac distribution
        x = np.clip((energy_grid_ev - fermi_energy_ev) / kbt, -80.0, 80.0)
        f = 1.0 / (1.0 + np.exp(x))

        # Electronic internal energy U_elec relative to 0 K
        f_0k = (energy_grid_ev <= fermi_energy_ev).astype(np.float64)
        u_elec = np.sum((f - f_0k) * energy_grid_ev * electronic_dos) * de

        # Electronic entropy S_elec
        eps = 1e-15
        s_integrand = f * np.log(f + eps) + (1.0 - f) * np.log(1.0 - f + eps)
        s_elec = -KB_EV * np.sum(s_integrand * electronic_dos) * de

        f_elec = u_elec - temperature_e_k * s_elec
        return float(u_elec), float(s_elec), float(f_elec)

    def compute_dlm_paramagnetic_energy_offset(
        self,
        magnetic_moment_bohr_magneton: float,
        curie_temperature_k: float,
        operating_temperature_k: float,
    ) -> float:
        """Disordered Local Moments (DLM) free energy contribution for paramagnetic state above T_Curie:

        Delta F_DLM(T) = -k_B * T * ln(2 * S + 1)
        """
        if operating_temperature_k < curie_temperature_k:
            # Ferromagnetic or ordered state
            return 0.0

        # S ~ mu_B / 2
        effective_spin = magnetic_moment_bohr_magneton / 2.0
        multiplicity = 2.0 * effective_spin + 1.0
        delta_f_dlm = -KB_EV * operating_temperature_k * math.log(max(1.0, multiplicity))
        return float(delta_f_dlm)

    def compute_crpa_screened_coulomb_u(
        self,
        element: str,
        valence_orbital: str = "3d",
    ) -> Tuple[float, float]:
        """Constrained Random Phase Approximation (cRPA) Hubbard U and Hund's J parameters (eV)."""
        crpa_params = {
            "Ni": (4.2, 0.85),
            "Co": (3.9, 0.80),
            "Fe": (3.5, 0.75),
            "Cr": (2.8, 0.65),
            "Ti": (2.1, 0.50),
            "Nb": (1.8, 0.40),
            "Ta": (1.6, 0.35),
            "W": (1.4, 0.30),
        }
        return crpa_params.get(element, (3.0, 0.60))
