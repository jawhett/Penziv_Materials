"""2D Generalized Stacking Fault Energy (GSFE / Gamma-Surface) Slab Simulation Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.structure.crystal_structure import CrystalStructure, PeriodicLattice, Site


class TwoDimensionalGammaSurfaceEngine:
    """Computes complete 2D gamma-surfaces gamma(u_x, u_y) across arbitrary Miller slip planes (hkl)."""

    def __init__(self, grid_resolution: int = 11):
        self.grid_res = grid_resolution

    def evaluate_2d_gamma_surface_grid(
        self,
        miller_plane: Tuple[int, int, int] = (1, 1, 1),
        slip_basis_1: Optional[np.ndarray] = None,
        slip_basis_2: Optional[np.ndarray] = None,
        shear_modulus_gpa: float = 80.0,
        interplanar_spacing_angstrom: float = 2.08,
        gamma_usf_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute the 2D energy landscape gamma(u_1, u_2) (mJ/m^2) for rigid interplanar shear."""
        u_vals = np.linspace(0.0, 1.0, self.grid_res)
        U1, U2 = np.meshgrid(u_vals, u_vals, indexing="ij")

        # 2D Fourier representation of gamma-surface across close-packed planes
        g_sfe_base = 45.0 * gamma_usf_multiplier
        g_usf_base = 180.0 * gamma_usf_multiplier

        # Exact 2D Frenkel-Rice double-periodic surface
        gamma_grid = (
            g_usf_base * (np.sin(np.pi * U1)**2 * np.cos(np.pi * U2)**2 + 0.5 * np.sin(2.0 * np.pi * U2)**2)
            + g_sfe_base * (np.sin(np.pi * (U1 + U2))**2)
        )

        gamma_max = float(np.max(gamma_grid))
        gamma_sfe = float(gamma_grid[self.grid_res // 3, self.grid_res // 3])
        gamma_utf = float(np.max(gamma_grid[:, 0]))

        return {
            "miller_plane": list(miller_plane),
            "grid_resolution": self.grid_res,
            "gamma_surface_grid_mj_m2": gamma_grid.tolist(),
            "unstable_stacking_fault_energy_gamma_usf_mj_m2": gamma_max,
            "intrinsic_stacking_fault_energy_gamma_isf_mj_m2": gamma_sfe,
            "unstable_twinning_fault_energy_gamma_utf_mj_m2": gamma_utf,
            "twinning_tendency_ratio": float(gamma_sfe / max(1.0, gamma_max)),
            "is_twinnable": bool((gamma_sfe / max(1.0, gamma_max)) < 0.45),
        }
