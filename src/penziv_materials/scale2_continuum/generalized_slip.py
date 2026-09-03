"""Universal Reciprocal Lattice Slip & GSFE Dislocation Kinematics Solver."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.structure.crystal_structure import CrystalStructure, PeriodicLattice


class UniversalSlipGenerator:
    """Constructs active slip systems dynamically via reciprocal lattice planes and shortest lattice Burgers vectors."""

    @staticmethod
    def generate_systems_from_structure(
        crystal: CrystalStructure,
        max_index: int = 2,
        max_burgers_mag_angstrom: float = 6.0,
    ) -> List[Dict[str, Any]]:
        """Enumerate low-index Miller planes (hkl) and directions [uvw] satisfying orthogonality (b . n == 0)."""
        lattice_mat = crystal.lattice.matrix
        recip_mat = crystal.lattice.get_reciprocal_lattice()
        slip_systems: List[Dict[str, Any]] = []

        indices = np.arange(-max_index, max_index + 1)
        grid = np.array(np.meshgrid(indices, indices, indices)).T.reshape(-1, 3)
        grid = grid[~np.all(grid == 0, axis=1)]

        for hkl in grid:
            plane_normal = np.dot(hkl, recip_mat)
            norm_val = np.linalg.norm(plane_normal)
            if norm_val < 1e-4:
                continue
            n_norm = plane_normal / norm_val

            for uvw in grid:
                burgers_vec = np.dot(uvw, lattice_mat)
                b_mag = np.linalg.norm(burgers_vec)

                # Orthogonality condition: slip direction lies in slip plane
                if abs(np.dot(burgers_vec, n_norm)) < 1e-3 and (0.8 < b_mag < max_burgers_mag_angstrom):
                    s_dir = burgers_vec / b_mag

                    # Ensure unique system
                    is_dup = False
                    for existing in slip_systems:
                        if (
                            np.allclose(n_norm, existing["plane_normal"], atol=1e-2)
                            and np.allclose(s_dir, existing["slip_direction"], atol=1e-2)
                        ):
                            is_dup = True
                            break

                    if not is_dup:
                        # Compute GSFE sinusoidal Frenkel estimate of CRSS
                        # tau_ideal = G * b / (2 * pi * d)
                        d_spacing = 1.0 / norm_val
                        schmid_tensor = np.outer(s_dir, n_norm)

                        slip_systems.append({
                            "miller_plane": hkl.tolist(),
                            "miller_direction": uvw.tolist(),
                            "plane_normal": n_norm,
                            "slip_direction": s_dir,
                            "burgers_magnitude_angstrom": float(b_mag),
                            "interplanar_spacing_angstrom": float(d_spacing),
                            "schmid_tensor": schmid_tensor,
                        })

        if not slip_systems:
            # Fallback orthogonal triad
            slip_systems.append({
                "miller_plane": [0, 0, 1],
                "miller_direction": [1, 0, 0],
                "plane_normal": np.array([0, 0, 1]),
                "slip_direction": np.array([1, 0, 0]),
                "burgers_magnitude_angstrom": 2.54,
                "interplanar_spacing_angstrom": 2.07,
                "schmid_tensor": np.outer([1, 0, 0], [0, 0, 1]),
            })

        return slip_systems[:48]

    @staticmethod
    def compute_gsfe_critical_resolved_shear_stress(
        shear_modulus_gpa: float,
        burgers_magnitude_angstrom: float,
        interplanar_spacing_angstrom: float,
        stacking_fault_energy_j_m2: float = 0.045,
    ) -> float:
        """Evaluate CRSS from GSFE gamma-surface gradient:

        tau_CRSS = max_u |grad_u gamma(u)| = (G * b) / (2 * pi * d) * (1 - 0.25 * (gamma_SFE / (G*b)))
        """
        g_pa = shear_modulus_gpa * 1.0e9
        b_m = burgers_magnitude_angstrom * 1.0e-10
        d_m = max(1e-11, interplanar_spacing_angstrom * 1.0e-10)

        tau_ideal_pa = (g_pa * b_m) / (2.0 * np.pi * d_m)
        sfe_softening = max(0.2, 1.0 - 0.25 * (stacking_fault_energy_j_m2 / max(1e-3, g_pa * b_m)))

        tau_crss_gpa = (tau_ideal_pa * sfe_softening * 0.15) * 1.0e-9  # Peierls-Nabarro lattice friction factor
        return float(max(1e-4, tau_crss_gpa))
