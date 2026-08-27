"""Scale 2: 3D Anisotropic Phase-Field Fracture & Non-Local Gradient Damage Mechanics."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class NonLocalDamageMechanics:
    """Solves 3D anisotropic phase-field fracture and gradient-enhanced damage evolution across RVE voxels."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        characteristic_length_l0_um: float = 2.5,
        characteristic_length_lc_um: Optional[float] = None,
        critical_fracture_energy_gc_j_m2: float = 45.0,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.l0 = characteristic_length_lc_um if characteristic_length_lc_um is not None else characteristic_length_l0_um
        self.gc = critical_fracture_energy_gc_j_m2

    def solve_1d_helmholtz_nonlocal(
        self,
        local_equivalent_strain: np.ndarray,
        mesh_spacing_dx: float = 1.0,
        l_c: Optional[float] = None,
    ) -> np.ndarray:
        """1D Helmholtz non-local strain regularization."""
        eps_loc = np.asarray(local_equivalent_strain, dtype=np.float64)
        n = len(eps_loc)
        if n < 3:
            return eps_loc

        length_c = l_c if l_c is not None else self.l0
        diag = (1.0 + 2.0 * (length_c / mesh_spacing_dx) ** 2) * np.ones(n)
        off_diag = -((length_c / mesh_spacing_dx) ** 2) * np.ones(n - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        A[0, 0] = 1.0 + (length_c / mesh_spacing_dx) ** 2
        A[-1, -1] = 1.0 + (length_c / mesh_spacing_dx) ** 2

        eps_nonlocal = np.linalg.solve(A, eps_loc)
        return np.clip(eps_nonlocal, 0.0, 1.0)

    def solve_nonlocal_equivalent_strain_1d(
        self,
        local_equivalent_strain: np.ndarray,
        mesh_spacing_dx: float = 1.0,
    ) -> np.ndarray:
        """Alias for 1D non-local equivalent strain calculation."""
        return self.solve_1d_helmholtz_nonlocal(local_equivalent_strain, mesh_spacing_dx=mesh_spacing_dx, l_c=self.l0)

    def compute_damage_variable(
        self,
        nonlocal_strain: np.ndarray,
        damage_threshold_eps0: float = 0.002,
        critical_strain_eps_f: float = 0.05,
    ) -> np.ndarray:
        """Compute scalar damage degradation variable d in [0, 1)."""
        eps = np.asarray(nonlocal_strain, dtype=np.float64)
        d = np.where(
            eps <= damage_threshold_eps0,
            0.0,
            (critical_strain_eps_f / (critical_strain_eps_f - damage_threshold_eps0)) * (1.0 - (damage_threshold_eps0 / np.maximum(1e-6, eps))),
        )
        return np.clip(d, 0.0, 0.99)

    def solve_3d_phase_field_fracture_step(
        self,
        damage_field: np.ndarray,
        strain_tensor_field: np.ndarray,
        youngs_modulus_gpa: float = 200.0,
        poisson_ratio: float = 0.30,
        anisotropic_tensor_A: Optional[np.ndarray] = None,
        dt: float = 0.01,
        mobility: float = 5.0,
    ) -> Dict[str, Any]:
        """Solve 3D anisotropic phase-field fracture equation."""
        d = damage_field.copy()
        nx, ny, nz = self.nx, self.ny, self.nz

        lam = (youngs_modulus_gpa * poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)) * 1.0e3
        mu = youngs_modulus_gpa / (2.0 * (1.0 + poisson_ratio)) * 1.0e3

        tr_eps = np.trace(strain_tensor_field, axis1=-2, axis2=-1)
        tr_eps_pos = np.maximum(0.0, tr_eps)
        psi_plus = 0.5 * lam * (tr_eps_pos**2) + mu * np.sum(np.maximum(0.0, strain_tensor_field)**2, axis=(-2, -1))

        lap_d = (
            np.roll(d, 1, axis=0) + np.roll(d, -1, axis=0)
            + np.roll(d, 1, axis=1) + np.roll(d, -1, axis=1)
            + np.roll(d, 1, axis=2) + np.roll(d, -1, axis=2)
            - 6.0 * d
        )

        gc_mpa_um = self.gc * 1.0e-3
        driving_force = (2.0 * (1.0 - d) / max(1e-4, gc_mpa_um)) * psi_plus
        spatial_regularization = (d / self.l0) - self.l0 * lap_d

        d_dot = mobility * np.maximum(0.0, driving_force - spatial_regularization)
        new_d = np.clip(d + dt * d_dot, d, 1.0)

        max_damage = float(np.max(new_d))
        broken_volume_fraction = float(np.mean(new_d >= 0.95))

        return {
            "damage_field": new_d,
            "max_damage_parameter": max_damage,
            "broken_volume_fraction": broken_volume_fraction,
            "is_macroscopically_failed": bool(broken_volume_fraction >= 0.10 or max_damage >= 0.98),
            "tensile_strain_energy_density_mj_m3": float(np.mean(psi_plus)),
        }
