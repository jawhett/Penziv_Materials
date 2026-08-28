"""Multi-Phase Complex Heterogeneous RVE Generator (RSA Particulates, IPC Sponges & Core-Shell Architectures)."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field


class MultiPhaseRVEResult(BaseModel):
    """Generated 3D multi-phase microstructural volume element."""
    architecture_type: str
    grid_shape: Tuple[int, int, int]
    num_phases: int
    volume_fractions: Dict[int, float]
    phase_id_map: List[List[List[int]]]


class MultiPhaseComplexRVEGenerator:
    """Generates complex heterogeneous microstructures: RSA particulate composites, GRF interpenetrating sponges, and core-shell architectures."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (32, 32, 32),
        dx_um: float = 0.5,
        random_seed: int = 42,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx_um
        self.rng = np.random.RandomState(random_seed)

    def generate_particulate_composite_rsa(
        self,
        target_volume_fraction: float = 0.20,
        particle_radius_um: float = 2.0,
        matrix_phase_id: int = 0,
        reinforcement_phase_id: int = 1,
        max_attempts: int = 2000,
    ) -> np.ndarray:
        """Generate 3D particulate composite via periodic Random Sequential Adsorption (RSA)."""
        grid = np.full((self.nx, self.ny, self.nz), matrix_phase_id, dtype=np.int32)
        r_voxels = max(1.5, particle_radius_um / self.dx)
        target_voxels = int(target_volume_fraction * self.nx * self.ny * self.nz)
        
        placed_centers: List[np.ndarray] = []
        current_voxels = 0

        # Coordinate mesh for sphere rasterization
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        z = np.arange(self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        for _ in range(max_attempts):
            if current_voxels >= target_voxels:
                break

            candidate = self.rng.uniform(0, [self.nx, self.ny, self.nz])
            
            # Check overlap with existing particles under PBC
            overlap = False
            for p in placed_centers:
                diff = np.abs(candidate - p)
                diff = np.minimum(diff, np.array([self.nx, self.ny, self.nz]) - diff)
                dist = np.linalg.norm(diff)
                if dist < 2.0 * r_voxels:
                    overlap = True
                    break

            if not overlap:
                placed_centers.append(candidate)
                # Rasterize sphere under PBC
                diff_x = np.abs(X - candidate[0])
                diff_x = np.minimum(diff_x, self.nx - diff_x)
                diff_y = np.abs(Y - candidate[1])
                diff_y = np.minimum(diff_y, self.ny - diff_y)
                diff_z = np.abs(Z - candidate[2])
                diff_z = np.minimum(diff_z, self.nz - diff_z)
                
                dist_sq = diff_x**2 + diff_y**2 + diff_z**2
                sphere_mask = dist_sq <= r_voxels**2
                grid[sphere_mask] = reinforcement_phase_id
                current_voxels = int(np.sum(grid == reinforcement_phase_id))

        return grid

    def generate_interpenetrating_phase_composite_grf(
        self,
        target_volume_fraction: float = 0.50,
        correlation_length_um: float = 3.0,
        num_modes: int = 100,
    ) -> np.ndarray:
        """Generate 3D Bi-Continuous Interpenetrating Phase Composite (IPC) via Gaussian Random Field level-set thresholding."""
        grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        k0 = 2.0 * np.pi / (correlation_length_um / self.dx)

        # Generate Gaussian Random Field: phi(r) = sum_m A_m cos(k_m . r + theta_m)
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        z = np.arange(self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        for _ in range(num_modes):
            # Random wavevector on sphere of radius ~ k0
            dir_vec = self.rng.normal(size=3)
            dir_vec /= np.linalg.norm(dir_vec)
            k_vec = dir_vec * k0 * self.rng.uniform(0.8, 1.2)
            phase = self.rng.uniform(0, 2.0 * np.pi)
            grid += np.cos(k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z + phase)

        grid /= np.sqrt(num_modes)

        # Threshold at quantile matching target volume fraction
        threshold = np.quantile(grid, 1.0 - target_volume_fraction)
        phase_map = np.where(grid >= threshold, 1, 0).astype(np.int32)
        return phase_map

    def generate_core_shell_nanocomposite(
        self,
        core_radius_um: float = 1.5,
        shell_thickness_um: float = 0.8,
        num_particles: int = 10,
    ) -> np.ndarray:
        """Generate Core-Shell particulate morphology (Phase 0: Matrix, Phase 1: Shell, Phase 2: Core)."""
        grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)
        r_core_vox = max(1.0, core_radius_um / self.dx)
        r_total_vox = max(r_core_vox + 0.5, (core_radius_um + shell_thickness_um) / self.dx)

        x = np.arange(self.nx)
        y = np.arange(self.ny)
        z = np.arange(self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        for _ in range(num_particles):
            center = self.rng.uniform(0, [self.nx, self.ny, self.nz])
            diff_x = np.minimum(np.abs(X - center[0]), self.nx - np.abs(X - center[0]))
            diff_y = np.minimum(np.abs(Y - center[1]), self.ny - np.abs(Y - center[1]))
            diff_z = np.minimum(np.abs(Z - center[2]), self.nz - np.abs(Z - center[2]))
            dist_sq = diff_x**2 + diff_y**2 + diff_z**2

            # Apply Shell (Phase 1)
            shell_mask = (dist_sq <= r_total_vox**2) & (dist_sq > r_core_vox**2)
            grid[shell_mask] = 1

            # Apply Core (Phase 2)
            core_mask = dist_sq <= r_core_vox**2
            grid[core_mask] = 2

        return grid
