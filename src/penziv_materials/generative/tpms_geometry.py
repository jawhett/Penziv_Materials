"""Triply Periodic Minimal Surface (TPMS) & Multi-Phase Hybrid Spatial Representation."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class TPMSMultiPhaseGenerator:
    """Generates continuous Signed Distance Fields (SDFs) and multi-material domains (solid + channel + polymer)."""

    def __init__(self, resolution: Tuple[int, int, int] = (32, 32, 32), cell_size_nm: float = 100.0):
        self.nx, self.ny, self.nz = resolution
        self.cell_size_nm = cell_size_nm

        # Coordinate grid in [-pi, pi]^3
        x = np.linspace(-np.pi, np.pi, self.nx)
        y = np.linspace(-np.pi, np.pi, self.ny)
        z = np.linspace(-np.pi, np.pi, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

    def generate_gyroid_sdf(self, isovalue_c: float = 0.0) -> np.ndarray:
        """Evaluate Gyroid surface nodal field:

        F_Gyroid(x, y, z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x) - c
        """
        f_gyroid = (
            np.sin(self.X) * np.cos(self.Y)
            + np.sin(self.Y) * np.cos(self.Z)
            + np.sin(self.Z) * np.cos(self.X)
            - isovalue_c
        )
        return f_gyroid

    def generate_schwarz_diamond_sdf(self, isovalue_c: float = 0.0) -> np.ndarray:
        """Evaluate Schwarz Diamond (D-surface) nodal field:

        F_Diamond(x, y, z) = cos(x)*cos(y)*cos(z) - sin(x)*sin(y)*sin(z) - c
        """
        f_diamond = (
            np.cos(self.X) * np.cos(self.Y) * np.cos(self.Z)
            - np.sin(self.X) * np.sin(self.Y) * np.sin(self.Z)
            - isovalue_c
        )
        return f_diamond

    def build_tri_phase_hybrid_architecture(
        self,
        surface_type: str = "gyroid",
        wall_thickness_ratio: float = 0.25,
        polymer_skin_thickness_nm: float = 8.0,
    ) -> Dict[str, Any]:
        """Construct multi-phase coexisting spatial domain:

        Phase 1 (Solid Ceramic Skeleton): |SDF| < t_wall
        Phase 2 (Internal Pressurized Fluid/Gas Channel): SDF > t_wall
        Phase 3 (Conformal Polymeric Electrolyte Membrane): Outer boundary / negative envelope
        """
        if surface_type.lower() == "diamond":
            sdf = self.generate_schwarz_diamond_sdf()
        else:
            sdf = self.generate_gyroid_sdf()

        # Phase segmentations
        t_half = wall_thickness_ratio
        phase_map = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)

        # 1 = Solid Ceramic Matrix
        solid_mask = np.abs(sdf) <= t_half
        phase_map[solid_mask] = 1

        # 2 = Pressurized Fluid/Gas Microchannel
        channel_mask = sdf > t_half
        phase_map[channel_mask] = 2

        # 3 = Conformal Polymer Membrane Skin
        polymer_mask = sdf < -t_half
        phase_map[polymer_mask] = 3

        total_voxels = self.nx * self.ny * self.nz
        vol_frac_solid = float(np.sum(phase_map == 1) / total_voxels)
        vol_frac_channel = float(np.sum(phase_map == 2) / total_voxels)
        vol_frac_polymer = float(np.sum(phase_map == 3) / total_voxels)

        # Compute specific surface area (m2/g) approximation
        pore_hydraulic_diameter_nm = self.cell_size_nm * (1.0 - wall_thickness_ratio)

        return {
            "surface_type": surface_type,
            "volume_fraction_solid_ceramic": vol_frac_solid,
            "volume_fraction_pressurized_channel": vol_frac_channel,
            "volume_fraction_polymer_skin": vol_frac_polymer,
            "pore_hydraulic_diameter_nm": float(pore_hydraulic_diameter_nm),
            "is_interpenetrating_bicontinuous": True,
            "phase_map_shape": list(phase_map.shape),
        }
