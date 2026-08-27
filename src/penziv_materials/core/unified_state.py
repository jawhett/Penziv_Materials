"""Universal Multiscale Phase-Geometry-Field Topological Tensor Container."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.core.models import CrystalSystem


class UnifiedMaterialState(BaseModel):
    """Universal phase-geometry-field topological tensor container.

    Unifies periodic 3D unit cells, 2D thin films / surfaces, amorphous boxes,
    and 3D discrete voxel RVEs with complete constitutive rank-2, rank-3, and rank-4 tensors.
    """

    name: str = "Penziv-Unified-Material"
    topology_type: str = "3D_Periodic_Crystal"  # "3D_Periodic_Crystal", "2D_Thin_Film", "Amorphous_Box", "Voxel_RVE"
    formula: str = "Ni3Al"
    space_group: str = "Pm-3m"
    space_group_number: int = 221
    magnetic_space_group: Optional[str] = None

    # Atomic / Network Coordinates and Symmetries
    atomic_numbers: List[int] = Field(default_factory=list)
    species: List[str] = Field(default_factory=list)
    fractional_coordinates: List[List[float]] = Field(default_factory=list)
    cartesian_coordinates: List[List[float]] = Field(default_factory=list)
    wyckoff_multiplicities: List[int] = Field(default_factory=list)
    partial_occupancies: List[float] = Field(default_factory=list)
    magnetic_moments: List[List[float]] = Field(default_factory=list)  # (N, 3) Bohr magnetons

    # Metric & Geometry
    lattice_matrix_angstrom: List[List[float]] = Field(default_factory=lambda: np.eye(3).tolist())
    box_dimensions_angstrom: List[float] = Field(default_factory=lambda: [10.0, 10.0, 10.0])
    volume_ang3: float = 1000.0

    # Constitutive Tensors (SI and Standard Units)
    elastic_stiffness_c_voigt_gpa: List[List[float]] = Field(default_factory=lambda: (np.eye(6) * 160.0).tolist())
    thermal_conductivity_tensor_w_m_k: List[List[float]] = Field(default_factory=lambda: (np.eye(3) * 35.0).tolist())
    dielectric_permittivity_tensor: List[List[float]] = Field(default_factory=lambda: (np.eye(3) * 12.5).tolist())
    piezoelectric_tensor_pc_n: Optional[List[List[List[float]]]] = None  # (3, 3, 3)
    eigenstrain_tensor: Optional[List[List[float]]] = None  # (3, 3)

    # Microstructural Fields (for RVE voxel grids)
    order_parameters_eta: Optional[List[float]] = None
    local_orientation_euler_rad: Optional[List[float]] = None
    damage_parameter_d: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def get_elastic_tensor_rank4(self) -> np.ndarray:
        """Convert 6x6 Voigt matrix to rank-4 elasticity tensor C_{ijkl} in GPa."""
        voigt = np.asarray(self.elastic_stiffness_c_voigt_gpa, dtype=np.float64)
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        C4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for a in range(6):
            i, j = voigt_map[a]
            for b in range(6):
                k, l = voigt_map[b]
                val = voigt[a, b]
                C4[i, j, k, l] = val
                C4[j, i, k, l] = val
                C4[i, j, l, k] = val
                C4[j, i, l, k] = val
        return C4

    def compute_acoustic_tensor(self, wavevector_n: np.ndarray, prestress_sigma_gpa: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute generalized acoustic tensor Lambda_{ik}(N) = C_{ijkl} N_j N_l + sigma_{jl} N_j N_l delta_{ik} under finite pre-stress."""
        N = wavevector_n / np.linalg.norm(wavevector_n)
        C4 = self.get_elastic_tensor_rank4()
        Lambda = np.zeros((3, 3), dtype=np.float64)

        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for l in range(3):
                        Lambda[i, k] += C4[i, j, k, l] * N[j] * N[l]
                        if prestress_sigma_gpa is not None and i == k:
                            Lambda[i, k] += prestress_sigma_gpa[j, l] * N[j] * N[l]
        return Lambda
