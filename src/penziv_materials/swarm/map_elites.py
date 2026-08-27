"""Quality-Diversity (QD) MAP-Elites Behavioral Niche Illumination Swarm Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class MAPElitesSwarmEngine:
    """Quality-Diversity (QD) illumination algorithm populating a discrete behavioral feature archive."""

    def __init__(
        self,
        grid_dim_x: int = 10,  # Behavioral feature 1: Ionic Conductivity log10(sigma_ion)
        grid_dim_y: int = 10,  # Behavioral feature 2: Structural Complexity / Porosity
        grid_dim_z: int = 10,  # Behavioral feature 3: Hybrid Interfacial Compliance
    ):
        self.dim_x = grid_dim_x
        self.dim_y = grid_dim_y
        self.dim_z = grid_dim_z

        # Archive stores the highest-fitness candidate for each (x, y, z) niche
        self.archive: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.fitness_grid = np.full((self.dim_x, self.dim_y, self.dim_z), -np.inf)

    def compute_behavioral_descriptors(
        self,
        ionic_conductivity_ms_cm: float,
        channel_volume_fraction: float,
        matrix_compliance_gpa_inv: float,
    ) -> Tuple[int, int, int]:
        """Map continuous candidate descriptors to discrete behavioral grid coordinates (i, j, k)."""
        # Descriptor 1: log10(sigma_ion in mS/cm) in [-3.0, 2.0]
        log_sigma = np.log10(max(1e-4, ionic_conductivity_ms_cm))
        idx_x = int(np.clip((log_sigma - (-3.0)) / (5.0 / self.dim_x), 0, self.dim_x - 1))

        # Descriptor 2: Channel volume fraction in [0.0, 0.8]
        idx_y = int(np.clip(channel_volume_fraction / (0.8 / self.dim_y), 0, self.dim_y - 1))

        # Descriptor 3: Compliance in [0.01, 0.20] GPa^-1
        idx_z = int(np.clip((matrix_compliance_gpa_inv - 0.01) / (0.19 / self.dim_z), 0, self.dim_z - 1))

        return idx_x, idx_y, idx_z

    def add_candidate_to_archive(
        self,
        candidate_data: Dict[str, Any],
        fitness_score: float,
        ionic_conductivity_ms_cm: float,
        channel_volume_fraction: float,
        matrix_compliance_gpa_inv: float,
    ) -> bool:
        """Attempt to insert candidate into its behavioral niche; replaces incumbent if fitness is superior."""
        coords = self.compute_behavioral_descriptors(
            ionic_conductivity_ms_cm=ionic_conductivity_ms_cm,
            channel_volume_fraction=channel_volume_fraction,
            matrix_compliance_gpa_inv=matrix_compliance_gpa_inv,
        )

        current_best_fitness = self.fitness_grid[coords]
        if fitness_score > current_best_fitness:
            self.fitness_grid[coords] = fitness_score
            self.archive[coords] = {
                "candidate": candidate_data,
                "fitness": fitness_score,
                "coords": coords,
                "sigma_ion": ionic_conductivity_ms_cm,
                "porosity": channel_volume_fraction,
                "compliance": matrix_compliance_gpa_inv,
            }
            return True
        return False

    def get_archive_statistics(self) -> Dict[str, Any]:
        """Compute coverage, total quality score (QD-Score), and maximum fitness in archive."""
        num_occupied = len(self.archive)
        total_cells = self.dim_x * self.dim_y * self.dim_z
        coverage = num_occupied / max(1, total_cells)

        finite_fitnesses = [v["fitness"] for v in self.archive.values() if np.isfinite(v["fitness"])]
        qd_score = float(np.sum(finite_fitnesses)) if finite_fitnesses else 0.0
        max_fitness = float(np.max(finite_fitnesses)) if finite_fitnesses else -np.inf

        return {
            "occupied_niches": num_occupied,
            "total_niches": total_cells,
            "archive_coverage": float(coverage),
            "qd_score": qd_score,
            "max_fitness": max_fitness,
        }
