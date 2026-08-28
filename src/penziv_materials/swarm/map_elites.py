"""Quality-Diversity (QD) MAP-Elites Behavioral Niche Illumination Swarm Engine (Domain-Agnostic N-Dimensional)."""

from typing import Dict, Tuple, List, Optional, Any, Callable, Sequence, Union
import numpy as np


class MAPElitesSwarmEngine:
    """Domain-Agnostic Quality-Diversity (QD) illumination algorithm populating an arbitrary N-dimensional behavioral feature archive."""

    # Built-in Domain Descriptor Presets
    PRESET_DESCRIPTORS = {
        "thermals_space": [
            ("kappa_xx_w_m_k", (0.1, 500.0), True),        # log scale
            ("melting_point_k", (300.0, 4000.0), False),
            ("cte_ppm_k", (0.5, 30.0), False),
            ("fracture_toughness_k1c", (0.5, 50.0), False),
        ],
        "semiconductors_optics": [
            ("bandgap_eg_ev", (0.0, 6.0), False),
            ("electron_effective_mass", (0.05, 2.0), False),
            ("relative_permittivity", (1.0, 50.0), False),
            ("carrier_mobility_cm2_v_s", (1.0, 5000.0), True),
        ],
        "batteries_interfaces": [
            ("ionic_conductivity_ms_cm", (1e-4, 100.0), True),
            ("voltage_window_v", (1.0, 6.0), False),
            ("interface_energy_j_m2", (0.01, 1.5), False),
            ("shear_modulus_gpa", (5.0, 200.0), False),
        ],
    }

    def __init__(
        self,
        grid_dimensions: Optional[Sequence[int]] = None,
        descriptor_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        descriptor_names: Optional[Sequence[str]] = None,
        # Backward compatibility defaults for 3D battery grid
        grid_dim_x: int = 10,
        grid_dim_y: int = 10,
        grid_dim_z: int = 10,
    ):
        if grid_dimensions is not None:
            self.dimensions = tuple(grid_dimensions)
            self.n_dims = len(self.dimensions)
        else:
            self.dimensions = (grid_dim_x, grid_dim_y, grid_dim_z)
            self.n_dims = 3

        self.dim_x = self.dimensions[0]
        self.dim_y = self.dimensions[1] if self.n_dims > 1 else 1
        self.dim_z = self.dimensions[2] if self.n_dims > 2 else 1

        if descriptor_bounds is not None:
            self.bounds = list(descriptor_bounds)
        else:
            # Default bounds for 3D battery space
            self.bounds = [(-3.0, 2.0), (0.0, 0.8), (0.01, 0.20)]

        if descriptor_names is not None:
            self.descriptor_names = list(descriptor_names)
        else:
            self.descriptor_names = [f"descriptor_{i+1}" for i in range(self.n_dims)]

        # Archive stores the highest-fitness candidate for each discrete coordinate tuple
        self.archive: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.fitness_grid = np.full(self.dimensions, -np.inf)

    @classmethod
    def from_preset(
        cls,
        preset_name: str = "batteries_interfaces",
        bins_per_dim: int = 8,
    ) -> "MAPElitesSwarmEngine":
        """Instantiate a domain-specialized QD engine from standard presets."""
        preset = cls.PRESET_DESCRIPTORS.get(preset_name.lower())
        if preset is None:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(cls.PRESET_DESCRIPTORS.keys())}")

        names = [p[0] for p in preset]
        bounds = [p[1] for p in preset]
        dims = [bins_per_dim] * len(preset)

        return cls(
            grid_dimensions=dims,
            descriptor_bounds=bounds,
            descriptor_names=names,
        )

    def compute_n_dim_coordinates(self, descriptor_values: Union[Sequence[float], Dict[str, float]]) -> Tuple[int, ...]:
        """Map arbitrary continuous candidate descriptors to discrete N-D behavioral grid coordinates."""
        if isinstance(descriptor_values, dict):
            vals = [descriptor_values.get(name, 0.0) for name in self.descriptor_names]
        else:
            vals = list(descriptor_values)

        coords = []
        for i, val in enumerate(vals[:self.n_dims]):
            low, high = self.bounds[i] if i < len(self.bounds) else (0.0, 1.0)
            n_bins = self.dimensions[i]
            # Discretize into bin [0, n_bins - 1]
            idx = int(np.clip((val - low) / max(1e-12, (high - low) / n_bins), 0, n_bins - 1))
            coords.append(idx)

        while len(coords) < self.n_dims:
            coords.append(0)

        return tuple(coords)

    def compute_behavioral_descriptors(
        self,
        ionic_conductivity_ms_cm: float,
        channel_volume_fraction: float,
        matrix_compliance_gpa_inv: float,
    ) -> Tuple[int, int, int]:
        """Backward-compatible 3D coordinate mapping."""
        log_sigma = float(np.log10(max(1e-4, ionic_conductivity_ms_cm)))
        coords = self.compute_n_dim_coordinates([log_sigma, channel_volume_fraction, matrix_compliance_gpa_inv])
        return coords[0], coords[1], coords[2]

    def add_candidate(
        self,
        candidate_data: Dict[str, Any],
        fitness_score: float,
        descriptors: Union[Sequence[float], Dict[str, float]],
    ) -> bool:
        """Add candidate to N-dimensional behavioral niche archive; replaces incumbent if fitness improves."""
        coords = self.compute_n_dim_coordinates(descriptors)
        current_best = self.fitness_grid[coords]

        if fitness_score > current_best:
            self.fitness_grid[coords] = fitness_score
            self.archive[coords] = {
                "candidate": candidate_data,
                "fitness": float(fitness_score),
                "coords": coords,
                "descriptors": descriptors if isinstance(descriptors, dict) else dict(zip(self.descriptor_names, descriptors)),
            }
            return True
        return False

    def add_candidate_to_archive(
        self,
        candidate_data: Dict[str, Any],
        fitness_score: float,
        ionic_conductivity_ms_cm: float,
        channel_volume_fraction: float,
        matrix_compliance_gpa_inv: float,
    ) -> bool:
        """Backward-compatible candidate insertion."""
        log_sigma = float(np.log10(max(1e-4, ionic_conductivity_ms_cm)))
        coords = self.compute_n_dim_coordinates([log_sigma, channel_volume_fraction, matrix_compliance_gpa_inv])

        current_best_fitness = self.fitness_grid[coords]
        if fitness_score > current_best_fitness:
            self.fitness_grid[coords] = fitness_score
            self.archive[coords] = {
                "candidate": candidate_data,
                "fitness": float(fitness_score),
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
        total_cells = int(np.prod(self.dimensions))
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
            "dimensions": list(self.dimensions),
            "descriptor_names": self.descriptor_names,
        }
