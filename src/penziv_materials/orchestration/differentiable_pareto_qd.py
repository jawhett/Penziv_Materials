"""High-Dimensional Centroidal Voronoi Tessellation (CVT-MAP-Elites) & Continuous Pareto QD Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.core.models import MaterialCandidate
from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator


class CVTNicheCell(BaseModel):
    """Centroidal Voronoi cell in high-dimensional latent descriptor space."""
    centroid_index: int
    centroid_vector: List[float]
    occupied: bool = False
    candidate_name: Optional[str] = None
    performance_vector: Optional[List[float]] = None
    composite_fitness: float = -float("inf")


class DifferentiableContinuousParetoQDEngine:
    """High-Dimensional Centroidal Voronoi Tessellation (CVT-MAP-Elites) across continuous latent property manifolds."""

    def __init__(
        self,
        latent_dim: int = 8,
        num_centroids: int = 50,
        random_seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.num_centroids = num_centroids
        self.orchestrator = MetaOrchestrator()
        self.centroids = self._initialize_k_means_centroids(random_seed)
        self.cells: List[CVTNicheCell] = [
            CVTNicheCell(centroid_index=i, centroid_vector=self.centroids[i].tolist())
            for i in range(num_centroids)
        ]

    def _initialize_k_means_centroids(self, seed: int) -> np.ndarray:
        """Generate uniform Centroidal Voronoi centroids on unit hypercube [0, 1]^D."""
        np.random.seed(seed)
        # Sample points and run k-means relaxation
        pts = np.random.uniform(0.0, 1.0, (self.num_centroids * 20, self.latent_dim))
        centroids = pts[:self.num_centroids].copy()

        for _ in range(15):
            dists = np.linalg.norm(pts[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1)
            closest = np.argmin(dists, axis=1)
            for k in range(self.num_centroids):
                cluster_pts = pts[closest == k]
                if len(cluster_pts) > 0:
                    centroids[k] = np.mean(cluster_pts, axis=0)

        return centroids

    def compute_latent_embedding(self, candidate: MaterialCandidate) -> np.ndarray:
        """Map candidate multiscale state to normalized latent descriptor vector in [0, 1]^D."""
        vec = np.zeros(self.latent_dim, dtype=np.float64)

        # Feature 0: Formation energy normalized
        vec[0] = float(np.clip((float(candidate.quantum.formation_energy_ev_atom) + 2.0) / 4.0, 0.0, 1.0)) if candidate.quantum else 0.5
        # Feature 1: Yield strength normalized
        vec[1] = float(np.clip(float(candidate.continuum.yield_strength_mpa) / 2000.0, 0.0, 1.0)) if candidate.continuum else 0.5
        # Feature 2: Fracture toughness normalized
        vec[2] = float(np.clip(float(candidate.continuum.fracture_toughness_k_ic_mpa_sqrt_m) / 100.0, 0.0, 1.0)) if candidate.continuum else 0.5
        # Feature 3: Creep resistance
        creep = candidate.continuum.steady_state_creep_rate_s_inv if candidate.continuum else 1e-8
        vec[3] = float(np.clip((-np.log10(max(1e-20, creep))) / 20.0, 0.0, 1.0))
        # Feature 4: Exergy normalized
        exergy = candidate.process.min_ore_extraction_exergy_mj_kg if candidate.process else 150.0
        vec[4] = float(np.clip(float(exergy) / 500.0, 0.0, 1.0))

        return vec

    def evaluate_composite_fitness(self, candidate: MaterialCandidate) -> Tuple[float, List[float]]:
        """Evaluate Pareto performance vector [YS, K_IC, -log(creep)] and scalar fitness."""
        ys = float(candidate.continuum.yield_strength_mpa) if candidate.continuum else 500.0
        kic = float(candidate.continuum.fracture_toughness_k_ic_mpa_sqrt_m) if candidate.continuum else 20.0
        creep = float(candidate.continuum.steady_state_creep_rate_s_inv) if candidate.continuum else 1e-7
        score = ys * 0.001 + kic * 0.02 - np.log10(max(1e-15, creep)) * 0.1
        return score, [ys, kic, -np.log10(max(1e-15, creep))]

    def add_candidate_to_cvt_archive(self, candidate: MaterialCandidate) -> bool:
        """Assign candidate to closest Centroidal Voronoi niche cell and update elite."""
        latent = self.compute_latent_embedding(candidate)
        dists = np.linalg.norm(self.centroids - latent, axis=1)
        cell_idx = int(np.argmin(dists))

        score, p_vec = self.evaluate_composite_fitness(candidate)
        cell = self.cells[cell_idx]

        if not cell.occupied or score > cell.composite_fitness:
            cell.occupied = True
            cell.candidate_name = candidate.name
            cell.performance_vector = p_vec
            cell.composite_fitness = score
            return True

        return False

    def execute_cvt_map_elites_search(
        self,
        base_elements: List[str],
        num_evaluations: int = 8,
    ) -> Dict[str, Any]:
        """Execute high-dimensional CVT-MAP-Elites discovery search."""
        k = len(base_elements)
        samples = np.random.dirichlet(np.ones(k) * 1.3, size=num_evaluations)
        novel_discoveries = 0

        for i, s in enumerate(samples):
            comp = {elem: float(round(frac, 4)) for elem, frac in zip(base_elements, s)}
            cand_name = f"Penziv-CVT-{i+1:04d}"
            cand = self.orchestrator.run_forward_multiscale_prediction(cand_name, comp)
            if self.add_candidate_to_cvt_archive(cand):
                novel_discoveries += 1

        occupied_count = sum(1 for c in self.cells if c.occupied)
        best_cell = max((c for c in self.cells if c.occupied), key=lambda c: c.composite_fitness, default=None)

        return {
            "total_cvt_cells": self.num_centroids,
            "occupied_cvt_cells": occupied_count,
            "archive_coverage_percent": float(100.0 * occupied_count / self.num_centroids),
            "novel_discoveries_added": novel_discoveries,
            "top_elite_candidate": best_cell.candidate_name if best_cell else None,
            "max_composite_fitness": best_cell.composite_fitness if best_cell else 0.0,
            "is_cvt_search_successful": bool(occupied_count > 0),
        }
