"""Unsupervised Latent Quality-Diversity (QD) Discovery with Property Tensor Autoencoding."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.core.models import MaterialCandidate
from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator


class LatentQDArchiveNode(BaseModel):
    """Candidate representation stored in dynamic continuous latent manifold."""
    candidate_name: str
    composition: Dict[str, float]
    latent_coordinates: List[float]
    multiscale_property_vector: List[float]
    composite_fitness_score: float
    epistemic_uncertainty: float = 0.0


class UnsupervisedLatentQDEngine:
    """Unsupervised Latent Quality-Diversity Engine discovering orthogonal behavioral axes across arbitrary material classes (semiconductors, superalloys, dielectrics, battery conductors)."""

    def __init__(self, latent_dim: int = 3, max_archive_size: int = 300):
        self.latent_dim = latent_dim
        self.max_archive_size = max_archive_size
        self.archive: List[LatentQDArchiveNode] = []
        self.orchestrator = MetaOrchestrator()
        self._projection_matrix: Optional[np.ndarray] = None

    def extract_normalized_property_vector(self, candidate: MaterialCandidate) -> np.ndarray:
        """Extract multi-dimensional normalized property vector across quantum, continuum, thermal, and electronic tiers."""
        vec = []
        # 1. Formation energy (eV/atom)
        vec.append(float(candidate.quantum.formation_energy_ev_atom) if candidate.quantum else -0.5)
        # 2. Bulk modulus proxy (GPa)
        vec.append(float(candidate.continuum.yield_strength_mpa) / 500.0 if candidate.continuum else 1.0)
        # 3. Fracture toughness
        vec.append(float(candidate.continuum.fracture_toughness_k_ic_mpa_sqrt_m) / 50.0 if candidate.continuum else 0.5)
        # 4. Creep resistance (-log10 creep)
        creep = candidate.continuum.steady_state_creep_rate_s_inv if candidate.continuum else 1e-8
        vec.append(-float(np.log10(max(1e-18, creep))) / 18.0)
        # 5. Ore extraction exergy
        exergy = candidate.process.min_ore_extraction_exergy_mj_kg if candidate.process else 150.0
        vec.append(float(exergy) / 300.0)

        return np.asarray(vec, dtype=np.float64)

    def update_latent_projection_manifold(self):
        """Fit PCA projection to uncover orthogonal behavioral axes dynamically from evaluated candidate distribution."""
        if len(self.archive) < self.latent_dim + 2:
            # Default orthogonal projection
            p_dim = len(self.archive[0].multiscale_property_vector) if self.archive else 5
            self._projection_matrix = np.eye(p_dim, self.latent_dim)
            return

        x_mat = np.array([node.multiscale_property_vector for node in self.archive])
        # Centering
        x_centered = x_mat - np.mean(x_mat, axis=0)
        # SVD / PCA
        _, _, vh = np.linalg.svd(x_centered, full_matrices=False)
        self._projection_matrix = vh[:self.latent_dim].T

    def project_to_latent_space(self, prop_vector: np.ndarray) -> np.ndarray:
        """Project high-dimensional property vector onto unsupervised latent manifold."""
        if self._projection_matrix is None:
            self.update_latent_projection_manifold()
        p_mat = self._projection_matrix if self._projection_matrix is not None else np.eye(len(prop_vector), self.latent_dim)
        return np.dot(prop_vector, p_mat)

    def evaluate_fitness(self, candidate: MaterialCandidate) -> float:
        """Compute composite objective fitness."""
        score = 0.0
        if candidate.continuum:
            score += candidate.continuum.yield_strength_mpa * 0.001
            score += candidate.continuum.fracture_toughness_k_ic_mpa_sqrt_m * 0.03
        if candidate.quantum:
            score -= candidate.quantum.formation_energy_ev_atom * 0.5
        return float(score)

    def add_candidate(self, candidate: MaterialCandidate, niche_radius: float = 0.20) -> bool:
        """Add candidate to unsupervised latent archive if discovering a novel niche or dominating an existing niche."""
        prop_vec = self.extract_normalized_property_vector(candidate)
        latent_coord = self.project_to_latent_space(prop_vec)
        fit = self.evaluate_fitness(candidate)

        node = LatentQDArchiveNode(
            candidate_name=candidate.name,
            composition=candidate.composition,
            latent_coordinates=latent_coord.tolist(),
            multiscale_property_vector=prop_vec.tolist(),
            composite_fitness_score=fit,
            epistemic_uncertainty=0.01,
        )

        if not self.archive:
            self.archive.append(node)
            return True

        coords = np.array([n.latent_coordinates for n in self.archive])
        dists = np.linalg.norm(coords - latent_coord, axis=1)
        min_idx = int(np.argmin(dists))

        if dists[min_idx] > niche_radius:
            self.archive.append(node)
            if len(self.archive) % 5 == 0:
                self.update_latent_projection_manifold()
            return True
        elif fit > self.archive[min_idx].composite_fitness_score:
            self.archive[min_idx] = node
            return True

        return False

    def execute_unsupervised_discovery(
        self,
        base_elements: List[str],
        num_candidates: int = 6,
    ) -> Dict[str, Any]:
        """Execute unsupervised latent quality-diversity discovery."""
        k = len(base_elements)
        samples = np.random.dirichlet(np.ones(k) * 1.5, size=num_candidates)
        novel_niches = 0

        for i, s in enumerate(samples):
            comp = {elem: float(round(frac, 4)) for elem, frac in zip(base_elements, s)}
            cand_name = f"Penziv-LatentQD-{len(self.archive)+1:04d}"
            cand = self.orchestrator.run_forward_multiscale_prediction(cand_name, comp)
            if self.add_candidate(cand):
                novel_niches += 1

        top_candidate = max(self.archive, key=lambda n: n.composite_fitness_score) if self.archive else None

        return {
            "total_archive_size": len(self.archive),
            "novel_niches_discovered": novel_niches,
            "top_candidate_name": top_candidate.candidate_name if top_candidate else None,
            "max_fitness": top_candidate.composite_fitness_score if top_candidate else 0.0,
            "is_unsupervised_manifold_active": bool(self._projection_matrix is not None),
        }
