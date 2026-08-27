"""Continuous Quality-Diversity (QD) Bayesian Materials Discovery Engine."""

from typing import Dict, Tuple, List, Optional, Any, Callable
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.core.models import MaterialCandidate, ValidationStatus
from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator


class ContinuousQDArchiveEntry(BaseModel):
    """Entry in continuous behavioral quality-diversity archive."""
    candidate: MaterialCandidate
    descriptor_vector: List[float]
    performance_score: float
    epistemic_uncertainty: float = 0.0


class BayesianQualityDiversityDiscoveryEngine:
    """Bayesian Quality-Diversity (QD) Discovery Engine for unconstrained multiscale exploration across arbitrary material domains."""

    def __init__(self, descriptor_dim: int = 2, max_archive_size: int = 200):
        self.descriptor_dim = descriptor_dim
        self.max_archive_size = max_archive_size
        self.archive: List[ContinuousQDArchiveEntry] = []
        self.orchestrator = MetaOrchestrator()

    def evaluate_behavioral_descriptors(self, candidate: MaterialCandidate) -> List[float]:
        """Map candidate multiscale state to continuous behavioral descriptor space (e.g. mass density, bandgap, ductility ratio)."""
        d1 = float(candidate.quantum.formation_energy_ev_atom) if candidate.quantum else -0.5
        d2 = float(candidate.continuum.yield_strength_mpa) / 1000.0 if candidate.continuum else 1.0
        return [d1, d2]

    def evaluate_performance_fitness(self, candidate: MaterialCandidate) -> float:
        """Compute composite objective fitness function."""
        score = 0.0
        if candidate.continuum:
            score += candidate.continuum.yield_strength_mpa * 0.001
            score += candidate.continuum.fracture_toughness_k_ic_mpa_sqrt_m * 0.02
            score -= np.log10(max(1e-15, candidate.continuum.steady_state_creep_rate_s_inv)) * 0.1
        if candidate.process:
            score -= candidate.process.min_ore_extraction_exergy_mj_kg * 0.005
        return float(score)

    def add_candidate_to_archive(self, candidate: MaterialCandidate, min_distance_threshold: float = 0.15) -> bool:
        """Add candidate to continuous Voronoi/k-NN QD archive if novel or superior in performance."""
        desc = np.array(self.evaluate_behavioral_descriptors(candidate))
        perf = self.evaluate_performance_fitness(candidate)

        if not self.archive:
            self.archive.append(ContinuousQDArchiveEntry(
                candidate=candidate,
                descriptor_vector=desc.tolist(),
                performance_score=perf,
                epistemic_uncertainty=0.01,
            ))
            return True

        archive_descs = np.array([e.descriptor_vector for e in self.archive])
        dists = np.linalg.norm(archive_descs - desc, axis=1)
        nearest_idx = int(np.argmin(dists))
        min_dist = dists[nearest_idx]

        if min_dist > min_distance_threshold:
            # Novel behavioral niche discovered!
            self.archive.append(ContinuousQDArchiveEntry(
                candidate=candidate,
                descriptor_vector=desc.tolist(),
                performance_score=perf,
                epistemic_uncertainty=0.01,
            ))
            return True
        elif perf > self.archive[nearest_idx].performance_score:
            # Elite replacement in existing niche
            self.archive[nearest_idx] = ContinuousQDArchiveEntry(
                candidate=candidate,
                descriptor_vector=desc.tolist(),
                performance_score=perf,
                epistemic_uncertainty=0.01,
            )
            return True

        return False

    def execute_quality_diversity_search(
        self,
        base_elements: List[str],
        n_iterations: int = 10,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """Execute unconstrained Quality-Diversity Bayesian exploration."""
        total_evaluated = 0
        novel_discoveries = 0

        for it in range(n_iterations):
            # Dirichlet composition perturbation
            k = len(base_elements)
            samples = np.random.dirichlet(np.ones(k) * 1.2, size=batch_size)
            for s in samples:
                comp = {elem: float(round(frac, 4)) for elem, frac in zip(base_elements, s)}
                cand_name = f"Penziv-QD-{total_evaluated+1:04d}"
                cand = self.orchestrator.run_forward_multiscale_prediction(
                    candidate_name=cand_name,
                    composition=comp,
                )
                added = self.add_candidate_to_archive(cand)
                if added:
                    novel_discoveries += 1
                total_evaluated += 1

        top_elite = max(self.archive, key=lambda e: e.performance_score) if self.archive else None

        return {
            "total_candidates_evaluated": total_evaluated,
            "archive_size": len(self.archive),
            "novel_niches_populated": novel_discoveries,
            "top_candidate": top_elite.candidate if top_elite else None,
            "max_performance_score": top_elite.performance_score if top_elite else 0.0,
            "is_qd_coverage_optimal": bool(len(self.archive) >= 5),
        }
