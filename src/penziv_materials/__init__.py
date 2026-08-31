"""Penziv Materials: Autonomous Multiscale First-Principles Materials Property Prediction Framework.

Version: 3.2.0-PROD
"""

__version__ = "3.2.0"
__author__ = "jawhett"

from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator
from penziv_materials.orchestration.discovery_engine import (
    AlloyDiscoveryEngine,
    DiscoveryTargetConstraints,
    ParetoDiscoveryResult,
)
from penziv_materials.validation.born_stability import BornStabilityValidator
from penziv_materials.validation.handshake_gates import HandshakeGatekeeper
from penziv_materials.adapters.standard_adapters import (
    SymmetryAdapter,
    PhaseDiagramAdapter,
    CalphadAdapter,
    TopologyAdapter,
    ElasticityAdapter,
)
from penziv_materials.scale5_quantum.surrogate_hierarchy import (
    SurrogateTier,
    SurrogateResult,
    HeuristicPrescreenFilter,
    UniversalMLIPCalculator,
    AbInitioDFTDriver,
    TieredSurrogateOrchestrator,
)

__all__ = [
    "__version__",
    "MetaOrchestrator",
    "AlloyDiscoveryEngine",
    "DiscoveryTargetConstraints",
    "ParetoDiscoveryResult",
    "BornStabilityValidator",
    "HandshakeGatekeeper",
    "SymmetryAdapter",
    "PhaseDiagramAdapter",
    "CalphadAdapter",
    "TopologyAdapter",
    "ElasticityAdapter",
    "SurrogateTier",
    "SurrogateResult",
    "HeuristicPrescreenFilter",
    "UniversalMLIPCalculator",
    "AbInitioDFTDriver",
    "TieredSurrogateOrchestrator",
]
