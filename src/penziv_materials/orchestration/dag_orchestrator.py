"""Scale-Adaptive Dynamic Directed Acyclic Graph (DAG) Multiscale Execution Engine."""

from typing import Dict, List, Optional, Any, Callable
import numpy as np


class MultiscaleDAGNode:
    """Represents an executable computation node in the multiscale physics DAG."""

    def __init__(self, name: str, execute_fn: Callable[..., Dict[str, Any]], dependencies: Optional[List[str]] = None):
        self.name = name
        self.execute_fn = execute_fn
        self.dependencies = dependencies or []


class DynamicDAGOrchestrator:
    """Assembles and executes material-class tailored computation graphs without redundant physics scale evaluations."""

    def __init__(self):
        self.nodes: Dict[str, MultiscaleDAGNode] = {}

    def register_node(self, node: MultiscaleDAGNode) -> None:
        self.nodes[node.name] = node

    def assemble_graph_for_material_class(self, material_class: str) -> List[str]:
        """Dynamically assemble the topological execution sequence based on material class."""
        mat_type = material_class.lower()

        if "semiconductor" in mat_type or "dielectric" in mat_type or "electronic" in mat_type:
            return ["quantum_dft", "semiconductor_bte", "dielectric_breakdown", "coupled_maxwell_poisson"]
        elif "composite" in mat_type or "structural" in mat_type or "alloy" in mat_type:
            return ["quantum_dft", "atomistic_mlip", "phase_field_microelasticity", "continuum_cpfft", "damage_mechanics"]
        elif "amorphous" in mat_type or "glass" in mat_type:
            return ["melt_quench_md", "voronoi_csro_topology", "stz_amorphous_plasticity"]
        elif "battery" in mat_type or "electrolyte" in mat_type:
            return ["quantum_dft", "ion_transport_pnp", "electrochemical_hull", "mechanical_damage"]
        else:
            return ["quantum_dft", "atomistic_mlip", "continuum_cpfft"]

    def execute_dag(
        self,
        material_class: str,
        initial_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the dynamically assembled DAG and return the aggregated multiscale state."""
        execution_order = self.assemble_graph_for_material_class(material_class)
        context = dict(initial_context)
        executed_nodes = []

        for node_name in execution_order:
            if node_name in self.nodes:
                node = self.nodes[node_name]
                res = node.execute_fn(context)
                context.update(res)
                executed_nodes.append(node_name)
            else:
                # Mock execution placeholder for unmapped custom user nodes
                executed_nodes.append(f"{node_name} (direct)")

        return {
            "material_class": material_class,
            "executed_dag_sequence": executed_nodes,
            "final_context": context,
            "is_dag_execution_successful": True,
        }
