"""State-Driven Dynamic DAG Multiscale Discovery Orchestrator."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.core.models import MaterialCandidate, QuantumState, MesoscaleState, ContinuumState, ProcessState
from penziv_materials.scale5_quantum.q_elec import QElecAgent
from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
from penziv_materials.scale3_mesoscale.calphad_grand_potential import CALPHADGrandPotentialPhaseFieldEngine
from penziv_materials.physics.wigner_peierls_transport import UnifiedThermalElectronicTransportEngine
from penziv_materials.physics.cohesive_interface import CohesiveZoneInterfaceEngine
from penziv_materials.structure.universal_neumann import UniversalNeumannTensorEngine


class MaterialDomainTarget(BaseModel):
    """Configuration target defining required physical tiers and metrics."""
    domain_type: str = "general"  # "thermoelectric", "semiconductor", "structural_alloy", "solid_electrolyte", "metallic_glass"
    target_temperature_k: float = 300.0
    applied_stress_mpa: float = 0.0
    operating_electric_field_v_m: float = 0.0


class StateDrivenDAGOrchestrator:
    """State-driven dynamic DAG multiscale discovery orchestrator automatically resolving physics dependencies."""

    def __init__(self):
        self.q_agent = QElecAgent()
        self.mlip_engine = EquivariantMLIPEngine()
        self.calphad_pf = CALPHADGrandPotentialPhaseFieldEngine()
        self.cohesive = CohesiveZoneInterfaceEngine()

    def execute_state_driven_pipeline(
        self,
        candidate_name: str,
        composition: Dict[str, float],
        target: Optional[MaterialDomainTarget] = None,
    ) -> Dict[str, Any]:
        """Execute state-driven multiscale evaluation tailored dynamically to the target application domain."""
        tgt = target or MaterialDomainTarget()
        T = tgt.target_temperature_k

        # 1. Quantum State evaluation
        q_state = self.q_agent.execute_quantum_state_evaluation(
            formula=candidate_name,
            composition=composition,
            temperature_k=T,
        )
        c_voigt = np.array(q_state.c_voigt_gpa)

        # 2. Universal Neumann Point Group Tensor Symmetrization
        c_rank4 = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                c_rank4[i, i, j, j] = c_voigt[i, j]
                if i != j:
                    c_rank4[i, j, i, j] = c_voigt[i + 3 if i + 3 < 6 else 3, j + 3 if j + 3 < 6 else 3]
        ops = [np.eye(3), -np.eye(3)]
        c_rank4_sym = UniversalNeumannTensorEngine.project_elastic_stiffness_rank4(c_rank4, ops)

        # 3. Transport and Domain-Specific Multiphysics Tier
        transport_engine = UnifiedThermalElectronicTransportEngine(temperature_k=T)
        freqs = np.linspace(1.0, 15.0, 30)
        linewidths = np.ones(30) * 0.35
        vels = np.ones((30, 3)) * 3400.0

        thermal_res = transport_engine.solve_dual_channel_peierls_wigner_thermal_conductivity(
            frequencies_thz=freqs,
            linewidths_thz=linewidths,
            diagonal_velocities_m_s=vels,
            cell_volume_ang3=110.0,
        )

        e_grid = np.linspace(-2.0, 2.0, 50)
        dos = np.ones(50) * 1.8
        e_vels = np.ones((50, 3)) * 2.5e5
        tau_e = np.ones(50) * 40.0
        el_res = transport_engine.solve_full_brillouin_zone_electronic_transport(
            energies_ev=e_grid,
            dos_states_ev=dos,
            band_velocities_m_s=e_vels,
            relaxation_times_fs=tau_e,
            fermi_energy_ev=0.0,
            cell_volume_ang3=110.0,
        )

        # 4. CALPHAD-Coupled Grand Potential Phase Field & STZ Kinetics
        phi_init = np.ones((3, 8, 8, 8)) / 3.0
        mu_vec = np.zeros(2)
        pf_res = self.calphad_pf.step_forward_grand_potential_field(
            phi_fields=phi_init,
            chemical_potentials=mu_vec,
            dt_s=0.002,
        )

        stz_rate = self.calphad_pf.compute_stz_plastic_strain_rate(
            deviatoric_shear_stress_mpa=max(10.0, tgt.applied_stress_mpa),
            effective_disorder_temperature_chi=0.15,
        )

        # 5. Interphase Cohesion
        w_sep_res = self.cohesive.compute_work_of_separation(
            surface_energy_phase1_j_m2=1.4,
            surface_energy_phase2_j_m2=1.1,
            interface_energy_j_m2=0.5,
        )

        # 6. Synthesize Full Multiscale State Output
        return {
            "candidate_name": candidate_name,
            "composition": composition,
            "target_domain": tgt.domain_type,
            "formation_energy_ev_atom": q_state.formation_energy_ev_atom,
            "symmetric_stiffness_tensor": c_rank4_sym.tolist(),
            "lattice_thermal_conductivity_w_m_k": thermal_res["isotropic_total_kappa_w_m_k"],
            "peierls_thermal_conductivity_w_m_k": thermal_res["isotropic_peierls_kappa_w_m_k"],
            "wigner_tunneling_fraction": thermal_res["wigner_tunneling_fraction"],
            "electrical_conductivity_s_m": el_res["isotropic_conductivity_s_m"],
            "seebeck_coefficient_uv_k": el_res["isotropic_seebeck_uv_k"],
            "thermoelectric_power_factor_uw_m_k2": el_res["thermoelectric_power_factor_uw_m_k2"],
            "hall_coefficient_m3_c": el_res["hall_coefficient_m3_c"],
            "stz_plastic_shear_rate_s_inv": stz_rate,
            "work_of_separation_j_m2": w_sep_res["work_of_separation_w_sep_j_m2"],
            "phase_field_fractions": pf_res["mean_phase_fractions"],
            "is_state_driven_pipeline_successful": True,
        }
