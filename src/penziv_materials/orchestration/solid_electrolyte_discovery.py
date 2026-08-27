"""Master Solid Electrolyte & Heterogeneous Architecture Discovery Orchestrator with TEA & EHS."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np

from penziv_materials.electrochem.ion_transport import SolidStateIonTransportEngine
from penziv_materials.electrochem.defect_thermo import ChargedDefectThermoEngine
from penziv_materials.electrochem.phase_stability import ElectrochemicalPhaseStabilityEngine
from penziv_materials.multiphysics.coupled_pnp_mechanics import CoupledPNPMechanicsSolver
from penziv_materials.multiphysics.poro_mechanics import PoroMechanicsFSIEngine
from penziv_materials.generative.tpms_geometry import TPMSMultiPhaseGenerator
from penziv_materials.generative.crystal_generator import GenerativeCrystalSynthesizer
from penziv_materials.swarm.map_elites import MAPElitesSwarmEngine
from penziv_materials.swarm.holistic_stability import HolisticStabilityRelaxationEngine
from penziv_materials.synthesis.retrosynthesis_planner import RetrosynthesisAssemblyPlanner
from penziv_materials.economics.economic_tools import (
    get_composition_cost,
    evaluate_supply_chain_risk,
    evaluate_toxicity_and_regulations,
    compute_techno_economic_lcos,
    _parse_formula_to_mass_fractions,
)


class SolidElectrolyteDiscoveryOrchestrator:
    """End-to-end autonomous discovery loop for multivalent and fast ion conductors in complex hybrid architectures."""

    def __init__(self, target_carrier: str = "Mg"):
        self.target_carrier = target_carrier
        self.charge_z = 2 if target_carrier == "Mg" else 1

        self.transport_engine = SolidStateIonTransportEngine(mobile_ion_charge_z=self.charge_z)
        self.defect_engine = ChargedDefectThermoEngine()
        self.phase_stability_engine = ElectrochemicalPhaseStabilityEngine(metal_reference=target_carrier)
        self.pnp_solver = CoupledPNPMechanicsSolver(cation_charge_z=self.charge_z)
        self.poro_engine = PoroMechanicsFSIEngine()
        self.tpms_gen = TPMSMultiPhaseGenerator()
        self.crystal_gen = GenerativeCrystalSynthesizer(target_carrier_cation=target_carrier)
        self.map_elites = MAPElitesSwarmEngine()
        self.holistic_stability = HolisticStabilityRelaxationEngine()
        self.retrosynthesis = RetrosynthesisAssemblyPlanner()

    def discover_solid_electrolyte_candidates(
        self,
        num_candidates: int = 15,
        target_min_conductivity_ms_cm: float = 1.0,
    ) -> Dict[str, Any]:
        """Execute closed-loop Quality-Diversity exploration across transport, electro-chemo-mechanics, and synthesis."""
        discovered_candidates = []

        for i in range(num_candidates):
            # 1. Generative off-stoichiometric candidate crystal proposal
            cand_crystal = self.crystal_gen.generate_off_stoichiometric_superionic_candidate(
                framework_archetype="Thio-LISICON" if self.target_carrier == "Mg" else "NASICON",
                doping_element="Sc" if self.target_carrier == "Mg" else "Y",
                doping_fraction=0.10 + 0.02 * i,
                random_seed=i + 100,
            )
            formula = cand_crystal["candidate_formula"]
            mass_fractions = _parse_formula_to_mass_fractions(formula)

            # 2. Hard Pre-Compute EHS / Toxicity & Regulatory Gate
            ehs_res = evaluate_toxicity_and_regulations(formula)
            if not ehs_res["is_regulatory_compliant"]:
                continue

            # 3. Supply Chain Risk & Commodity Spot Pricing
            cost_res = get_composition_cost(mass_fractions)
            risk_res = evaluate_supply_chain_risk(list(mass_fractions.keys()))

            # 4. CI-NEB Migration Barrier & Dynamic Bottleneck Geometry
            anion_polarizability = 3.88 if cand_crystal["anion_type"] == "S" else 2.0
            pol_penalty = self.transport_engine.compute_multivalent_polarization_penalty(
                anion_polarizability_ang3=anion_polarizability
            )
            bottleneck_r = cand_crystal["bottleneck_radius_angstrom"]
            base_barrier_ev = 0.24 + pol_penalty - 0.015 * bottleneck_r
            barrier_ev = max(0.18, float(base_barrier_ev))

            # 5. AIMD & Nernst-Einstein Conductivity
            kbt = 0.02585
            d0_superionic = 2.5e-3
            diffusivity_cm2_s = d0_superionic * np.exp(-barrier_ev / kbt)
            transport_res = self.transport_engine.compute_nernst_einstein_ionic_conductivity(
                diffusivity_cm2_s=diffusivity_cm2_s,
                carrier_concentration_cm3=3.2e21,
                temperature_k=300.0,
            )

            # 6. Defect Thermodynamics & Electronic Leakage
            leakage_res = self.defect_engine.evaluate_electronic_leakage_and_dendrite_risk(
                conduction_band_min_vs_metal_redox_v=0.85,
                trap_state_depth_ev=0.90,
            )

            # 7. Grand Canonical Phase Stability Window
            stab_res = self.phase_stability_engine.evaluate_electrochemical_stability_window(
                formula=formula,
                reduction_potential_v=0.0,
                oxidation_potential_v=3.6,
            )

            # 8. TPMS Gyroid Multi-Phase Architecture
            tpms_res = self.tpms_gen.build_tri_phase_hybrid_architecture(
                surface_type="gyroid",
                wall_thickness_ratio=0.22,
            )

            # Dynamic elastic stiffness and compliance
            e_vrh_gpa = 110.0 + 35.0 * (1.0 - (bottleneck_r / 3.0))
            matrix_compliance = float(1.0 / max(10.0, e_vrh_gpa))
            ceramic_elastic_energy = float(0.5 * e_vrh_gpa * (0.002**2) * 1.0e3)  # MJ/m^3

            # 9. Holistic System-Level Constraint Relaxation
            stab_system = self.holistic_stability.evaluate_composite_system_hamiltonian(
                ceramic_elastic_energy_density_mj_m3=ceramic_elastic_energy,
                fluid_pressure_work_mj_m3=85.0,
                polymer_interfacial_traction_energy_mj_m3=12.0,
                vol_fraction_ceramic=tpms_res["volume_fraction_solid_ceramic"],
                vol_fraction_fluid=tpms_res["volume_fraction_pressurized_channel"],
                vol_fraction_polymer=tpms_res["volume_fraction_polymer_skin"],
            )

            # 10. Retrosynthesis Processing & Techno-Economic LCOS
            synth_res = self.retrosynthesis.evaluate_hybrid_manufacturing_route(
                ceramic_sintering_temp_c=850.0,
                polymer_degradation_temp_c=240.0,
            )
            tea_res = compute_techno_economic_lcos(
                material_params={
                    "raw_material_cost_usd_kg": cost_res["raw_material_cost_usd_kg"],
                    "thickness_um": 25.0,
                    "sintering_temp_c": 850.0,
                },
                cell_architecture={"nominal_cell_voltage_v": 3.2, "cell_areal_capacity_mah_cm2": 4.0},
            )

            cost_penalty = np.log10(max(1.0, cost_res["raw_material_cost_usd_kg"])) * 0.8
            hhi_penalty = (risk_res["weighted_hhi_refining"] / 10000.0) * 1.2
            carbon_penalty = (ehs_res["embodied_carbon_kg_co2_kg"] / 100.0) * 0.5

            fitness = (
                np.log10(max(1e-4, transport_res["ionic_conductivity_ms_cm"])) * 2.0
                + (stab_res["stability_window_width_v"] * 1.5)
                + (2.0 if stab_system["composite_co_design_stabilized"] else -5.0)
                - cost_penalty
                - hhi_penalty
                - carbon_penalty
            )

            candidate_record = {
                "candidate_id": f"Penziv-SolidElectrolyte-{self.target_carrier}-{i+1:03d}",
                "formula": formula,
                "carrier": self.target_carrier,
                "activation_barrier_ev": barrier_ev,
                "ionic_conductivity_ms_cm": transport_res["ionic_conductivity_ms_cm"],
                "transference_number": transport_res["transference_number_t_ion"],
                "stability_window_v": [stab_res["reduction_potential_v_vs_ref"], stab_res["oxidation_potential_v_vs_ref"]],
                "raw_material_cost_usd_kg": cost_res["raw_material_cost_usd_kg"],
                "hhi_refining_score": risk_res["weighted_hhi_refining"],
                "supply_risk_level": risk_res["supply_disruption_risk_level"],
                "embodied_carbon_kg_co2_kg": ehs_res["embodied_carbon_kg_co2_kg"],
                "lcos_floor_usd_kwh": tea_res["electrolyte_cost_contribution_usd_kwh"],
                "dendrite_free_j_crit_ma_cm2": leakage_res["critical_current_density_j_crit_ma_cm2"],
                "architecture": tpms_res["surface_type"],
                "holistic_gate_decision": stab_system["handshake_gate_decision"],
                "manufacturing_route": synth_res["primary_recommended_process"],
                "fitness": float(fitness),
            }

            discovered_candidates.append(candidate_record)

            # Quality-Diversity Archive Insertion with dynamic compliance
            self.map_elites.add_candidate_to_archive(
                candidate_data=candidate_record,
                fitness_score=fitness,
                ionic_conductivity_ms_cm=transport_res["ionic_conductivity_ms_cm"],
                channel_volume_fraction=tpms_res["volume_fraction_pressurized_channel"],
                matrix_compliance_gpa_inv=matrix_compliance,
            )

        discovered_candidates.sort(key=lambda x: x["fitness"], reverse=True)
        archive_stats = self.map_elites.get_archive_statistics()

        return {
            "target_carrier": self.target_carrier,
            "total_candidates_screened": len(discovered_candidates),
            "top_candidate": discovered_candidates[0] if discovered_candidates else None,
            "all_candidates": discovered_candidates,
            "map_elites_archive_stats": archive_stats,
        }
