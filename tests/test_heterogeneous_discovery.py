"""Unit tests for Solid Electrolytes, Coupled Multiphysics, Generative Topology, and QD Swarms."""

import unittest
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
from penziv_materials.orchestration.solid_electrolyte_discovery import SolidElectrolyteDiscoveryOrchestrator


class TestHeterogeneousDiscovery(unittest.TestCase):
    def test_ion_transport_cineb_and_conductivity(self):
        engine = SolidStateIonTransportEngine(mobile_ion_charge_z=2)
        # CI-NEB path
        path = np.array([0.0, 0.15, 0.32, 0.18, 0.05])
        neb_res = engine.compute_ci_neb_migration_barrier(path)
        self.assertAlmostEqual(neb_res["activation_energy_ev"], 0.32)

        # Polarization penalty
        penalty = engine.compute_multivalent_polarization_penalty(anion_polarizability_ang3=3.88)
        self.assertGreater(penalty, 0.0)

        # AIMD MSD
        traj = np.random.randn(20, 10, 3) * 0.5
        d_cm2_s, msd_final, msd_arr = engine.compute_msd_and_diffusivity_aimd(traj)
        self.assertGreaterEqual(d_cm2_s, 0.0)

        # Nernst-Einstein
        sigma_res = engine.compute_nernst_einstein_ionic_conductivity(
            diffusivity_cm2_s=1.0e-7,
            carrier_concentration_cm3=1.0e21,
            temperature_k=300.0,
        )
        self.assertIn("ionic_conductivity_ms_cm", sigma_res)
        self.assertGreater(sigma_res["transference_number_t_ion"], 0.90)

    def test_defect_thermodynamics_fnv(self):
        defect_engine = ChargedDefectThermoEngine(band_gap_ev=4.5)
        fnv_corr = defect_engine.compute_fnv_image_charge_correction(defect_charge_q=2)
        self.assertGreater(fnv_corr, 0.0)

        formation = defect_engine.compute_defect_formation_energy(
            e_defect_dft_ev=-120.5,
            e_pristine_dft_ev=-125.0,
            defect_charge_q=2,
            fermi_level_ef_ev=1.5,
            chemical_potential_deltas={"Mg": -1.2},
            stoichiometry_deltas={"Mg": -1},
        )
        self.assertIn("defect_formation_energy_ev", formation)

        leakage = defect_engine.evaluate_electronic_leakage_and_dendrite_risk(
            conduction_band_min_vs_metal_redox_v=0.9
        )
        self.assertTrue(leakage["is_electronically_insulating"])

    def test_grand_canonical_phase_stability(self):
        stab = ElectrochemicalPhaseStabilityEngine(metal_reference="Mg")
        phi = stab.compute_grand_potential(-5.5, num_metal_atoms_per_formula=1.0, applied_voltage_vs_metal_v=1.5)
        self.assertIsInstance(phi, float)

        res = stab.evaluate_electrochemical_stability_window("MgSc(PS4)3", 0.0, 3.8)
        self.assertTrue(res["is_thermodynamically_stable_vs_anode"])
        self.assertTrue(res["is_high_voltage_cathode_stable"])

    def test_coupled_pnp_mechanics(self):
        pnp = CoupledPNPMechanicsSolver(grid_points=20)
        c_cation = np.linspace(1e21, 2e21, 20)
        phi, e_field, lambda_d = pnp.solve_space_charge_potential_1d(c_cation, 1.5e21)
        self.assertEqual(len(phi), 20)
        self.assertGreater(lambda_d, 0.0)

        j_bv = pnp.evaluate_butler_volmer_current_density(overpotential_eta_v=0.05)
        self.assertGreater(j_bv, 0.0)

        stress = pnp.compute_chemo_mechanical_stress_coupling(
            elastic_strain=np.eye(3) * 0.001,
            concentration_change_mol_m3=100.0,
            electric_field_v_m=1.0e6,
        )
        self.assertIn("total_stress_mpa", stress)

    def test_poro_mechanics_fsi(self):
        poro = PoroMechanicsFSIEngine(gas_pressure_mpa=3.0)
        kn_res = poro.compute_knudsen_diffusion_coefficient(pore_diameter_nm=50.0)
        self.assertIn("knudsen_number", kn_res)

        wall = poro.evaluate_channel_wall_hydrostatic_support(applied_external_compressive_stress_mpa=50.0)
        self.assertTrue(wall["is_mechanically_stabilized"])

    def test_tpms_multi_phase_geometry(self):
        tpms = TPMSMultiPhaseGenerator(resolution=(12, 12, 12))
        res = tpms.build_tri_phase_hybrid_architecture(surface_type="gyroid")
        self.assertTrue(res["is_interpenetrating_bicontinuous"])
        self.assertGreater(res["volume_fraction_solid_ceramic"], 0.0)
        self.assertGreater(res["volume_fraction_pressurized_channel"], 0.0)

    def test_crystal_synthesizer(self):
        gen = GenerativeCrystalSynthesizer(target_carrier_cation="Mg")
        cand = gen.generate_off_stoichiometric_superionic_candidate()
        self.assertIn("candidate_formula", cand)
        self.assertEqual(cand["target_carrier"], "Mg")

    def test_map_elites_qd_swarm(self):
        swarm = MAPElitesSwarmEngine(grid_dim_x=5, grid_dim_y=5, grid_dim_z=5)
        inserted = swarm.add_candidate_to_archive(
            candidate_data={"name": "TestElectrolyte"},
            fitness_score=8.5,
            ionic_conductivity_ms_cm=5.2,
            channel_volume_fraction=0.35,
            matrix_compliance_gpa_inv=0.05,
        )
        self.assertTrue(inserted)
        stats = swarm.get_archive_statistics()
        self.assertEqual(stats["occupied_niches"], 1)

    def test_holistic_stability_relaxation(self):
        relax = HolisticStabilityRelaxationEngine()
        eval_res = relax.evaluate_composite_system_hamiltonian(
            ceramic_elastic_energy_density_mj_m3=140.0,
            fluid_pressure_work_mj_m3=90.0,
            polymer_interfacial_traction_energy_mj_m3=10.0,
            vol_fraction_ceramic=0.4,
            vol_fraction_fluid=0.4,
            vol_fraction_polymer=0.2,
        )
        self.assertTrue(eval_res["composite_co_design_stabilized"])
        self.assertEqual(eval_res["handshake_gate_decision"], "ACCEPTED_VIA_HOLISTIC_RELAXATION")

    def test_retrosynthesis_planner(self):
        planner = RetrosynthesisAssemblyPlanner()
        synth = planner.evaluate_hybrid_manufacturing_route(ceramic_sintering_temp_c=800.0)
        self.assertTrue(synth["is_synthetically_feasible"])
        self.assertIn("SEQUENTIAL_COLD_SINTERING_AND_INFILTRATION", synth["primary_recommended_process"])

    def test_solid_electrolyte_discovery_loop(self):
        orchestrator = SolidElectrolyteDiscoveryOrchestrator(target_carrier="Mg")
        res = orchestrator.discover_solid_electrolyte_candidates(num_candidates=3)
        self.assertEqual(res["total_candidates_screened"], 3)
        self.assertIsNotNone(res["top_candidate"])


if __name__ == "__main__":
    unittest.main()
