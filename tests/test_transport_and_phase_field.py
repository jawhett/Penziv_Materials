"""Unit tests for Universal Neumann Tensors, Wigner-Peierls Transport, CALPHAD Grand Potential Phase-Field, and Cohesive Zone Interface Mechanics."""

import unittest
import numpy as np

from penziv_materials.structure.universal_neumann import UniversalNeumannTensorEngine
from penziv_materials.physics.wigner_peierls_transport import UnifiedThermalElectronicTransportEngine
from penziv_materials.scale3_mesoscale.calphad_grand_potential import CALPHADGrandPotentialPhaseFieldEngine
from penziv_materials.physics.cohesive_interface import CohesiveZoneInterfaceEngine


class TestTransportAndPhaseField(unittest.TestCase):
    def setUp(self):
        self.transport = UnifiedThermalElectronicTransportEngine(temperature_k=300.0)
        self.calphad_pf = CALPHADGrandPotentialPhaseFieldEngine(num_phases=3, grid_shape=(8, 8, 8))
        self.cohesive = CohesiveZoneInterfaceEngine(temperature_k=300.0)

    def test_universal_neumann_arbitrary_rank_projection(self):
        # Rank 2
        d2 = np.random.uniform(1.0, 10.0, (3, 3))
        ops = [np.eye(3), -np.eye(3)]
        p2 = UniversalNeumannTensorEngine.project_tensor(d2, ops, rank=2)
        self.assertEqual(p2.shape, (3, 3))

        # Rank 3
        d3 = np.random.uniform(1.0, 10.0, (3, 3, 3))
        p3 = UniversalNeumannTensorEngine.project_piezoelectric_rank3(d3, ops)
        self.assertEqual(p3.shape, (3, 3, 3))

        # Rank 4
        c4 = np.random.uniform(10.0, 200.0, (3, 3, 3, 3))
        p4 = UniversalNeumannTensorEngine.project_elastic_stiffness_rank4(c4, ops)
        self.assertEqual(p4.shape, (3, 3, 3, 3))
        # Verify major symmetry C_ijkl == C_klij
        np.testing.assert_allclose(p4, np.swapaxes(np.swapaxes(p4, 0, 2), 1, 3), atol=1e-5)

    def test_dual_channel_peierls_wigner_thermal_transport(self):
        freqs = np.linspace(1.0, 15.0, 25)
        gammas = np.ones(25) * 0.4
        vels = np.ones((25, 3)) * 3200.0

        res = self.transport.solve_dual_channel_peierls_wigner_thermal_conductivity(
            frequencies_thz=freqs,
            linewidths_thz=gammas,
            diagonal_velocities_m_s=vels,
            cell_volume_ang3=100.0,
        )
        self.assertIn("kappa_total_tensor_w_m_k", res)
        self.assertIn("isotropic_total_kappa_w_m_k", res)
        self.assertIn("wigner_tunneling_fraction", res)
        self.assertGreater(res["isotropic_total_kappa_w_m_k"], 0.0)

    def test_full_brillouin_zone_electronic_transport(self):
        e_bins = np.linspace(-1.5, 1.5, 40)
        dos = np.ones(40) * 1.5
        vels = np.ones((40, 3)) * 2.0e5
        tau = np.ones(40) * 30.0

        res = self.transport.solve_full_brillouin_zone_electronic_transport(
            energies_ev=e_bins,
            dos_states_ev=dos,
            band_velocities_m_s=vels,
            relaxation_times_fs=tau,
            fermi_energy_ev=0.0,
            cell_volume_ang3=100.0,
        )
        self.assertIn("electrical_conductivity_tensor_s_m", res)
        self.assertIn("thermoelectric_power_factor_uw_m_k2", res)
        self.assertIn("hall_coefficient_m3_c", res)
        self.assertGreater(res["isotropic_conductivity_s_m"], 0.0)

    def test_calphad_grand_potential_phase_field(self):
        phi_init = np.ones((3, 8, 8, 8)) / 3.0
        chem_pot = np.zeros(2)
        res = self.calphad_pf.step_forward_grand_potential_field(
            phi_fields=phi_init,
            chemical_potentials=chem_pot,
            dt_s=0.002,
        )
        self.assertTrue(res["is_calphad_coupled"])
        self.assertEqual(len(res["mean_phase_fractions"]), 3)
        self.assertAlmostEqual(sum(res["mean_phase_fractions"]), 1.0, places=4)

        # STZ amorphous plasticity check
        gamma_dot = self.calphad_pf.compute_stz_plastic_strain_rate(
            deviatoric_shear_stress_mpa=650.0,
            effective_disorder_temperature_chi=0.18,
        )
        self.assertGreater(gamma_dot, 0.0)

    def test_cohesive_zone_and_pnp_biot(self):
        w_res = self.cohesive.compute_work_of_separation(
            surface_energy_phase1_j_m2=1.2,
            surface_energy_phase2_j_m2=0.8,
            interface_energy_j_m2=0.4,
        )
        self.assertEqual(w_res["work_of_separation_w_sep_j_m2"], 1.6)
        self.assertTrue(w_res["is_thermodynamically_adherent"])

        trac = self.cohesive.evaluate_exponential_traction_separation(
            normal_opening_delta_n_nm=0.3,
            shear_opening_delta_t_nm=0.1,
            work_of_separation_j_m2=1.6,
        )
        self.assertIn("normal_traction_t_n_mpa", trac)
        self.assertGreater(trac["peak_cohesive_strength_mpa"], 0.0)

        # PNP-Biot fluxes
        c_field = np.ones((8, 8, 8)) * 1000.0
        phi_field = np.zeros((8, 8, 8))
        phi_field[4:, :, :] = 0.5
        sig_field = np.zeros((8, 8, 8))
        pnp_res = self.cohesive.solve_coupled_pnp_biot_fluxes(
            concentration_field_mol_m3=c_field,
            electric_potential_v=phi_field,
            hydrostatic_stress_field_mpa=sig_field,
        )
        self.assertIn("max_ion_flux_mol_m2_s", pnp_res)
        self.assertGreater(pnp_res["max_ion_flux_mol_m2_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
