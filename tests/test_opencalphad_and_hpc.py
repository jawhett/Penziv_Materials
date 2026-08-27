"""Unit tests for OpenCALPHAD TDB Engine and First-Principles HPC Dispatcher."""

import unittest
import numpy as np

from penziv_materials.thermodynamics.opencalphad_tdb import OpenCALPHADTDBEngine
from penziv_materials.meta_bridge.hpc_dispatch import FirstPrinciplesHPCDispatcher


class TestOpenCALPHADAndHPC(unittest.TestCase):
    def setUp(self):
        self.calphad = OpenCALPHADTDBEngine()
        self.hpc = FirstPrinciplesHPCDispatcher(cluster_partition="gpu-a100", num_nodes=2)

    def test_calphad_tdb_parser_and_minimizer(self):
        sample_tdb = """
        ELEMENT NI FCC_A1 58.6934 $
        ELEMENT CR BCC_A2 51.9961 $
        ELEMENT AL FCC_A1 26.9815 $
        PHASE FCC_A1 % 1 1.0 $
        PHASE BCC_A2 % 1 1.0 $
        PHASE GAMMA_PRIME % 2 0.75 0.25 $
        PARAMETER G(FCC_A1,NI;0) 298.15 -5120.0 + 120.0*T - 25.0*T*LN(T);
        """
        parse_res = self.calphad.parse_tdb_content(sample_tdb)
        self.assertTrue(parse_res["is_tdb_valid"])
        self.assertGreater(parse_res["num_phases"], 0)

        min_res = self.calphad.minimize_multicomponent_gibbs_energy(
            overall_composition={"Ni": 0.70, "Al": 0.20, "Cr": 0.10},
            temperature_k=1123.15,
        )
        self.assertIn("stable_primary_phase", min_res)
        self.assertIn("equilibrium_phase_fractions", min_res)
        self.assertAlmostEqual(sum(min_res["equilibrium_phase_fractions"].values()), 1.0, places=4)

    def test_first_principles_hpc_dispatch_and_ingestion(self):
        lat = np.eye(3) * 3.85
        species = ["Ni", "Al"]
        fracs = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        job = self.hpc.trigger_automated_first_principles_dispatch(
            formula="NiAl",
            lattice_matrix=lat,
            atomic_species=species,
            fractional_coords=fracs,
        )
        self.assertEqual(job.status, "SUBMITTED")
        self.assertIn("&CONTROL", job.input_deck_content)
        self.assertIn("#SBATCH", job.slurm_script_content)

        ingest_res = self.hpc.ingest_completed_dft_ground_truth(
            job_id=job.job_id,
            total_energy_ev=-14.85,
            atomic_forces_ev_ang=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        self.assertEqual(ingest_res["status"], "INGESTED")
        self.assertTrue(ingest_res["is_active_learning_updated"])
        self.assertGreater(ingest_res["total_training_pool_size"], 0)


if __name__ == "__main__":
    unittest.main()
