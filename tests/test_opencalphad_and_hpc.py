"""Unit tests for PyCALPHAD Adapter and First-Principles HPC Dispatcher."""

import unittest
import numpy as np

from penziv_materials.adapters.standard_adapters import CalphadAdapter
from penziv_materials.meta_bridge.hpc_dispatch import FirstPrinciplesHPCDispatcher

MINIMAL_NI_AL_TDB = """
ELEMENT /- ELECTRON_GAS 0.0000E+00 0.0000E+00 0.0000E+00!
ELEMENT VA VACUUM 0.0000E+00 0.0000E+00 0.0000E+00!
ELEMENT NI FCC_A1 5.8693E+01 4.7870E+03 2.9796E+01!
ELEMENT AL FCC_A1 2.6982E+01 4.5400E+03 2.8300E+01!
SPECIES NI NI1!
SPECIES AL AL1!
PHASE FCC_A1 % 1 1.0 !
CONSTITUENT FCC_A1 :AL,NI: !
PARAMETER G(FCC_A1,AL;0) 298.15 -7976.15+137.093038*T-24.3671976*T*LN(T)-.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 6000.0 N !
PARAMETER G(FCC_A1,NI;0) 298.15 -5179.15+117.854*T-22.096*T*LN(T)-.0048407*T**2; 6000.0 N !
"""


class TestOpenCALPHADAndHPC(unittest.TestCase):
    def setUp(self):
        self.hpc = FirstPrinciplesHPCDispatcher(cluster_partition="gpu-a100", num_nodes=2)

    def test_calphad_tdb_parser_and_minimizer(self):
        min_res = CalphadAdapter.evaluate_gibbs_equilibrium(
            elements=["NI", "AL"],
            phases=["FCC_A1"],
            temperature_k=1123.15,
            tdb_file_content=MINIMAL_NI_AL_TDB,
        )
        self.assertEqual(min_res["backend"], "pycalphad")
        self.assertIn("stable_phases", min_res)
        self.assertIn("gibbs_energy_j_mol", min_res)

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
