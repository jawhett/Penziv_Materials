"""Automated First-Principles HPC Dispatcher & Active Learning Data Pool Ingestion."""

import datetime
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.structure.crystal_structure import CrystalStructure


class HPCCalculationJob(BaseModel):
    """Metadata container for dispatched first-principles HPC calculations."""
    job_id: str
    calculation_type: str  # "DFT_SCF", "DFT_RELAX", "CI_NEB", "PHONON_BTE"
    code_engine: str       # "QUANTUM_ESPRESSO", "VASP"
    system_formula: str
    input_deck_content: str
    slurm_script_content: str
    status: str = "SUBMITTED"  # "SUBMITTED", "RUNNING", "COMPLETED", "INGESTED"
    submitted_at: str
    ground_truth_energy_ev: Optional[float] = None
    ground_truth_forces_ev_ang: Optional[List[List[float]]] = None


class FirstPrinciplesHPCDispatcher:
    """Automates native input deck generation for Quantum ESPRESSO & VASP and ingests ground truth into active learning pool."""

    def __init__(self, cluster_partition: str = "gpu-a100", num_nodes: int = 2):
        self.partition = cluster_partition
        self.nodes = num_nodes
        self.dispatched_jobs: List[HPCCalculationJob] = []
        self.training_data_pool: List[Dict[str, Any]] = []

    def generate_quantum_espresso_pw_deck(
        self,
        formula: str,
        lattice_matrix: np.ndarray,
        atomic_species: List[str],
        fractional_coords: np.ndarray,
        ecutwfc_ry: float = 65.0,
        ecutrho_ry: float = 520.0,
        k_grid: Tuple[int, int, int] = (6, 6, 6),
    ) -> str:
        """Generate production-ready Quantum ESPRESSO pw.x input deck."""
        unique_species = sorted(list(set(atomic_species)))
        n_atoms = len(atomic_species)
        n_typ = len(unique_species)

        lines = [
            "&CONTROL",
            "  calculation = 'scf',",
            "  restart_mode = 'from_scratch',",
            f"  prefix = '{formula}',",
            "  pseudo_dir = './pseudo',",
            "  outdir = './tmp',",
            "  tstress = .true.,",
            "  tprnfor = .true.,",
            "/",
            "&SYSTEM",
            f"  ibrav = 0, nat = {n_atoms}, ntyp = {n_typ},",
            f"  ecutwfc = {ecutwfc_ry:.1f}, ecutrho = {ecutrho_ry:.1f},",
            "  occupations = 'smearing', smearing = 'cold', degauss = 0.02,",
            "/",
            "&ELECTRONS",
            "  conv_thr = 1.0d-8,",
            "  mixing_beta = 0.7d0,",
            "/",
            "CELL_PARAMETERS (angstrom)",
        ]
        for row in lattice_matrix:
            lines.append(f"  {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}")

        lines.append("ATOMIC_SPECIES")
        for sp in unique_species:
            lines.append(f"  {sp:3s} 1.0 {sp}.pbe-n-kjpaw_psl.1.0.0.UPF")

        lines.append("ATOMIC_POSITIONS (crystal)")
        for sp, frac in zip(atomic_species, fractional_coords):
            lines.append(f"  {sp:3s} {frac[0]:12.8f} {frac[1]:12.8f} {frac[2]:12.8f}")

        lines.append(f"K_POINTS (automatic)\n  {k_grid[0]} {k_grid[1]} {k_grid[2]} 0 0 0")
        return "\n".join(lines)

    def generate_slurm_submission_script(
        self,
        job_name: str,
        walltime_hours: int = 12,
        num_gpus: int = 4,
    ) -> str:
        """Generate high-performance SLURM batch submission script."""
        return "\n".join([
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --gpus-per-node={num_gpus}",
            f"#SBATCH --time={walltime_hours:02d}:00:00",
            "#SBATCH --output=slurm-%j.out",
            "",
            "module purge",
            "module load quantum-espresso/7.2-cuda-openmpi",
            "",
            "export OMP_NUM_THREADS=1",
            f"srun -n {self.nodes * num_gpus} pw.x -input pw.in > pw.out",
        ])

    def trigger_automated_first_principles_dispatch(
        self,
        formula: str,
        lattice_matrix: np.ndarray,
        atomic_species: List[str],
        fractional_coords: np.ndarray,
        calculation_type: str = "DFT_SCF",
    ) -> HPCCalculationJob:
        """Create calculation decks and dispatch computation to HPC queue upon epistemic trigger."""
        job_id = f"HPC-DFT-{len(self.dispatched_jobs)+1:05d}"
        pw_deck = self.generate_quantum_espresso_pw_deck(
            formula=formula,
            lattice_matrix=lattice_matrix,
            atomic_species=atomic_species,
            fractional_coords=fractional_coords,
        )
        slurm_script = self.generate_slurm_submission_script(job_name=f"{formula}_{job_id}")

        job = HPCCalculationJob(
            job_id=job_id,
            calculation_type=calculation_type,
            code_engine="QUANTUM_ESPRESSO",
            system_formula=formula,
            input_deck_content=pw_deck,
            slurm_script_content=slurm_script,
            status="SUBMITTED",
            submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        self.dispatched_jobs.append(job)
        return job

    def ingest_completed_dft_ground_truth(
        self,
        job_id: str,
        total_energy_ev: float,
        atomic_forces_ev_ang: List[List[float]],
    ) -> Dict[str, Any]:
        """Ingest converged ab initio ground truth into active learning pool to retrain MLIP surrogate models."""
        for job in self.dispatched_jobs:
            if job.job_id == job_id:
                job.status = "COMPLETED"
                job.ground_truth_energy_ev = total_energy_ev
                job.ground_truth_forces_ev_ang = atomic_forces_ev_ang

                record = {
                    "job_id": job_id,
                    "formula": job.system_formula,
                    "energy_ev": total_energy_ev,
                    "forces": atomic_forces_ev_ang,
                    "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                self.training_data_pool.append(record)

                return {
                    "job_id": job_id,
                    "status": "INGESTED",
                    "total_training_pool_size": len(self.training_data_pool),
                    "is_active_learning_updated": True,
                }

        raise KeyError(f"Job ID {job_id} not found.")
