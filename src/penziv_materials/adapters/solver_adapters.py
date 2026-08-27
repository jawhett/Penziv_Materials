"""Commercial & Open-Source Solver Adapters with Subprocess Runners & Output Log Ingestion."""

import os
import re
import subprocess
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class SolverAdapterBridge:
    """Translates scale-bridging data packets into native solver inputs, executes cluster jobs, and parses output logs."""

    def __init__(self):
        self.supported_solvers = [
            "VASP",
            "QUANTUM_ESPRESSO",
            "LAMMPS",
            "MOOSE_PHASE_FIELD",
            "DAMASK_CPFFT",
            "WARP3D",
            "ABAQUS_UMAT",
            "THERMO_CALC_TQ",
        ]

    def generate_quantum_espresso_input(
        self,
        formula: str,
        lattice_parameter_angstrom: float,
        ecutwfc_ry: float = 80.0,
    ) -> str:
        """Generate formatted pw.x input card for Quantum ESPRESSO self-consistent DFT."""
        qe_input = f"""&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '{formula}',
  pseudo_dir = './pseudo',
  outdir = './tmp',
  tstress = .true.,
  tprnfor = .true.
/
&SYSTEM
  ibrav = 2,
  celldm(1) = {lattice_parameter_angstrom * 1.8897259886:.6f},
  nat = 1,
  ntyp = 1,
  ecutwfc = {ecutwfc_ry:.1f},
  occupations = 'smearing',
  smearing = 'm-p',
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-8,
  mixing_beta = 0.7
/
ATOMIC_SPECIES
  Ni  58.6934  Ni.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Ni 0.00 0.00 0.00
K_POINTS (automatic)
  16 16 16 0 0 0
"""
        return qe_input

    def generate_lammps_neb_script(
        self,
        potential_file: str = "potential.eam.alloy",
        spring_constant: float = 1.0,
        num_replicas: int = 16,
    ) -> str:
        """Generate formatted LAMMPS CI-NEB script for minimum energy pathway calculations."""
        lammps_script = f"""# LAMMPS CI-NEB Migration Path Calculation
units           metal
atom_style      atomic
atom_modify     map array
boundary        p p p

read_data       initial.data
pair_style      eam/alloy
pair_coeff      * * {potential_file} Ni

fix             1 all neb {spring_constant:.2f}
fix             2 all neb/ci {spring_constant:.2f}

timestep        0.001
neb             1.0e-6 1.0e-4 1000 1000 100 final final.coords
"""
        return lammps_script

    def generate_damask_material_config(
        self,
        c11_gpa: float,
        c12_gpa: float,
        c44_gpa: float,
        tau_0_mpa: float,
    ) -> str:
        """Generate DAMASK material.config card for CPFFT crystal plasticity simulations."""
        damask_yaml = f"""# DAMASK Crystal Plasticity Configuration
phase:
  Matrix_gamma:
    lattice: cF
    mechanical:
      elastic:
        type: Hooke
        C_11: {c11_gpa * 1e9:.2e}
        C_12: {c12_gpa * 1e9:.2e}
        C_44: {c44_gpa * 1e9:.2e}
      plastic:
        type: phenopowerlaw
        N_sl: [12]
        tau_0_sl: [{tau_0_mpa * 1e6:.2e}]
        h_0_sl_sl: 200.0e6
        a_sl: 2.25
        n_sl: 20.0
"""
        return damask_yaml

    def generate_slurm_submission_script(
        self,
        job_name: str,
        solver_cmd: str,
        num_nodes: int = 2,
        num_tasks_per_node: int = 64,
        walltime_hours: int = 12,
    ) -> str:
        """Generate HPC Slurm job submission script for automated cluster dispatch."""
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_tasks_per_node}
#SBATCH --time={walltime_hours:02d}:00:00
#SBATCH --partition=compute
#SBATCH --output={job_name}_%j.log
#SBATCH --error={job_name}_%j.err

module load intel oneapi openmpi/4.1.5

echo "Starting job {job_name} at $(date)"
mpirun -np $(( {num_nodes} * {num_tasks_per_node} )) {solver_cmd}
echo "Completed job {job_name} at $(date)"
"""
        return slurm_script

    def parse_quantum_espresso_scf_output(self, log_content: str) -> Dict[str, Any]:
        """Parse Quantum ESPRESSO pw.x standard output log to extract total energy, forces, stress tensor, and Fermi energy."""
        results: Dict[str, Any] = {
            "converged": False,
            "total_energy_ry": None,
            "total_energy_ev": None,
            "fermi_energy_ev": None,
            "total_force_ry_au": None,
            "pressure_kbar": None,
        }

        if "convergence has been achieved" in log_content:
            results["converged"] = True

        energy_match = re.search(r"!\s+total energy\s+=\s+([-\d\.]+)\s+Ry", log_content)
        if energy_match:
            e_ry = float(energy_match.group(1))
            results["total_energy_ry"] = e_ry
            results["total_energy_ev"] = e_ry * 13.605693122994

        fermi_match = re.search(r"the Fermi energy is\s+([-\d\.]+)\s+ev", log_content, re.IGNORECASE)
        if fermi_match:
            results["fermi_energy_ev"] = float(fermi_match.group(1))

        force_match = re.search(r"Total force\s+=\s+([-\d\.]+)", log_content)
        if force_match:
            results["total_force_ry_au"] = float(force_match.group(1))

        press_match = re.search(r"P=\s+([-\d\.]+)\s+kbar", log_content)
        if press_match:
            results["pressure_kbar"] = float(press_match.group(1))

        return results

    def parse_lammps_neb_log(self, log_content: str) -> Dict[str, float]:
        """Parse LAMMPS CI-NEB log to extract forward and reverse activation migration barriers."""
        results = {"forward_barrier_ev": 0.0, "reverse_barrier_ev": 0.0, "reaction_energy_ev": 0.0}
        fwd_match = re.search(r"Forward barrier\s+=\s+([-\d\.]+)\s+eV", log_content)
        if fwd_match:
            results["forward_barrier_ev"] = float(fwd_match.group(1))

        rev_match = re.search(r"Backward barrier\s+=\s+([-\d\.]+)\s+eV", log_content)
        if rev_match:
            results["reverse_barrier_ev"] = float(rev_match.group(1))

        results["reaction_energy_ev"] = results["forward_barrier_ev"] - results["reverse_barrier_ev"]
        return results

    def execute_local_subprocess(
        self,
        command_args: List[str],
        cwd: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> Tuple[int, str, str]:
        """Execute a local physics solver subprocess safely with timeout and returncode capture."""
        try:
            res = subprocess.run(
                command_args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Execution timed out after {timeout_seconds}s"
        except Exception as e:
            return -1, "", str(e)
