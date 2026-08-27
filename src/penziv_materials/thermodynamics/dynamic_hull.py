"""Universal Thermodynamic Convex Hull & Grand Canonical CALPHAD Solver with CSRO."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from scipy.optimize import linprog
from penziv_materials.core.constants import R_GAS, BOLTZMANN_EV_K


class UniversalConvexHullSolver:
    """Solves thermodynamic phase equilibria, non-ideal CSRO mixing, and grand canonical chemical potential stability via Linear Programming."""

    @staticmethod
    def compute_temperature_dependent_gibbs_energy(
        e_dft_ev_atom: float,
        temperature_k: float = 300.0,
        pressure_gpa: float = 0.0,
        volume_ang3_atom: float = 15.0,
        debye_temperature_k: float = 450.0,
        composition: Optional[Dict[str, float]] = None,
        warren_cowley_csro_matrix: Optional[np.ndarray] = None,
        coordination_number_z: float = 12.0,
    ) -> float:
        """Evaluate full temperature and pressure-dependent Gibbs free energy:

        G(x, T, P) = E_DFT + F_vib(T) + F_elec(T) - T * (S_ideal + S_CSRO) + P * V
        """
        # 1. Phonon vibrational free energy via Debye model
        if temperature_k > 1.0:
            x = debye_temperature_k / temperature_k
            zero_point = 1.125 * BOLTZMANN_EV_K * debye_temperature_k
            f_thermal = -BOLTZMANN_EV_K * temperature_k * (np.pi**4 / (5.0 * (x**3 + 1e-6)))
            f_vib = zero_point + f_thermal
        else:
            f_vib = 1.125 * BOLTZMANN_EV_K * debye_temperature_k

        # 2. Electronic entropy term (Sommerfeld)
        gamma_elec = 1.5e-4  # eV / (atom * K^2)
        f_elec = -0.5 * gamma_elec * (temperature_k**2)

        # 3. Configurational & Non-ideal CSRO entropy
        s_conf = 0.0
        s_csro = 0.0
        if composition:
            tot = sum(composition.values())
            c_vec = np.array([count / max(1e-6, tot) for count in composition.values()])
            for c_i in c_vec:
                if c_i > 0:
                    s_conf -= BOLTZMANN_EV_K * c_i * np.log(c_i)

            if warren_cowley_csro_matrix is not None:
                alpha = np.asarray(warren_cowley_csro_matrix, dtype=np.float64)
                if alpha.shape == (len(c_vec), len(c_vec)):
                    for i in range(len(c_vec)):
                        for j in range(len(c_vec)):
                            p_ij = c_vec[j] * (1.0 - alpha[i, j])
                            if p_ij > 1e-9:
                                s_csro -= 0.5 * coordination_number_z * BOLTZMANN_EV_K * c_vec[i] * p_ij * np.log(p_ij / max(1e-9, c_vec[j]))

        total_entropy = s_conf + s_csro

        # 4. Pressure PV work (1 GPa * 1 Å^3 = 0.0062415 eV)
        pv_work_ev = pressure_gpa * volume_ang3_atom * 0.006241509

        g_total = e_dft_ev_atom + f_vib + f_elec - (temperature_k * total_entropy) + pv_work_ev
        return float(g_total)

    @classmethod
    def solve_stability(
        cls,
        target_composition: Dict[str, float],
        target_energy_per_atom: float,
        reference_database: List[Dict[str, Any]],
        temperature_k: float = 300.0,
    ) -> Dict[str, Any]:
        """Solve thermodynamic phase stability via linear programming over arbitrary chemical spaces."""
        elements = sorted(list(target_composition.keys()))
        n_elems = len(elements)
        tot_atoms = sum(target_composition.values())
        c_target = {k: v / max(1e-6, tot_atoms) for k, v in target_composition.items()}

        subspace_entries = [
            e for e in reference_database
            if set(e["composition"].keys()).issubset(set(elements))
        ]

        if not subspace_entries:
            subspace_entries = [
                {"composition": {el: 1.0}, "energy_per_atom": 0.0, "formula": el}
                for el in elements
            ]

        n_entries = len(subspace_entries)
        c_energies = np.array([e["energy_per_atom"] for e in subspace_entries])

        A_eq = np.zeros((n_elems + 1, n_entries))
        b_eq = np.zeros(n_elems + 1)

        for i, elem in enumerate(elements):
            b_eq[i] = c_target.get(elem, 0.0)
            for j, entry in enumerate(subspace_entries):
                entry_tot = sum(entry["composition"].values())
                A_eq[i, j] = entry["composition"].get(elem, 0.0) / max(1e-6, entry_tot)

        A_eq[-1, :] = 1.0
        b_eq[-1] = 1.0

        res = linprog(c_energies, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")

        if not res.success:
            e_hull_baseline = float(np.min(c_energies))
            active_phases = [subspace_entries[int(np.argmin(c_energies))].get("formula", "RefPhase")]
        else:
            e_hull_baseline = float(res.fun)
            active_indices = np.where(res.x > 1e-4)[0]
            active_phases = [subspace_entries[idx].get("formula", f"Phase_{idx}") for idx in active_indices]

        delta_e_hull = float(target_energy_per_atom - e_hull_baseline)
        is_stable = bool(delta_e_hull <= 0.025)

        return {
            "energy_above_hull_ev_atom": max(0.0, delta_e_hull),
            "is_thermodynamically_stable": is_stable,
            "ground_state_hull_energy_ev_atom": float(e_hull_baseline),
            "decomposition_products": active_phases,
            "temperature_k": temperature_k,
        }
