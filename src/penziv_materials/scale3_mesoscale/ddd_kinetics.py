"""Discrete Dislocation Dynamics (DDD) & Peach-Koehler Dislocation Mobility Engine."""

from typing import Dict, Tuple, List, Optional
import numpy as np


class DiscreteDislocationEngine:
    """Nodal dislocation dynamics solver with dynamic Cottrell solute drag and Peach-Koehler forces."""

    def __init__(self, burgers_vector_m: float = 2.54e-10):
        self.b_vec_m = burgers_vector_m

    def compute_peach_koehler_force(
        self,
        stress_tensor_pa: np.ndarray,
        burgers_vector: np.ndarray,
        line_sense_tangent: np.ndarray,
    ) -> np.ndarray:
        """Evaluate exact Peach-Koehler force per unit dislocation line length:

        f_PK = (sigma · b) × xi
        """
        # Traction vector t = sigma · b
        traction = np.matmul(stress_tensor_pa, burgers_vector)
        # Force f = t × xi
        f_pk = np.cross(traction, line_sense_tangent)
        return f_pk

    def compute_dynamic_cottrell_drag_coefficient(
        self,
        solute_concentration: float,
        temperature_k: float,
        dislocation_velocity_m_s: float,
        base_drag_pa_s: float = 1.0e-4,
    ) -> float:
        """Dynamic Cottrell atmosphere solute drag B_drag(c, T, v):

        B_total = B_phonon(T) + B_solute(c, T) / (1 + (v / v_crit)^2)
        """
        # Phonon drag increases linearly with temperature
        b_phonon = base_drag_pa_s * (temperature_k / 300.0)

        # Non-linear solute breakaway
        v_crit = 0.05  # m/s
        b_solute = 5.0e-3 * solute_concentration * np.exp(1200.0 / max(1.0, temperature_k))
        b_effective_solute = b_solute / (1.0 + (dislocation_velocity_m_s / v_crit) ** 2)

        b_total = b_phonon + b_effective_solute
        return float(b_total)

    def advance_dislocation_nodes(
        self,
        node_positions_m: np.ndarray,
        stress_tensor_pa: np.ndarray,
        burgers_vector: np.ndarray,
        line_tangents: np.ndarray,
        temperature_k: float,
        solute_concentration: float,
        dt_s: float = 1.0e-10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance dislocation nodal positions: v = B^-1 · F_PK, r(t+dt) = r(t) + v * dt."""
        n_nodes = len(node_positions_m)
        velocities = np.zeros((n_nodes, 3), dtype=np.float64)

        for i in range(n_nodes):
            f_pk = self.compute_peach_koehler_force(
                stress_tensor_pa=stress_tensor_pa,
                burgers_vector=burgers_vector,
                line_sense_tangent=line_tangents[i],
            )
            f_mag = np.linalg.norm(f_pk)
            # Estimate initial velocity
            v_est = f_mag / 1.0e-4
            b_drag = self.compute_dynamic_cottrell_drag_coefficient(
                solute_concentration=solute_concentration,
                temperature_k=temperature_k,
                dislocation_velocity_m_s=v_est,
            )
            v_node = f_pk / max(1e-12, b_drag)
            velocities[i] = v_node

        new_positions = node_positions_m + velocities * dt_s
        return new_positions, velocities
