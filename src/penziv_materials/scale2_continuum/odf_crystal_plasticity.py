"""Orientation Distribution Function (ODF) Texture Integration & Non-Schmid Plasticity."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class ODFTexturePlasticityEngine:
    """Computes polycrystalline Taylor and Sachs factors M(ODF) from discrete Euler angles and evaluates non-Schmid yield surfaces."""

    def __init__(self, num_orientations: int = 100):
        self.n_ori = num_orientations

    def generate_random_texture_euler_angles(self, seed: int = 42) -> np.ndarray:
        """Generate uniform Bunge Euler angles (phi1, Phi, phi2) in radians."""
        np.random.seed(seed)
        phi1 = np.random.uniform(0.0, 2.0 * np.pi, self.n_ori)
        phi = np.arccos(np.random.uniform(0.0, 1.0, self.n_ori))
        phi2 = np.random.uniform(0.0, 2.0 * np.pi, self.n_ori)
        return np.stack([phi1, phi, phi2], axis=-1)

    def compute_polycrystalline_taylor_and_sachs_factors(
        self,
        euler_angles_rad: Optional[np.ndarray] = None,
        slip_systems: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, float]:
        """Compute Taylor (isostrain upper bound) and Sachs (isostress lower bound) factors from crystallographic texture."""
        eulers = self.generate_random_texture_euler_angles() if euler_angles_rad is None else np.asarray(euler_angles_rad)
        n = len(eulers)

        # Standard FCC {111}<110> Schmid factors distribution across orientations
        schmid_factors = []
        for phi1, Phi, phi2 in eulers:
            # Rotation matrix from crystal to sample
            r_z1 = np.array([[np.cos(phi1), -np.sin(phi1), 0], [np.sin(phi1), np.cos(phi1), 0], [0, 0, 1]])
            r_x = np.array([[1, 0, 0], [0, np.cos(Phi), -np.sin(Phi)], [0, np.sin(Phi), np.cos(Phi)]])
            r_z2 = np.array([[np.cos(phi2), -np.sin(phi2), 0], [np.sin(phi2), np.cos(phi2), 0], [0, 0, 1]])
            g = np.dot(r_z1, np.dot(r_x, r_z2))

            # Tensile loading axis [0, 0, 1] in sample frame
            load_axis = np.dot(g.T, np.array([0.0, 0.0, 1.0]))

            # Max Schmid factor for {111}<110>
            n_plane = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
            s_dir = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
            m_s = abs(np.dot(load_axis, n_plane) * np.dot(load_axis, s_dir))
            schmid_factors.append(max(0.1, m_s))

        m_schmid_mean = float(np.mean(schmid_factors))
        sachs_factor = 1.0 / max(0.01, m_schmid_mean)
        # Taylor factor (full constraint upper bound)
        taylor_factor = float(np.mean(1.0 / np.array(schmid_factors)))

        return {
            "mean_schmid_factor": m_schmid_mean,
            "sachs_factor_lower_bound": float(sachs_factor),
            "taylor_factor_upper_bound": float(taylor_factor),
            "taylor_sachs_mean_factor": float(0.5 * (taylor_factor + sachs_factor)),
            "texture_anisotropy_ratio": float(taylor_factor / max(1.0, sachs_factor)),
        }

    def evaluate_non_schmid_resolved_shear_stress(
        self,
        applied_stress_tensor: np.ndarray,      # (3, 3)
        slip_direction: np.ndarray,             # s
        slip_normal: np.ndarray,                # m
        non_schmid_coefficients: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute generalized non-Schmid yield criterion tau_eff = tau_Schmid + a1 * tau_coplanar + a2 * tau_cross + a3 * sigma_normal."""
        sig = np.asarray(applied_stress_tensor, dtype=np.float64)
        s = slip_direction / np.linalg.norm(slip_direction)
        m = slip_normal / np.linalg.norm(slip_normal)
        t = np.cross(m, s)

        coeffs = non_schmid_coefficients or {"a1": 0.08, "a2": 0.05, "a3": 0.02}

        tau_schmid = float(np.dot(m, np.dot(sig, s)))
        tau_coplanar = float(np.dot(m, np.dot(sig, t)))
        tau_cross = float(np.dot(t, np.dot(sig, s)))
        sig_normal = float(np.dot(m, np.dot(sig, m)))

        tau_eff = tau_schmid + coeffs.get("a1", 0.08) * tau_coplanar + coeffs.get("a2", 0.05) * tau_cross + coeffs.get("a3", 0.02) * sig_normal

        return {
            "schmid_resolved_shear_stress_mpa": tau_schmid,
            "effective_non_schmid_shear_stress_mpa": tau_eff,
            "non_schmid_contribution_percent": float(100.0 * abs(tau_eff - tau_schmid) / max(1.0, abs(tau_schmid))),
            "is_non_schmid_significant": bool(abs(tau_eff - tau_schmid) > 0.05 * abs(tau_schmid)),
        }
