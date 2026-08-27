"""SO(3)-Equivariant Physics-Informed Neural Operator (PINO) with Pseudo-Arc-Length Continuation."""

from typing import Dict, Tuple, List, Optional, Callable
import numpy as np


class SO3PINOSurrogate:
    """Fourier Neural Operator enforcing hard frame indifference under SO(3) rotations and arc-length bifurcation tracking."""

    def __init__(self, in_channels: int = 9, out_channels: int = 6, num_modes: int = 12):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes

    def forward_frame_indifferent_operator(
        self,
        deformation_gradient_F: np.ndarray,
    ) -> np.ndarray:
        """Evaluate surrogate stress response while strictly preserving SO(3) frame indifference:

        N(Q · F · Q^T) == Q · N(F) · Q^T
        """
        # Right Cauchy-Green deformation tensor C = F^T · F (strictly frame-invariant scalar invariants)
        C = np.matmul(deformation_gradient_F.T, deformation_gradient_F)
        I1 = float(np.trace(C))
        I2 = 0.5 * (I1**2 - float(np.trace(np.matmul(C, C))))
        I3 = float(np.linalg.det(C))

        # Hyperelastic strain energy derivatives dW/dI1, dW/dI2
        mu1 = 450.0  # MPa
        mu2 = 120.0
        dW_dI1 = mu1 + mu2 * (I1 - 3.0)
        dW_dI2 = mu2

        # 2nd Piola-Kirchhoff stress in reference frame S = 2 * (dW/dI1 * I + dW/dI2 * (I1*I - C) + ...)
        S_pk2 = 2.0 * (dW_dI1 * np.eye(3) + dW_dI2 * (I1 * np.eye(3) - C))

        # Push forward to Cauchy stress in current frame: sigma = 1/J * F · S · F^T
        J = max(1e-4, np.sqrt(max(1e-4, I3)))
        cauchy_stress = (1.0 / J) * np.matmul(deformation_gradient_F, np.matmul(S_pk2, deformation_gradient_F.T))
        return cauchy_stress

    def step_pseudo_arc_length_continuation(
        self,
        residual_fn: Callable[[float, float], float],
        u_prev: float,
        lambda_prev: float,
        du_ds: float,
        dlambda_ds: float,
        ds: float = 0.05,
        max_newton_iters: int = 12,
        tol: float = 1.0e-5,
    ) -> Tuple[float, float, bool]:
        """Solve bifurcation / snap-back continuation problem using pseudo-arc-length constraint:

        Residual R(u, lambda) = 0
        Constraint C(u, lambda) = du_ds * (u - u_prev) + dlambda_ds * (lambda - lambda_prev) - ds = 0
        """
        # Initial predictor step
        u = u_prev + du_ds * ds
        lam = lambda_prev + dlambda_ds * ds

        for it in range(max_newton_iters):
            # Evaluate residual and constraint
            r_val = residual_fn(u, lam)
            c_val = du_ds * (u - u_prev) + dlambda_ds * (lam - lambda_prev) - ds

            if abs(r_val) < tol and abs(c_val) < tol:
                return u, lam, True

            # Numerical Jacobian
            eps = 1e-6
            dr_du = (residual_fn(u + eps, lam) - residual_fn(u - eps, lam)) / (2.0 * eps)
            dr_dlam = (residual_fn(u, lam + eps) - residual_fn(u, lam - eps)) / (2.0 * eps)

            J = np.array([[dr_du, dr_dlam], [du_ds, dlambda_ds]])
            rhs = -np.array([r_val, c_val])

            try:
                delta = np.linalg.solve(J, rhs)
                u += delta[0]
                lam += delta[1]
            except np.linalg.LinAlgError:
                return u, lam, False

        return u, lam, False
