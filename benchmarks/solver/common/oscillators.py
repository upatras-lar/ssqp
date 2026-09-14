from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class DuffingConfig:
    # Problem size / structure
    m: int = 4
    # Number of oscillators:
    #   nx = 2*m
    #   nu = m

    coupling: Literal["local", "global"] = "local"

    # Dynamics
    dt: float = 0.05

    damping: float = 0.15  # d
    stiffness: float = 1.0  # k
    cubic: float = 0.25  # alpha
    coupling_strength: float = 0.4  # c

    # For local coupling:
    #
    # fixed:
    #     c * (q[i-1] - 2 q[i] + q[i+1])
    #
    # grid:
    #     c * (m+1)^2 * (...)
    #
    # "fixed" is preferable for pure solver scaling.
    # "grid" corresponds more closely to refining a fixed spatial domain,
    # but makes the dynamics increasingly stiff as m grows.
    local_scaling: Literal["fixed", "grid"] = "fixed"

    # Stage cost
    q_weight: float = 1.0
    v_weight: float = 0.1
    u_weight: float = 1e-2

    # Optional nonlinear cost:
    # beta/4 * sum(q_i - q_ref_i)^4
    quartic_cost: float = 0.0

    # Terminal cost
    terminal_q_weight: float = 10.0
    terminal_v_weight: float = 1.0
    terminal_quartic_cost: float = 0.0

    # Scaling
    # Dividing cost by m prevents the objective magnitude from
    # automatically growing as the number of oscillators increases.
    normalize_cost_by_m: bool = True

    # Useful if you want:
    #
    # stage_cost_scale = dt
    #
    # for a Riemann approximation of an integral cost.
    stage_cost_scale: float = 1.0
    terminal_cost_scale: float = 1.0


class CoupledDuffingOCP:
    """
    Scalable nonlinear optimal-control benchmark.

    State:
        x = [q_1, ..., q_m, v_1, ..., v_m]

    Control:
        u = [u_1, ..., u_m]

    Continuous dynamics:
        qdot = v

        vdot_i =
            -d v_i
            -k q_i
            -alpha q_i^3
            + coupling_i(q)
            + u_i

    Discrete dynamics:
        x_next = x + dt * f(x, u)

    Derivative convention:
        z = [x; u]

    dynamics_hessians(x,u)[i] = Hessian of F_i w.r.t. z

    weighted_dynamics_hessian(x,u,lam)
        = Hessian of lam.T @ F(x,u)
    """

    def __init__(self, cfg: DuffingConfig):
        if cfg.m < 1:
            raise ValueError("m must be >= 1")

        self.cfg = cfg

        self.m = cfg.m
        self.nx = 2 * self.m
        self.nu = self.m
        self.nz = self.nx + self.nu

        self._I = np.eye(self.m)
        self._L = self._make_local_laplacian()

    # Utilities
    def _make_local_laplacian(self):
        """
        Fixed-end discrete Laplacian:

            q_0     = 0
            q_{m+1} = 0

        L q gives:
            q_{i-1} - 2 q_i + q_{i+1}
        """
        L = -2.0 * np.eye(self.m)

        if self.m > 1:
            i = np.arange(self.m - 1)
            L[i, i + 1] = 1.0
            L[i + 1, i] = 1.0

        return L

    def _check_xu(self, x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        if x.shape != (self.nx,):
            raise ValueError(f"x must have shape ({self.nx},), got {x.shape}")

        if u.shape != (self.nu,):
            raise ValueError(f"u must have shape ({self.nu},), got {u.shape}")

        return x, u

    def _cost_normalization(self):
        if self.cfg.normalize_cost_by_m:
            return 1.0 / self.m
        return 1.0

    def initial_state(
        self,
        amplitude: float = 1.0,
        mode: int = 1,
    ):
        """
        Dimension-independent sinusoidal initial condition.

        q_i = A sin(mode*pi*i/(m+1))
        v_i = 0
        """
        i = np.arange(1, self.m + 1)

        q = amplitude * np.sin(mode * np.pi * i / (self.m + 1))

        v = np.zeros(self.m)

        return np.concatenate([q, v])

    # Coupling
    def _local_gain(self):
        c = self.cfg.coupling_strength

        if self.cfg.local_scaling == "grid":
            c *= (self.m + 1) ** 2

        return c

    def coupling(self, q):
        """
        Coupling acceleration only.
        """
        q = np.asarray(q)

        if self.cfg.coupling == "local":
            return self._local_gain() * (self._L @ q)

        if self.cfg.coupling == "global":
            # delta[i,j] = q_j - q_i
            delta = q[None, :] - q[:, None]

            # 1/m normalization keeps the coupling force O(1)
            # as the number of oscillators grows.
            c = self.cfg.coupling_strength / self.m

            return c * np.sin(delta).sum(axis=1)

        raise ValueError(f"Unknown coupling: {self.cfg.coupling}")

    def coupling_jacobian(self, q):
        """
        Returns dcoupling/dq, shape (m,m).
        """
        q = np.asarray(q)

        if self.cfg.coupling == "local":
            return self._local_gain() * self._L

        # Global coupling:
        # c/m * sum_j sin(q_j - q_i)

        delta = q[None, :] - q[:, None]

        c = self.cfg.coupling_strength / self.m

        J = c * np.cos(delta)

        # j = i term has zero derivative because
        # q_i - q_i == 0 identically.
        np.fill_diagonal(J, 0.0)

        # diagonal derivative is minus sum of off-diagonal
        np.fill_diagonal(J, -J.sum(axis=1))

        return J

    # Continuous dynamics

    def continuous_dynamics(self, x, u):
        """
        f(x,u), where xdot = f(x,u).
        """
        x, u = self._check_xu(x, u)

        q = x[: self.m]
        v = x[self.m :]

        a = (
            -self.cfg.damping * v
            - self.cfg.stiffness * q
            - self.cfg.cubic * q**3
            + self.coupling(q)
            + u
        )

        return np.concatenate([v, a])

    def continuous_jacobians(self, x, u):
        """
        Returns:

            Fx = df/dx   shape (nx,nx)
            Fu = df/du   shape (nx,nu)
        """
        x, u = self._check_xu(x, u)

        q = x[: self.m]

        Fx = np.zeros((self.nx, self.nx))
        Fu = np.zeros((self.nx, self.nu))

        # qdot = v
        Fx[: self.m, self.m :] = self._I

        # acceleration derivative wrt q
        Aq = self.coupling_jacobian(q).copy()

        Aq += np.diag(-self.cfg.stiffness - 3.0 * self.cfg.cubic * q**2)

        Fx[self.m :, : self.m] = Aq

        # acceleration derivative wrt v
        Fx[self.m :, self.m :] = -self.cfg.damping * self._I

        # acceleration derivative wrt u
        Fu[self.m :, :] = self._I

        return Fx, Fu

    def continuous_hessians(self, x, u):
        """
        Full second-derivative tensor.

        Returns H with shape:

            (nx, nz, nz)

        where:

            H[i,:,:] = Hessian of f_i(x,u)

        and:

            z = [x; u]

        WARNING:
            This tensor scales as O(nx * nz^2).

        For large systems, prefer:
            continuous_weighted_hessian(...)
        """
        x, u = self._check_xu(x, u)

        q = x[: self.m]

        H = np.zeros((self.nx, self.nz, self.nz))

        # Duffing cubic term
        # -alpha q_i^3
        # d2/dq_i2 = -6 alpha q_i

        for i in range(self.m):
            H[self.m + i, i, i] += -6.0 * self.cfg.cubic * q[i]

        # Global coupling second derivatives

        if self.cfg.coupling == "global":

            c = self.cfg.coupling_strength / self.m

            for i in range(self.m):

                # Hessian of acceleration_i wrt q
                Hi = H[
                    self.m + i,
                    : self.m,
                    : self.m,
                ]

                for j in range(self.m):

                    if j == i:
                        continue

                    # For:
                    #
                    # c sin(q_j - q_i)
                    #
                    # Hessian contribution is:
                    #
                    # -c sin(q_j-q_i)
                    # (e_j-e_i)(e_j-e_i)^T

                    s = -c * np.sin(q[j] - q[i])

                    Hi[i, i] += s
                    Hi[j, j] += s

                    Hi[i, j] -= s
                    Hi[j, i] -= s

        return H

    def continuous_weighted_hessian(
        self,
        x,
        u,
        lam,
    ):
        """
        Hessian of:

            lam.T @ f(x,u)

        with respect to:

            z = [x;u]

        Returns shape:

            (nz,nz)

        This is usually what SQP/IPM solvers actually need.

        It avoids constructing the full:
            (nx,nz,nz)
        dynamics Hessian tensor.
        """
        x, u = self._check_xu(x, u)

        lam = np.asarray(lam, dtype=float)

        if lam.shape != (self.nx,):
            raise ValueError(f"lam must have shape ({self.nx},)")

        q = x[: self.m]

        # Only acceleration equations are nonlinear.
        lam_v = lam[self.m :]

        H = np.zeros((self.nz, self.nz))

        Hq = H[: self.m, : self.m]

        # Duffing cubic
        idx = np.arange(self.m)

        Hq[idx, idx] += lam_v * (-6.0 * self.cfg.cubic * q)

        # Global coupling
        if self.cfg.coupling == "global":

            c = self.cfg.coupling_strength / self.m

            for i in range(self.m):

                wi = lam_v[i]

                if wi == 0.0:
                    continue

                for j in range(self.m):

                    if i == j:
                        continue

                    s = wi * -c * np.sin(q[j] - q[i])

                    Hq[i, i] += s
                    Hq[j, j] += s

                    Hq[i, j] -= s
                    Hq[j, i] -= s

        return H

    # Discrete OCP dynamics

    def dynamics(self, x, u):
        """
        Forward-Euler discrete dynamics:

            F(x,u) = x + dt*f(x,u)
        """
        x = np.asarray(x, dtype=float)

        return x + self.cfg.dt * self.continuous_dynamics(x, u)

    def dynamics_jacobians(self, x, u):
        """
        Returns:

            A = dF/dx
            B = dF/du
        """
        Fx, Fu = self.continuous_jacobians(x, u)

        A = np.eye(self.nx) + self.cfg.dt * Fx

        B = self.cfg.dt * Fu

        return A, B

    def dynamics_hessians(self, x, u):
        """
        Full tensor:

            H[i] = Hessian_z F_i

        shape:

            (nx,nz,nz)
        """
        return self.cfg.dt * self.continuous_hessians(x, u)

    def weighted_dynamics_hessian(
        self,
        x,
        u,
        lam,
    ):
        """
        Hessian of:

            lam.T @ F(x,u)

        wrt:

            z = [x;u]
        """
        return self.cfg.dt * self.continuous_weighted_hessian(
            x,
            u,
            lam,
        )

    # Stage cost
    def stage_cost(
        self,
        x,
        u,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ):
        """
        l(x,u) =

            1/2 wq ||q-qref||^2
          + 1/2 wv ||v-vref||^2
          + 1/2 wu ||u-uref||^2
          + beta/4 sum_i (q_i-qref_i)^4
        """
        x, u = self._check_xu(x, u)

        if x_ref is None:
            x_ref = np.zeros(self.nx)

        if u_ref is None:
            u_ref = np.zeros(self.nu)

        x_ref = np.asarray(x_ref)
        u_ref = np.asarray(u_ref)

        q_err = x[: self.m] - x_ref[: self.m]
        v_err = x[self.m :] - x_ref[self.m :]
        u_err = u - u_ref

        value = (
            0.5 * self.cfg.q_weight * np.dot(q_err, q_err)
            + 0.5 * self.cfg.v_weight * np.dot(v_err, v_err)
            + 0.5 * self.cfg.u_weight * np.dot(u_err, u_err)
            + 0.25 * self.cfg.quartic_cost * np.sum(q_err**4)
        )

        scale = self.cfg.stage_cost_scale * self._cost_normalization()

        return scale * value

    def stage_cost_derivatives(
        self,
        x,
        u,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ):
        """
        Returns:

            value
            grad_z
            Hessian_z

        where:

            z = [x;u]
        """
        x, u = self._check_xu(x, u)

        if x_ref is None:
            x_ref = np.zeros(self.nx)

        if u_ref is None:
            u_ref = np.zeros(self.nu)

        x_ref = np.asarray(x_ref)
        u_ref = np.asarray(u_ref)

        q_err = x[: self.m] - x_ref[: self.m]
        v_err = x[self.m :] - x_ref[self.m :]
        u_err = u - u_ref

        scale = self.cfg.stage_cost_scale * self._cost_normalization()

        value = self.stage_cost(
            x,
            u,
            x_ref,
            u_ref,
        )

        # Gradient
        g = np.zeros(self.nz)

        g[: self.m] = (
            self.cfg.q_weight * q_err + self.cfg.quartic_cost * q_err**3
        )

        g[self.m : self.nx] = self.cfg.v_weight * v_err

        g[self.nx :] = self.cfg.u_weight * u_err

        g *= scale

        # Hessian
        H = np.zeros((self.nz, self.nz))

        idx_q = np.arange(self.m)

        H[idx_q, idx_q] = (
            self.cfg.q_weight + 3.0 * self.cfg.quartic_cost * q_err**2
        )

        idx_v = self.m + np.arange(self.m)

        H[idx_v, idx_v] = self.cfg.v_weight

        idx_u = self.nx + np.arange(self.nu)

        H[idx_u, idx_u] = self.cfg.u_weight

        H *= scale

        return value, g, H

    # Terminal cost
    def terminal_cost(
        self,
        x,
        x_ref: Optional[np.ndarray] = None,
    ):
        if x_ref is None:
            x_ref = np.zeros(self.nx)

        x = np.asarray(x)
        x_ref = np.asarray(x_ref)

        q_err = x[: self.m] - x_ref[: self.m]
        v_err = x[self.m :] - x_ref[self.m :]

        value = (
            0.5 * self.cfg.terminal_q_weight * np.dot(q_err, q_err)
            + 0.5 * self.cfg.terminal_v_weight * np.dot(v_err, v_err)
            + 0.25 * self.cfg.terminal_quartic_cost * np.sum(q_err**4)
        )

        scale = self.cfg.terminal_cost_scale * self._cost_normalization()

        return scale * value

    def terminal_cost_derivatives(
        self,
        x,
        x_ref: Optional[np.ndarray] = None,
    ):
        """
        Returns:

            value
            grad_x
            Hessian_x
        """
        if x_ref is None:
            x_ref = np.zeros(self.nx)

        x = np.asarray(x)
        x_ref = np.asarray(x_ref)

        q_err = x[: self.m] - x_ref[: self.m]
        v_err = x[self.m :] - x_ref[self.m :]

        scale = self.cfg.terminal_cost_scale * self._cost_normalization()

        value = self.terminal_cost(x, x_ref)

        g = np.zeros(self.nx)

        g[: self.m] = (
            self.cfg.terminal_q_weight * q_err
            + self.cfg.terminal_quartic_cost * q_err**3
        )

        g[self.m :] = self.cfg.terminal_v_weight * v_err

        g *= scale

        H = np.zeros((self.nx, self.nx))

        idx_q = np.arange(self.m)

        H[idx_q, idx_q] = (
            self.cfg.terminal_q_weight
            + 3.0 * self.cfg.terminal_quartic_cost * q_err**2
        )

        idx_v = self.m + np.arange(self.m)

        H[idx_v, idx_v] = self.cfg.terminal_v_weight

        H *= scale

        return value, g, H
