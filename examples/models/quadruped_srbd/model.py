import ssqpy
import ssqpy.model.dynamics as dynamics
import ssqpy.model.constraints as constraints
import ssqpy.model.costs as costs

import numpy as np

from pathlib import Path
import os

import models.quadruped_hlplanner.config as config
import models.utils as utils

# Pinocchio model creation
import pinocchio as pin

examples_env = os.environ.get("PYTHONPATH", "")
sep = os.pathsep
examples_dir_str = Path(
    [p for p in examples_env.split(sep) if "examples" in p][0]
)

pin_model, _, _ = pin.buildModelsFromMJCF(
    str(examples_dir_str / Path("assets/mjcf/go2/go2.xml"))
)
pin_data = pin_model.createData()


# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------
def _p(x):
    return x[0:3]


def _q(x):
    return x[3:7]


def _v(x):
    return x[7:10]


def _w(x):
    return x[10:13]


def _pack(p, q, v, w):
    return np.hstack((p, q / np.linalg.norm(q), v, w))


# ---------------------------------------------------------------------------
# Tangent state accessors
# ---------------------------------------------------------------------------
def _dp(dx):
    return dx[0:3]


def _dq(dx):
    return dx[3:6]


def _dv(dx):
    return dx[6:9]


def _dw(dx):
    return dx[9:12]


# ---------------------------------------------------------------------------
# Model Dynamics for SRB system
# ---------------------------------------------------------------------------
class SRBDynamics(dynamics.StageDynamics):
    def __init__(self, dt, p_ref, mass, inertia):
        super().__init__()
        self.dt = dt
        self.p_ref = p_ref

        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)

    def _foot_positions(self):
        return {
            foot: self.p_ref[foot][0] for foot in config.FRAME_NAMES.keys()
        }

    def update_p_ref(self, p_ref):
        self.p_ref = p_ref

    def output_size(self):
        return config.SRBDParameters.nq

    def state_size(self):
        return config.SRBDParameters.nq

    def state_diff_size(self):
        return config.SRBDParameters.ndq

    def control_size(self):
        return config.SRBDParameters.nu

    def state_right_plus(self, x, dx):
        pn = _p(x) + _dp(dx)
        qn = utils.quat_integrate(_q(x), _dq(dx))
        vn = _v(x) + _dv(dx)
        wn = _w(x) + _dw(dx)

        return _pack(pn, qn, vn, wn)

    def state_right_minus(self, x, y):
        dp = _p(x) - _p(y)
        R_y = utils.quat_to_rotation_matrix(_q(y))
        R_x = utils.quat_to_rotation_matrix(_q(x))
        R_err = R_y.T @ R_x
        angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        if abs(angle) < 1e-8:
            dtheta = np.zeros(3)
        else:
            dtheta = (
                angle
                / (2.0 * np.sin(angle))
                * np.array(
                    [
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ]
                )
            )
        dv = _v(x) - _v(y)
        dw = _w(x) - _w(y)
        return np.hstack((dp, dtheta, dv, dw))

    def evaluate(self, x, u):
        fp = self._foot_positions()
        p, q, v, w = _p(x), _q(x), _v(x), _w(x)
        R = utils.quat_to_rotation_matrix(q)

        I_world = R @ self.inertia @ R.T
        I_world_inv = R @ self.inertia_inv @ R.T

        f_total = np.zeros(3)
        tau_total = np.zeros(3)
        for i, foot in enumerate(config.FRAME_NAMES.keys()):
            f_i = u[3 * i : 3 * i + 3]
            r_i = fp[foot] - p
            f_total += f_i
            tau_total += np.cross(r_i, f_i)

        # Velocity update
        v_new = v + self.dt * (
            f_total / self.mass + config.SRBDParameters.GRAVITY
        )
        w_new = w + self.dt * I_world_inv @ (
            tau_total - np.cross(w, I_world @ w)
        )

        # Position / orientation update with new velocities
        p_new = p + self.dt * v_new
        q_new = utils.quat_integrate(q, self.dt * w_new)

        return _pack(p_new, q_new, v_new, w_new)

    def gradient(self, x, u):
        """Analytical Jacobians for semi-implicit Euler SRBD.

        Returns
        -------
        dX : (ndq, ndq)  d(x⁺ ⊖ x⁺₀) / d(δx)  in tangent coordinates
        dU : (ndq, nu)   d(x⁺ ⊖ x⁺₀) / d(u)

        Tangent layout: [δp(3) | δθ(3) | δv(3) | δω(3)]
        Control layout: [f_FL(3) | f_FR(3) | f_RL(3) | f_RR(3)]

        Derivation
        ----------
        Semi-implicit Euler:
            v⁺ = v + dt·(Σfᵢ/m + g)
            ω⁺ = ω + dt·Iw⁻¹·(τ - ωxIwω)    τ = Σ rᵢxfᵢ,  rᵢ = p_footᵢ - p
            p⁺ = p + dt·v⁺
            q⁺ = quat_mul(q, dq(dt·ω⁺))

        Key sub-block derivations:
            ∂ω⁺/∂ω   = I + dt·Iw⁻¹·( skew(Iwω) - skew(ω)·Iw )
            ∂ω⁺/∂p   = dt·Iw⁻¹·Σ skew(fᵢ)          [rᵢ = p_foot - p, ∂rᵢ/∂p = -I]
            ∂ω⁺/∂θ   = dt·Iw⁻¹·( h·ωᵀ - (ω·h)I - skew(ω)·Iw·skew(ω) )
                        where h = Iw·ω,  from variation δIw = [δθ]xIw - Iw[δθ]x
            ∂θ⁺/∂θ   = R(dq)ᵀ  (SO(3) transport)  + dt·Jr·∂ω⁺/∂θ
                        R(dq)ᵀ from: (q⊕δθ)⊗dq ⊖ q⊗dq = R(dq)ᵀ δθ  (right trivialisation)
            ∂θ⁺/∂*   = dt·Jr(dt·ω⁺)·∂ω⁺/∂*   for * ∈ {p, ω}
            Jr(φ)     = I - (1-cos|φ|)/|φ|²·[φ]x + (|φ|-sin|φ|)/|φ|³·[φ]×²
        """
        p, q, v, w = _p(x), _q(x), _v(x), _w(x)
        R = utils.quat_to_rotation_matrix(q)
        dt = self.dt

        I_w = R @ self.inertia @ R.T
        I_w_inv = R @ self.inertia_inv @ R.T

        # ---- replicate forward pass to get ω⁺ and dq ----------------------
        fp = self._foot_positions()
        r = {foot: fp[foot] - p for foot in config.FRAME_NAMES.keys()}

        f_total = np.zeros(3)
        tau = np.zeros(3)
        for i, foot in enumerate(config.FRAME_NAMES.keys()):
            f_i = u[3 * i : 3 * i + 3]
            f_total += f_i
            tau += np.cross(r[foot], f_i)

        h = I_w @ w  # angular momentum, world frame
        gyro = np.cross(w, h)
        w_new = w + dt * (I_w_inv @ (tau - gyro))

        phi = dt * w_new  # rotation vector for quat update
        angle = np.linalg.norm(phi)

        # Rotation matrix R(dq) corresponding to the quaternion increment dq
        if angle < 1e-10:
            R_dq = np.eye(3)
            J_r = np.eye(3) - 0.5 * utils.skew(phi)
        else:
            axis = phi / angle
            K = utils.skew(axis)
            K2 = K @ K
            R_dq = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K2

            S_phi = utils.skew(phi)
            J_r = (
                np.eye(3)
                - (1.0 - np.cos(angle)) / angle**2 * S_phi
                + (angle - np.sin(angle)) / angle**3 * (S_phi @ S_phi)
            )

        # ---- ∂ω⁺/∂ω  -------------------------------------------------------
        # ω⁺ = ω + dt·Iw⁻¹·(τ − ω×h)
        # ∂(−ω×h)/∂ω  =  skew(h) − skew(ω)·Iw   [h = Iw ω, so also ∂h/∂ω = Iw]
        S_w = utils.skew(w)
        S_h = utils.skew(h)
        A_ww = np.eye(3) + dt * (I_w_inv @ (S_h - S_w @ I_w))

        # ---- ∂ω⁺/∂θ  -------------------------------------------------------
        # From δIw = [δθ]×·Iw − Iw·[δθ]×  (Lie derivative of Iw w.r.t. orientation):
        # δ(−ω×Iwω) = −[ω]×[δθ]×h + [ω]×Iw[δθ]×ω
        # First term:  −ω×(δθ×h) = h(ω·δθ) − (ω·h)δθ   →  (h·ωᵀ − (ω·h)I)δθ
        # Second term: [ω]×Iw[δθ]×ω = −[ω]×Iw[ω]×δθ    →  −(S_w·Iw·S_w)δθ
        dgyro_dtheta = (
            np.outer(h, w) - np.dot(w, h) * np.eye(3) - S_w @ I_w @ S_w
        )
        A_wtheta = dt * (I_w_inv @ dgyro_dtheta)

        # ---- ∂ω⁺/∂p  -------------------------------------------------------
        # τ = Σ rᵢ×fᵢ,  rᵢ = p_footᵢ − p
        # ∂(rᵢ×fᵢ)/∂p·δp = (−δp)×fᵢ  →  ∂τ/∂p = Σ skew(fᵢ)
        dtau_dp = sum(
            utils.skew(u[3 * i : 3 * i + 3])
            for i in range(len(config.FRAME_NAMES.keys()))
        )
        A_wp = dt * (I_w_inv @ dtau_dp)

        # ---- assemble dX (ndq × ndq) ---------------------------------------
        dX = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

        # δp⁺ = δp + dt·δv⁺
        dX[0:3, 0:3] = np.eye(3)
        dX[0:3, 6:9] = dt * np.eye(3)

        # δθ⁺ = R(dq)ᵀ·δθ  +  dt·Jr·(A_wp·δp + A_wtheta·δθ + A_ww·δω)
        dX[3:6, 0:3] = dt * (J_r @ A_wp)
        dX[3:6, 3:6] = R_dq.T + dt * (J_r @ A_wtheta)
        dX[3:6, 9:12] = dt * (J_r @ A_ww)

        # δv⁺ = δv  (f_total has no state dependence with fixed foot positions)
        dX[6:9, 6:9] = np.eye(3)

        # δω⁺
        dX[9:12, 0:3] = A_wp
        dX[9:12, 3:6] = A_wtheta
        dX[9:12, 9:12] = A_ww

        # ---- assemble dU (ndq × nu) ----------------------------------------
        dU = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

        for i, foot in enumerate(config.FRAME_NAMES.keys()):
            S_ri = utils.skew(r[foot])
            col = 3 * i

            # ∂v⁺/∂fᵢ = dt/m · I
            dU[6:9, col : col + 3] = (dt / self.mass) * np.eye(3)

            # ∂p⁺/∂fᵢ = dt · ∂v⁺/∂fᵢ = dt²/m · I
            dU[0:3, col : col + 3] = (dt**2 / self.mass) * np.eye(3)

            # ∂ω⁺/∂fᵢ = dt · Iw⁻¹ · skew(rᵢ)    [τ = Σ rᵢ×fᵢ, ∂(rᵢ×fᵢ)/∂fᵢ = skew(rᵢ)]
            dU_wi = dt * (I_w_inv @ S_ri)
            dU[9:12, col : col + 3] = dU_wi

            # ∂θ⁺/∂fᵢ = dt · Jr · ∂ω⁺/∂fᵢ
            dU[3:6, col : col + 3] = dt * (J_r @ dU_wi)

        return dX, dU

    def hessprod_xx(self, x, u, lam):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessprod_xu(self, x, u, lam):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessprod_uu(self, x, u, lam):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# ---------------------------------------------------------------------------
# Model Cost for SRBD
# ---------------------------------------------------------------------------

# Weights
w_p = 50.0  # CoM position tracking
w_v = 10.0  # CoM velocity tracking
w_o = 100.0  # Orientation tracking
w_w = 1.0  # Angular velocity regularisation
w_f = 1e-4  # Contact force regularisation

Q_p = w_p * np.eye(3)
Q_v = w_v * np.eye(3)
Q_o = w_o * np.eye(3)
Q_w = w_w * np.eye(3)
Q_f = w_f * np.eye(config.SRBDParameters.nu)


class SRBCost(costs.StageCost):
    """Tracking + regularisation cost for the SRBD model.

    References
    ----------
    com_ref  : np.ndarray (3,)  — desired CoM position
    xb_ref   : np.ndarray (7,)  — desired base pose [p(3), q(4)]
    p_ref    : dict  foot → [pos(3), is_swing(bool)]
    terminal : bool
    """

    def __init__(
        self,
        com_ref,
        xb_ref,
        p_ref,
        terminal=False,
    ):
        super().__init__()
        self.update_references(com_ref, xb_ref, p_ref)
        self.terminal = terminal

    def update_references(self, com_ref, xb_ref, p_ref):
        self.com_ref = com_ref
        self.xb_ref = xb_ref
        self.p_ref = p_ref

    def evaluate(self, x, u):
        cost = 0.0

        # CoM position tracking
        p_err = _p(x) - self.com_ref
        cost += p_err @ Q_p @ p_err

        # Orientation tracking
        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_err = utils.quat_mul(q_ref_inv, _q(x))
        cost += w_o * np.dot(q_err[:3], q_err[:3])

        # Angular velocity regularisation
        cost += _w(x) @ Q_w @ _w(x)

        if not self.terminal:
            # Contact force regularisation
            cost += u @ Q_f @ u

        return cost

    def gradient(self, x, u):
        dx = np.zeros(config.SRBDParameters.ndq)
        du = np.zeros(config.SRBDParameters.nu)

        # CoM position
        p_err = _p(x) - self.com_ref
        dx[0:3] += 2.0 * Q_p @ p_err

        # Orientation
        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_err = utils.quat_mul(q_ref_inv, _q(x))
        dx[3:6] += 2.0 * w_o * q_err[:3]  # local tangent approximation

        # Angular velocity
        dx[9:12] += 2.0 * Q_w @ _w(x)

        if not self.terminal:
            # Force regularisation
            du += 2.0 * Q_f @ u

        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))
        Hxx[0:3, 0:3] += 2.0 * Q_p
        Hxx[3:6, 3:6] += 2.0 * w_o * np.eye(3)
        Hxx[9:12, 9:12] += 2.0 * Q_w
        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        Huu = np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))
        if not self.terminal:
            Huu += 2.0 * Q_f
        return Huu


# ---------------------------------------------------------------------------
# SRBD Constraints
#
#   Per stance foot (is_swing == False):
#     ·  f_z  ≥  f_z_min                         (normal force lower bound)
#     ·  f_z  ≤  f_z_max                          (normal force upper bound)
#     ·  |f_x| ≤ μ f_z   →  -μ f_z ≤ f_x ≤ μ f_z  (linearised friction cone)
#     ·  |f_y| ≤ μ f_z   →  -μ f_y ≤ f_x ≤ μ f_y
#
#   Per swing foot:
#     ·  f = 0   enforced via tight equality bounds
#
#   Body height (z ≥ z_min)
#
# Constraint vector layout (per foot, 6 entries):
#   [ f_z,  f_x - μ f_z,  f_x + μ f_z,
#           f_y - μ f_z,  f_y + μ f_z,  f_z ]   (redundant f_z for clarity)
#
# For implementation simplicity the constraints are linearised as box
# constraints on a transformed control.  The standard friction pyramid uses:
#
#   c_i = [ f_x_i + μ f_z_i,   (≥ 0)
#           -f_x_i + μ f_z_i,  (≥ 0)
#            f_y_i + μ f_z_i,  (≥ 0)
#           -f_y_i + μ f_z_i,  (≥ 0)
#            f_z_i           ]  (≥ f_min, ≤ f_max)
#
# giving 5 scalar constraints per foot → 20 total, plus 1 height = 21.
# ---------------------------------------------------------------------------

MU = 0.6  # friction coefficient
F_Z_MIN = 1.0  # minimum normal force when in stance [N]
F_Z_MAX = 250.0  # maximum normal force [N]
Z_MIN = 0.20  # minimum CoM height [m]

# 5 friction/force constraints per foot + 1 height = 21
NC_SRB = 5 * len(config.FRAME_NAMES.keys()) + 1


class SRBConstraints(constraints.StageConstraints):
    """Friction-pyramid + height constraints for SRBD."""

    def __init__(self, p_ref):
        super().__init__()
        self.p_ref = p_ref

    def update_pref(self, p_ref):
        self.p_ref = p_ref

    def _is_swing(self, foot):
        return self.p_ref[foot][1]

    def output_size(self):
        return NC_SRB

    def lower_bound(self):
        lb = []
        for foot in config.FRAME_NAMES.keys():
            if self._is_swing(foot):
                lb += [0.0, 0.0, 0.0, 0.0, 0.0]  # all forces → 0
            else:
                lb += [0.0, 0.0, 0.0, 0.0, F_Z_MIN]
        lb.append(Z_MIN)
        return np.array(lb)

    def upper_bound(self):
        ub = []
        for foot in config.FRAME_NAMES.keys():
            if self._is_swing(foot):
                ub += [0.0, 0.0, 0.0, 0.0, 0.0]  # all forces → 0
            else:
                ub += [np.inf, np.inf, np.inf, np.inf, F_Z_MAX]
        ub.append(np.inf)
        return np.array(ub)

    def evaluate(self, x, u):
        c = []
        for i, foot in enumerate(config.FRAME_NAMES.keys()):
            fx, fy, fz = u[3 * i], u[3 * i + 1], u[3 * i + 2]
            c.append(fx + MU * fz)  # ≥ 0
            c.append(-fx + MU * fz)  # ≥ 0
            c.append(fy + MU * fz)  # ≥ 0
            c.append(-fy + MU * fz)  # ≥ 0
            c.append(fz)  # ≥ F_Z_MIN (stance) or = 0 (swing)
        c.append(_p(x)[2])  # CoM height ≥ Z_MIN
        return np.array(c)

    def gradient(self, x, u):
        dX = np.zeros((NC_SRB, config.SRBDParameters.ndq))
        dU = np.zeros((NC_SRB, config.SRBDParameters.nu))

        for i, foot in enumerate(config.FRAME_NAMES.keys()):
            row = 5 * i
            col = 3 * i  # u index for f_x of this foot
            #  f_x + μ f_z
            dU[row, col] = 1.0
            dU[row, col + 2] = MU
            # -f_x + μ f_z
            dU[row + 1, col] = -1.0
            dU[row + 1, col + 2] = MU
            #  f_y + μ f_z
            dU[row + 2, col + 1] = 1.0
            dU[row + 2, col + 2] = MU
            # -f_y + μ f_z
            dU[row + 3, col + 1] = -1.0
            dU[row + 3, col + 2] = MU
            #  f_z
            dU[row + 4, col + 2] = 1.0

        # CoM height → state index 2  (p_z in the tangent space is dx[2])
        dX[NC_SRB - 1, 2] = 1.0

        return dX, dU

    def hessprod_xx(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))
