import ssqpy
import ssqpy.model.dynamics as dynamics
import ssqpy.model.constraints as constraints
import ssqpy.model.costs as costs

import numpy as np

import models.quadruped_hlplanner.config as config
import models.utils as utils

# ---------------------------------------------------------------------------
# State / tangent / control dimensions
#
#   State  x   = [ p_com(3), q(4), v_com(3), ω(3), p_FL(3)..p_RR(3) ]
#   nx  = 13 + 12 = 25
#
#   Tangent dx  = [ δp(3), δθ(3), δv(3), δω(3), δp_FL(3)..δp_RR(3) ]
#   ndx = 12 + 12 = 24
#
#   Control u   = [ f_FL(3)..f_RR(3), v_FL(3)..v_RR(3) ]
#   nu  = 12 + 12 = 24
# ---------------------------------------------------------------------------
FEET = list(config.FRAME_NAMES.keys())  # [FL, FR, RL, RR]
NF = len(FEET)  # 4

# Tangent-space offsets
_O_P_COM = 0
_O_THETA = 3
_O_V_COM = 6
_O_OMEGA = 9
_O_P_FEET = 12  # δp_footᵢ at [12 + 3i : 15 + 3i]

# Control offsets
_O_F = 0  # f_footᵢ at [3i     : 3i+3]
_O_V = 12  # v_footᵢ at [12+3i  : 15+3i]


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


def _pf(x, i):
    return x[13 + 3 * i : 16 + 3 * i]


def _pack(p, q, v, w, p_feet):
    return np.hstack((p, q / np.linalg.norm(q), v, w, *p_feet))


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
class SRBDynamics(dynamics.StageDynamics):
    """Single Rigid Body Dynamics with foot positions as states.

    State   x = [p_com(3), q(4), v_com(3), ω(3), p_feet(12)]
    Control u = [f_feet(12), v_feet(12)]

    Semi-implicit Euler:
        v_com⁺   = v_com  + dt·(Σfᵢ/m + g)
        ω⁺       = ω      + dt·Iw⁻¹·(Σ rᵢ×fᵢ − ω×Iw·ω)
        p_com⁺   = p_com  + dt·v_com⁺
        q⁺       = quat_integrate(q, dt·ω⁺)
        p_footᵢ⁺ = p_footᵢ + dt·v_footᵢ

    where rᵢ = p_footᵢ − p_com from state (not a reference).

    Constraints enforce:
        stance: fᵢ in friction cone,  v_footᵢ = 0
        swing:  fᵢ = 0,               v_footᵢ free
    """

    def __init__(self, dt, mass, inertia):
        super().__init__()
        self.dt = dt
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)

    def output_size(self):
        return config.SRBDParameters.nq

    def state_size(self):
        return config.SRBDParameters.nq

    def state_diff_size(self):
        return config.SRBDParameters.ndq

    def control_size(self):
        return config.SRBDParameters.nu

    def state_right_plus(self, x, dx):
        pn = _p(x) + dx[_O_P_COM : _O_P_COM + 3]
        qn = utils.quat_integrate(_q(x), dx[_O_THETA : _O_THETA + 3])
        vn = _v(x) + dx[_O_V_COM : _O_V_COM + 3]
        wn = _w(x) + dx[_O_OMEGA : _O_OMEGA + 3]
        pfn = [
            _pf(x, i) + dx[_O_P_FEET + 3 * i : _O_P_FEET + 3 * i + 3]
            for i in range(NF)
        ]
        return _pack(pn, qn, vn, wn, pfn)

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
        dpf = np.hstack([_pf(x, i) - _pf(y, i) for i in range(NF)])
        return np.hstack((dp, dtheta, dv, dw, dpf))

    def evaluate(self, x, u):
        p, q, v, w = _p(x), _q(x), _v(x), _w(x)
        R = utils.quat_to_rotation_matrix(q)
        dt = self.dt

        I_w = R @ self.inertia @ R.T
        I_w_inv = R @ self.inertia_inv @ R.T

        f_total = np.zeros(3)
        tau_total = np.zeros(3)
        for i in range(NF):
            f_i = u[_O_F + 3 * i : _O_F + 3 * i + 3]
            r_i = _pf(x, i) - p
            f_total += f_i
            tau_total += np.cross(r_i, f_i)

        v_new = v + dt * (f_total / self.mass + config.SRBDParameters.GRAVITY)
        w_new = w + dt * (I_w_inv @ (tau_total - np.cross(w, I_w @ w)))
        p_new = p + dt * v_new
        q_new = utils.quat_integrate(q, dt * w_new)

        pf_new = [
            _pf(x, i) + dt * u[_O_V + 3 * i : _O_V + 3 * i + 3]
            for i in range(NF)
        ]

        return _pack(p_new, q_new, v_new, w_new, pf_new)

    def gradient(self, x, u):
        """Analytical Jacobians for the extended semi-implicit Euler SRBD.

        Returns
        -------
        dX : (ndq, ndq)
        dU : (ndq, nu)

        Tangent layout : [δp(3) | δθ(3) | δv(3) | δω(3) | δp_feet(12)]
        Control layout : [f_feet(12) | v_feet(12)]

        Foot-position coupling into CoM dynamics
        -----------------------------------------
        rᵢ = p_footᵢ − p_com  lives in state, so:

            ∂(rᵢ×fᵢ)/∂p_footᵢ = −skew(fᵢ)   [∂rᵢ/∂p_footᵢ = +I]
            ∂(rᵢ×fᵢ)/∂p_com   = +skew(fᵢ)   [∂rᵢ/∂p_com   = −I]

        giving the cross-coupling blocks:
            ∂ω⁺/∂p_footᵢ = −dt·Iw⁻¹·skew(fᵢ)
            ∂θ⁺/∂p_footᵢ =  dt·Jr·∂ω⁺/∂p_footᵢ

        Foot position blocks (decoupled across feet, decoupled from CoM):
            dX: ∂p_footᵢ⁺/∂p_footᵢ = I
            dU: ∂p_footᵢ⁺/∂v_footᵢ = dt·I
        """
        p, q, v, w = _p(x), _q(x), _v(x), _w(x)
        R = utils.quat_to_rotation_matrix(q)
        dt = self.dt

        I_w = R @ self.inertia @ R.T
        I_w_inv = R @ self.inertia_inv @ R.T

        ndq = config.SRBDParameters.ndq
        nu = config.SRBDParameters.nu

        # ---- replicate forward-pass quantities needed for gradient ----------
        r = [_pf(x, i) - p for i in range(NF)]

        tau = sum(
            np.cross(r[i], u[_O_F + 3 * i : _O_F + 3 * i + 3])
            for i in range(NF)
        )
        h = I_w @ w
        w_new = w + dt * (I_w_inv @ (tau - np.cross(w, h)))

        phi = dt * w_new
        angle = np.linalg.norm(phi)

        if angle < 1e-10:
            R_dq = np.eye(3)
            J_r = np.eye(3) - 0.5 * utils.skew(phi)
        else:
            axis = phi / angle
            K = utils.skew(axis)
            R_dq = (
                np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
            )
            S_phi = utils.skew(phi)
            J_r = (
                np.eye(3)
                - (1.0 - np.cos(angle)) / angle**2 * S_phi
                + (angle - np.sin(angle)) / angle**3 * (S_phi @ S_phi)
            )

        # ---- CoM sub-block gradients ----------------------------------------

        S_w = utils.skew(w)
        S_h = utils.skew(h)

        # ∂ω⁺/∂ω
        A_ww = np.eye(3) + dt * (I_w_inv @ (S_h - S_w @ I_w))

        # ∂ω⁺/∂θ
        dgyro_dtheta = (
            np.outer(h, w) - np.dot(w, h) * np.eye(3) - S_w @ I_w @ S_w
        )
        A_wtheta = dt * (I_w_inv @ dgyro_dtheta)

        # ∂ω⁺/∂p_com = dt·Iw⁻¹·Σ skew(fᵢ)
        dtau_dp_com = sum(
            utils.skew(u[_O_F + 3 * i : _O_F + 3 * i + 3]) for i in range(NF)
        )
        A_wp_com = dt * (I_w_inv @ dtau_dp_com)

        # ∂ω⁺/∂p_footᵢ = −dt·Iw⁻¹·skew(fᵢ)
        A_wp_feet = [
            -dt * (I_w_inv @ utils.skew(u[_O_F + 3 * i : _O_F + 3 * i + 3]))
            for i in range(NF)
        ]

        # ---- assemble dX ---------------------------------------------------
        dX = np.zeros((ndq, ndq))

        # δp_com⁺
        dX[_O_P_COM : _O_P_COM + 3, _O_P_COM : _O_P_COM + 3] = np.eye(3)
        dX[_O_P_COM : _O_P_COM + 3, _O_V_COM : _O_V_COM + 3] = dt * np.eye(3)

        # δθ⁺
        dX[_O_THETA : _O_THETA + 3, _O_P_COM : _O_P_COM + 3] = dt * (
            J_r @ A_wp_com
        )
        dX[_O_THETA : _O_THETA + 3, _O_THETA : _O_THETA + 3] = R_dq.T + dt * (
            J_r @ A_wtheta
        )
        dX[_O_THETA : _O_THETA + 3, _O_OMEGA : _O_OMEGA + 3] = dt * (
            J_r @ A_ww
        )
        for i in range(NF):
            cs = _O_P_FEET + 3 * i
            dX[_O_THETA : _O_THETA + 3, cs : cs + 3] = dt * (
                J_r @ A_wp_feet[i]
            )

        # δv_com⁺
        dX[_O_V_COM : _O_V_COM + 3, _O_V_COM : _O_V_COM + 3] = np.eye(3)

        # δω⁺
        dX[_O_OMEGA : _O_OMEGA + 3, _O_P_COM : _O_P_COM + 3] = A_wp_com
        dX[_O_OMEGA : _O_OMEGA + 3, _O_THETA : _O_THETA + 3] = A_wtheta
        dX[_O_OMEGA : _O_OMEGA + 3, _O_OMEGA : _O_OMEGA + 3] = A_ww
        for i in range(NF):
            cs = _O_P_FEET + 3 * i
            dX[_O_OMEGA : _O_OMEGA + 3, cs : cs + 3] = A_wp_feet[i]

        # δp_footᵢ⁺  (identity — foot velocity is control, not state)
        for i in range(NF):
            cs = _O_P_FEET + 3 * i
            dX[cs : cs + 3, cs : cs + 3] = np.eye(3)

        # ---- assemble dU ---------------------------------------------------
        dU = np.zeros((ndq, nu))

        for i in range(NF):
            cf = _O_F + 3 * i
            cv = _O_V + 3 * i
            S_ri = utils.skew(r[i])

            # ∂v_com⁺/∂fᵢ = dt/m · I
            dU[_O_V_COM : _O_V_COM + 3, cf : cf + 3] = (
                dt / self.mass
            ) * np.eye(3)

            # ∂p_com⁺/∂fᵢ = dt²/m · I
            dU[_O_P_COM : _O_P_COM + 3, cf : cf + 3] = (
                dt**2 / self.mass
            ) * np.eye(3)

            # ∂ω⁺/∂fᵢ = dt·Iw⁻¹·skew(rᵢ)
            dU_wi = dt * (I_w_inv @ S_ri)
            dU[_O_OMEGA : _O_OMEGA + 3, cf : cf + 3] = dU_wi

            # ∂θ⁺/∂fᵢ = dt·Jr·∂ω⁺/∂fᵢ
            dU[_O_THETA : _O_THETA + 3, cf : cf + 3] = dt * (J_r @ dU_wi)

            # ∂p_footᵢ⁺/∂v_footᵢ = dt·I
            dU[_O_P_FEET + 3 * i : _O_P_FEET + 3 * i + 3, cv : cv + 3] = (
                dt * np.eye(3)
            )

        return dX, dU

    def hessprod_xx(self, x, u, lam):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessprod_xu(self, x, u, lam):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessprod_uu(self, x, u, lam):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


# CoM Position Tracking Cost
class CoMTrackingCost(costs.StageCost):
    def __init__(self, com_ref, weight):
        super().__init__()
        self.com_ref = com_ref
        self.Q_p = weight * np.eye(3)

    def evaluate(self, x, u):
        p_err = _p(x) - self.com_ref
        return p_err @ self.Q_p @ p_err

    def gradient(self, x, u):
        dx = np.zeros(config.SRBDParameters.ndq)
        du = np.zeros(config.SRBDParameters.nu)
        dx[_O_P_COM : _O_P_COM + 3] = 2.0 * self.Q_p @ (_p(x) - self.com_ref)
        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))
        Hxx[_O_P_COM : _O_P_COM + 3, _O_P_COM : _O_P_COM + 3] = 2.0 * self.Q_p
        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# Orientation Tracking Cost
class OrientationTrackingCost(costs.StageCost):
    def __init__(self, xb_ref, weight):
        super().__init__()
        self.xb_ref = xb_ref
        self.w_o = weight

    def evaluate(self, x, u):
        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_err = utils.quat_mul(q_ref_inv, _q(x))
        return self.w_o * np.dot(q_err[:3], q_err[:3])

    def gradient(self, x, u):
        dx = np.zeros(config.SRBDParameters.ndq)
        du = np.zeros(config.SRBDParameters.nu)
        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_err = utils.quat_mul(q_ref_inv, _q(x))
        dx[_O_THETA : _O_THETA + 3] = 2.0 * self.w_o * q_err[:3]
        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))
        Hxx[_O_THETA : _O_THETA + 3, _O_THETA : _O_THETA + 3] = (
            2.0 * self.w_o * np.eye(3)
        )
        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# CoM Velocity Regularization
class VelocityCost(costs.StageCost):
    def __init__(self, weight):
        super().__init__()
        self.Q_v = weight * np.eye(3)

    def evaluate(self, x, u):
        return _v(x) @ self.Q_v @ _v(x)

    def gradient(self, x, u):
        dx = np.zeros(config.SRBDParameters.ndq)
        du = np.zeros(config.SRBDParameters.nu)
        dx[_O_V_COM : _O_V_COM + 3] = 2.0 * self.Q_v @ _v(x)
        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))
        Hxx[_O_V_COM : _O_V_COM + 3, _O_V_COM : _O_V_COM + 3] = 2.0 * self.Q_v
        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# CoM Angular Velocity Regularization
class AngularVelocityCost(costs.StageCost):
    def __init__(self, weight):
        super().__init__()
        self.Q_w = weight * np.eye(3)

    def evaluate(self, x, u):
        return _w(x) @ self.Q_w @ _w(x)

    def gradient(self, x, u):
        dx = np.zeros(config.SRBDParameters.ndq)
        du = np.zeros(config.SRBDParameters.nu)
        dx[_O_OMEGA : _O_OMEGA + 3] = 2.0 * self.Q_w @ _w(x)
        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))
        Hxx[_O_OMEGA : _O_OMEGA + 3, _O_OMEGA : _O_OMEGA + 3] = 2.0 * self.Q_w
        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# Foot Position Tracking Cost
class FootTrackingCost(costs.StageCost):
    def __init__(self, p_ref, weight):
        super().__init__()
        self.p_ref = p_ref
        self.Q_pf = weight * np.eye(3)

    def evaluate(self, x, u):
        cost = 0.0
        for i, foot in enumerate(FEET):
            pf_err = _pf(x, i) - self.p_ref[foot][0]
            cost += pf_err @ self.Q_pf @ pf_err
        return cost

    def gradient(self, x, u):
        dx = np.zeros(config.SRBDParameters.ndq)
        du = np.zeros(config.SRBDParameters.nu)
        for i, foot in enumerate(FEET):
            cs = _O_P_FEET + 3 * i
            dx[cs : cs + 3] = (
                2.0 * self.Q_pf @ (_pf(x, i) - self.p_ref[foot][0])
            )
        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))
        for i, foot in enumerate(FEET):
            cs = _O_P_FEET + 3 * i
            Hxx[cs : cs + 3, cs : cs + 3] = 2.0 * self.Q_pf
        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# Control Regularization (Forces & Velocities)
class ControlRegularizationCost(costs.StageCost):
    def __init__(self, p_ref, weight_forces, weight_fvel, terminal=False):
        super().__init__()
        self.p_ref = p_ref
        self.Q_f = weight_forces * np.eye(3)
        self.Q_vf = weight_fvel * np.eye(3)
        self.terminal = terminal

    def evaluate(self, x, u):
        if self.terminal:
            return 0.0
        cost = 0.0
        for i, foot in enumerate(FEET):
            f_i = u[_O_F + 3 * i : _O_F + 3 * i + 3]
            cost += f_i @ self.Q_f @ f_i
            fv_i = u[_O_V + 3 * i : _O_V + 3 * i + 3]
            cost += fv_i @ self.Q_vf @ fv_i
        return cost

    def gradient(self, x, u):
        dx = np.zeros(config.SRBDParameters.ndq)
        du = np.zeros(config.SRBDParameters.nu)
        if self.terminal:
            return dx, du

        for i, foot in enumerate(FEET):
            cf = _O_F + 3 * i
            du[cf : cf + 3] = 2.0 * self.Q_f @ u[cf : cf + 3]
            cvf = _O_V + 3 * i
            du[cvf : cvf + 3] = 2.0 * self.Q_vf @ u[cvf : cvf + 3]
        return dx, du

    def hessian_xx(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        Huu = np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))
        if self.terminal:
            return Huu
        for i, foot in enumerate(FEET):
            cf = _O_F + 3 * i
            Huu[cf : cf + 3, cf : cf + 3] = 2.0 * self.Q_f
            cvf = _O_V + 3 * i
            Huu[cvf : cvf + 3, cvf : cvf + 3] = 2.0 * self.Q_vf
        return Huu


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


# 1. CoM Height Constraint
class CoMHeightConstraint(constraints.StageConstraints):
    def __init__(self, z_min):
        super().__init__()
        self.z_min = z_min

    def output_size(self):
        return 1

    def lower_bound(self):
        return np.array([self.z_min])

    def upper_bound(self):
        return np.array([np.inf])

    def evaluate(self, x, u):
        return np.array([_p(x)[2]])

    def gradient(self, x, u):
        dX = np.zeros((1, config.SRBDParameters.ndq))
        dU = np.zeros((1, config.SRBDParameters.nu))
        dX[0, _O_P_COM + 2] = 1.0
        return dX, dU

    def hessprod_xx(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# 2. Friction Pyramid & Stance Force Constraint
class FrictionAndForcesConstraint(constraints.StageConstraints):
    def __init__(self, p_ref, foot_idx, mu=0.6, fz_min=1.0, fz_max=250.0):
        super().__init__()
        self.p_ref = p_ref
        self.foot_idx = foot_idx
        self.foot_name = FEET[foot_idx]
        self.mu = mu
        self.fz_min = fz_min
        self.fz_max = fz_max

    def _is_swing(self):
        return self.p_ref[self.foot_name][1]

    def output_size(self):
        return 5

    def lower_bound(self):
        lb = np.zeros(5)
        if not self._is_swing():
            lb[4] = self.fz_min
        return lb

    def upper_bound(self):
        if self._is_swing():
            return np.zeros(5)
        else:
            ub = np.full(5, np.inf)
            ub[4] = self.fz_max
            return ub

    def evaluate(self, x, u):
        cf = _O_F + 3 * self.foot_idx
        fx, fy, fz = u[cf], u[cf + 1], u[cf + 2]

        c = np.zeros(5)
        c[0] = fx + self.mu * fz
        c[1] = -fx + self.mu * fz
        c[2] = fy + self.mu * fz
        c[3] = -fy + self.mu * fz
        c[4] = fz
        return c

    def gradient(self, x, u):
        dX = np.zeros((5, config.SRBDParameters.ndq))
        dU = np.zeros((5, config.SRBDParameters.nu))

        cf = _O_F + 3 * self.foot_idx
        dU[0, cf] = 1.0
        dU[0, cf + 2] = self.mu
        dU[1, cf] = -1.0
        dU[1, cf + 2] = self.mu
        dU[2, cf + 1] = 1.0
        dU[2, cf + 2] = self.mu
        dU[3, cf + 1] = -1.0
        dU[3, cf + 2] = self.mu
        dU[4, cf + 2] = 1.0

        return dX, dU

    def hessprod_xx(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))


# 3. Foot Velocity Constraint
class FootVelocityConstraint(constraints.StageConstraints):
    def __init__(self, foot_idx):
        super().__init__()
        self.foot_idx = foot_idx

    def output_size(self):
        return 3

    def lower_bound(self):
        return np.zeros(3)

    def upper_bound(self):
        return np.zeros(3)

    def evaluate(self, x, u):
        cv = _O_V + 3 * self.foot_idx
        return u[cv : cv + 3]

    def gradient(self, x, u):
        dX = np.zeros((3, config.SRBDParameters.ndq))
        dU = np.zeros((3, config.SRBDParameters.nu))

        cv = _O_V + 3 * self.foot_idx
        dU[0:3, cv : cv + 3] = np.eye(3)
        return dX, dU

    def hessprod_xx(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))
