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
# Cost
# ---------------------------------------------------------------------------

w_p = 50.0  # CoM position tracking
w_v = 10.0  # CoM velocity tracking
w_o = 100.0  # Orientation tracking
w_w = 1.0  # Angular velocity regularisation
w_f = 1e-4  # Contact force regularisation
w_pf = 10.0  # Swing foot position tracking
w_vf = 1.0  # Swing foot velocity tracking (control regularisation)

Q_p = w_p * np.eye(3)
Q_v = w_v * np.eye(3)
Q_w = w_w * np.eye(3)
Q_pf = w_pf * np.eye(3)
Q_f = w_f * np.eye(3)
Q_vf = w_vf * np.eye(3)


class SRBCost(costs.StageCost):
    """Tracking + regularisation cost for the SRBD model.

    Parameters
    ----------
    com_ref : (3,)  desired CoM position
    xb_ref  : (7,)  desired base pose [p(3), q(4)]
    p_ref   : dict  foot → [pos(3), is_swing(bool), vel_ref(3)]
                    vel_ref used as swing foot velocity reference (control reg.)
    terminal : bool
    """

    def __init__(self, com_ref, xb_ref, p_ref, terminal=False):
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

        for i, foot in enumerate(FEET):
            is_swing = self.p_ref[foot][1]
            if is_swing:
                # Swing: track reference foot position (state)
                pf_err = _pf(x, i) - self.p_ref[foot][0]
                cost += pf_err @ Q_pf @ pf_err
            else:
                if not self.terminal:
                    # Stance: regularise contact force
                    f_i = u[_O_F + 3 * i : _O_F + 3 * i + 3]
                    cost += f_i @ Q_f @ f_i

        return cost

    def gradient(self, x, u):
        ndq = config.SRBDParameters.ndq
        nu = config.SRBDParameters.nu
        dx = np.zeros(ndq)
        du = np.zeros(nu)

        # CoM position
        dx[_O_P_COM : _O_P_COM + 3] += 2.0 * Q_p @ (_p(x) - self.com_ref)

        # Orientation
        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_err = utils.quat_mul(q_ref_inv, _q(x))
        dx[_O_THETA : _O_THETA + 3] += 2.0 * w_o * q_err[:3]

        # Angular velocity
        dx[_O_OMEGA : _O_OMEGA + 3] += 2.0 * Q_w @ _w(x)

        for i, foot in enumerate(FEET):
            is_swing = self.p_ref[foot][1]
            cs = _O_P_FEET + 3 * i
            cv = _O_V + 3 * i
            cf = _O_F + 3 * i
            if is_swing:
                dx[cs : cs + 3] += (
                    2.0 * Q_pf @ (_pf(x, i) - self.p_ref[foot][0])
                )
            else:
                if not self.terminal:
                    du[cf : cf + 3] += 2.0 * Q_f @ u[cf : cf + 3]

        return dx, du

    def hessian_xx(self, x, u):
        ndq = config.SRBDParameters.ndq
        Hxx = np.zeros((ndq, ndq))
        Hxx[_O_P_COM : _O_P_COM + 3, _O_P_COM : _O_P_COM + 3] += 2.0 * Q_p
        Hxx[_O_THETA : _O_THETA + 3, _O_THETA : _O_THETA + 3] += (
            2.0 * w_o * np.eye(3)
        )
        Hxx[_O_OMEGA : _O_OMEGA + 3, _O_OMEGA : _O_OMEGA + 3] += 2.0 * Q_w
        for i, foot in enumerate(FEET):
            if self.p_ref[foot][1]:
                cs = _O_P_FEET + 3 * i
                Hxx[cs : cs + 3, cs : cs + 3] += 2.0 * Q_pf
        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessian_uu(self, x, u):
        nu = config.SRBDParameters.nu
        Huu = np.zeros((nu, nu))
        if not self.terminal:
            for i, foot in enumerate(FEET):
                is_swing = self.p_ref[foot][1]
                if not is_swing:
                    cf = _O_F + 3 * i
                    Huu[cf : cf + 3, cf : cf + 3] += 2.0 * Q_f
        return Huu


# ---------------------------------------------------------------------------
# Constraints
#
# Per foot (8 entries):
#   [0:5]  friction pyramid / force-zero  (same structure as before)
#   [5:8]  foot velocity vᵢ  (control): = 0 for stance, free for swing
# Plus 1 CoM height.
# Total: 8×4 + 1 = 33
# ---------------------------------------------------------------------------

MU = 0.6
F_Z_MIN = 1.0
F_Z_MAX = 250.0
Z_MIN = 0.20

NC_PER_FOOT = 8
NC_SRB = NC_PER_FOOT * NF + 1


class SRBConstraints(constraints.StageConstraints):
    """Friction pyramid, foot zero-velocity (stance), zero-force (swing),
    and CoM height constraints.

    Update each MPC step via ``update_contact_schedule``.
    """

    def __init__(self, p_ref):
        super().__init__()
        self.p_ref = p_ref

    def update_contact_schedule(self, p_ref):
        self.p_ref = p_ref

    def _is_swing(self, foot):
        return self.p_ref[foot][1]

    def output_size(self):
        return NC_SRB

    def lower_bound(self):
        lb = []
        for foot in FEET:
            if self._is_swing(foot):
                lb += [0.0, 0.0, 0.0, 0.0, 0.0]  # force = 0
                lb += [-np.inf, -np.inf, -np.inf]  # velocity free
            else:
                lb += [0.0, 0.0, 0.0, 0.0, F_Z_MIN]  # friction pyramid
                lb += [0.0, 0.0, 0.0]  # velocity = 0
        lb.append(Z_MIN)
        return np.array(lb)

    def upper_bound(self):
        ub = []
        for foot in FEET:
            if self._is_swing(foot):
                ub += [0.0, 0.0, 0.0, 0.0, 0.0]  # force = 0
                ub += [np.inf, np.inf, np.inf]  # velocity free
            else:
                ub += [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    F_Z_MAX,
                ]  # friction pyramid
                ub += [0.0, 0.0, 0.0]  # velocity = 0
        ub.append(np.inf)
        return np.array(ub)

    def evaluate(self, x, u):
        c = []
        for i, foot in enumerate(FEET):
            fx, fy, fz = (
                u[_O_F + 3 * i],
                u[_O_F + 3 * i + 1],
                u[_O_F + 3 * i + 2],
            )

            # Friction pyramid (5)
            c.append(fx + MU * fz)
            c.append(-fx + MU * fz)
            c.append(fy + MU * fz)
            c.append(-fy + MU * fz)
            c.append(fz)

            # Foot velocity (3) — control
            c.extend(u[_O_V + 3 * i : _O_V + 3 * i + 3].tolist())

        c.append(_p(x)[2])
        return np.array(c)

    def gradient(self, x, u):
        ndq = config.SRBDParameters.ndq
        nu = config.SRBDParameters.nu
        dX = np.zeros((NC_SRB, ndq))
        dU = np.zeros((NC_SRB, nu))

        for i, foot in enumerate(FEET):
            row = NC_PER_FOOT * i
            cf = _O_F + 3 * i
            cv = _O_V + 3 * i

            # Friction pyramid (rows 0–4)
            dU[row, cf] = 1.0
            dU[row, cf + 2] = MU
            dU[row + 1, cf] = -1.0
            dU[row + 1, cf + 2] = MU
            dU[row + 2, cf + 1] = 1.0
            dU[row + 2, cf + 2] = MU
            dU[row + 3, cf + 1] = -1.0
            dU[row + 3, cf + 2] = MU
            dU[row + 4, cf + 2] = 1.0

            # Foot velocity (rows 5–7) — control
            dU[row + 5 : row + 8, cv : cv + 3] = np.eye(3)

        # CoM height
        dX[NC_SRB - 1, _O_P_COM + 2] = 1.0

        return dX, dU

    def hessprod_xx(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.ndq))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.ndq, config.SRBDParameters.nu))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((config.SRBDParameters.nu, config.SRBDParameters.nu))
