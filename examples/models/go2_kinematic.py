import ssqpy
import ssqpy.model.dynamics as dynamics
import ssqpy.model.constraints as constraints
import ssqpy.model.costs as costs

import pinocchio as pin
import numpy as np

import model_utils

# Pinocchio model creation
# ------------------------
pin_model, pin_data = None, None


def load_model(xml_path):
    global pin_model, pin_data
    pin_model, _, _ = pin.buildModelsFromMJCF(xml_path)
    pin_data = pin_model.createData()


# Parameters
# ----------
nj = 12  # number of actuated joints
nq = 7 + nj  # configuration size
ndq = 6 + nj  # linearized configuration size

frame_names = {
    "FL": "FL_foot",
    "FR": "FR_foot",
    "RL": "RL_foot",
    "RR": "RR_foot",
}

### High-level planner ###
### ================== ###

init_pos = np.array([0.0, 0.0, 0.389])
init_joint_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

init_config = np.zeros(nq)
init_config[:3] = init_pos
init_config[6] = 1.0
init_config[7:] = init_joint_config

# Contact Scheduling parameters
# -----------------------------
T = 20
beta = 0.5

T_swing = int(beta * T)
T_stance = T - T_swing

foot_phases = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}
hip_to_foot_offsets = {
    "FL": np.array((0.6, 0.14, -0.36892682)),
    "FR": np.array((0.6, -0.14, -0.36892682)),
    "RL": np.array((0.1, 0.14, -0.36892682)),
    "RR": np.array((0.1, -0.14, -0.36892682)),
}

# Reference Generation
# --------------------
Kv = 0.1
Kw = 0.1
h_sw = 0.1


class ReferenceState:
    initialized = False

    liftoff = {}
    touchdown = {}


def generate_references_MPC(
    v_cmd, config_0, vel_0, k0, H, dt, ssqp_model=None
):
    translations = {}

    if ssqp_model is None:
        pin.forwardKinematics(pin_model, pin_data, config_0)
        pin.updateFramePlacements(pin_model, pin_data)

        for foot in foot_phases:
            translations[foot] = pin_data.oMf[
                pin_model.getFrameId(frame_names[foot])
            ].translation
    else:
        for foot in foot_phases:
            translations[foot], _ = ssqp_model.forwardKinematics(
                config_0, frame_names[foot]
            )

    if not ReferenceState.initialized:
        ReferenceState.liftoff = translations.copy()
        ReferenceState.touchdown = {
            foot: ReferenceState.liftoff[foot].copy() for foot in foot_phases
        }

        ReferenceState.initialized = True

    # Local copies
    liftoff = {f: v.copy() for f, v in ReferenceState.liftoff.items()}
    touchdown = {f: v.copy() for f, v in ReferenceState.touchdown.items()}

    # Actual initial velocity for x/y and yaw
    vel_0_3 = np.hstack((vel_0[:2], 0.0))
    wz_0 = vel_0[2]

    # Commanded velocity for x/y and yaw
    Rz_0 = model_utils.yaw_to_Rz(model_utils.quat_to_yaw(config_0[3:7]))
    v_cmd_3 = Rz_0 @ np.hstack((v_cmd[:2], 0.0))
    wz_cmd = v_cmd[2]

    # CoM reference
    # -------------
    if ssqp_model is None:
        com_0 = pin.centerOfMass(pin_model, pin_data, config_0)
    else:
        com_0 = ssqp_model.centerOfMass(config_0)

    com_ref = [com_0]
    for _ in range(H):
        com_ref.append(com_ref[-1] + dt * v_cmd_3)

    # Base pose reference
    # -------------------
    xb_0 = config_0[:7]
    yaw = model_utils.quat_to_yaw(config_0[3:7])
    xb_ref = [xb_0]
    for _ in range(H):
        p_ref_k = xb_ref[-1][:3] + dt * v_cmd_3
        yaw += dt * wz_cmd
        xb_ref.append(
            np.hstack((p_ref_k, model_utils.euler_to_quat(0.0, 0.0, yaw)))
        )

    # Foot trajectory references
    # --------------------------
    p_ref = {
        foot: [
            [
                translations[foot].copy(),
                is_swinging(k0, foot_phases[foot])[0],
            ]
        ]
        for foot in foot_phases
    }

    for k in range(H):
        t = k0 + k

        yaw_k = model_utils.quat_to_yaw(xb_ref[k][3:7])
        Rz = model_utils.yaw_to_Rz(yaw_k)

        for foot in foot_phases:
            swing_at_t, swapped = is_swinging(t, foot_phases[foot])

            if not swing_at_t:
                if swapped:
                    # Foot just landed – freeze its position
                    if k == 0:
                        liftoff[foot] = translations[foot].copy()
                    else:
                        liftoff[foot] = touchdown[foot].copy()
                p_ref[foot].append([liftoff[foot].copy(), False])
                continue

            # Update predicted touchdown for this swing
            if swapped:
                touchdown[foot] = (
                    com_ref[k]
                    + Rz @ hip_to_foot_offsets[foot]
                    - T_stance * dt / 2 * vel_0_3
                    + Kv * (v_cmd_3 - vel_0_3)
                    + Kw
                    * (wz_0 - wz_cmd)
                    * Rz
                    @ np.array(
                        (
                            -hip_to_foot_offsets[foot][1],
                            hip_to_foot_offsets[foot][0],
                            0.0,
                        )
                    )
                )
                touchdown[foot][2] = -0.0002611

            # Normalized swing phase
            phi = (t / T + foot_phases[foot]) % 1
            phi_sw = phi / beta

            swing_alpha = 3 * phi_sw**2 - 2 * phi_sw**3

            p_ref_k = (1 - swing_alpha) * liftoff[
                foot
            ] + swing_alpha * touchdown[foot]
            p_ref_k[2] += 4 * h_sw * phi_sw * (1 - phi_sw)

            p_ref[foot].append([p_ref_k, True])

        if k == 0:
            ReferenceState.liftoff = {f: v.copy() for f, v in liftoff.items()}
            ReferenceState.touchdown = {
                f: v.copy() for f, v in touchdown.items()
            }

    return com_ref, xb_ref, p_ref


def generate_references(v_cmd, config_0, vel_0, k0, H, dt, ssqp_model=None):
    # Last liftoff location for each foot, along with its liftoff time
    translations = {}

    if ssqp_model is None:
        pin.forwardKinematics(pin_model, pin_data, config_0)
        pin.updateFramePlacements(pin_model, pin_data)

        for foot in foot_phases:
            translations[foot] = pin_data.oMf[
                pin_model.getFrameId(frame_names[foot])
            ].translation
    else:
        for foot in foot_phases:
            translations[foot], _ = ssqp_model.forwardKinematics(
                config_0, frame_names[foot]
            )

    liftoff = translations.copy()

    # Next touchdown location for each foot, along with its touchdown time
    touchdown = {foot: liftoff[foot].copy() for foot in foot_phases}

    # Actual initial velocity for x and y axes, and angular (yaw)
    vel_0_3 = np.hstack((vel_0[:2], 0.0))
    wz_0 = vel_0[2]

    # Commanded velocity for x and y axes, and angular (yaw)
    v_cmd_3 = np.hstack((v_cmd[:2], 0.0))
    wz_cmd = v_cmd[2]

    # Generate CoM reference from velocity command
    if ssqp_model is None:
        com_0 = pin.centerOfMass(pin_model, pin_data, config_0)
    else:
        com_0 = ssqp_model.centerOfMass(config_0)

    com_ref = [com_0]
    for _ in range(H):
        # Follow velocity command with CoM
        com_ref_k = com_ref[-1] + dt * v_cmd_3
        com_ref.append(com_ref_k)

    # Generate base position reference
    xb_0 = config_0[:7]
    yaw = model_utils.quat_to_yaw(config_0[3:7])

    xb_ref = [xb_0]
    for _ in range(H):
        # Follow velocity command with base position reference
        p_ref_k = xb_ref[-1][:3] + dt * v_cmd_3
        # Follow angular (yaw) command with rotation angle
        yaw += dt * wz_cmd
        xb_ref.append(
            np.hstack((p_ref_k, model_utils.euler_to_quat(0.0, 0.0, yaw)))
        )

    # Generate feet trajectory references

    # Get initial positions via forward kinematics
    # Also keep a flag noting if foot is in swing for each position
    p_ref = {
        foot: [
            [
                translations[foot].copy(),
                False,
            ]
        ]
        for foot in foot_phases
    }

    p_ref["FL"][0][1] = True
    p_ref["RR"][0][1] = True

    for k in range(H):
        t = k0 + k

        yaw_k = model_utils.quat_to_yaw(xb_ref[k][3:7])
        Rz = model_utils.yaw_to_Rz(yaw_k)

        for foot in foot_phases:
            # Use scheduling to check if foot is swinging at step t, and if it
            # just swapped swinging state
            swing_at_t, swapped = is_swinging(t, foot_phases[foot])

            if not swing_at_t:
                if swapped:
                    # If the foot just swapped from swinging to stance,keep the
                    # last touchdown position as the next stance position
                    liftoff[foot] = touchdown[foot].copy()

                # Keep the stance position as this step's reference
                p_ref[foot].append([liftoff[foot].copy(), False])
                continue

            # If the foot is swinging, update the touchdown position for step k
            touchdown[foot] = (
                com_ref[k]
                + Rz @ hip_to_foot_offsets[foot]
                + T_stance / 2 * vel_0_3
                + Kv * (vel_0_3 - v_cmd_3)
                + Kw
                * (wz_0 - wz_cmd)
                * Rz
                @ np.array(
                    (
                        -hip_to_foot_offsets[foot][1],
                        hip_to_foot_offsets[foot][0],
                        0.0,
                    )
                )
            )

            # Calculate the normalized swing phase
            phi = (t / T + foot_phases[foot]) % 1
            phi_sw = phi / beta

            # Get an interpolation alpha for the swing
            swing_alpha = 3 * phi_sw**2 - 2 * phi_sw**3

            # Interpolate between previous liftoff and caluclated touchdown
            # positions to get current position reference
            p_ref_k = (1 - swing_alpha) * liftoff[
                foot
            ] + swing_alpha * touchdown[foot]

            # For z reference, also add a height according to phase
            p_ref_k[2] += 4 * h_sw * phi_sw * (1 - phi_sw)

            p_ref[foot].append([p_ref_k, True])

    return com_ref, xb_ref, p_ref


# Whether a foot with initial phase phi_0 is swinging at time t
# Returns: (is_swinging, just_swapped)
# just_swapped is True if the foot's current swing state is different to that of
# the previous time step
def is_swinging(t, phi_0):
    prev_phi = ((t - 1) / T + phi_0) % 1
    phi = (t / T + phi_0) % 1

    at_prev = prev_phi < beta
    at_t = phi < beta

    return at_t, at_t != at_prev


### ssqpy Model ###
### =========== ###


# Dynamics for Kinematic model
# ----------------------------
class Kinematics(dynamics.StageDynamics):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def output_size(self):
        return nq

    def state_size(self):
        return nq

    def state_diff_size(self):
        return ndq

    def control_size(self):
        return ndq

    def state_right_plus(self, x, dx):
        return pin.integrate(pin_model, x, dx)

    def state_right_minus(self, x, y):
        return pin.difference(pin_model, y, x)

    def evaluate(self, x, u):
        return self.state_right_plus(x, u * self.dt)

    def gradient(self, x, u):
        dX = pin.dIntegrate(
            pin_model, x, u * self.dt, pin.ArgumentPosition.ARG0
        )
        dU = (
            pin.dIntegrate(
                pin_model, x, u * self.dt, pin.ArgumentPosition.ARG1
            )
            * self.dt
        )
        return dX, dU

    def hessprod_xx(self, x, u, lam):
        return np.zeros((ndq, ndq))

    def hessprod_xu(self, x, u, lam):
        return np.zeros((ndq, ndq))

    def hessprod_uu(self, x, u, lam):
        return np.zeros((ndq, ndq))


# Cost for Kinematic model
# ------------------------
q_c = 1.0  # CoM tracking weight
q_b = 1.0  # Body position tracking weight
q_o = 1.0  # Orientation tracking weight
q_st = 200.0  # Stance feet position tracking weight
q_sw = 10.0  # Swing feet position tracking weight
q_q = 1e-3  # Configuration regularization weight
q_v = 1e-3  # Velocity regularization weight
# TODO: validity check
q_sp = 50.0  # Support polygon (line segment) CoM weight
l_sigma = 0.5

Q_c = q_c * np.eye(3)
Q_b = q_b * np.eye(3)
Q_o = q_o * np.eye(3)
Q_st = q_st * np.eye(3)
Q_sw = q_sw * np.eye(3)
Q_q = q_q * np.eye(nq - 7)
Q_v = q_v * np.eye(nq - 7)
Q_sp = q_sp * np.eye(2)


class KinematicCost(costs.StageCost):
    def __init__(self, com_ref, xb_ref, p_ref, qj_nom, vj_nom, terminal=False):
        super().__init__()
        self.update_references(com_ref, xb_ref, p_ref, qj_nom, vj_nom)
        self.terminal = terminal

    def update_references(self, com_ref, xb_ref, p_ref, qj_nom, vj_nom):
        self.com_ref = com_ref
        self.xb_ref = xb_ref
        self.p_ref = p_ref
        self.qj_nom = qj_nom
        self.vj_nom = vj_nom

    def evaluate(self, x, u):
        cost = 0.0

        # CoM tracking cost
        com = pin.centerOfMass(pin_model, pin_data, x)
        com_error = com - self.com_ref
        cost += com_error.T @ Q_c @ com_error

        # Base Pose tracking cost
        xb_error = x[:3] - self.xb_ref[:3]
        cost += xb_error.T @ Q_b @ xb_error

        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_error = model_utils.quat_mul(q_ref_inv, x[3:7])
        cost += q_o * self.xb_ref[6] * np.dot(q_error[:3], q_error[:3])

        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        feet_pos = {
            foot: pin_data.oMf[
                pin_model.getFrameId(frame_names[foot])
            ].translation
            for foot in frame_names
        }

        for foot in frame_names:
            p_error = feet_pos[foot] - self.p_ref[foot][0]
            if self.p_ref[foot][1]:  # swing
                cost += p_error.T @ Q_sw @ p_error
            else:  # stance
                cost += p_error.T @ Q_st @ p_error

        # Joint configuration regularization
        qj_error = x[7:nq] - self.qj_nom
        cost += qj_error.T @ Q_q @ qj_error

        # Joint velocity regularization
        if not self.terminal:
            vj_error = u[6:ndq] - self.vj_nom
            cost += vj_error.T @ Q_v @ vj_error

        # Find feet that are standing
        ground_feet = []
        for foot in frame_names:
            if not self.p_ref[foot][1]:
                ground_feet.append(foot)

        # CoM support line segment cost
        if len(ground_feet) == 2:
            pf1_xy = feet_pos[ground_feet[0]][:2]
            pf2_xy = feet_pos[ground_feet[1]][:2]

            l = pf1_xy + l_sigma * (pf2_xy - pf1_xy)
            com_xy = com[:2]

            com_sp_error = com_xy - l

            cost += com_sp_error.T @ Q_sp @ com_sp_error

        return cost

    def gradient(self, x, u):
        dx = np.zeros(ndq)
        du = np.zeros(ndq)

        # CoM tracking
        com = pin.centerOfMass(pin_model, pin_data, x)
        com_error = com - self.com_ref

        com_jac = pin.jacobianCenterOfMass(pin_model, pin_data, x)
        dx += 2.0 * com_jac.T @ Q_c @ com_error

        # Base Pose tracking
        xb_error = x[:3] - self.xb_ref[:3]

        Jxb = np.zeros((3, ndq))
        Jxb[:, 0:3] = np.eye(3)
        dx += 2.0 * Jxb.T @ Q_b @ xb_error

        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_error = model_utils.quat_mul(q_ref_inv, x[3:7])

        dx[3:6] += q_o * self.xb_ref[6] * q_error[:3]

        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        # Foot tracking
        for foot in frame_names:
            frame_id = pin_model.getFrameId(frame_names[foot])

            p = pin_data.oMf[frame_id].translation
            p_error = p - self.p_ref[foot][0]

            J6 = pin.computeFrameJacobian(
                pin_model,
                pin_data,
                x,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            Jp = J6[:3, :]

            if self.p_ref[foot][1]:
                dx += 2.0 * Jp.T @ Q_sw @ p_error
            else:
                dx += 2.0 * Jp.T @ Q_st @ p_error

        # Joint configuration
        qj_error = x[7:nq] - self.qj_nom
        dx[6:ndq] += 2.0 * Q_q @ qj_error

        # Joint velocity
        if not self.terminal:
            vj_error = u[6:ndq] - self.vj_nom
            du[6:ndq] += 2.0 * Q_v @ vj_error

        # Find feet that are standing
        ground_feet = []
        for foot in frame_names:
            if not self.p_ref[foot][1]:
                ground_feet.append(foot)

        # CoM support line segment cost
        if len(ground_feet) == 2:
            frame1 = pin_model.getFrameId(frame_names[ground_feet[0]])
            frame2 = pin_model.getFrameId(frame_names[ground_feet[1]])

            pf1_xy = pin_data.oMf[frame1].translation[:2]
            pf2_xy = pin_data.oMf[frame2].translation[:2]

            l = pf1_xy + l_sigma * (pf2_xy - pf1_xy)

            com_xy = com[:2]
            com_sp_error = com_xy - l

            Jf1 = pin.computeFrameJacobian(
                pin_model,
                pin_data,
                x,
                frame1,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:2, :]

            Jf2 = pin.computeFrameJacobian(
                pin_model,
                pin_data,
                x,
                frame2,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:2, :]

            Jl = (1.0 - l_sigma) * Jf1 + l_sigma * Jf2

            Je = com_jac[:2, :] - Jl

            dx += 2.0 * Je.T @ Q_sp @ com_sp_error

        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((ndq, ndq))

        # CoM tracking
        Jcom = pin.jacobianCenterOfMass(pin_model, pin_data, x)
        Hxx += 2.0 * Jcom.T @ Q_c @ Jcom

        # Base pose tracking
        Jxb = np.zeros((3, ndq))
        Jxb[:, 0:3] = np.eye(3)
        Hxx += 2.0 * Jxb.T @ Q_b @ Jxb

        J = np.zeros((3, 4))
        J[:, :3] = np.eye(3)

        H_q = q_o * self.xb_ref[6] * (J.T @ J)
        Hxx[3:7, 3:7] += H_q

        # Foot tracking
        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        for foot in frame_names:
            frame_id = pin_model.getFrameId(frame_names[foot])

            J6 = pin.computeFrameJacobian(
                pin_model,
                pin_data,
                x,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            Jp = J6[:3, :]

            if self.p_ref[foot][1]:
                Hxx += 2.0 * Jp.T @ Q_sw @ Jp
            else:
                Hxx += 2.0 * Jp.T @ Q_st @ Jp

        # Joint posture regularization
        Hxx[6:ndq, 6:ndq] += 2.0 * Q_q

        # Find feet that are standing
        ground_feet = []
        for foot in frame_names:
            if not self.p_ref[foot][1]:
                ground_feet.append(foot)

        # CoM support line segment cost
        if len(ground_feet) == 2:
            frame1 = pin_model.getFrameId(frame_names[ground_feet[0]])
            frame2 = pin_model.getFrameId(frame_names[ground_feet[1]])

            Jf1 = pin.computeFrameJacobian(
                pin_model,
                pin_data,
                x,
                frame1,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:2, :]

            Jf2 = pin.computeFrameJacobian(
                pin_model,
                pin_data,
                x,
                frame2,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:2, :]

            Jl = (1.0 - l_sigma) * Jf1 + l_sigma * Jf2
            Je = Jcom[:2, :] - Jl

            Hxx += 2.0 * Je.T @ Q_sp @ Je

        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((ndq, ndq))

    def hessian_uu(self, x, u):
        Huu = np.zeros((ndq, ndq))

        if not self.terminal:
            Huu[6:, 6:] = 2.0 * Q_v

        return Huu


class LineSegmentCost(costs.StageCost):
    def __init__(self, model, p_ref, q_sp, l_sigma, terminal=False):
        super().__init__()
        self.model = model
        self.p_ref = p_ref
        self.Q_sp = q_sp * np.eye(2)
        self.l_sigma = l_sigma

        if terminal:
            self.u_size = 0
        else:
            self.u_size = ndq

    def update_reference(self, p_ref):
        self.p_ref = p_ref

    def evaluate(self, x, u):
        cost = 0.0

        # Find feet that are standing
        ground_feet = []
        for foot in frame_names:
            if not self.p_ref[foot][1]:
                ground_feet.append(frame_names[foot])

        com = self.model.centerOfMass(x)

        # CoM support line segment cost
        if len(ground_feet) == 2:
            pf1_xy = self.model.forwardKinematics(x, ground_feet[0])[0][:2]
            pf2_xy = self.model.forwardKinematics(x, ground_feet[1])[0][:2]

            l = pf1_xy + self.l_sigma * (pf2_xy - pf1_xy)
            com_xy = com[:2]

            com_sp_error = com_xy - l

            cost += com_sp_error.T @ self.Q_sp @ com_sp_error

        return cost

    def gradient(self, x, u):
        dx = np.zeros(ndq)
        du = np.zeros(self.u_size)

        # Find feet that are standing
        ground_feet = []
        for foot in frame_names:
            if not self.p_ref[foot][1]:
                ground_feet.append(frame_names[foot])

        # CoM support line segment cost
        if len(ground_feet) == 2:
            com = self.model.centerOfMass(x)
            com_jac = self.model.jacobianCenterOfMass(x)

            pf1_xy = self.model.forwardKinematics(x, ground_feet[0])[0][:2]
            pf2_xy = self.model.forwardKinematics(x, ground_feet[1])[0][:2]

            l = pf1_xy + self.l_sigma * (pf2_xy - pf1_xy)

            com_xy = com[:2]
            com_sp_error = com_xy - l

            Jf1 = self.model.frameJacobian(x, ground_feet[0])[:2, :]
            Jf2 = self.model.frameJacobian(x, ground_feet[1])[:2, :]

            Jl = (1.0 - self.l_sigma) * Jf1 + self.l_sigma * Jf2

            Je = com_jac[:2, :] - Jl

            dx += 2.0 * Je.T @ self.Q_sp @ com_sp_error

        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((ndq, ndq))

        # Find feet that are standing
        ground_feet = []
        for foot in frame_names:
            if not self.p_ref[foot][1]:
                ground_feet.append(frame_names[foot])

        # CoM support line segment cost
        if len(ground_feet) == 2:
            com_jac = self.model.jacobianCenterOfMass(x)

            Jf1 = self.model.frameJacobian(x, ground_feet[0])[:2, :]
            Jf2 = self.model.frameJacobian(x, ground_feet[1])[:2, :]

            Jl = (1.0 - self.l_sigma) * Jf1 + self.l_sigma * Jf2
            Je = com_jac[:2, :] - Jl

            Hxx += 2.0 * Je.T @ self.Q_sp @ Je

        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((ndq, self.u_size))

    def hessian_uu(self, x, u):
        return np.zeros((self.u_size, self.u_size))


# Constraints for Kinematic model
# -------------------------------
nc = nj + nj + 1

zb_min = 0.35
zb_max = 0.4


class KinematicConstraints(constraints.StageConstraints):
    def output_size(self):
        return nc

    def lower_bound(self):
        return np.hstack(
            (
                pin_model.lowerPositionLimit[7:],
                -pin_model.velocityLimit[6:],
                zb_min,
            )
        )

    def upper_bound(self):
        return np.hstack(
            (
                pin_model.upperPositionLimit[7:],
                pin_model.velocityLimit[6:],
                zb_max,
            )
        )

    def evaluate(self, x, u):
        return np.hstack((x[7:], u[6:], x[2]))

    def gradient(self, x, u):
        dX = np.zeros((nc, ndq))
        dU = np.zeros((nc, ndq))

        dX[:nj, 6:ndq] += np.eye(nj)

        dU[nj : 2 * nj, 6:ndq] += np.eye(nj)

        dX[2 * nj, 2] = 1.0

        return dX, dU

    def hessprod_xx(self, x, u, mu):
        return np.zeros((ndq, ndq))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((ndq, ndq))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((ndq, ndq))
