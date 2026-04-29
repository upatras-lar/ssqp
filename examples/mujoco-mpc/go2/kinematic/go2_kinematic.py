import ssqpy
import ssqpy.model.dynamics as dynamics
import ssqpy.model.constraints as constraints
import ssqpy.model.costs as costs

import numpy as np

import pinocchio as pin

from time import time

pin_model, _, _ = pin.buildModelsFromMJCF("../../../assets/mjcf/go2/go2.xml")
pin_data = pin_model.createData()


# Helpers
# -------
def yaw_to_Rz(yaw):
    return np.array(
        (
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def quat_to_yaw(quat):
    qx, qy, qz, qw = quat
    return np.atan2(2.0 * (qz * qw + qx * qy), -1.0 + 2.0 * (qw**2 + qx**2))


def yaw_tangent_jacobian(q, eps=1e-8):
    """
    Returns dyaw/d(delta_q) in tangent space.

    Output shape: (ndq,)
    """
    J = np.zeros(ndq)

    yaw0 = quat_to_yaw(q[3:7])

    for i in range(ndq):
        dq = np.zeros(ndq)
        dq[i] = eps

        q_plus = pin.integrate(pin_model, q, dq)

        yaw_plus = quat_to_yaw(q_plus[3:7])

        # wrap angle difference
        dyaw = np.arctan2(
            np.sin(yaw_plus - yaw0),
            np.cos(yaw_plus - yaw0),
        )

        J[i] = dyaw / eps

    return J


def quat_wxyz_to_xyzw(wxyz):
    xyzw = np.zeros(4)
    xyzw[:3] = wxyz[1:]
    xyzw[3] = wxyz[0]
    return xyzw


def quat_xyzw_to_wxyz(xyzw):
    wxyz = np.zeros(4)
    wxyz[0] = xyzw[3]
    wxyz[1:] = xyzw[:3]
    return wxyz


# Parameters
# ----------
dt = 0.01  # timestep
MPC_H = 500  # MPC optimization horizon

nj = 12  # number of actuated joints
nq = 7 + nj  # configuration size
ndq = 6 + nj  # linearized configuration size

frame_names = {
    "FL": "FL_foot",
    "FR": "FR_foot",
    "RL": "RL_foot",
    "RR": "RR_foot",
}

init_pos = np.array([0.0, 0.0, 0.389])
init_joint_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

init_config = np.zeros(nq)
init_config[:3] = init_pos
init_config[6] = 1.0
init_config[7:] = init_joint_config

# Contact Scheduler
# -----------------
# TODO: parameter values
T = 120
beta = 0.4

T_swing = int(beta * T)
T_stance = T - T_swing

foot_phases = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}
hip_to_foot_offsets = {
    "FL": np.array((0.1934, 0.142, -0.36892682)),
    "FR": np.array((0.1934, -0.142, -0.36892682)),
    "RL": np.array((-0.1934, 0.142, -0.36892682)),
    "RR": np.array((-0.1934, -0.142, -0.36892682)),
}


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


# Reference Generation
# --------------------
Kv = 1.0
Kw = 0.1
h_sw = 0.4

# Last liftoff location for each foot, along with its liftoff time
pin.forwardKinematics(pin_model, pin_data, init_config)
pin.updateFramePlacements(pin_model, pin_data)

liftoff = {
    foot: [
        pin_data.oMf[
            pin_model.getFrameId(frame_names[foot])
        ].translation.copy(),
        0.0,
    ]
    for foot in foot_phases
}

# Next touchdown location for each foot, along with its touchdown time
touchdown = {foot: liftoff[foot].copy() for foot in liftoff}


def generate_references(v_cmd, config_0, vel_0, t0):
    # Actual initial velocity for x and y axes, and angular (yaw)
    vel_0_3 = np.hstack((vel_0[:2], 0.0))
    wz_0 = vel_0[2]

    # Commanded velocity for x and y axes, and angular (yaw)
    v_cmd_3 = np.hstack((v_cmd[:2], 0.0))
    wz_cmd = v_cmd[2]

    # Generate CoM reference from velocity command
    com_0 = pin.centerOfMass(pin_model, pin_data, config_0)

    com_ref = [com_0]
    for _ in range(MPC_H):
        # Follow velocity command with CoM
        com_ref_k = com_ref[-1] + dt * v_cmd_3
        com_ref.append(com_ref_k)

    # Generate base position reference
    yaw_0 = quat_to_yaw(config_0[3:7])
    xb_0 = np.hstack((config_0[:3], 0.0, 0.0, yaw_0))

    xb_ref = [xb_0]
    for _ in range(MPC_H):
        # Follow velocity command with base position reference
        p_ref_k = xb_ref[-1][:3] + dt * v_cmd_3
        # Follow angular (yaw) command with rotation angle
        yaw_ref_k = xb_ref[-1][-1] + dt * wz_cmd
        xb_ref.append(np.hstack((p_ref_k, 0.0, 0.0, yaw_ref_k)))

    # Generate feet trajectory references
    pin.forwardKinematics(pin_model, pin_data, config_0)
    pin.updateFramePlacements(pin_model, pin_data)

    # Get initial positions via forward kinematics
    # Also keep a flag noting if foot is in swing for each position
    p_ref = {
        foot: [
            [
                pin_data.oMf[
                    pin_model.getFrameId(frame_names[foot])
                ].translation.copy(),
                0,
            ]
        ]
        for foot in foot_phases
    }

    p_ref["FL"][0][1] = 1
    p_ref["RR"][0][1] = 1

    for k in range(MPC_H):
        t = t0 + k

        Rz = yaw_to_Rz(xb_ref[k][-1])

        for foot in foot_phases:
            # Use scheduling to check if foot is swinging at step t, and if it
            # just swapped swinging state
            swing_at_t, swapped = is_swinging(t, foot_phases[foot])

            if not swing_at_t:
                if swapped:
                    # If the foot just swapped from swinging to stance,keep the
                    # last touchdown position as the next stance position
                    liftoff[foot][:] = touchdown[foot][:]

                # Keep the stance position as this step's reference
                ref = [liftoff[foot][0].copy(), 0]
                p_ref[foot].append(ref)
                continue

            if swapped:
                # If the foot just swapped from stance to swinging, calculate
                # the next touchdown time
                touchdown[foot][1] = t + T_swing - 1

            # If the foot is swinging, update the touchdown position for step k
            touchdown[foot][0] = (
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
            phi_sw = (t - liftoff[foot][1]) / (
                touchdown[foot][1] - liftoff[foot][1]
            )

            # Get an interpolation alpha for the swing
            swing_alpha = 3 * phi_sw**2 - 2 * phi_sw**3

            # Interpolate between previous liftoff and caluclated touchdown
            # positions to get current position reference
            p_ref_k = (1 - swing_alpha) * liftoff[foot][
                0
            ] + swing_alpha * touchdown[foot][0]

            # For z reference, also add a height according to phase
            p_ref_k[2] += 4 * h_sw * phi_sw * (1 - phi_sw)

            p_ref[foot].append([p_ref_k, 1])

    return com_ref, xb_ref, p_ref


# Dynamics for Kinematic model
# ----------------------------
class KinematicDynamics(dynamics.StageDynamics):
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
        return self.state_right_plus(x, u * dt)

    def gradient(self, x, u):
        dX = pin.dIntegrate(pin_model, x, u * dt, pin.ArgumentPosition.ARG0)
        dU = (
            pin.dIntegrate(pin_model, x, u * dt, pin.ArgumentPosition.ARG1)
            * dt
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
q_c = 1.0
q_b = 0.5
q_st = 1.0
q_sw = 0.01
q_q = 1e-2
q_v = 1e-3

Q_c = q_c * np.eye(3)
Q_b = q_b * np.eye(6)
Q_st = q_st * np.eye(3)
Q_sw = q_sw * np.eye(3)
Q_q = q_q * np.eye(nq - 7)
Q_v = q_v * np.eye(nq - 7)


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
        x_pose = np.hstack((x[:3], 0.0, 0.0, quat_to_yaw(x[3:7])))
        xb_error = x_pose - self.xb_ref
        cost += xb_error.T @ Q_b @ xb_error

        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        feet_pos = {
            foot: pin_data.oMf[
                pin_model.getFrameId(frame_names[foot])
            ].translation
            for foot in foot_phases
        }

        for foot in foot_phases:
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
        yaw = quat_to_yaw(x[3:7])
        x_pose = np.hstack((x[:3], 0.0, 0.0, yaw))
        xb_error = x_pose - self.xb_ref

        Jxb = np.zeros((6, ndq))
        Jxb[0:3, 0:3] = np.eye(3)
        Jxb[5, :] = yaw_tangent_jacobian(x)
        dx += 2.0 * Jxb.T @ Q_b @ xb_error

        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        # Foot tracking
        for foot in foot_phases:
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

        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((ndq, ndq))

        # CoM tracking
        Jcom = pin.jacobianCenterOfMass(pin_model, pin_data, x)
        Hxx += 2.0 * Jcom.T @ Q_c @ Jcom

        # Base pose tracking
        Jxb = np.zeros((6, ndq))
        Jxb[0:3, 0:3] = np.eye(3)
        Jxb[5, :] = yaw_tangent_jacobian(x)
        Hxx += 2.0 * Jxb.T @ Q_b @ Jxb

        # Foot tracking
        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        for foot in foot_phases:
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

        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((ndq, ndq))

    def hessian_uu(self, x, u):
        Huu = np.zeros((ndq, ndq))

        if not self.terminal:
            Huu[6:, 6:] = 2.0 * Q_v

        return Huu


# Constraints for Kinematic model
nc = nj + nj + 1

zb_min = 0.25
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


v_cmd = np.array((0.5, 0.0, 0.0))

dt = 0.01

model = ssqpy.model.Model(MPC_H, dt)

com_ref, xb_ref, p_ref = generate_references(
    v_cmd, init_config, np.zeros(3), 0.0
)

constr = KinematicConstraints()

stage_costs = []
for k in range(MPC_H):
    stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

    cost = KinematicCost(
        com_ref[k], xb_ref[k], stage_p_ref, init_config[7:], np.zeros(nj)
    )
    stage_costs.append(cost)

    model.addCost(k, cost)
    model.addConstraints(k, constr)

stage_p_ref = {foot: p_ref[foot][MPC_H] for foot in p_ref}
cost = KinematicCost(
    com_ref[MPC_H],
    xb_ref[MPC_H],
    stage_p_ref,
    init_config[7:],
    np.zeros(nj),
    terminal=True,
)

model.addCost(MPC_H, cost)

model.finalize(
    dynamics=[KinematicDynamics() for _ in range(MPC_H + 1)],
    init_states=[init_config],
)

custom_properties = ssqpy.model.customProperties()
custom_properties.nq = nq
custom_properties.nv = ndq
custom_properties.nu = ndq
custom_properties.njoints = nj
custom_properties.floating_base = True
custom_properties.ordered_joints = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
custom_properties.joint_types = ["revolute"] * 12

model.setCustomProperties(custom_properties)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)
state, x, u = solver.solve()
print("Solver state:", state)
solver.write("tmp.json")


"""
# MPC & Simulation
dt = 0.002
mpc_dt = 0.1

v_cmd = np.array((0.0, 0.0, 0.0))

model = ssqpy.model.Model(MPC_H, mpc_dt)

com_ref, xb_ref, p_ref = generate_references(
    v_cmd, init_config, np.zeros(3), 0.0
)

constr = KinematicConstraints()

stage_costs = []
for k in range(MPC_H):
    stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

    cost = KinematicCost(
        com_ref[k], xb_ref[k], stage_p_ref, init_config[7:], np.zeros(nj)
    )
    stage_costs.append(cost)

    model.addCost(k, cost)
    model.addConstraints(k, constr)

stage_p_ref = {foot: p_ref[foot][MPC_H] for foot in p_ref}
cost = KinematicCost(
    com_ref[MPC_H],
    xb_ref[MPC_H],
    stage_p_ref,
    init_config[7:],
    np.zeros(nj),
    terminal=True,
)

model.addCost(MPC_H, cost)

model.finalize(
    dynamics=[KinematicDynamics() for _ in range(MPC_H + 1)],
    init_states=[init_config],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=400)

import mujoco
import mujoco.viewer

mjmodel = mujoco.MjModel.from_xml_path("../../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_config.copy()
mjdata.qpos[3:7] = quat_xyzw_to_wxyz(mjdata.qpos[3:7])

np.set_printoptions(suppress=True)

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        print(f"\rSimulation time: {mjdata.time:.2f}s")
        frame_start = time()

        new_pos = mjdata.qpos.copy()
        new_pos[3:7] = quat_wxyz_to_xyzw(new_pos[3:7])

        pre = time()

        mjdata.qpos = mpc.step(new_pos)[0].stage(1)

        elapsed = time() - pre

        print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

print()

"""
