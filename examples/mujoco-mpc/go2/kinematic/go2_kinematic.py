import ssqpy
import ssqpy.model.dynamics as dynamics
import ssqpy.model.constraints as constraints
import ssqpy.model.costs as costs

import numpy as np

import pinocchio as pin


# Helpers
# -------
def yaw_to_Rz(yaw):
    return np.array(
        (
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 0.0),
        )
    )


# Parameters
# ----------
dt = 0.01  # timestep
MPC_H = 25  # MPC optimization horizon

nj = 12  # number of actuated joints
nq = 7 + nj  # configuration size
ndq = 6 + nj  # linearized configuration size

v_cmd = np.array((0.1, 0.1))  # sample velocity command

# Contact Scheduler
# -----------------
# TODO: parameter values
T = 50
beta = 0.5

foot_phases = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}
hip_to_foot_ratios = {"FL": 1.0, "FR": 1.0, "RL": 1.0, "RR": 1.0}


def is_swinging(t, phi_0):
    phi = (t / T + phi_0) % 1
    return phi < beta


# Reference Generation
# --------------------
Kv = 0.1


def generate_references(v_cmd, config_0, vel_0, t):
    # Commanded velocity for x and y axes, and angular (yaw)
    v_cmd_3 = np.hstack((v_cmd[:2], 0.0))
    wz_cmd = v_cmd[2]

    # Generate CoM reference from velocity command
    com_0 = pin.centerOfMass(pin_model, pin_data, config_0)

    com_ref = [com_0]
    for _ in range(MPC_H):
        com_ref_k = com_ref[-1] + dt * v_cmd_3
        com_ref.append(com_ref_k)

    # Generate base position reference
    qx, qy, qz, qw = config_0[3:7]

    yaw_0 = np.atan2(2.0 * (qz * qw + qx * qy), -1.0 + 2.0 * (qw**2 + qx**2))
    xb_0 = np.hstack((config_0[:3], 0.0, 0.0, yaw_0))

    xb_ref = [xb_0]
    for _ in range(MPC_H):
        p_ref_k = xb_ref[-1][:3] + dt * v_cmd_3
        yaw_ref_k = xb_ref[-1][-1] + dt * wz_cmd
        xb_ref.append(np.hstack((p_ref_k, 0.0, 0.0, yaw_ref_k)))

    # Generate feet trajectory references
    pin.forwardKinematics(pin_model, pin_data, config_0)

    p_ref = {
        "FL": [pin_data.oMf[pin_model.getFrameId("FL_foot")].translation()],
        "FR": [pin_data.oMf[pin_model.getFrameId("FR_foot")].translation()],
        "RL": [pin_data.oMf[pin_model.getFrameId("RL_foot")].translation()],
        "RR": [pin_data.oMf[pin_model.getFrameId("RR_foot")].translation()],
    }

    touchdown_references = {}

    Rz = yaw_to_Rz(yaw_0)

    for foot in foot_phases:
        if not is_swinging(t, foot_phases[foot]):
            touchdown_references[foot] = p_ref[foot][0]
            continue

        # TODO: turning yaw-rate correction term
        touchdown_references[foot] = (
            p_ref[foot][0]
            + Rz * hip_to_foot_ratios[foot]
            + (1 - beta) * T / 2 * vel_0[:3]
            + Kv * (vel_0[:3] - v_cmd_3)
        )

    for k in range(MPC_H):
        actual_t = t + k
        for foot in foot_phases:
            if not is_swinging(actual_t, foot_phases[foot]):
                p_ref[foot].append(p_ref[foot][-1])
                continue

    return com_ref, xb_ref, p_ref


pin_model, _, _ = pin.buildModelsFromMJCF("../../../assets/mjcf/go2/go2.xml")
pin_data = pin_model.createData()


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
        return nq

    def state_right_plus(self, x, dx):
        return pin.integrate(pin_model, x, dx)

    def state_right_minus(self, x, y):
        return pin.difference(pin_model, y, x)

    def evaluate(self, x, u):
        return self.state_right_plus(x, u * dt)

    def gradient(self, x, u):
        dX = pin.dIntegrate(pin_model, x, u * dt, 0)
        dU = pin.dIntegrate(pin_model, x, u * dt, 1) * dt
        return dX, dU

    def hessprod_xx(self, x, u):
        return np.zeros((ndq, ndq))

    def hessprod_xu(self, x, u):
        return np.zeros((ndq, ndq))

    def hessprod_uu(self, x, u):
        return np.zeros((ndq, ndq))


# Cost for Kinematic model
# ------------------------
Q_c = np.eye(3)
Q_b = np.eye(3)


class KinematicCost(costs.StageCost):
    def __init__(self, com_ref, xb_ref, p_ref):
        self.update_references(com_ref, xb_ref, p_ref)

    def update_references(self, com_ref, xb_ref, p_ref):
        self.com_ref = com_ref
        self.xb_ref = xb_ref
        self.p_ref = p_ref

    def evaluate(self, x, u):
        cost = 0.0

        com = pin.centerOfMass(pin_model, pin_data, x)
        com_error = com - self.com_ref
        cost += com_error.T @ Q_c @ com_error

        xb_error = x[:3] - self.xb_ref
        cost += xb_error.T @ Q_b @ xb_error
