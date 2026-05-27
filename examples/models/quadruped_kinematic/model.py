import ssqpy
import ssqpy.model.dynamics as dynamics
import ssqpy.model.constraints as constraints
import ssqpy.model.costs as costs

import os
from sys import argv
from pathlib import Path

import numpy as np
import pinocchio as pin

import models.quadruped_hlplanner.config as config
import models.utils as utils

examples_env = os.environ.get("PYTHONPATH", "")
sep = os.pathsep
examples_dir_str = Path(
    [p for p in examples_env.split(sep) if "examples" in p][0]
)

if len(argv) > 1:
    pin_model, _, _ = pin.buildModelsFromMJCF(
        str(examples_dir_str / Path(argv[1]))
    )
else:
    pin_model, _, _ = pin.buildModelsFromMJCF(
        str(examples_dir_str / Path("assets/mjcf/go2/go2.xml"))
    )

pin_data = pin_model.createData()


# Dynamics for Kinematic model
# ----------------------------
class Kinematics(dynamics.StageDynamics):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def output_size(self):
        return config.KinematicParameters.nq

    def state_size(self):
        return config.KinematicParameters.nq

    def state_diff_size(self):
        return config.KinematicParameters.ndq

    def control_size(self):
        return config.KinematicParameters.ndq

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
        return np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

    def hessprod_xu(self, x, u, lam):
        return np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

    def hessprod_uu(self, x, u, lam):
        return np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )


# Cost for Kinematic model
# ------------------------
q_c = 1.0  # CoM tracking weight
q_b = 1.0  # Body position tracking weight
q_o = 1.0  # Orientation tracking weight
q_st = 200.0  # Stance feet position tracking weight
q_sw = 10.0  # Swing feet position tracking weight
q_q = 1e-3  # Configuration regularization weight
q_v = 1e-3  # Velocity regularization weight
q_sp = 50.0  # Support polygon (line segment) CoM weight
l_sigma = 0.5

Q_c = q_c * np.eye(3)
Q_b = q_b * np.eye(3)
Q_o = q_o * np.eye(3)
Q_st = q_st * np.eye(3)
Q_sw = q_sw * np.eye(3)
Q_q = q_q * np.eye(config.KinematicParameters.nq - 7)
Q_v = q_v * np.eye(config.KinematicParameters.nq - 7)
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
        q_error = utils.quat_mul(q_ref_inv, x[3:7])
        cost += q_o * self.xb_ref[6] * np.dot(q_error[:3], q_error[:3])

        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        feet_pos = {
            foot: pin_data.oMf[
                pin_model.getFrameId(config.FRAME_NAMES[foot])
            ].translation
            for foot in config.FRAME_NAMES
        }

        for foot in config.FRAME_NAMES:
            p_error = feet_pos[foot] - self.p_ref[foot][0]
            if self.p_ref[foot][1]:  # swing
                cost += p_error.T @ Q_sw @ p_error
            else:  # stance
                cost += p_error.T @ Q_st @ p_error

        # Joint configuration regularization
        qj_error = x[7 : config.KinematicParameters.nq] - self.qj_nom
        cost += qj_error.T @ Q_q @ qj_error

        # Joint velocity regularization
        if not self.terminal:
            vj_error = u[6 : config.KinematicParameters.ndq] - self.vj_nom
            cost += vj_error.T @ Q_v @ vj_error

        # Find feet that are standing
        ground_feet = []
        for foot in config.FRAME_NAMES:
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
        dx = np.zeros(config.KinematicParameters.ndq)
        du = np.zeros(config.KinematicParameters.ndq)

        # CoM tracking
        com = pin.centerOfMass(pin_model, pin_data, x)
        com_error = com - self.com_ref

        com_jac = pin.jacobianCenterOfMass(pin_model, pin_data, x)
        dx += 2.0 * com_jac.T @ Q_c @ com_error

        # Base Pose tracking
        xb_error = x[:3] - self.xb_ref[:3]

        Jxb = np.zeros((3, config.KinematicParameters.ndq))
        Jxb[:, 0:3] = np.eye(3)
        dx += 2.0 * Jxb.T @ Q_b @ xb_error

        q_ref = self.xb_ref[3:7]
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]])
        q_error = utils.quat_mul(q_ref_inv, x[3:7])

        dx[3:6] += q_o * self.xb_ref[6] * q_error[:3]

        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        # Foot tracking
        for foot in config.FRAME_NAMES:
            frame_id = pin_model.getFrameId(config.FRAME_NAMES[foot])

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
        qj_error = x[7 : config.KinematicParameters.nq] - self.qj_nom
        dx[6 : config.KinematicParameters.ndq] += 2.0 * Q_q @ qj_error

        # Joint velocity
        if not self.terminal:
            vj_error = u[6 : config.KinematicParameters.ndq] - self.vj_nom
            du[6 : config.KinematicParameters.ndq] += 2.0 * Q_v @ vj_error

        # Find feet that are standing
        ground_feet = []
        for foot in config.FRAME_NAMES:
            if not self.p_ref[foot][1]:
                ground_feet.append(foot)

        # CoM support line segment cost
        if len(ground_feet) == 2:
            frame1 = pin_model.getFrameId(config.FRAME_NAMES[ground_feet[0]])
            frame2 = pin_model.getFrameId(config.FRAME_NAMES[ground_feet[1]])

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
        Hxx = np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

        # CoM tracking
        Jcom = pin.jacobianCenterOfMass(pin_model, pin_data, x)
        Hxx += 2.0 * Jcom.T @ Q_c @ Jcom

        # Base pose tracking
        Jxb = np.zeros((3, config.KinematicParameters.ndq))
        Jxb[:, 0:3] = np.eye(3)
        Hxx += 2.0 * Jxb.T @ Q_b @ Jxb

        J = np.zeros((3, 4))
        J[:, :3] = np.eye(3)

        H_q = q_o * self.xb_ref[6] * (J.T @ J)
        Hxx[3:7, 3:7] += H_q

        # Foot tracking
        pin.forwardKinematics(pin_model, pin_data, x)
        pin.updateFramePlacements(pin_model, pin_data)

        for foot in config.FRAME_NAMES:
            frame_id = pin_model.getFrameId(config.FRAME_NAMES[foot])

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
        Hxx[
            6 : config.KinematicParameters.ndq,
            6 : config.KinematicParameters.ndq,
        ] += (
            2.0 * Q_q
        )

        # Find feet that are standing
        ground_feet = []
        for foot in config.FRAME_NAMES:
            if not self.p_ref[foot][1]:
                ground_feet.append(foot)

        # CoM support line segment cost
        if len(ground_feet) == 2:
            frame1 = pin_model.getFrameId(config.FRAME_NAMES[ground_feet[0]])
            frame2 = pin_model.getFrameId(config.FRAME_NAMES[ground_feet[1]])

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
        return np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

    def hessian_uu(self, x, u):
        Huu = np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

        if not self.terminal:
            Huu[6:, 6:] = 2.0 * Q_v

        return Huu


# Constraints for Kinematic model
# -------------------------------
nc = config.KinematicParameters.nj + config.KinematicParameters.nj + 1

zb = 0.35


class KinematicConstraints(constraints.StageConstraints):
    def output_size(self):
        return nc

    def lower_bound(self):
        return np.hstack(
            (
                pin_model.lowerPositionLimit[7:],
                -pin_model.velocityLimit[6:],
                zb,
            )
        )

    def upper_bound(self):
        return np.hstack(
            (
                pin_model.upperPositionLimit[7:],
                pin_model.velocityLimit[6:],
                zb,
            )
        )

    def evaluate(self, x, u):
        return np.hstack((x[7:], u[6:], x[2]))

    def gradient(self, x, u):
        dX = np.zeros((nc, config.KinematicParameters.ndq))
        dU = np.zeros((nc, config.KinematicParameters.ndq))

        dX[
            : config.KinematicParameters.nj, 6 : config.KinematicParameters.ndq
        ] += np.eye(config.KinematicParameters.nj)

        dU[
            config.KinematicParameters.nj : 2 * config.KinematicParameters.nj,
            6 : config.KinematicParameters.ndq,
        ] += np.eye(config.KinematicParameters.nj)

        dX[2 * config.KinematicParameters.nj, 2] = 1.0

        return dX, dU

    def hessprod_xx(self, x, u, mu):
        return np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

    def hessprod_xu(self, x, u, mu):
        return np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

    def hessprod_uu(self, x, u, mu):
        return np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )
