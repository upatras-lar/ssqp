from ssqpy.model.costs import StageCost

import os
from sys import argv

from pathlib import Path

import numpy as np

import models.utils as utils
from . import config


class ReferenceState:
    initialized = False

    liftoff = {}
    touchdown = {}


def generate_references_MPC(v_cmd, config_0, vel_0, k0, H, dt, ssqp_model):
    translations = {}

    for foot in config.ContactScheduling.foot_phases:
        translations[foot], _ = ssqp_model.forwardKinematics(
            config_0, config.FRAME_NAMES[foot]
        )

    if not ReferenceState.initialized:
        ReferenceState.liftoff = translations.copy()
        ReferenceState.touchdown = {
            foot: ReferenceState.liftoff[foot].copy()
            for foot in config.ContactScheduling.foot_phases
        }

        ReferenceState.initialized = True

    # Local copies
    liftoff = {f: v.copy() for f, v in ReferenceState.liftoff.items()}
    touchdown = {f: v.copy() for f, v in ReferenceState.touchdown.items()}

    # Actual initial velocity for x/y and yaw
    vel_0_3 = np.hstack((vel_0[:2], 0.0))
    wz_0 = vel_0[2]

    # Commanded velocity for x/y and yaw
    Rz_0 = utils.yaw_to_Rz(utils.quat_to_yaw(config_0[3:7]))
    v_cmd_3 = Rz_0 @ np.hstack((v_cmd[:2], 0.0))
    wz_cmd = v_cmd[2]

    # CoM reference
    # -------------
    com_0 = ssqp_model.centerOfMass(config_0)

    com_ref = [com_0]
    for _ in range(H):
        com_ref.append(com_ref[-1] + dt * v_cmd_3)

    # Base pose reference
    # -------------------
    xb_0 = config_0[:7]
    yaw = utils.quat_to_yaw(config_0[3:7])
    xb_ref = [xb_0]
    for _ in range(H):
        p_ref_k = xb_ref[-1][:3] + dt * v_cmd_3
        yaw += dt * wz_cmd
        xb_ref.append(np.hstack((p_ref_k, utils.euler_to_quat(0.0, 0.0, yaw))))

    # Foot trajectory references
    # --------------------------
    p_ref = {
        foot: [
            [
                translations[foot].copy(),
                is_swinging(k0, config.ContactScheduling.foot_phases[foot])[0],
            ]
        ]
        for foot in config.ContactScheduling.foot_phases
    }

    for k in range(H):
        t = k0 + k

        yaw_k = utils.quat_to_yaw(xb_ref[k][3:7])
        Rz = utils.yaw_to_Rz(yaw_k)

        for foot in config.ContactScheduling.foot_phases:
            swing_at_t, swapped = is_swinging(
                t, config.ContactScheduling.foot_phases[foot]
            )

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
                    + Rz @ config.ContactScheduling.hip_to_foot_offsets[foot]
                    - config.ContactScheduling.T_stance * dt / 2 * vel_0_3
                    + config.ReferenceGeneration.Kv * (v_cmd_3 - vel_0_3)
                    + config.ReferenceGeneration.Kw
                    * (wz_0 - wz_cmd)
                    * Rz
                    @ np.array(
                        (
                            -config.ContactScheduling.hip_to_foot_offsets[
                                foot
                            ][1],
                            config.ContactScheduling.hip_to_foot_offsets[foot][
                                0
                            ],
                            0.0,
                        )
                    )
                )
                touchdown[foot][2] = 0.0  # -0.0002611

            # Normalized swing phase
            phi = (
                t / config.ContactScheduling.T
                + config.ContactScheduling.foot_phases[foot]
            ) % 1
            phi_sw = phi / config.ContactScheduling.beta

            swing_alpha = 3 * phi_sw**2 - 2 * phi_sw**3

            p_ref_k = (1 - swing_alpha) * liftoff[
                foot
            ] + swing_alpha * touchdown[foot]
            p_ref_k[2] += (
                4 * config.ReferenceGeneration.h_sw * phi_sw * (1 - phi_sw)
            )

            p_ref[foot].append([p_ref_k, True])

        if k == 0:
            ReferenceState.liftoff = {f: v.copy() for f, v in liftoff.items()}
            ReferenceState.touchdown = {
                f: v.copy() for f, v in touchdown.items()
            }

    return com_ref, xb_ref, p_ref


def generate_references(v_cmd, config_0, vel_0, k0, H, dt, ssqp_model):
    # Last liftoff location for each foot, along with its liftoff time
    translations = {}

    for foot in config.ContactScheduling.foot_phases:
        translations[foot], _ = ssqp_model.forwardKinematics(
            config_0, config.FRAME_NAMES[foot]
        )

    liftoff = translations.copy()

    # Next touchdown location for each foot, along with its touchdown time
    touchdown = {
        foot: liftoff[foot].copy()
        for foot in config.ContactScheduling.foot_phases
    }

    # Actual initial velocity for x and y axes, and angular (yaw)
    vel_0_3 = np.hstack((vel_0[:2], 0.0))
    wz_0 = vel_0[2]

    # Commanded velocity for x and y axes, and angular (yaw)
    v_cmd_3 = np.hstack((v_cmd[:2], 0.0))
    wz_cmd = v_cmd[2]

    # Generate CoM reference from velocity command
    com_0 = ssqp_model.centerOfMass(config_0)

    com_ref = [com_0]
    for _ in range(H):
        # Follow velocity command with CoM
        com_ref_k = com_ref[-1] + dt * v_cmd_3
        com_ref.append(com_ref_k)

    # Generate base position reference
    xb_0 = config_0[:7]
    yaw = utils.quat_to_yaw(config_0[3:7])

    xb_ref = [xb_0]
    for _ in range(H):
        # Follow velocity command with base position reference
        p_ref_k = xb_ref[-1][:3] + dt * v_cmd_3
        # Follow angular (yaw) command with rotation angle
        yaw += dt * wz_cmd
        xb_ref.append(np.hstack((p_ref_k, utils.euler_to_quat(0.0, 0.0, yaw))))

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
        for foot in config.ContactScheduling.foot_phases
    }

    p_ref["FL"][0][1] = True
    p_ref["RR"][0][1] = True

    for k in range(H):
        t = k0 + k

        yaw_k = utils.quat_to_yaw(xb_ref[k][3:7])
        Rz = utils.yaw_to_Rz(yaw_k)

        for foot in config.ContactScheduling.foot_phases:
            # Use scheduling to check if foot is swinging at step t, and if it
            # just swapped swinging state
            swing_at_t, swapped = is_swinging(
                t, config.ContactScheduling.foot_phases[foot]
            )

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
                + Rz @ config.ContactScheduling.hip_to_foot_offsets[foot]
                + config.ContactScheduling.T_stance / 2 * vel_0_3
                + config.ReferenceGeneration.Kv * (vel_0_3 - v_cmd_3)
                + config.ReferenceGeneration.Kw
                * (wz_0 - wz_cmd)
                * Rz
                @ np.array(
                    (
                        -config.ContactScheduling.hip_to_foot_offsets[foot][1],
                        config.ContactScheduling.hip_to_foot_offsets[foot][0],
                        0.0,
                    )
                )
            )

            # Calculate the normalized swing phase
            phi = (
                t / config.ContactScheduling.T
                + config.ContactScheduling.foot_phases[foot]
            ) % 1
            phi_sw = phi / config.ContactScheduling.beta

            # Get an interpolation alpha for the swing
            swing_alpha = 3 * phi_sw**2 - 2 * phi_sw**3

            # Interpolate between previous liftoff and caluclated touchdown
            # positions to get current position reference
            p_ref_k = (1 - swing_alpha) * liftoff[
                foot
            ] + swing_alpha * touchdown[foot]

            # For z reference, also add a height according to phase
            p_ref_k[2] += (
                4 * config.ReferenceGeneration.h_sw * phi_sw * (1 - phi_sw)
            )

            p_ref[foot].append([p_ref_k, True])

    return com_ref, xb_ref, p_ref


# Whether a foot with initial phase phi_0 is swinging at time t
# Returns: (is_swinging, just_swapped)
# just_swapped is True if the foot's current swing state is different to that of
# the previous time step
def is_swinging(t, phi_0):
    prev_phi = ((t - 1) / config.ContactScheduling.T + phi_0) % 1
    phi = (t / config.ContactScheduling.T + phi_0) % 1

    at_prev = prev_phi < config.ContactScheduling.beta
    at_t = phi < config.ContactScheduling.beta

    return at_t, at_t != at_prev


class LineSegmentCost(StageCost):
    def __init__(self, model, p_ref, q_sp, l_sigma, terminal=False):
        super().__init__()
        self.model = model
        self.p_ref = p_ref
        self.Q_sp = q_sp * np.eye(2)
        self.l_sigma = l_sigma

        if terminal:
            self.u_size = 0
        else:
            self.u_size = config.KinematicParameters.ndq

    def update_reference(self, p_ref):
        self.p_ref = p_ref

    def evaluate(self, x, u):
        cost = 0.0

        # Find feet that are standing
        ground_feet = []
        for foot in config.FRAME_NAMES:
            if not self.p_ref[foot][1]:
                ground_feet.append(config.FRAME_NAMES[foot])

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
        dx = np.zeros(config.KinematicParameters.ndq)
        du = np.zeros(self.u_size)

        # Find feet that are standing
        ground_feet = []
        for foot in config.FRAME_NAMES:
            if not self.p_ref[foot][1]:
                ground_feet.append(config.FRAME_NAMES[foot])

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
        Hxx = np.zeros(
            (config.KinematicParameters.ndq, config.KinematicParameters.ndq)
        )

        # Find feet that are standing
        ground_feet = []
        for foot in config.FRAME_NAMES:
            if not self.p_ref[foot][1]:
                ground_feet.append(config.FRAME_NAMES[foot])

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
        return np.zeros((config.KinematicParameters.ndq, self.u_size))

    def hessian_uu(self, x, u):
        return np.zeros((self.u_size, self.u_size))
