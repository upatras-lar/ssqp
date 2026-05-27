"""
Configuration file for the Quadruped models
Determines various options about how the model is loaded and its properties
"""

import numpy as np

# Whether to import pinocchio
# Also determines whether the robot model is loaded via Python
WITH_PINOCCHIO = True


# Initial base position
INIT_BASE_POS = np.array([0.0, 0.0, 0.389])
# Initial joint configuration
INIT_JOINT_CONFIG = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

# Initial generalized configuration vector of the robot
INIT_CONFIG = np.zeros(19)
INIT_CONFIG[:3] = INIT_BASE_POS
INIT_CONFIG[6] = 1.0
INIT_CONFIG[7:] = INIT_JOINT_CONFIG


# Kinematic-specific Parameters
class KinematicParameters:
    nj = 12  # number of actuated joints
    nq = 7 + nj  # configuration size
    ndq = 6 + nj  # linearized configuration size


# SRBD-specific parameters
class SRBDParameters:
    GRAVITY = np.array((0.0, 0.0, -9.81))

    nq = 25  # state size (p, q, v, w, 4 * feet_pos, 4 * feet_vel)
    ndq = 24  # tangent state size (dp, dq, dv, dw, 4 * d_feet_pos, 4 * d_feet_vel)
    nu = 24  # controls / contact forces (4 feet * 3)

    mass = 15.206408
    inertia = np.array(
        (
            (0.22180955, 0.00012166, 0.00323384),
            (0.00012166, 0.48982369, -0.0000312),
            (0.00323384, -0.0000312, 0.54306506),
        )
    )

    INIT_STATE = np.hstack((INIT_CONFIG[:7], np.zeros(6)))


# Dictionary mapping foot frame names used to XML frame names
FRAME_NAMES = {
    "FL": "FL_foot",
    "FR": "FR_foot",
    "RL": "RL_foot",
    "RR": "RR_foot",
}


# Contact Scheduling parameters
class ContactScheduling:
    # Number of time steps for each stance - swing cycle
    T = 30
    # Percentage of `T` corresponding to swing
    beta = 0.5

    # Number of time steps for each swing
    T_swing = None
    # Number of time steps for each stance
    T_stance = None

    # Phases (range [0.0 - 1.0]) for each foot
    # Each phase acts as a percentage of the cycle, at which point the swing
    # begins
    foot_phases = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}

    # Offsets of the positions of each foot relative to the hip
    hip_to_foot_offsets = {
        "FL": np.array((0.3, 0.14, -0.36892682)),
        "FR": np.array((0.3, -0.14, -0.36892682)),
        "RL": np.array((-0.15, 0.14, -0.36892682)),
        "RR": np.array((-0.15, -0.14, -0.36892682)),
    }

    @staticmethod
    def calculate():
        ContactScheduling.T_swing = int(
            ContactScheduling.beta * ContactScheduling.T
        )

        ContactScheduling.T_stance = (
            ContactScheduling.T - ContactScheduling.T_swing
        )


ContactScheduling.calculate()


# Reference Generation parameters
class ReferenceGeneration:
    # Weight for following the linear velocity command
    Kv = 0.1
    # Weight for following the angular velocity command
    Kw = 0.1
    # Maximum height of swing
    h_sw = 0.1
    # Range within which to keep base z-position
    zb_min = 0.35
    zb_max = 0.35
    # Maximum commanded velocities
    max_vel_front = 1.0
    max_vel_back = 0.5
    max_vel_side = 0.75
    max_vel_yaw = 1.0
