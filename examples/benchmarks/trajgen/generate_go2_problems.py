import ssqpy
from ssqpy.model import costs, constraints
import numpy as np
from math import pi
import sys

sys.path.insert(0, "../../trajgen/python/utils")

from GaitManager import GaitManager


def yaw_to_quaternion(yaw):
    q = np.zeros(4)
    q[2] = np.sin(yaw / 2.0)
    q[3] = np.cos(yaw / 2.0)
    return q


def rotation_matrix_to_quaternion(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # Calculate quaternion components
    q0 = 0.5 * np.sqrt(1 + r11 + r22 + r33)
    q1 = 0.5 * np.sqrt(1 + r11 - r22 - r33) * np.sign(r32 - r23)
    q2 = 0.5 * np.sqrt(1 - r11 + r22 - r33) * np.sign(r13 - r31)
    q3 = 0.5 * np.sqrt(1 - r11 - r22 + r33) * np.sign(r21 - r12)

    return np.array([q0, q1, q2, q3])


def generate_go2_walk_problems(n_problems, dynamics_mode, seed=42):
    rng = np.random.default_rng(seed)

    problems = []

    # Model Parameters
    nq = 19
    nv = 18
    nu = 18
    HORIZON = 300
    dt = 0.01

    # Weights
    C_WGT = 1e-1
    V_WGT = 1e-2
    U_WGT = 1e-3
    F_WGT = 1e-5
    TF_WGT = 5.0
    ROT_WGT = 5.0

    GAIT = GaitManager.setWalk
    CYCLE_STEPS = 60
    FINAL_STANDING_STEPS = 10

    z_height = 0.3689268220122

    for i in range(n_problems):

        # -------------------------
        # Random initial / final pose
        # Distance between them ≈ 1m
        # -------------------------
        x0, y0 = rng.uniform(-1.0, 1.0, size=2)

        # Random direction
        direction = rng.uniform(-pi, pi)

        # Slight noise around 1m if desired
        distance = rng.uniform(0.9, 1.1)

        xf = x0 + distance * np.cos(direction)
        yf = y0 + distance * np.sin(direction)

        yaw0 = rng.uniform(-pi, pi)
        yawf = rng.uniform(-pi, pi)

        base_pos = np.array([x0, y0, z_height])
        final_pos = np.array([xf, yf, z_height])
        final_R = np.array(
            [
                [np.cos(yawf), -np.sin(yawf), 0],
                [np.sin(yawf), np.cos(yawf), 0],
                [0, 0, 1],
            ]
        )

        # print("Problem:", i)
        # print(
        # "--shaded-model-base",
        # *final_pos,
        # *rotation_matrix_to_quaternion(final_R)
        # )

        # -------------------------
        # Initial state
        # -------------------------
        init = np.zeros(37)
        init[:3] = base_pos
        init[3:7] = yaw_to_quaternion(yaw0)
        init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

        # -------------------------
        # Create model
        # -------------------------
        model = ssqpy.model.Model(
            HORIZON,
            dt,
            urdf_path="../../assets/urdf/go2/go2.urdf",
            free_flyer=True,
            solver_mode=dynamics_mode,
        )

        # -------------------------
        # Costs
        # -------------------------
        config_cost = costs.SquaredConfigurationErrorCost(
            model, init[7:19], nq, nv, C_WGT, True
        )
        joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
        control_cost = costs.SquaredControlCost(nv, nu, U_WGT)
        force_cost = costs.SquaredForceCost(nv, nu, F_WGT)

        for k in range(HORIZON):
            model.addCost(k, joint_vel_cost)
            model.addCost(k, control_cost)
            model.addCost(k, force_cost)

        for k in range(HORIZON + 1):
            model.addCost(k, config_cost)

        # Strong terminal velocity regularization
        model.addCost(
            HORIZON,
            costs.SquaredJointVelocityCost(nv, 1e3, True),
        )

        # Final translation target
        model.addCost(
            HORIZON,
            costs.FrameSquaredPoseErrorCost(
                model, "base", final_pos, final_R, nq, nv, TF_WGT, ROT_WGT
            ),
        )

        # Final base velocity constraint
        model.addConstraints(
            HORIZON,
            constraints.FrameLinearVelocityBound(
                model, "base", np.zeros(3), np.zeros(3), nq, nv, True
            ),
        )

        # -------------------------
        # Gait
        # -------------------------
        init_controls = []

        gait_manager = GaitManager(
            model, nq, nv, ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        )

        if CYCLE_STEPS is None:
            init_controls.extend(
                GAIT(gait_manager, 0, HORIZON - FINAL_STANDING_STEPS - 1)
            )
        else:
            init_controls.extend(
                GAIT(
                    gait_manager,
                    0,
                    HORIZON - FINAL_STANDING_STEPS - 1,
                    cycle_steps=CYCLE_STEPS,
                )
            )

        for k in range(HORIZON - FINAL_STANDING_STEPS, HORIZON):
            init_controls.extend(gait_manager.addStandingStage(k))

        model.finalize(init_states=init)

        problems.append(model)

    return problems
