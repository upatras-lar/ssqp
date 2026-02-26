import ssqpy
import numpy as np
from math import pi, sin, cos

nq = 9
nv = 9
dt = 0.01


def get_panda_problems(dynamics_mode):
    problems = []

    problems.append(get_panda_circle_problem(dynamics_mode))
    problems.append(get_panda_obstacle_problem(dynamics_mode))

    return problems


def get_panda_circle_problem(dynamics_mode):
    H = 400

    T_WGT = 3.0
    R_WGT = 1.0
    V_WGT = 3e-3
    U_WGT = 3e-3

    constraint_hlines = [0.1, 0.41]
    constraint_vlines = [-0.15, 0.15]

    cx = 0.3
    cy = 0
    cz = 0.26
    cr = 0.22

    DISTANCE = 2 * pi
    ω = DISTANCE / (H * dt)
    φ0 = 0

    traj = []

    for k in range(H + 1):
        φk = φ0 + ω * k * dt
        traj.append(np.array([cx, cy + cr * sin(φk), cz + cr * cos(φk)]))

    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    circle_problem = ssqpy.model.Model(
        H,
        dt,
        urdf_path="../../assets/urdf/panda/panda.urdf",
        dynamics_mode=dynamics_mode,
    )

    init_state = np.zeros(2 * nq)
    init_state[:nq] = [
        0.0013,
        -0.9115,
        -0.0010,
        -2.6316,
        -0.0008,
        1.7431,
        0.7862,
        0.0,
        0.0,
    ]

    lb = np.array((-np.inf, constraint_vlines[0], constraint_hlines[0]))
    ub = np.array((np.inf, constraint_vlines[1], constraint_hlines[1]))
    box_constraint = ssqpy.model.constraints.FramePositionBound(
        circle_problem, "panda_ee", lb, ub, nq, nv
    )

    for k in range(H):
        circle_problem.addCost(
            k,
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                circle_problem, traj[k], Rd, "panda_ee", nq, nq, T_WGT, R_WGT
            ),
        )

        circle_problem.addCost(
            k, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
        )
        circle_problem.addCost(
            k, ssqpy.model.costs.SquaredControlCost(nq, U_WGT)
        )
        circle_problem.addConstraints(k, box_constraint)

    circle_problem.addCost(
        H,
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            circle_problem, traj[H], Rd, "panda_ee", nq, nq, T_WGT, R_WGT
        ),
    )

    circle_problem.addCost(
        H, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
    )

    circle_problem.addConstraints(H, box_constraint)

    circle_problem.finalize(init_states=init_state)

    return circle_problem


def get_panda_obstacle_problem(dynamics_mode):
    H = 400

    T_WGT = 10.0
    R_WGT = 10.0
    V_WGT = 1e-1
    U_WGT = 1e-2

    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    obstacle_problem = ssqpy.model.Model(
        H,
        dt,
        urdf_path="../../assets/urdf/panda/panda.urdf",
        dynamics_mode=dynamics_mode,
    )

    init_state = np.zeros(2 * nq)
    init_state[:nq] = [
        -0.3958,
        0.1247,
        -0.3661,
        -2.6497,
        0.1197,
        2.7639,
        -0.0852,
        0.0,
        0.0,
    ]

    joint_vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
    control_cost = ssqpy.model.costs.SquaredControlCost(nq, U_WGT)
    rot_cost = ssqpy.model.costs.FrameSquaredRotationErrorCost(
        obstacle_problem, Rd, "panda_ee", nq, nq, R_WGT
    )
    pose_cost = ssqpy.model.costs.FrameSquaredPoseErrorCost(
        obstacle_problem,
        np.array((0.3, 0.3, 0.1)),
        Rd,
        "panda_ee",
        nq,
        nq,
        T_WGT,
        R_WGT,
    )

    eef_bound = ssqpy.model.constraints.FramePositionBound(
        obstacle_problem,
        "panda_ee",
        np.array((-np.inf, -np.inf, 0.4)),
        np.array((np.inf, np.inf, np.inf)),
        nq,
        nv,
    )

    for k in range(H):
        obstacle_problem.addCost(k, rot_cost)
        obstacle_problem.addCost(k, joint_vel_cost)
        obstacle_problem.addCost(k, control_cost)

        if k > 50 and k < H - 50:
            obstacle_problem.addConstraints(k, eef_bound)

    obstacle_problem.addCost(H, pose_cost)
    obstacle_problem.addCost(H, joint_vel_cost)

    obstacle_problem.finalize(init_states=init_state)

    return obstacle_problem
