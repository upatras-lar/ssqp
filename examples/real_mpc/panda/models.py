import ssqpy
import numpy as np


def get_target_model(dynamics_mode, target_pos=np.array((0.46, 0.35, 0.0))):
    """
    Get a model that leads the Panda's end effector to the desired target
    position
    """

    MPC_H = 10
    dt = 0.1

    T_WGT = 100.0
    R_WGT = 20.0
    V_WGT = 1e-2
    U_WGT = 1e-5
    C_WGT = np.array([0.0, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])

    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    desired_pose = np.zeros((4, 4))
    desired_pose[:3, :3] = Rd
    desired_pose[:3, 3] = target_pos
    desired_pose[3, 3] = 1

    model = ssqpy.model.Model(
        MPC_H,
        dt,
        from_preset=ssqpy.model.Preset.Panda,
        solver_mode=dynamics_mode,
    )

    nq = model.getnq()
    nv = model.getnv()

    init_state = np.zeros(nq + nv)
    init_state[:nq] = [0.0] * 9

    constraint_vlines = [-np.inf, 0.1]

    lb = np.array((-np.inf, constraint_vlines[0], -np.inf))
    ub = np.array((np.inf, constraint_vlines[1], np.inf))
    box_constraint = ssqpy.model.constraints.FrameTranslationBound(
        model, "panda_ee", lb, ub
    )

    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, init_state[:nq], C_WGT
    )
    pose_error = ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, "panda_ee", desired_pose, T_WGT, R_WGT
    )

    for k in range(MPC_H):
        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, pose_error)
        model.addCost(k, config_cost)

        model.addConstraints(k, box_constraint)

    model.addCost(MPC_H, pose_error)
    model.addCost(MPC_H, v_cost)

    model.addConstraints(MPC_H, box_constraint)

    def update_target_pos(new_pos):
        if type(new_pos) is int:
            return

        pose_error.set_desired_translation(new_pos)

    return model, update_target_pos


def get_reset_model(dynamics_mode):
    """
    Get a model that leads the Panda to the initial configuration (config0)

    That initial configuration is
        config0 = [0.0013, -0.9115, -0.0010, -2.6316, -0.0008, 1.7431, 0.7862]
    """

    MPC_H = 10
    dt = 0.1

    C_WGT = 1.0
    V_WGT = 0.05
    U_WGT = 1e-3

    model = ssqpy.model.Model(
        MPC_H,
        dt,
        from_preset=ssqpy.model.Preset.Panda,
        solver_mode=dynamics_mode,
    )

    nq = model.getnq()
    nv = model.getnv()

    target_init_state = np.zeros(nq + nv)
    target_init_state[:nq] = [
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

    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
    init_config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, target_init_state[:nq], C_WGT
    )

    for k in range(MPC_H):
        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, init_config_cost)

    model.addCost(MPC_H, v_cost)
    model.addCost(MPC_H, init_config_cost)

    return model, lambda _: None


def get_circle_model(dynamics_mode, constrain=False):
    """
    Get a model that performs a circular motion in the yz-plane with the Panda's
    end effector
    """

    H = 400
    dt = 0.01

    cx = 0.42
    cy = 0
    cz = 0.36
    cr = 0.36

    DISTANCE = 2 * np.pi
    ω = DISTANCE / (H * dt)
    φ0 = 0

    square_side_hlf = cr / np.sqrt(2)
    constraint_hlines = [cz - square_side_hlf, cz + square_side_hlf]
    constraint_vlines = [cy - square_side_hlf, cy + square_side_hlf]

    traj = []

    for k in range(H + 1):
        φk = φ0 + ω * k * dt
        traj.append(np.array([cx, cy + cr * np.sin(φk), cz + cr * np.cos(φk)]))

    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    MPC_H = 10
    MPC_dt = 0.1

    T_WGT = 10.0
    R_WGT = 10.0
    V_WGT = 1e-2
    U_WGT = 1e-3
    C_WGT = 0.16

    model = ssqpy.model.Model(
        MPC_H,
        MPC_dt,
        from_preset=ssqpy.model.Preset.Panda,
        solver_mode=dynamics_mode,
    )

    nq = model.getnq()
    nv = model.getnv()

    init_state = np.zeros(nq + nv)
    init_state[:nq] = [
        0.0013,
        -0.3281,
        -0.0010,
        -2.1711,
        -0.0008,
        1.7431,
        0.7862,
        0.0,
        0.0,
    ]

    pose_costs = []

    lb = np.array((-np.inf, constraint_vlines[0], constraint_hlines[0]))
    ub = np.array((np.inf, constraint_vlines[1], constraint_hlines[1]))
    box_constraint = ssqpy.model.constraints.FrameTranslationBound(
        model, "panda_ee", lb, ub
    )

    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, init_state[:nq], C_WGT
    )

    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)

    for k in range(MPC_H):
        pose_costs.append(
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                model, "panda_ee", traj[k], Rd, T_WGT, R_WGT
            )
        )

        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, config_cost)
        model.addCost(k, pose_costs[-1])

        if constrain:
            model.addConstraints(k, box_constraint)

    pose_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, "panda_ee", traj[MPC_H], Rd, T_WGT, R_WGT
        )
    )

    model.addCost(MPC_H, v_cost)
    model.addCost(MPC_H, pose_costs[-1])

    if constrain:
        model.addConstraints(MPC_H, box_constraint)

    def update_pose_costs(current_stage):
        for k in range(current_stage, current_stage + MPC_H):
            pose_costs[k - current_stage].set_desired_translation(traj[k % H])

    return model, update_pose_costs


def get_sequence_model(dynamics_mode, constrain=False):
    """
    Get a model that follows a sequence of points with the Panda's end effector
    """
    H = 3800

    points = [
        np.array((0.3, 0.0, 0.3)),
        np.array((0.3, 0.3, 0.5)),
        np.array((0.3, 0.3, 0.1)),
        np.array((0.3, -0.4, 0.3)),
        np.array((0.0, -0.4, 0.4)),
        np.array((-0.2, -0.4, 0.0)),
        np.array((0.3, -0.4, 0.6)),
        np.array((0.3, 0.0, 0.6)),
        np.array((0.3, 0.4, 0.1)),
        np.array((-0.2, 0.4, 0.6)),
        np.array((-0.2, 0.4, 0.0)),
        np.array((0.2, 0.3, -0.2)),
        np.array((0.4, 0.5, 0.0)),
        np.array((0.4, 0.5, 0.6)),
        np.array((0.5, 0.0, 0.3)),
        np.array((0.5, -0.2, 0.1)),
        np.array((0.5, -0.3, -0.1)),
        np.array((0.5, -0.4, -0.25)),
        np.array((0.62, -0.45, 0.6)),
        np.array((0.45, -0.45, 0.6)),
    ]

    point_steps = H // len(points)

    traj = []

    for k in range(H + 1):
        k_between = (k % point_steps) / point_steps
        prev_point = points[k_between % len(points)]
        next_point = points[(k_between + 1) % len(points)]

        traj.append(
            (k_between / point_steps) * next_point
            + (1 - k_between / point_steps) * prev_point
        )

    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    MPC_H = 10
    MPC_dt = 0.1

    T_WGT = 10.0
    R_WGT = 10.0
    V_WGT = 1e-2
    U_WGT = 1e-3
    C_WGT = 0.16

    model = ssqpy.model.Model(
        MPC_H,
        MPC_dt,
        from_preset=ssqpy.model.Preset.Panda,
        solver_mode=dynamics_mode,
    )

    nq = model.getnq()
    nv = model.getnv()

    init_state = np.zeros(nq + nv)
    init_state[:nq] = [
        0.0013,
        -0.3281,
        -0.0010,
        -2.1711,
        -0.0008,
        1.7431,
        0.7862,
        0.0,
        0.0,
    ]

    pose_costs = []

    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, init_state[:nq], C_WGT
    )

    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)

    for k in range(MPC_H):
        pose_costs.append(
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                model, "panda_ee", traj[k], Rd, T_WGT, R_WGT
            )
        )

        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, config_cost)
        model.addCost(k, pose_costs[-1])

    pose_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, "panda_ee", traj[MPC_H], Rd, T_WGT, R_WGT
        )
    )

    model.addCost(MPC_H, v_cost)
    model.addCost(MPC_H, pose_costs[-1])

    def update_pose_costs(current_stage):
        for k in range(current_stage, current_stage + MPC_H):
            pose_costs[k - current_stage].set_desired_translation(traj[k % H])

    return model, update_pose_costs
