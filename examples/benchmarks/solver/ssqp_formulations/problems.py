import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np


def pendubot(use_hpipm=False):
    H = 2500
    dt = 0.001

    V_WGT = 1e-4
    U_WGT = 1e-4
    C_WGT = 1.0

    DESIRED_CONFIG = np.array((np.pi, 0.0))

    model = ssqpy.model.Model(
        H,
        dt,
        urdf_path="urdf/double_pendulum/pendubot.urdf",
        actuated_joints=[0],
        solver_mode=ssqpy.model.SolverMode.ForwardDynamics,
    )

    vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, DESIRED_CONFIG, C_WGT
    )

    for k in range(H):
        model.addCost(k, vel_cost)
        model.addCost(k, u_cost)

        if k > H - 500:
            model.addCost(k, config_cost)

    model.finalize(terminal_cost=config_cost)

    ssqp_params = ssqpy.solvers.ssqpParams()
    ssqp_params.tolerance = 1e-2

    if use_hpipm:
        ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
        hpipm_params = ssqpy.solvers.hpipmParams()
        hpipm_params.tol_comp = 1e-3
        hpipm_params.tol_stat = 1e-3
        hpipm_params.tol_eq = 1e-3
        hpipm_params.tol_ineq = 1e-3
        ssqp_params.hpipmParams = hpipm_params
    else:
        admm_params = ssqpy.solvers.admmParams()
        admm_params.max_iter = 100
        admm_params.abs_tolerance = 1e-3
        admm_params.rel_tolerance = 1e-3
        admm_params.adaptive_rho_tolerance = 1
        admm_params.rho_update_iter_freq = 25
        admm_params.warm_start = True
        ssqp_params.admmParams = admm_params

    solver = ssqpy.solvers.ssqp(model, ssqp_params)

    solver.solve()

    solver_data_dir = "ssqp_hpipm" if use_hpipm else "ssqp_admm"
    solver.write(f"data/{solver_data_dir}/pendubot.json")


def go2_standing(use_hpipm=False):
    # Model Parameters
    H = 100
    dt = 0.01

    # Weights
    A_WGT = 10.0
    V_WGT = 1e-2
    U_WGT = 1e-4
    F_WGT = 1e-5

    MAX_NORMAL_FORCE = 50
    FOOT_EQ_TOL = 0.01

    # Initial and final positions
    base_pos = np.array([0.0, 0.0, 0.3879793820122])
    base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

    # Initial configuration
    init = np.zeros(37)
    init[:3] = base_pos
    init[6] = 1.0
    init[7:19] = base_config

    # Create an SSQP model for the Go2
    model = ssqpy.model.Model(
        H,
        dt,
        urdf_path="urdf/go2/go2.urdf",
        free_flyer=True,
    )

    # Define regularization costs
    config_cost = costs.SquaredConfigurationErrorCost(
        model, base_config, A_WGT, "Configuration cost"
    )
    joint_vel_cost = costs.SquaredJointVelocityCost(
        model, V_WGT, "Velocity cost"
    )
    control_cost = costs.SquaredControlCost(model, U_WGT, "Control cost")
    force_cost = costs.SquaredForceCost(model, F_WGT, "Force cost")

    # Define velocity bounds for feet
    # Used in conjunction with contact constraints
    fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model,
        "FL_foot",
        np.zeros(3),
        np.zeros(3),
        name="FL foot velocity bound",
    )

    fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model,
        "FR_foot",
        np.zeros(3),
        np.zeros(3),
        name="FR foot velocity bound",
    )

    rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model,
        "RL_foot",
        np.zeros(3),
        np.zeros(3),
        name="RL foot velocity bound",
    )

    rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model,
        "RR_foot",
        np.zeros(3),
        np.zeros(3),
        name="RR foot velocity bound",
    )

    for k in range(H):
        # Add regularization costs
        model.addCost(k, config_cost)
        model.addCost(k, joint_vel_cost)
        model.addCost(k, control_cost)
        model.addCost(k, force_cost)

        # Add a standing phase (4 feet on the ground)
        model.addContactConstraint(
            k,
            "FL_foot",
            name="FL foot contact constraint",
            max_normal_force=MAX_NORMAL_FORCE,
            eq_tol=FOOT_EQ_TOL,
        )
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(
            k,
            "FR_foot",
            name="FR foot contact constraint",
            max_normal_force=MAX_NORMAL_FORCE,
            eq_tol=FOOT_EQ_TOL,
        )
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(
            k,
            "RL_foot",
            name="RL foot contact constraint",
            max_normal_force=MAX_NORMAL_FORCE,
            eq_tol=FOOT_EQ_TOL,
        )
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(
            k,
            "RR_foot",
            name="RR foot contact constraint",
            max_normal_force=MAX_NORMAL_FORCE,
            eq_tol=FOOT_EQ_TOL,
        )
        model.addConstraints(k, rr_foot_vel_bound)

    model.addCost(H, config_cost)
    model.addConstraints(
        H,
        constraints.FrameLinearVelocityBound(
            model, "base", np.zeros(3), np.zeros(3)
        ),
    )

    # Finalize the model with our state and controls
    model.finalize(init_states=init)

    ssqp_params = ssqpy.solvers.ssqpParams()
    ssqp_params.tolerance = 1e-2

    if use_hpipm:
        ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
        hpipm_params = ssqpy.solvers.hpipmParams()
        hpipm_params.tol_stat = 1e-3
        hpipm_params.tol_comp = 1e-3
        hpipm_params.tol_eq = 1e-3
        hpipm_params.tol_ineq = 1e-3
        ssqp_params.hpipmParams = hpipm_params
    else:
        admm_params = ssqpy.solvers.admmParams()
        admm_params.abs_tolerance = 1e-3
        admm_params.rel_tolerance = 1e-3
        admm_params.max_iter = 100
        admm_params.warm_start = True
        ssqp_params.admmParams = admm_params

    # Create a solver with the model
    solver = ssqpy.solvers.ssqp(model, ssqp_params)

    solver.solve()

    solver_data_dir = "ssqp_hpipm" if use_hpipm else "ssqp_admm"
    solver.write(f"data/{solver_data_dir}/go2_standing.json")
