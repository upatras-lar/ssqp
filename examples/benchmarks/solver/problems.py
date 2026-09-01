import numpy as np


def make_pendubot_problem(problem_wrapper_t, constructor_parameters):
    problem_wrapper = problem_wrapper_t(
        2500,
        0.001,
        "urdf/double_pendulum/pendubot.urdf",
        constructor_parameters,
    )

    V_WGT = 1e-4
    U_WGT = 1e-4
    C_WGT = 1.0

    DESIRED_CONFIG = np.array((np.pi, 0.0))

    for k in range(problem_wrapper.H):
        problem_wrapper.addSquaredJointVelocityCost(k, V_WGT)
        problem_wrapper.addSquaredControlCost(k, U_WGT)

        if k > 4 * problem_wrapper.H // 5:
            problem_wrapper.addSquaredConfigurationErrorCost(
                k, DESIRED_CONFIG, C_WGT
            )

    problem_wrapper.addSquaredConfigurationErrorCost(
        problem_wrapper.H, DESIRED_CONFIG, C_WGT
    )

    return problem_wrapper


def make_go2_standing_problem(problem_wrapper_t, constructor_parameters):
    problem_wrapper = problem_wrapper_t(
        100, 0.01, "urdf/go2/go2.urdf", constructor_parameters
    )

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

    for k in range(problem_wrapper.H):
        problem_wrapper.addSquaredConfigurationErrorCost(k, base_config, A_WGT)
        problem_wrapper.addSquaredJointVelocityCost(k, V_WGT)
        problem_wrapper.addSquaredControlCost(k, U_WGT)
        problem_wrapper.addSquaredForceCost(k, F_WGT)

        problem_wrapper.addContactConstraint(
            k, "FL_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "FL_foot", np.zeros(3), np.zeros(3)
        )

        problem_wrapper.addContactConstraint(
            k, "FR_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "FR_foot", np.zeros(3), np.zeros(3)
        )

        problem_wrapper.addContactConstraint(
            k, "RL_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "RL_foot", np.zeros(3), np.zeros(3)
        )

        problem_wrapper.addContactConstraint(
            k, "RR_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "RR_foot", np.zeros(3), np.zeros(3)
        )

    problem_wrapper.addSquaredConfigurationErrorCost(
        problem_wrapper.H, base_config, A_WGT
    )
    problem_wrapper.addFrameLinearVelocityBound(
        problem_wrapper.H, "base", np.zeros(3), np.zeros(3)
    )

    problem_wrapper.finalize({"init_states": init})

    return problem_wrapper


def make_go2_jump_problem(problem_wrapper_t, constructor_parameters):
    problem_wrapper = problem_wrapper_t(
        150, 0.01, "urdf/go2/go2.urdf", constructor_parameters
    )

    START_STEP = 60
    END_STEP = 90

    A_WGT = 1e-3
    V_WGT = 1e-2
    U_WGT = 1e-4
    F_WGT = 1e-3
    C_WGT = 1e-3

    MAX_NORMAL_FORCE = 100
    FOOT_EQ_TOL = 0.01

    # Initial and final positions
    base_pos = np.array([0.0, 0.0, 0.3879793820122])
    base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

    # Initial configuration
    init = np.zeros(37)
    init[:3] = base_pos
    init[6] = 1.0
    init[7:19] = base_config

    for k in range(problem_wrapper.H):
        # Regularization costs
        problem_wrapper.addSquaredJointAngleCost(k, A_WGT)
        problem_wrapper.addSquaredJointVelocityCost(k, V_WGT)
        problem_wrapper.addSquaredControlCost(k, U_WGT)
        problem_wrapper.addSquaredForceCost(k, F_WGT)
        problem_wrapper.addSquaredConfigurationErrorCost(k, base_config, C_WGT)

        problem_wrapper.addFrameTranslationBound(
            k,
            "FL_foot",
            np.array((-np.inf, 0.1, -0.01)),
            np.array((np.inf, np.inf, np.inf)),
        )
        problem_wrapper.addFrameTranslationBound(
            k,
            "FR_foot",
            np.array((-np.inf, -np.inf, -0.01)),
            np.array((np.inf, -0.1, np.inf)),
        )
        problem_wrapper.addFrameTranslationBound(
            k,
            "RL_foot",
            np.array((-np.inf, 0.1, -0.01)),
            np.array((np.inf, np.inf, np.inf)),
        )
        problem_wrapper.addFrameTranslationBound(
            k,
            "RR_foot",
            np.array((-np.inf, -np.inf, -0.01)),
            np.array((np.inf, -0.1, np.inf)),
        )

        problem_wrapper.addFrameTranslationBound(
            k,
            "Head_lower",
            np.array((-np.inf, -np.inf, 0.1)),
            np.array((np.inf, np.inf, np.inf)),
        )

        if k < START_STEP or k >= END_STEP:
            problem_wrapper.addContactConstraint(
                k, "FR_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
            )
            problem_wrapper.addFrameLinearVelocityBound(
                k, "FR_foot", np.zeros(3), np.zeros(3)
            )

            problem_wrapper.addContactConstraint(
                k, "RL_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
            )
            problem_wrapper.addFrameLinearVelocityBound(
                k, "RL_foot", np.zeros(3), np.zeros(3)
            )

            problem_wrapper.addContactConstraint(
                k, "RR_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
            )
            problem_wrapper.addFrameLinearVelocityBound(
                k, "RR_foot", np.zeros(3), np.zeros(3)
            )

    problem_wrapper.addSquaredJointVelocityCost(problem_wrapper.H, V_WGT)

    problem_wrapper.addFrameLinearVelocityBound(
        problem_wrapper.H, "base", np.zeros(3), np.zeros(3)
    )
    problem_wrapper.addFrameTranslationBound(
        problem_wrapper.H,
        "Head_lower",
        np.array((-np.inf, -np.inf, 0.1)),
        np.array((np.inf, np.inf, np.inf)),
    )

    problem_wrapper.addFrameTranslationBound(
        problem_wrapper.H,
        "FL_foot",
        np.array((0.1934, 0.142, -0.00002682)),
        np.array((0.1934, 0.142, -0.00002682)),
    )
    problem_wrapper.addFrameTranslationBound(
        problem_wrapper.H,
        "FR_foot",
        np.array((0.1934, -0.142, -0.00002682)),
        np.array((0.1934, -0.142, -0.00002682)),
    )
    problem_wrapper.addFrameTranslationBound(
        problem_wrapper.H,
        "RL_foot",
        np.array((-0.1934, 0.142, -0.00002682)),
        np.array((-0.1934, 0.142, -0.00002682)),
    )
    problem_wrapper.addFrameTranslationBound(
        problem_wrapper.H,
        "RR_foot",
        np.array((-0.1934, -0.142, -0.00002682)),
        np.array((-0.1934, -0.142, -0.00002682)),
    )

    problem_wrapper.finalize({"init_states": init})

    return problem_wrapper
