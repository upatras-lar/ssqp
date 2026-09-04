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

    A_WGT = 100.0
    V_WGT = 1e-4
    U_WGT = 1e-8
    F_WGT = 1e-10

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

    A_WGT = 1e-6
    V_WGT = 1e-4
    U_WGT = 1e-8
    F_WGT = 1e-6
    C_WGT = 1e-6

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


def make_panda_circle_problem(problem_wrapper_t, constructor_parameters):
    """Ported from examples/trajgen/python/panda_circle.py's hand-written
    ssqpy script: the Panda arm's end effector traces a circular path on
    the yz-plane (x held constant) while its orientation is held fixed,
    and its position is additionally kept inside a rectangular box nested
    within that circle. Fixed-base and fully actuated, so unlike
    pendubot/go2 it needs no special constructor_parameters.
    """
    H = 400
    problem_wrapper = problem_wrapper_t(
        H, 0.01, "urdf/panda/panda.urdf", constructor_parameters
    )

    T_WGT = 9.0
    R_WGT = 1.0
    V_WGT = 9e-6
    # The original trajgen script used SolverMode.InverseDynamics, where u
    # is joint acceleration; this benchmark forces ForwardDynamics, where
    # u is joint torque. U_WGT is retuned for that scale rather than
    # reused verbatim -- at the original 3e-3, torque cost dominated
    # tracking cost and every backend converged to "barely move" instead
    # of the circle.
    U_WGT = 9e-10

    # Box constraint on the end effector's y/z position, nested within the
    # circle it's meant to trace.
    CONSTRAINT_VLINES = (-0.15, 0.15)
    CONSTRAINT_HLINES = (0.1, 0.41)
    box_lb = np.array((-np.inf, CONSTRAINT_VLINES[0], CONSTRAINT_HLINES[0]))
    box_ub = np.array((np.inf, CONSTRAINT_VLINES[1], CONSTRAINT_HLINES[1]))

    # Circle center/radius for the traced end effector path (yz-plane).
    CX, CY, CZ, CR = 0.3, 0.0, 0.26, 0.22
    omega = 2 * np.pi / (H * problem_wrapper.dt)

    traj = [
        np.array(
            [
                CX,
                CY + CR * np.sin(omega * k * problem_wrapper.dt),
                CZ + CR * np.cos(omega * k * problem_wrapper.dt),
            ]
        )
        for k in range(H + 1)
    ]

    # Desired end effector orientation, held fixed throughout the motion.
    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    # Nominal arm configuration (7 arm joints + 2 fingers), also used as
    # the sole initial state.
    init = np.zeros(18)
    init[:9] = [
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

    for k in range(H):
        problem_wrapper.addSquaredFramePoseErrorCost(
            k, "panda_ee", traj[k], Rd, T_WGT, R_WGT
        )
        problem_wrapper.addSquaredJointVelocityCost(k, V_WGT)
        problem_wrapper.addSquaredControlCost(k, U_WGT)
        problem_wrapper.addFrameTranslationBound(k, "panda_ee", box_lb, box_ub)

    problem_wrapper.addSquaredFramePoseErrorCost(
        H, "panda_ee", traj[H], Rd, T_WGT, R_WGT
    )
    problem_wrapper.addSquaredJointVelocityCost(H, V_WGT)
    problem_wrapper.addFrameTranslationBound(H, "panda_ee", box_lb, box_ub)

    problem_wrapper.finalize({"init_states": init})

    return problem_wrapper


def make_go2_frog_problem(problem_wrapper_t, constructor_parameters):
    """Ported from examples/trajgen/python/go2_frog.py's hand-written ssqpy
    script onto the generalized ProblemWrapper interface. Unlike
    make_go2_jump_problem's single synchronized flight phase, this is a
    longer (3s) motion made of four successive hops with staggered,
    per-leg liftoff/landing and brief full-flight windows, ending 2m
    forward of the start.

    Two aspects of the original script have no equivalent in the
    ProblemWrapper interface and are deliberately dropped rather than
    approximated, to keep this problem solvable identically by every
    backend:
      - a loose per-step bound on the base's pitch/roll (originally via a
        rotation-bound primitive the interface doesn't expose), and
      - a soft cost pulling each foot's height toward the ground when not
        otherwise constrained (originally a frame-translation cost with a
        z-only weight, likewise not expressible through this interface).
    """
    H = 300
    problem_wrapper = problem_wrapper_t(
        H, 0.01, "urdf/go2/go2.urdf", constructor_parameters
    )

    A_WGT = 1e-14
    V_WGT = 1e-8
    U_WGT = 2.5e-9
    F_WGT = 2.5e-7

    MAX_NORMAL_FORCE = 100
    FOOT_EQ_TOL = 0.01

    FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    # Contact phase schedule: (start_step, end_step, feet_in_contact).
    # Four successive hop cycles with staggered per-leg liftoff, each
    # separated by a brief full-flight window (empty feet tuple).
    CONTACT_SCHEDULE = [
        (0, 24, ("FL_foot", "FR_foot", "RL_foot", "RR_foot")),
        (24, 30, ("FR_foot", "RL_foot", "RR_foot")),
        (30, 45, ("RL_foot", "RR_foot")),
        (45, 54, ()),
        (54, 69, ("FL_foot", "FR_foot")),
        (69, 81, ("FL_foot", "FR_foot", "RL_foot", "RR_foot")),
        (81, 87, ("FR_foot", "RL_foot", "RR_foot")),
        (87, 96, ("RL_foot", "RR_foot")),
        (96, 111, ()),
        (111, 117, ("FL_foot",)),
        (117, 129, ("FL_foot", "FR_foot")),
        (129, 141, ("FL_foot", "FR_foot", "RR_foot")),
        (141, 174, ("FL_foot", "FR_foot", "RL_foot", "RR_foot")),
        (174, 177, ("FR_foot", "RL_foot", "RR_foot")),
        (177, 183, ("RL_foot", "RR_foot")),
        (183, 204, ()),
        (204, 210, ("FR_foot",)),
        (210, 213, ("FL_foot", "FR_foot")),
        (213, 228, ("FL_foot", "FR_foot", "RL_foot", "RR_foot")),
        (228, 234, ("FL_foot", "RR_foot")),
        (234, 240, ()),
        (240, 243, ("FR_foot",)),
        (243, 249, ("FR_foot", "RL_foot")),
        (249, 252, ()),
        (252, 300, ("FL_foot", "FR_foot", "RL_foot", "RR_foot")),
    ]

    def feet_in_contact(k):
        for start, end, feet in CONTACT_SCHEDULE:
            if start <= k < end:
                return feet
        return ()

    # Initial/nominal standing configuration -- same stance as
    # make_go2_standing_problem/make_go2_jump_problem.
    base_pos = np.array([0.0, 0.0, 0.3879793820122])
    base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

    init = np.zeros(37)
    init[:3] = base_pos
    init[6] = 1.0
    init[7:19] = base_config

    # Target: land 2m forward of the start, back at the nominal standing
    # height, at rest.
    final_pos = np.array([2.0, 0.0, base_pos[2]])

    for k in range(H):
        problem_wrapper.addSquaredJointAngleCost(k, A_WGT)
        problem_wrapper.addSquaredJointVelocityCost(k, V_WGT)
        problem_wrapper.addSquaredControlCost(k, U_WGT)
        problem_wrapper.addSquaredForceCost(k, F_WGT)

        for foot in FEET:
            problem_wrapper.addFrameTranslationBound(
                k,
                foot,
                np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
                np.array((np.inf, np.inf, np.inf)),
            )

        for foot in feet_in_contact(k):
            problem_wrapper.addContactConstraint(
                k, foot, MAX_NORMAL_FORCE, FOOT_EQ_TOL
            )
            problem_wrapper.addFrameLinearVelocityBound(
                k, foot, np.zeros(3), np.zeros(3)
            )

    problem_wrapper.addSquaredJointAngleCost(H, A_WGT)
    problem_wrapper.addSquaredJointVelocityCost(H, V_WGT)

    problem_wrapper.addFrameTranslationBound(H, "base", final_pos, final_pos)
    problem_wrapper.addFrameLinearVelocityBound(
        H, "base", np.zeros(3), np.zeros(3)
    )

    problem_wrapper.finalize({"init_states": init})

    return problem_wrapper
