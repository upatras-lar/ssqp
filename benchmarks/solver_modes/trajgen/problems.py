"""
Centroidal Voronoi Tessellation (CVT) sampler for n-DOF robot state space.
State: (q1, ..., qn, dq1, ..., dqn)
Domain: [q_lo, q_hi] x [v_lo, v_hi] per dimension (not necessarily symmetric).
Lloyd relaxation via Monte Carlo centroid estimation.
"""

import ssqpy
from ssqpy.model import costs, constraints
import numpy as np
from math import pi

from trajgen.python.utils.GaitManager import GaitManager


def cvt(
    n_generators: int,
    q_lo: np.ndarray = [-np.pi, -np.pi],
    q_hi: np.ndarray = [np.pi, np.pi],
    v_lo: np.ndarray = [-25.0, -25.0],
    v_hi: np.ndarray = [25.0, 25.0],
    n_iterations: int = 200,
    n_samples_per_iter: int = 10_000,
    seed: int = 42,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Run CVT Lloyd relaxation in a 2*nq dimensional state space.

    Parameters
    ----------
    n_generators      : number of CVT generators (output configurations)
    q_lo, q_hi        : (nq,) position lower / upper bounds
    v_lo, v_hi        : (nv,) velocity lower / upper bounds
    n_iterations      : max Lloyd iterations
    n_samples_per_iter: Monte Carlo samples per iteration
    seed              : RNG seed
    tol               : convergence threshold on max generator displacement

    Returns
    -------
    generators : (n_generators, 2*nq) array — columns [q1..qn, dq1..dqn]
    """
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    v_lo = np.asarray(v_lo, dtype=float)
    v_hi = np.asarray(v_hi, dtype=float)
    assert q_lo.shape == q_hi.shape == v_lo.shape == v_hi.shape

    lo = np.concatenate([q_lo, v_lo])
    hi = np.concatenate([q_hi, v_hi])
    ndim = len(lo)
    span = hi - lo

    rng = np.random.default_rng(seed)
    generators = rng.uniform(0.0, 1.0, size=(n_generators, ndim))

    for iteration in range(n_iterations):
        samples = rng.uniform(0.0, 1.0, size=(n_samples_per_iter, ndim))

        diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        assignments = np.argmin(np.sum(diff**2, axis=-1), axis=-1)

        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=np.int64)
        np.add.at(new_generators, assignments, samples)
        np.add.at(counts, assignments, 1)

        nonempty = counts > 0
        new_generators[nonempty] /= counts[nonempty, np.newaxis]
        new_generators[~nonempty] = generators[~nonempty]

        displacement = np.max(
            np.linalg.norm(new_generators - generators, axis=-1)
        )
        generators = new_generators

        if displacement < tol:
            print(
                f"Converged at iteration {iteration + 1} (disp={displacement:.2e})"
            )
            break
    else:
        print(f"Max iterations ({n_iterations}), disp={displacement:.2e}")

    return generators * span + lo


def control_cost(model, dynamics_mode, weight, k=0):
    # ForwardDynamics controls are torques, so SquaredControlCost already
    # penalizes torque. InverseDynamics/ReducedInverseDynamics controls are
    # accelerations, so use SquaredIDTorqueCost to penalize the torque they
    # imply instead, with the same weight. `k` only matters when the model
    # has forces at that stage; construct one cost per stage (not a single
    # shared instance) if forces vary across the stages it's added to
    if dynamics_mode == ssqpy.model.SolverMode.ForwardDynamics:
        return ssqpy.model.costs.SquaredControlCost(model, weight)
    return ssqpy.model.costs.SquaredIDTorqueCost(model, weight, k=k)


def generate_pendubot_problems(n, horizon, dynamics_mode):
    V_WGT = 1e-3
    U_WGT = 1e-4
    TIP_WGT = 5.0
    C_WGT = 4.0

    dt = 0.01

    problems = []

    init_states = cvt(n, v_lo=[-1.0, -1.0], v_hi=[1.0, 1.0])

    for i in range(n):
        model = ssqpy.model.Model(
            horizon,
            dt,
            urdf_path="../../../examples/assets/urdf/double_pendulum/pendubot.urdf",
            actuated_joints=[0],
            solver_mode=dynamics_mode,
        )

        nq = model.getnq()
        nv = model.getnv()
        nu = model.getnu()

        vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
        u_cost = control_cost(model, dynamics_mode, U_WGT)
        tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
            model, "tip", np.array((0.0, 0.0, 0.1)), TIP_WGT
        )
        config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
            model, np.array((np.pi, 0.0)), C_WGT
        )

        for k in range(horizon):
            model.addCost(k, vel_cost)
            model.addCost(k, u_cost)
            # model.addCost(k, tip_cost)

            if k > 100:
                model.addCost(k, config_cost)

        model.addCost(horizon, vel_cost)
        # model.addCost(horizon, tip_cost)
        model.addCost(horizon, config_cost)

        model.finalize(init_states=init_states[i])

        problems.append(model)

    return problems


def generate_panda_feasible_region_problems(
    n_problems, dynamics_mode, seed=42
):
    nq = 9
    dt = 0.01

    rng = np.random.default_rng(seed)

    H = 400

    T_WGT = 20.0
    R_WGT = 10.0
    V_WGT = 1e-1
    U_WGT = 1e-3

    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    # Start EEF position
    start_pos = np.array([0.0, 0.0, 0.4])

    init_state = np.zeros(2 * nq)
    init_state[:nq] = [
        -0.6155932121557595,
        -1.0844467142837122,
        -0.12995999754492005,
        -2.8648662500071036,
        -0.11678214046488461,
        1.7779025205888452,
        0.13315038657733308,
        4.72139930268247e-06,
        3.344468750010948e-05,
    ]

    problems = []

    for i in range(n_problems):

        obstacle_problem = ssqpy.model.Model(
            H,
            dt,
            urdf_path="../../../examples/assets/urdf/panda/panda.urdf",
            solver_mode=dynamics_mode,
        )

        random_offset = rng.uniform(-0.3, 0.3, size=3)
        final_goal = start_pos + random_offset

        pose_cost = ssqpy.model.costs.FrameSquaredPoseErrorCost(
            obstacle_problem,
            "panda_ee",
            final_goal,
            Rd,
            T_WGT,
            R_WGT,
        )

        rot_cost = ssqpy.model.costs.FrameSquaredRotationErrorCost(
            obstacle_problem, "panda_ee", Rd, R_WGT
        )

        joint_vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(
            obstacle_problem, V_WGT
        )
        u_cost = control_cost(obstacle_problem, dynamics_mode, U_WGT)

        # -----------------------------
        # Random obstacle along line
        # -----------------------------
        # Sample location along segment (avoid endpoints)
        alpha = rng.uniform(0.3, 0.7)
        obstacle_center = (1 - alpha) * start_pos + alpha * final_goal

        # Small box half-size (5–10 cm)
        half_size = rng.uniform(0.05, 0.1)

        obstacle_lower = obstacle_center - half_size
        obstacle_upper = obstacle_center + half_size

        obstacle_box = ssqpy.model.constraints.FrameTranslationBound(
            obstacle_problem, "panda_ee", obstacle_lower, obstacle_upper
        )

        # -----------------------------
        # Add costs & constraints
        # -----------------------------
        for k in range(H):
            obstacle_problem.addCost(k, rot_cost)
            obstacle_problem.addCost(k, joint_vel_cost)
            obstacle_problem.addCost(k, u_cost)

            if 50 < k < H - 50:
                obstacle_problem.addConstraints(k, obstacle_box)

        obstacle_problem.addCost(H, pose_cost)
        obstacle_problem.addCost(H, joint_vel_cost)

        obstacle_problem.finalize(init_states=init_state)

        problems.append(obstacle_problem)

    return problems


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
    U_WGT = 1e-4
    F_WGT = 1e-5
    TF_WGT = 10.0
    ROT_WGT = 5.0

    GAIT = GaitManager.setWalk
    CYCLE_STEPS = 60
    FINAL_STANDING_STEPS = 10

    MAX_NORMAL_FORCE = 80
    FOOT_EQ_TOL = 0.01

    z_height = 0.3879793820122

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
            urdf_path="../../../examples/assets/urdf/go2/go2.urdf",
            free_flyer=True,
            solver_mode=dynamics_mode,
        )

        # -------------------------
        # Costs
        # -------------------------
        config_cost = costs.SquaredConfigurationErrorCost(
            model, init[7:19], C_WGT
        )
        joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
        force_cost = costs.SquaredForceCost(model, F_WGT)

        for k in range(HORIZON):
            model.addCost(k, joint_vel_cost)
            model.addCost(k, force_cost)

        for k in range(HORIZON + 1):
            model.addCost(k, config_cost)

        # Strong terminal velocity regularization
        model.addCost(
            HORIZON,
            costs.SquaredJointVelocityCost(model, 1e3),
        )

        # Final translation target
        model.addCost(
            HORIZON,
            costs.FrameSquaredPoseErrorCost(
                model, "base", final_pos, final_R, TF_WGT, ROT_WGT
            ),
        )

        # Final base velocity constraint
        model.addConstraints(
            HORIZON,
            constraints.FrameLinearVelocityBound(
                model, "base", np.zeros(3), np.zeros(3)
            ),
        )

        # -------------------------
        # Gait
        # -------------------------
        init_controls = []

        gait_manager = GaitManager(
            model,
            nq,
            nv,
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            max_normal_force=MAX_NORMAL_FORCE,
            contact_equality_tolerance=FOOT_EQ_TOL,
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

        # Constructed per-stage (not shared) since the gait's contact
        # forces vary by stage, and control_cost snapshots the model's
        # forces at that stage when it builds a SquaredIDTorqueCost
        for k in range(HORIZON):
            model.addCost(k, control_cost(model, dynamics_mode, U_WGT, k))

        model.finalize(init_states=init)

        problems.append(model)

    return problems


def generate_go2_multi_jump_problems(dynamics_mode):
    HORIZON = 340
    dt = 0.01

    A_WGT = 1e-5
    V_WGT = 1e-3
    U_WGT = 1e-4
    F_WGT = 1e-4
    TF_WGT = 1e-1
    RF_WGT = 1e-1

    MAX_FORCE_NORMAL = 147

    # On-air step ranges for each of the 5 jumps
    JUMP_STEPS = [
        (50, 70),
        (100, 120),
        (150, 180),
        (210, 240),
        (270, 300),
    ]

    # Which feet to land on in between the jumps above
    LAND_SEQ = ["B", "R", "L", "F"]

    base_pos = np.array([0.0, 0.0, 0.3879793820122])
    final_pos = np.array([0.0, 0.0, 0.3879793820122])

    init = np.zeros(37)
    init[:3] = base_pos
    init[6] = 1.0
    init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

    model = ssqpy.model.Model(
        HORIZON,
        dt,
        urdf_path="../../../examples/assets/urdf/go2/go2.urdf",
        free_flyer=True,
        solver_mode=dynamics_mode,
    )

    joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
    joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
    force_cost = costs.SquaredForceCost(model, F_WGT)

    fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model, "FL_foot", np.zeros(3), np.zeros(3)
    )
    fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model, "FR_foot", np.zeros(3), np.zeros(3)
    )
    rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model, "RL_foot", np.zeros(3), np.zeros(3)
    )
    rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
        model, "RR_foot", np.zeros(3), np.zeros(3)
    )

    # Position bounds keeping on-air feet above the ground, and front/back
    # feet from getting too close to each other
    fl_foot_pos_bound = constraints.FrameTranslationBound(
        model,
        "FL_foot",
        np.array((-np.inf, 0.1, -0.01)),
        np.array((np.inf, np.inf, np.inf)),
    )
    fr_foot_pos_bound = constraints.FrameTranslationBound(
        model,
        "FR_foot",
        np.array((-np.inf, -np.inf, -0.01)),
        np.array((np.inf, -0.1, np.inf)),
    )
    rl_foot_pos_bound = constraints.FrameTranslationBound(
        model,
        "RL_foot",
        np.array((-np.inf, 0.1, -0.01)),
        np.array((np.inf, np.inf, np.inf)),
    )
    rr_foot_pos_bound = constraints.FrameTranslationBound(
        model,
        "RR_foot",
        np.array((-np.inf, -np.inf, -0.01)),
        np.array((np.inf, -0.1, np.inf)),
    )

    head_pos_bound = constraints.FrameTranslationBound(
        model,
        "Head_lower",
        np.array((-np.inf, -np.inf, 0.01)),
        np.array((np.inf, np.inf, np.inf)),
    )
    base_pos_bound = constraints.FrameTranslationBound(
        model,
        "base",
        np.array((-np.inf, -np.inf, 0.2)),
        np.array((np.inf, np.inf, np.inf)),
    )

    for k in range(HORIZON):
        model.addCost(k, joint_angle_cost)
        model.addCost(k, joint_vel_cost)
        model.addCost(k, force_cost)

        model.addConstraints(k, head_pos_bound)
        model.addConstraints(k, fl_foot_pos_bound)
        model.addConstraints(k, fr_foot_pos_bound)
        model.addConstraints(k, rl_foot_pos_bound)
        model.addConstraints(k, rr_foot_pos_bound)
        model.addConstraints(k, base_pos_bound)

        if k < JUMP_STEPS[0][0] or k >= JUMP_STEPS[-1][-1]:
            # Standing phase: all 4 feet on the ground
            model.addContactConstraint(
                k, "FL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fl_foot_vel_bound)
            model.addContactConstraint(
                k, "FR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fr_foot_vel_bound)
            model.addContactConstraint(
                k, "RL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rl_foot_vel_bound)
            model.addContactConstraint(
                k, "RR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rr_foot_vel_bound)
            continue

        # Determine whether this step is on-air, or landing after jump `i`
        on_air = True
        for i, jmp_range in enumerate(JUMP_STEPS):
            if k < jmp_range[0]:
                on_air = False
                break
            if k < jmp_range[1]:
                break

        if not on_air:
            land = LAND_SEQ[i - 1]

            if land in ("B", "L", "RL"):
                model.addContactConstraint(
                    k, "RL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
                )
                model.addConstraints(k, rl_foot_vel_bound)
            if land in ("B", "R", "RR"):
                model.addContactConstraint(
                    k, "RR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
                )
                model.addConstraints(k, rr_foot_vel_bound)
            if land in ("F", "L", "FL"):
                model.addContactConstraint(
                    k, "FL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
                )
                model.addConstraints(k, fl_foot_vel_bound)
            if land in ("F", "R", "FR"):
                model.addContactConstraint(
                    k, "FR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
                )
                model.addConstraints(k, fr_foot_vel_bound)

    model.addCost(HORIZON, joint_vel_cost)
    model.addCost(
        HORIZON,
        costs.FrameSquaredPoseErrorCost(
            model, "base", final_pos, np.eye(3), TF_WGT, RF_WGT
        ),
    )
    model.addConstraints(
        HORIZON,
        constraints.FrameLinearVelocityBound(
            model, "base", np.zeros(3), np.zeros(3)
        ),
    )

    # Constructed per-stage (not shared) since the jump sequence's contact
    # forces vary heavily across the horizon (4 feet down to none while
    # airborne), and control_cost snapshots the model's forces at that stage
    for k in range(HORIZON):
        model.addCost(k, control_cost(model, dynamics_mode, U_WGT, k))

    model.finalize(init_states=init)

    return [model]


def generate_g1_jump_problems(dynamics_mode):
    H = 200
    dt = 0.01

    A_WGT = 1e-2
    V_WGT = 1e-2
    U_WGT = 1e-4
    F_WGT = 1e-4
    T_BASE_WGT = 1.0
    R_BASE_WGT = 1.0

    MAX_NORMAL_FORCE = 160

    JUMP_START = 80
    JUMP_END = 120

    model = ssqpy.model.Model(
        H,
        dt,
        urdf_path="../../../examples/assets/urdf/g1/g1_23dof.urdf",
        free_flyer=True,
        solver_mode=dynamics_mode,
    )

    base_pos = np.array((0.0, 0.0, 0.78686375))

    init = np.zeros(model.getnq() + model.getnv())
    init[:3] = base_pos
    init[6] = 1.0

    joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
    joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
    force_cost = costs.SquaredForceCost(model, F_WGT)

    foot_frames = [
        "left_mm",
        "left_mp",
        "left_pp",
        "left_pm",
        "right_mm",
        "right_mp",
        "right_pp",
        "right_pm",
    ]
    foot_vel_bounds = {
        frame: constraints.FrameLinearVelocityBound(
            model, frame, np.zeros(3), np.zeros(3)
        )
        for frame in foot_frames
    }

    for k in range(H):
        model.addCost(k, joint_angle_cost)
        model.addCost(k, joint_vel_cost)
        model.addCost(k, force_cost)

        model.addCost(
            k,
            costs.FrameSquaredPoseErrorCost(
                model, "torso_link", base_pos, np.eye(3), T_BASE_WGT, R_BASE_WGT
            ),
        )

        if k < JUMP_START or k > JUMP_END:
            # Standing phase: both feet (4 points each) on the ground
            for frame in foot_frames:
                model.addContactConstraint(
                    k, frame, eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
                )
                model.addConstraints(k, foot_vel_bounds[frame])

    model.addCost(H, joint_angle_cost)
    model.addCost(H, joint_vel_cost)

    model.addConstraints(
        H,
        constraints.FrameLinearVelocityBound(
            model, "torso_link", np.zeros(3), np.zeros(3)
        ),
    )

    # Constructed per-stage (not shared) since contact forces vary (8
    # contacts while standing, none while airborne)
    for k in range(H):
        model.addCost(k, control_cost(model, dynamics_mode, U_WGT, k))

    model.finalize(init_states=init)

    return [model]
