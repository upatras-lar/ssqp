import ssqpy
import numpy as np

ssqpy.setSilentMode()

# Keep MPC_DT a multiple of the controller's command period, so that plan
# stages line up with commands
MPC_DT = 0.02
MPC_H = 10

T_WGT = 1.0
R_WGT = 1.0
C_WGT = 1e-2
V_WGT = 1e-3
U_WGT = 1e-5
F_WGT = 2e-4

CONTACT_MU = 0.6

# Speed limit on stance feet, keeping the plan from swinging planted legs
STANCE_VEL_TOL = 0.2

# Plan stages per swing phase
GAIT_PERIOD = 10

# Time from measuring the state until the plan solved from it is applied. The
# state is predicted this far ahead before solving, which makes up for the
# solve time. Solves should finish within it, since the walk tolerates little
# more delay than this
PREDICT_TIME = 0.015

# Time spent on all four feet before trotting, settling into the posture
START_TIME = 1.0
START_STAGES = round(START_TIME / MPC_DT)

# Default step height, and the range and rate of change for adjusting it with
# the D-pad
STEP_HEIGHT = 0.05
MIN_STEP_HEIGHT = 0.03
MAX_STEP_HEIGHT = 0.1
STEP_HEIGHT_RATE = 0.05

# Limits on the commanded velocity, and on how quickly the command may change
MAX_VEL_FRONT = 0.3
MAX_VEL_BACK = 0.3
MAX_VEL_SIDE = 0.2
MAX_VEL_YAW = 0.6
MAX_ACC = 0.6
MAX_YAW_ACC = 1.0

# Default base height, and the range and rate of change for adjusting it with
# the D-pad
BASE_HEIGHT = 0.34
MIN_BASE_HEIGHT = 0.28
MAX_BASE_HEIGHT = 0.38
HEIGHT_RATE = 0.05

# Joint gains around the planned joint positions and velocities, on top of the
# planned torques
JOINT_KP = 0.01
JOINT_KD = 2 * np.sqrt(JOINT_KP)

THIGH_LENGTH = 0.213
# Distance from the knee to the foot frame
CALF_LENGTH = 0.23648

FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Trot pairs, swinging in alternate gait periods
SWING_PAIRS = [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]]


def leg_config(height):
    # Hip, thigh and calf angles placing the foot straight below the hip
    calf = -np.arccos(
        (height**2 - THIGH_LENGTH**2 - CALF_LENGTH**2)
        / (2 * THIGH_LENGTH * CALF_LENGTH)
    )
    thigh = np.arcsin(-CALF_LENGTH * np.sin(calf) / height)
    return np.array([0.0, thigh, calf])


def rotation_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_rotation(xyzw):
    x, y, z, w = xyzw
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def rotation_to_quat(R):
    # Returns the xyzw quaternion of a rotation matrix
    w = 0.5 * np.sqrt(max(1.0 + np.trace(R), 1e-12))
    if w > 0.1:
        return np.array(
            [
                (R[2, 1] - R[1, 2]) / (4 * w),
                (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w),
                w,
            ]
        )

    # Near a half turn, extract the quaternion from the largest diagonal term
    i = np.argmax(np.diag(R))
    j, k = (i + 1) % 3, (i + 2) % 3
    q = np.zeros(4)
    q[i] = 0.5 * np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
    q[j] = (R[j, i] + R[i, j]) / (4 * q[i])
    q[k] = (R[k, i] + R[i, k]) / (4 * q[i])
    q[3] = (R[k, j] - R[j, k]) / (4 * q[i])
    return q


def get_controller(init_config):
    """
    Builds the dynamics walking MPC for the robot starting at init_config

    The robot only measures the base orientation, not its position, so the
    base position is estimated from the stance feet (leg odometry), assuming
    they stay where they landed on flat ground at zero height. The yaw is
    taken relative to the initial one. The base position in the given
    configurations is ignored

    Returns the model, the MPC, and a function to call once per plan stage,
    before solving. It takes the commanded velocity (forward, sideways and
    yaw), the D-pad state (x and y), and the measured configuration and
    velocity. The base linear velocity is estimated from the stance feet as
    well, so the one in the given velocity is ignored. It returns the
    estimated state. Predict it to when the plan will be applied before solving
    (see predict_state), and play the plan back from then (see plan_commands
    and playback)
    """
    model = ssqpy.model.Model(
        MPC_H,
        MPC_DT,
        mjcf_path="../../assets/mjcf/go2/go2.xml",
        free_flyer=True,
        solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
    )

    nq = model.getnq()

    # Foot positions relative to the base at the nominal posture, on the
    # ground
    nominal_config = np.zeros(nq)
    nominal_config[6] = 1.0
    nominal_config[7:] = np.tile(leg_config(BASE_HEIGHT), 4)
    foot_offsets = {
        name: model.forwardKinematics(nominal_config, name)[0]
        for name in FOOT_NAMES
    }
    for offset in foot_offsets.values():
        offset[2] = 0.0

    initial_yaw = np.arctan2(*quat_to_rotation(init_config[3:7])[[1, 0], 0])

    # Estimation, command and reference state, shared with the update
    # function
    est = {
        "stage": 0,
        "v_cmd": np.zeros(3),
        "base_height": BASE_HEIGHT,
        "step_height": STEP_HEIGHT,
        # Reference base poses (x, y and yaw) for the stages from pose_stage
        # onwards, extrapolated from the current command
        "pose_stage": 0,
        "poses": [np.zeros(3)],
        # Where each foot last landed
        "landed": {},
    }

    def feet_in_base(config):
        # Foot positions relative to the base, in the base frame
        local = config[:nq].copy()
        local[:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return {
            name: model.forwardKinematics(local, name)[0]
            for name in FOOT_NAMES
        }

    def estimate_state(config, vel, stance_feet):
        # Replaces the base pose and linear velocity with the leg odometry
        # estimate
        R = rotation_z(-initial_yaw) @ quat_to_rotation(config[3:7])
        feet = feet_in_base(config)

        base_pos = np.mean(
            [est["landed"][name] - R @ feet[name] for name in stance_feet],
            axis=0,
        )

        # Stance feet stay still, so the base moves opposite to how they move
        # relative to it, through the joints and the base's rotation
        step = 1e-4
        moved = config[:nq].copy()
        moved[7:] += step * vel[6:]
        feet_moved = feet_in_base(moved)
        base_vel = -np.mean(
            [
                (feet_moved[name] - feet[name]) / step
                + np.cross(vel[3:6], feet[name])
                for name in stance_feet
            ],
            axis=0,
        )

        state = np.hstack(
            (base_pos, rotation_to_quat(R), config[7:nq], base_vel, vel[3:])
        )
        return state, R, feet

    # Start with every foot on the ground where it currently is
    R0 = rotation_z(-initial_yaw) @ quat_to_rotation(init_config[3:7])
    feet0 = feet_in_base(init_config)
    height0 = -np.mean([(R0 @ p)[2] for p in feet0.values()])
    for name in FOOT_NAMES:
        est["landed"][name] = np.array([0.0, 0.0, height0]) + R0 @ feet0[name]
        est["landed"][name][2] = 0.0

    init_state = np.zeros(nq + model.getnv())
    init_state[:nq] = init_config[:nq]
    init_state[:3] = [0.0, 0.0, height0]
    init_state[3:7] = rotation_to_quat(R0)

    def predict_poses(pose, count):
        v_cmd = est["v_cmd"]
        predicted = [pose]
        for _ in range(count):
            x, y, yaw = predicted[-1]
            c, s = np.cos(yaw), np.sin(yaw)
            predicted.append(
                np.array(
                    [
                        x + (c * v_cmd[0] - s * v_cmd[1]) * MPC_DT,
                        y + (s * v_cmd[0] + c * v_cmd[1]) * MPC_DT,
                        yaw + v_cmd[2] * MPC_DT,
                    ]
                )
            )
        return predicted

    def foothold(name, period):
        # Where the foot lands at the end of its swing in the given period,
        # placed so that the base passes over the foot's nominal position
        # mid-stance
        stage = START_STAGES - est["pose_stage"]
        if stage + (period + 1) * GAIT_PERIOD <= 0:
            return est["landed"][name]
        x, y, yaw = est["poses"][stage + int((period + 1.5) * GAIT_PERIOD)]
        return np.array([x, y, 0.0]) + rotation_z(yaw) @ foot_offsets[name]

    def foot_reference(name, k):
        # Returns the foot's reference position at stage k and whether it
        # swings
        if k < START_STAGES:
            return est["landed"][name], False

        period, phase = divmod(k - START_STAGES, GAIT_PERIOD)
        phase /= GAIT_PERIOD

        if name not in SWING_PAIRS[period % 2]:
            return foothold(name, period - 1), False

        # Ease in and out of the swing, so that the foot lifts off and touches
        # down without sliding
        blend = 0.5 * (1 - np.cos(np.pi * phase))
        p = (1 - blend) * foothold(name, period - 2) + blend * foothold(
            name, period
        )
        p[2] += est["step_height"] * np.sin(np.pi * phase) ** 2
        return p, True

    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, nominal_config[7:], C_WGT
    )
    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
    f_cost = ssqpy.model.costs.SquaredForceCost(model, F_WGT)

    base_costs = []
    foot_funcs = {name: [] for name in FOOT_NAMES}

    for k in range(MPC_H):
        model.addCost(k, config_cost)
        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, f_cost)

        base_costs.append(
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                model, "base_link", init_state[:3], np.eye(3), T_WGT, R_WGT
            )
        )
        model.addCost(k, base_costs[-1])

        for name in FOOT_NAMES:
            foot_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
                model, name, est["landed"][name], T_WGT
            )
            foot_constraint = model.addContactConstraint(
                k, name, CONTACT_MU, eq_tol=0.02, max_normal_force=150
            )

            model.addCost(k, foot_cost)

            # The contact constraint lets stance feet move, so bound their
            # velocity as well. The first stage's velocity is measured, so it
            # is left free
            vel_bound = None
            if k > 0:
                vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
                    model,
                    name,
                    np.full(3, -STANCE_VEL_TOL),
                    np.full(3, STANCE_VEL_TOL),
                )
                model.addConstraints(k, vel_bound)

            foot_funcs[name].append((foot_cost, foot_constraint, vel_bound))

    def update_references():
        current_stage = est["stage"]

        # Look far enough ahead for the footholds of every swing in the
        # horizon
        est["poses"] = predict_poses(
            est["poses"][current_stage - est["pose_stage"]],
            MPC_H + 3 * GAIT_PERIOD,
        )
        est["pose_stage"] = current_stage

        config_cost.set_desired_configuration(
            np.tile(leg_config(est["base_height"]), 4)
        )

        for idx in range(MPC_H):
            k = current_stage + idx
            x, y, yaw = est["poses"][idx]

            base_costs[idx].set_desired_translation(
                np.array([x, y, est["base_height"]])
            )
            base_costs[idx].set_desired_rotation(rotation_z(yaw))

            for name in FOOT_NAMES:
                foot_pos, swinging = foot_reference(name, k)
                foot_cost, foot_constraint, vel_bound = foot_funcs[name][idx]

                # Swinging feet track their swing trajectory, while stance
                # feet stay in contact. The contact constraint only bounds the
                # foot's height, so the tracking cost also keeps stance feet
                # from sliding
                foot_constraint.set_enabled(not swinging)
                if vel_bound is not None:
                    vel_bound.set_enabled(not swinging)
                foot_cost.set_desired_translation(foot_pos)

    update_references()

    model.addCost(MPC_H, v_cost)

    model.finalize(init_states=[init_state])

    ssqp_params = ssqpy.solvers.ssqpParams()
    ssqp_params.tolerance = 1e-2
    ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

    hpipm_params = ssqpy.solvers.hpipmParams()
    hpipm_params.tol_stat = 1e-4
    hpipm_params.tol_comp = 1e-4
    hpipm_params.tol_eq = 1e-4
    hpipm_params.tol_ineq = 1e-4
    hpipm_params.prediction_correction = True
    ssqp_params.hpipmParams = hpipm_params

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=2, qp_iters=100)

    def update_func(v_cmd, dpad_x, dpad_y, config, vel):
        current_stage = est["stage"]
        gait_stage = current_stage - START_STAGES
        period = gait_stage // GAIT_PERIOD

        # Stay in place until trotting
        if gait_stage < 0:
            v_cmd = np.zeros(3)

        max_change = np.array([MAX_ACC, MAX_ACC, MAX_YAW_ACC]) * MPC_DT
        est["v_cmd"] += np.clip(
            np.array(v_cmd) - est["v_cmd"], -max_change, max_change
        )

        # The D-pad reports -1 while held up
        est["base_height"] = np.clip(
            est["base_height"] - dpad_y * HEIGHT_RATE * MPC_DT,
            MIN_BASE_HEIGHT,
            MAX_BASE_HEIGHT,
        )
        est["step_height"] = np.clip(
            est["step_height"] + dpad_x * STEP_HEIGHT_RATE * MPC_DT,
            MIN_STEP_HEIGHT,
            MAX_STEP_HEIGHT,
        )

        # Estimate the base from the feet that were in stance during the last
        # stage, which are all still on the ground
        stance_feet = FOOT_NAMES
        if gait_stage > 0:
            last_swing = SWING_PAIRS[((gait_stage - 1) // GAIT_PERIOD) % 2]
            stance_feet = [
                name for name in FOOT_NAMES if name not in last_swing
            ]
        state, R, feet = estimate_state(config, vel, stance_feet)

        # Feet finishing their swing keep the spot they actually landed on
        if gait_stage > 0 and gait_stage % GAIT_PERIOD == 0:
            for name in SWING_PAIRS[(period - 1) % 2]:
                est["landed"][name] = state[:3] + R @ feet[name]
                est["landed"][name][2] = 0.0

        update_references()
        est["stage"] += 1

        return state

    return model, mpc, update_func


def plan_commands(model, x, u, stages=3):
    """
    Returns the plan's states at the start of each of its first stages (and at
    the end of the last one), and the joint torques of its first stage
    """
    states = [np.array(x.stage(k)) for k in range(stages + 1)]
    torques = model.getActuatedTorqueVector(states[0], u.stage(0), 0)
    return states, torques


def plan_state(states, elapsed):
    # Interpolates the plan's states at the given time since the plan started
    progress = min(max(elapsed / MPC_DT, 0.0), len(states) - 1 - 1e-6)
    k = int(progress)
    blend = progress - k

    # q and -q are the same rotation, so blend toward the one nearer the
    # current stage's
    after = states[k + 1].copy()
    if states[k][3:7] @ after[3:7] < 0:
        after[3:7] *= -1

    state = (1 - blend) * states[k] + blend * after
    state[3:7] /= np.linalg.norm(state[3:7])
    return state


def predict_state(state, commands, start, end):
    """
    Predicts how the given state changes between two times since the plan in
    commands started, by applying the change the plan predicts over them
    """
    states, _ = commands
    before = plan_state(states, start)
    after = plan_state(states, end)

    predicted = state + after - before

    # Rotate the base by the plan's rotation in between, in the base frame
    rotation = quat_to_rotation(before[3:7]).T @ quat_to_rotation(after[3:7])
    predicted[3:7] = rotation_to_quat(quat_to_rotation(state[3:7]) @ rotation)
    return predicted


def playback(commands, elapsed):
    """
    Returns the joint position and velocity targets for the given time since
    the plan started, interpolated along the plan, and the joint torques of the
    plan's first stage, which stay applied until the next plan
    """
    states, torques = commands
    state = plan_state(states, elapsed)
    return state[7:19], state[25:37], torques
