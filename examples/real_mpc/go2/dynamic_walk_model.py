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
# Weight on the base's roll and pitch rates, which keeps each plan from
# correcting the whole base tilt error within one stage
W_WGT = 0.1

CONTACT_MU = 0.6
# Limit on each foot's normal force
MAX_FOOT_FORCE = 150.0

# Speed limit on stance feet, keeping the plan from swinging planted legs
STANCE_VEL_TOL = 0.2

# Plan stages per swing phase
GAIT_PERIOD = 12

# Time from measuring the state until the plan solved from it is applied. The
# state is predicted this far ahead before solving, which makes up for the
# solve time. Solves should finish within it, since the walk tolerates little
# more delay than this
PREDICT_TIME = 0.015

# Solve time limit, leaving room within PREDICT_TIME for the work around the
# solve. SQP iterations that would end past it are skipped
MAX_SOLVE_TIME = 0.012

# The robot stands on all four feet until it is commanded to move, and goes
# back to standing once the command has been at rest for IDLE_TIME. Commands
# below MOVE_THRESHOLD count as at rest. Stepping starts GAIT_START_STAGES
# after the move command, which gives the plan time to shift its weight
IDLE_TIME = 1.0
IDLE_STAGES = round(IDLE_TIME / MPC_DT)
MOVE_THRESHOLD = 0.02
GAIT_START_STAGES = 3

# Limits on the commanded velocity, and on how quickly the command may change
MAX_VEL_FRONT = 0.5
MAX_VEL_BACK = 0.3
MAX_VEL_SIDE = 0.2
MAX_VEL_YAW = 1.0
MAX_ACC = 0.6
MAX_YAW_ACC = 1.0

# Limit on the leg odometry base linear velocity, when used without the
# filter. It spikes when a stance foot slips or loses contact, and the plan
# follows such spikes
MAX_BASE_VEL = 0.6

# Whether the base position and linear velocity come from a Kalman filter,
# which integrates the accelerometer and corrects it with the leg odometry of
# every state message, rather than from the leg odometry alone
USE_FILTER = True
# Noise of the accelerometer (m/s^2), and of the leg odometry base velocity
# (m/s) and position (m) from each stance foot
FILTER_ACC_STD = 0.1
FILTER_VEL_STD = 0.1
FILTER_POS_STD = 0.02
# Stance feet whose odometry is further than this many standard deviations
# from the prediction are ignored, since they are slipping or not yet down
FILTER_GATE = 4.0
GRAVITY = np.array([0.0, 0.0, -9.81])

# Default base height, and the range and rate of change for adjusting it with
# the D-pad
BASE_HEIGHT = 0.34
MIN_BASE_HEIGHT = 0.28
MAX_BASE_HEIGHT = 0.38
HEIGHT_RATE = 0.05

# Default step height, and the range and rate of change for adjusting it with
# the D-pad
STEP_HEIGHT = 0.08
MIN_STEP_HEIGHT = 0.03
MAX_STEP_HEIGHT = 0.1
STEP_HEIGHT_RATE = 0.05

# Base pitch while tilting the front up or down, and the rate at which it
# changes, including back to level once the tilt stops
MAX_BASE_PITCH = 0.2
PITCH_RATE = 0.3

# Base roll while tilting the base to the left or right, and the rate at which
# it changes, including back to level once the tilt stops
MAX_BASE_ROLL = 0.5
ROLL_RATE = 0.5

# Base yaw relative to the walking direction while turning the front to the
# left or right, with the feet staying in place, and the rate at which it
# changes, including back to straight once the turn stops
MAX_BASE_YAW = 0.3
YAW_RATE = 0.5

# Small jump in place, started by pressing the jump button while standing
# level with the command at rest. The base crouches, pushes off, flies with all
# feet in the air, absorbs the landing and rises back to its height over
# JUMP_STAGES, while move and tilt commands are ignored. The phases are in plan
# stages. The push depth follows from the flight time, and the landing depth
# from how fast the base falls at touchdown. JUMP_ENABLED turns it on
JUMP_ENABLED = True
JUMP_STAGES = 120
JUMP_CROUCH_STAGES = 22
JUMP_PUSH_STAGES = 10
JUMP_FLIGHT_STAGES = 16
JUMP_LAND_STAGES = 12
# Base height at takeoff and touchdown
JUMP_TAKEOFF_HEIGHT = 0.34
# Weight holding the legs in their takeoff configuration while in the air,
# where the feet and the base's path can no longer be steered. The base's
# rotation is left free in the air as well, since the plan could only steer it
# by swinging the legs
JUMP_CONFIG_WGT = 100.0
# How far the feet pull up toward the base halfway through the flight, before
# they stretch back out for the landing
JUMP_TUCK = 0.04
# Weight on the joint velocities while in the air, which holds the legs still
# rather than letting them carry on with their motion from the push
JUMP_FLIGHT_VEL_WGT = 1e-2
# Limit on each foot's normal force over the landing, which has to stop the
# fall on whichever feet touch down first
JUMP_LAND_FORCE = 90.0
# With foot contacts measured, the landing also starts once a foot touches
# down rather than on schedule. Feet still in the air then keep swinging over
# the plan's next JUMP_TOUCH_STAGES, until they touch down as well
JUMP_TOUCH_STAGES = 0
# Stages after every foot has landed before the feet step back under the hips
JUMP_TROT_DELAY = 26

JUMP_SPEED = -GRAVITY[2] * JUMP_FLIGHT_STAGES * MPC_DT / 2
JUMP_CROUCH_HEIGHT = (
    JUMP_TAKEOFF_HEIGHT - JUMP_SPEED * JUMP_PUSH_STAGES * MPC_DT / 2
)
# Stages of the jump where the flight and the landing start
JUMP_TAKEOFF_STAGE = JUMP_CROUCH_STAGES + JUMP_PUSH_STAGES
JUMP_LAND_STAGE = JUMP_TAKEOFF_STAGE + JUMP_FLIGHT_STAGES

# Joint gains around the planned joint positions and velocities, on top of the
# planned torques
JOINT_KP = 6.0
JOINT_KD = 2 * np.sqrt(JOINT_KP)
# Joint gains in the unitree_mujoco simulator, which holds each command's PD
# torque until the next command, where the robot's gains make the legs shake
SIM_JOINT_KP = 0.1
SIM_JOINT_KD = 2 * np.sqrt(SIM_JOINT_KP)

# Whether the joint torques are recomputed every command from the measured
# state, rather than held from the plan's first stage
TICK_TORQUES = True
# Gains pulling the planned joint accelerations toward the planned joint
# positions and velocities, when recomputing the torques
TICK_KP = 25.0
TICK_KD = 10.0

# Rotor inertia each joint adds through its gearing, which the model leaves
# out. The torques add it for the planned joint accelerations
JOINT_ARMATURE = 0.03
# Rotor inertia compensated in the unitree_mujoco simulator, where the
# compensation makes the legs miss their plan with the simulator's soft gains
SIM_JOINT_ARMATURE = 0.0

# Motor torque limits (hip, thigh and calf of each leg)
TORQUE_LIMITS = np.tile([23.7, 23.7, 45.43], 4)

# Plan stages kept for playback
PLAN_STAGES = 3

THIGH_LENGTH = 0.213
# Distance from the knee to the foot frame
CALF_LENGTH = 0.23648

FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Gaits, as the groups of feet that take turns to swing for a gait period each
# and the share of the velocity limits they walk at. The walk lifts one foot at
# a time, so it steps slower
GAITS = {
    "trot": ([["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]], 1.0),
    "walk": ([["RL_foot"], ["FL_foot"], ["RR_foot"], ["FR_foot"]], 0.5),
}
# Gait at the start. The gait button cycles through them while standing
GAIT = "trot"


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


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def base_rotation(pitch, roll, yaw):
    # Base orientation relative to the walking direction
    return rotation_z(yaw) @ rotation_y(-pitch) @ rotation_x(roll)


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


def get_controller(init_config, simulation=False):
    """
    Builds the dynamics walking MPC for the robot starting at init_config,
    or for the unitree_mujoco simulator with simulation set

    The robot only measures the base orientation, not its position, so the
    base position is estimated from the stance feet (leg odometry), assuming
    they stay where they landed on flat ground at zero height. With
    USE_FILTER, a Kalman filter fuses this with the accelerometer. The yaw is
    taken relative to the initial one. The base position in the given
    configurations is ignored

    The robot stands on all four feet until the command moves it, and stands
    again once the command has been at rest for IDLE_TIME

    Returns the model, the MPC, and four functions:
     - update_func, to call once per plan stage, before solving. It takes the
       commanded velocity (forward, sideways and yaw), the D-pad state (x and
       y), the base orientation commands (pitch, roll and yaw, each 1 or -1
       while tilting the front up, tilting to the right or turning the front
       to the left, or the other way, and 0 otherwise), whether the jump
       button is held, the measured configuration and velocity, and
       optionally whether each foot is in contact (in FOOT_NAMES order). The
       base linear velocity is estimated from the stance feet as well, so the
       one in the given velocity is ignored. It returns the estimated state.
       Predict it to when the plan will be applied before solving (see
       predict_state)
     - plan_commands, taking the solved states and controls and returning the
       commands to play the plan back with (see playback)
     - command_torques, taking the commands, the time since the plan started
       and the measured configuration and velocity, and returning the joint
       torques to apply
     - filter_state, to call with every state message when USE_FILTER is
       set. It takes the measured configuration and velocity, the
       accelerometer reading (in the base frame) and the time in seconds
    """
    model = ssqpy.model.Model(
        MPC_H,
        MPC_DT,
        mjcf_path="../../assets/mjcf/go2/go2.xml",
        free_flyer=True,
        solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
    )

    nq = model.getnq()
    armature = SIM_JOINT_ARMATURE if simulation else JOINT_ARMATURE

    # Foot positions relative to the base at the nominal configuration, on
    # the ground
    nominal_config = np.zeros(nq)
    nominal_config[6] = 1.0
    nominal_config[7:] = np.tile(leg_config(BASE_HEIGHT), 4)
    foot_offsets = {
        name: model.forwardKinematics(nominal_config, name)[0]
        for name in FOOT_NAMES
    }
    for offset in foot_offsets.values():
        offset[2] = 0.0

    # Hip joint positions relative to the base, and the sideways offset from
    # each hip joint to its thigh joint
    hip_offsets = {
        name: model.forwardKinematics(
            nominal_config, name.replace("foot", "hip")
        )[0]
        for name in FOOT_NAMES
    }
    hip_length = abs(
        model.forwardKinematics(nominal_config, "FL_thigh")[0][1]
        - hip_offsets["FL_foot"][1]
    )

    def leg_ik(foot, side):
        # Hip, thigh and calf angles placing the foot at the given position
        # relative to the hip joint, in the base frame. side is 1 for the left
        # legs and -1 for the right ones. Feet out of reach get a straight leg
        # pointing at them
        x, y, z = foot
        length = np.sqrt(y**2 + z**2 - hip_length**2)
        hip = np.arctan2(z, y) - np.arctan2(-length, side * hip_length)
        hip = (hip + np.pi) % (2 * np.pi) - np.pi
        calf = -np.arccos(
            np.clip(
                (x**2 + length**2 - THIGH_LENGTH**2 - CALF_LENGTH**2)
                / (2 * THIGH_LENGTH * CALF_LENGTH),
                -1.0,
                1.0,
            )
        )
        thigh = np.arctan2(-x, length) - np.arctan2(
            CALF_LENGTH * np.sin(calf),
            THIGH_LENGTH + CALF_LENGTH * np.cos(calf),
        )
        return np.array([hip, thigh, calf])

    def stance_config(height, pitch, roll, yaw):
        # Joint angles keeping each foot on its nominal spot on the ground,
        # with the base at the given height and orientation
        R = base_rotation(pitch, roll, yaw)
        base = np.array([0.0, 0.0, height])
        return np.hstack(
            [
                leg_ik(
                    R.T @ (foot_offsets[name] - base) - hip_offsets[name],
                    1 if foot_offsets[name][1] > 0 else -1,
                )
                for name in FOOT_NAMES
            ]
        )

    initial_yaw = np.arctan2(*quat_to_rotation(init_config[3:7])[[1, 0], 0])

    # Estimation, command and reference state, shared with the update
    # function
    est = {
        "stage": 0,
        "v_cmd": np.zeros(3),
        "base_height": BASE_HEIGHT,
        "step_height": STEP_HEIGHT,
        # Positive pitch tilts the front of the base up, positive roll tilts
        # the base to the right, and positive yaw turns the front to the left
        "base_pitch": 0.0,
        "base_roll": 0.0,
        "base_yaw": 0.0,
        # Reference base poses (x, y and yaw) for the stages from pose_stage
        # onwards, extrapolated from the current command
        "pose_stage": 0,
        "poses": [np.zeros(3)],
        # Where each foot last landed
        "landed": {},
        # Current gait
        "gait": GAIT,
        "gait_held": False,
        # Stages where the gait starts and stops, None while standing or
        # while stepping without a planned stop
        "gait_start": None,
        "gait_stop": None,
        # Stages the command has been at rest for
        "idle_stages": 0,
        # First stage of the jump, None when not jumping
        "jump_start": None,
        "jump_held": False,
        # Base height and vertical velocity at touchdown, which the landing
        # starts from. Until the feet touch down, the planned ones
        "touchdown": None,
        # Feet still in the air once the landing has started, which swing
        # until the stage before touch_stage
        "falling_feet": set(),
        "touch_stage": 0,
    }

    def feet_in_base(config, kinematics=model):
        # Foot positions relative to the base, in the base frame
        local = config[:nq].copy()
        local[:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return {
            name: kinematics.forwardKinematics(local, name)[0]
            for name in FOOT_NAMES
        }

    def leg_odometry(config, vel, stance_feet, kinematics=model):
        # The base position and linear velocity (in the base frame) each
        # stance foot implies, if it stays where it landed
        R = rotation_z(-initial_yaw) @ quat_to_rotation(config[3:7])
        feet = feet_in_base(config, kinematics)

        positions = {
            name: est["landed"][name] - R @ feet[name] for name in stance_feet
        }

        # Stance feet stay still, so the base moves opposite to how they move
        # relative to it, through the joints and the base's rotation
        step = 1e-4
        moved = config[:nq].copy()
        moved[7:] += step * vel[6:]
        feet_moved = feet_in_base(moved, kinematics)
        velocities = {
            name: -(feet_moved[name] - feet[name]) / step
            - np.cross(vel[3:6], feet[name])
            for name in stance_feet
        }
        return R, feet, positions, velocities

    def estimate_state(config, vel, stance_feet):
        # Replaces the base pose and linear velocity with the filter's
        # estimate, or the leg odometry one
        R, feet, positions, velocities = leg_odometry(config, vel, stance_feet)
        if USE_FILTER:
            base_pos = est["filter"][1][:3]
            base_vel = R.T @ est["filter"][1][3:]
        else:
            base_pos = np.mean(list(positions.values()), axis=0)
            base_vel = np.mean(list(velocities.values()), axis=0)
            speed = np.linalg.norm(base_vel)
            if speed > MAX_BASE_VEL:
                base_vel *= MAX_BASE_VEL / speed

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

    # Filter time, estimate (base position and linear velocity, in the
    # yaw-aligned frame) and covariance, and the feet it takes as stance
    est["filter"] = (
        None,
        np.array([0.0, 0.0, height0, 0.0, 0.0, 0.0]),
        1e-4 * np.eye(6),
    )
    est["stance_feet"] = FOOT_NAMES

    # The filter runs alongside the solver, so it uses its own model
    filter_model = ssqpy.model.Model(
        1,
        MPC_DT,
        mjcf_path="../../assets/mjcf/go2/go2.xml",
        free_flyer=True,
        solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
    )

    def filter_state(config, vel, acc, t):
        last_t, x, P = est["filter"]
        reset = est.pop("filter_reset", None)
        if reset is not None:
            x = x.copy()
            x[2] = reset
        stance_feet = est["stance_feet"]
        R, _, positions, velocities = leg_odometry(
            config, vel, stance_feet, filter_model
        )

        # Predict with the accelerometer, which measures the base
        # acceleration minus gravity
        if last_t is not None:
            dt = min(max(t - last_t, 0.0), 0.05)
            a = R @ acc + GRAVITY
            x = x.copy()
            x[:3] += dt * x[3:] + 0.5 * dt**2 * a
            x[3:] += dt * a
            F = np.eye(6)
            F[:3, 3:] = dt * np.eye(3)
            Q = np.zeros((6, 6))
            Q[:3, :3] = dt**3 / 3 * np.eye(3)
            Q[:3, 3:] = Q[3:, :3] = dt**2 / 2 * np.eye(3)
            Q[3:, 3:] = dt * np.eye(3)
            P = F @ P @ F.T + FILTER_ACC_STD**2 * Q

        # Correct with the odometry of each stance foot that agrees with the
        # prediction
        rows, residuals, stds = [], [], []
        for name in stance_feet:
            for offset, value, std in (
                (0, positions[name], FILTER_POS_STD),
                (3, R @ velocities[name], FILTER_VEL_STD),
            ):
                residual = value - x[offset : offset + 3]
                S = P[offset : offset + 3, offset : offset + 3]
                S = S + std**2 * np.eye(3)
                if residual @ np.linalg.solve(S, residual) <= FILTER_GATE**2:
                    rows.append(offset)
                    residuals.append(residual)
                    stds.append(std)
        if rows:
            H = np.zeros((3 * len(rows), 6))
            for i, offset in enumerate(rows):
                H[3 * i : 3 * i + 3, offset : offset + 3] = np.eye(3)
            S = H @ P @ H.T + np.diag(np.repeat(np.square(stds), 3))
            K = np.linalg.solve(S, H @ P).T
            x = x + K @ np.concatenate(residuals)
            P = (np.eye(6) - K @ H) @ P

        est["filter"] = (t, x, P)

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

    def jump_stage(k):
        # Stage k's index within the jump, or None outside of it
        if est["jump_start"] is None:
            return None
        j = k - est["jump_start"]
        return j if 0 <= j < JUMP_STAGES else None

    def jump_base_height(j):
        # Base reference height at stage j of the jump
        height = est["base_height"]
        push = JUMP_CROUCH_STAGES
        flight = JUMP_TAKEOFF_STAGE
        land = JUMP_LAND_STAGE
        recover = land + JUMP_LAND_STAGES

        if j < push:
            blend = 0.5 * (1 - np.cos(np.pi * j / JUMP_CROUCH_STAGES))
            return (1 - blend) * height + blend * JUMP_CROUCH_HEIGHT
        if j < flight:
            t = (j - push) * MPC_DT
            acc = JUMP_SPEED / (JUMP_PUSH_STAGES * MPC_DT)
            return JUMP_CROUCH_HEIGHT + 0.5 * acc * t**2
        if j < land:
            t = (j - flight) * MPC_DT
            rise = JUMP_SPEED * t + 0.5 * GRAVITY[2] * t**2
            return JUMP_TAKEOFF_HEIGHT + rise
        # The landing stops the base's fall evenly, from where it touched down
        land_height, land_speed = est["touchdown"]
        duration = JUMP_LAND_STAGES * MPC_DT
        if j < recover:
            t = (j - land) * MPC_DT
            acc = -land_speed / duration
            return land_height + land_speed * t + 0.5 * acc * t**2
        phase = (j - recover) / (JUMP_STAGES - recover)
        blend = 0.5 * (1 - np.cos(np.pi * phase))
        landed = land_height + land_speed * duration / 2
        return (1 - blend) * landed + blend * height

    def in_flight(k):
        # Whether all feet are in the air at stage k
        j = jump_stage(k)
        return j is not None and JUMP_TAKEOFF_STAGE <= j < JUMP_LAND_STAGE

    def falling(name, k):
        # Whether the foot is still coming down at stage k of the landing
        return name in est["falling_feet"] and k < est["touch_stage"]

    def swing_feet(k):
        # The feet swinging at stage k
        if in_flight(k):
            return FOOT_NAMES
        landing = [name for name in FOOT_NAMES if falling(name, k)]
        if landing:
            return landing
        start, stop = est["gait_start"], est["gait_stop"]
        if start is None or k < start or (stop is not None and k >= stop):
            return []
        groups = GAITS[est["gait"]][0]
        return groups[((k - start) // GAIT_PERIOD) % len(groups)]

    def last_swing(name, period):
        # The latest gait period up to the given one where the foot swings
        groups = GAITS[est["gait"]][0]
        group = next(i for i, g in enumerate(groups) if name in g)
        return period - (period - group) % len(groups)

    def foothold(name, period):
        # Where the foot lands at the end of its swing in the given period of
        # the gait, placed so that the base passes over the foot's nominal
        # position mid-stance. Feet that are yet to swing stay where they are
        stage = est["gait_start"] - est["pose_stage"]
        if period < 0 or stage + (period + 1) * GAIT_PERIOD <= 0:
            return est["landed"][name]
        stance = len(GAITS[est["gait"]][0]) - 1
        middle = int((period + 1 + stance / 2) * GAIT_PERIOD)
        x, y, yaw = est["poses"][stage + middle]
        return np.array([x, y, 0.0]) + rotation_z(yaw) @ foot_offsets[name]

    def final_foothold(name):
        # Where the foot stands once the gait stops, which is where its last
        # swing lands
        periods = (est["gait_stop"] - est["gait_start"]) // GAIT_PERIOD
        return foothold(name, last_swing(name, periods - 1))

    def foot_reference(name, k):
        # Returns the foot's reference position at stage k and whether it
        # swings
        if in_flight(k) or falling(name, k):
            return est["landed"][name], True
        start, stop = est["gait_start"], est["gait_stop"]
        if start is None or k < start:
            return est["landed"][name], False
        if stop is not None and k >= stop:
            return final_foothold(name), False

        period, phase = divmod(k - start, GAIT_PERIOD)
        phase /= GAIT_PERIOD

        groups = GAITS[est["gait"]][0]
        if name not in groups[period % len(groups)]:
            return foothold(name, last_swing(name, period)), False

        # Ease in and out of the swing, so that the foot lifts off and touches
        # down without sliding
        blend = 0.5 * (1 - np.cos(np.pi * phase))
        p = (1 - blend) * foothold(
            name, period - len(groups)
        ) + blend * foothold(name, period)
        p[2] += est["step_height"] * np.sin(np.pi * phase) ** 2
        return p, True

    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, nominal_config[7:], C_WGT
    )
    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
    f_cost = ssqpy.model.costs.SquaredForceCost(model, F_WGT)
    w_cost = ssqpy.model.costs.FloatingBaseSquaredVelocityErrorCost(
        model, np.zeros(6), np.array([0.0, 0.0, 0.0, W_WGT, W_WGT, 0.0])
    )

    # Leg configuration through each stage of the flight, pulling the feet in
    # and out smoothly
    flight_configs = [
        stance_config(
            JUMP_TAKEOFF_HEIGHT
            - JUMP_TUCK * np.sin(np.pi * i / JUMP_FLIGHT_STAGES) ** 2,
            0.0,
            0.0,
            0.0,
        )
        for i in range(JUMP_FLIGHT_STAGES)
    ]

    base_costs = []
    flight_costs = []
    flight_vel_costs = []
    foot_funcs = {name: [] for name in FOOT_NAMES}

    for k in range(MPC_H):
        model.addCost(k, config_cost)
        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, f_cost)
        model.addCost(k, w_cost)

        base_costs.append(
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                model, "base_link", init_state[:3], np.eye(3), T_WGT, R_WGT
            )
        )
        model.addCost(k, base_costs[-1])

        flight_costs.append(
            ssqpy.model.costs.SquaredConfigurationErrorCost(
                model,
                flight_configs[0],
                JUMP_CONFIG_WGT,
            )
        )
        model.addCost(k, flight_costs[-1])

        flight_vel_costs.append(
            ssqpy.model.costs.SquaredJointVelocityCost(
                model, JUMP_FLIGHT_VEL_WGT
            )
        )
        model.addCost(k, flight_vel_costs[-1])

        for name in FOOT_NAMES:
            foot_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
                model, name, est["landed"][name], T_WGT
            )
            foot_constraint = model.addContactConstraint(
                k,
                name,
                CONTACT_MU,
                eq_tol=0.02,
                max_normal_force=MAX_FOOT_FORCE,
                no_slip=False,
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
            MPC_H + (len(GAITS[est["gait"]][0]) + 1) * GAIT_PERIOD,
        )
        est["pose_stage"] = current_stage

        config_cost.set_desired_configuration(
            stance_config(
                est["base_height"],
                est["base_pitch"],
                est["base_roll"],
                est["base_yaw"],
            )
        )

        for idx in range(MPC_H):
            k = current_stage + idx
            x, y, yaw = est["poses"][idx]

            j = jump_stage(k)
            height = est["base_height"] if j is None else jump_base_height(j)
            base_costs[idx].set_desired_translation(np.array([x, y, height]))
            base_costs[idx].set_desired_rotation(
                rotation_z(yaw)
                @ base_rotation(
                    est["base_pitch"], est["base_roll"], est["base_yaw"]
                )
            )

            # In the air, the legs hold their configuration rather than
            # following the feet's references. Before takeoff, the base's
            # reference through the flight sets the takeoff speed, but once in
            # the air its path can no longer be steered
            flight = in_flight(k)
            airborne = flight and in_flight(current_stage)
            base_costs[idx].set_translation_weight(0.0 if airborne else T_WGT)
            base_costs[idx].set_rotation_weight(0.0 if airborne else R_WGT)
            flight_costs[idx].set_enabled(flight)
            if flight:
                flight_costs[idx].set_desired_configuration(
                    flight_configs[j - JUMP_TAKEOFF_STAGE]
                )
            flight_vel_costs[idx].set_enabled(flight)

            land = JUMP_LAND_STAGE
            landing = j is not None and land <= j < land + JUMP_LAND_STAGES
            max_force = JUMP_LAND_FORCE if landing else MAX_FOOT_FORCE

            for name in FOOT_NAMES:
                foot_pos, swinging = foot_reference(name, k)
                foot_cost, foot_constraint, vel_bound = foot_funcs[name][idx]
                foot_constraint.set_max_normal_force(max_force)

                # Swinging feet track their swing trajectory, while stance
                # feet stay in contact. The contact constraint only bounds the
                # foot's height, so the tracking cost also keeps stance feet
                # from sliding
                foot_constraint.set_enabled(not swinging)
                if vel_bound is not None:
                    vel_bound.set_enabled(not swinging)
                foot_cost.set_enabled(not flight)
                foot_cost.set_desired_translation(foot_pos)

    update_references()

    model.addCost(MPC_H, v_cost)

    model.finalize(init_states=[init_state])

    # The torques are computed alongside the solver, so they use their own
    # model, which is never solved
    torque_model = ssqpy.model.Model(
        1,
        MPC_DT,
        mjcf_path="../../assets/mjcf/go2/go2.xml",
        free_flyer=True,
        solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
    )
    for name in FOOT_NAMES:
        torque_model.addContactConstraint(
            0,
            name,
            CONTACT_MU,
            eq_tol=0.02,
            max_normal_force=150,
            no_slip=False,
        )
    torque_model.addCost(
        0, ssqpy.model.costs.SquaredControlCost(torque_model, U_WGT)
    )
    torque_model.addCost(
        1, ssqpy.model.costs.SquaredJointVelocityCost(torque_model, V_WGT)
    )
    torque_model.finalize(init_states=[init_state])
    nv = torque_model.getnv()

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

    mpc = ssqpy.solvers.MPC(
        model,
        ssqp_params,
        sqp_iters=2,
        qp_iters=100,
        max_time=MAX_SOLVE_TIME,
    )

    def update_func(
        v_cmd,
        dpad_x,
        dpad_y,
        orientation,
        jump,
        switch_gait,
        config,
        vel,
        contacts=None,
    ):
        current_stage = est["stage"]
        moving = np.max(np.abs(v_cmd)) > MOVE_THRESHOLD

        # Jump on each new press of the jump button while standing still with
        # the base level and facing straight, and ignore move and tilt
        # commands until the jump is over
        level = not np.any(orientation) and not any(
            est[key] for key in ["base_pitch", "base_roll", "base_yaw"]
        )
        if (
            est["jump_start"] is not None
            and current_stage >= est["jump_start"] + JUMP_STAGES
        ):
            est["jump_start"] = None
        if (
            JUMP_ENABLED
            and jump
            and not est["jump_held"]
            and est["jump_start"] is None
            and est["gait_start"] is None
            and not moving
            and not np.any(est["v_cmd"])
            and level
        ):
            est["jump_start"] = current_stage + 1
            est["touchdown"] = (JUMP_TAKEOFF_HEIGHT, -JUMP_SPEED)
            est["falling_feet"] = set()
        est["jump_held"] = jump

        # Start the landing as soon as a foot touches down, and follow the
        # other feet down as they touch down too. The base's height comes
        # from the accelerometer alone in the air, so it is reset from the
        # feet that touched down. Their joints have yet to move on impact,
        # so they are no good for the velocity
        j = jump_stage(current_stage)
        takeoff = JUMP_TAKEOFF_STAGE
        touchdown = JUMP_LAND_STAGE
        if contacts is not None and j is not None:
            touching = [n for n, c in zip(FOOT_NAMES, contacts) if c]
            if takeoff + 1 < j < touchdown and touching:
                est["jump_start"] = current_stage - touchdown
                est["falling_feet"] = set(FOOT_NAMES) - set(touching)
                _, _, positions, _ = leg_odometry(config, vel, touching)
                reset = np.mean([positions[n][2] for n in touching])
                # The filter runs alongside, so it takes the reset as well
                filter_time, x, P = est["filter"]
                x = x.copy()
                x[2] = reset
                est["filter"] = (filter_time, x, P)
                est["filter_reset"] = reset
            elif touchdown <= j < touchdown + JUMP_LAND_STAGES:
                est["falling_feet"] -= set(touching)
            else:
                est["falling_feet"] = set()
            est["touch_stage"] = current_stage + JUMP_TOUCH_STAGES

        # The feet land out of place, so they step back under the hips once
        # they are all down, which also catches the robot if it landed
        # unsteadily. Without foot contacts, they are down on schedule
        j = jump_stage(current_stage)
        if (
            j is not None
            and touchdown <= j <= touchdown + JUMP_LAND_STAGES
            and not est["falling_feet"]
            and est["gait_start"] is None
        ):
            est["gait_start"] = current_stage + 1 + JUMP_TROT_DELAY
            est["gait_stop"] = (
                est["gait_start"] + len(GAITS[est["gait"]][0]) * GAIT_PERIOD
            )
        if est["jump_start"] is not None:
            moving = False
            orientation = [0, 0, 0]

        # Switch to the next gait on each new press of the gait button while
        # standing
        if (
            switch_gait
            and not est["gait_held"]
            and est["gait_start"] is None
            and est["jump_start"] is None
            and not moving
        ):
            names = list(GAITS)
            est["gait"] = names[(names.index(est["gait"]) + 1) % len(names)]
            print(f"Switched to the {est['gait']} gait")
        est["gait_held"] = switch_gait

        # Start stepping on a move command, and stop at the end of the
        # current period once the command has been at rest for a while
        est["idle_stages"] = 0 if moving else est["idle_stages"] + 1
        if moving:
            if est["gait_start"] is None:
                est["gait_start"] = current_stage + GAIT_START_STAGES
            est["gait_stop"] = None
        elif (
            est["idle_stages"] >= IDLE_STAGES
            and est["gait_start"] is not None
            and est["gait_stop"] is None
        ):
            periods = -(-(current_stage - est["gait_start"]) // GAIT_PERIOD)
            est["gait_stop"] = (
                est["gait_start"] + max(periods, 0) * GAIT_PERIOD
            )

        # Stay in place while not stepping
        stepping = (
            est["gait_start"] is not None
            and current_stage >= est["gait_start"]
            and est["gait_stop"] is None
        )
        if not stepping:
            v_cmd = np.zeros(3)
        v_cmd = np.array(v_cmd) * GAITS[est["gait"]][1]

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

        for key, command, limit, rate in zip(
            ["base_pitch", "base_roll", "base_yaw"],
            orientation,
            [MAX_BASE_PITCH, MAX_BASE_ROLL, MAX_BASE_YAW],
            [PITCH_RATE, ROLL_RATE, YAW_RATE],
        ):
            est[key] += np.clip(
                command * limit - est[key], -rate * MPC_DT, rate * MPC_DT
            )

        # Estimate the base from the feet that were in stance during the last
        # stage, which are all still on the ground
        last_swing = swing_feet(current_stage - 1)
        stance_feet = [name for name in FOOT_NAMES if name not in last_swing]
        state, R, feet = estimate_state(config, vel, stance_feet)

        # In the air, the landing moves to when the lowest foot is predicted
        # to reach the ground, so that the plan never reaches for the ground
        # before the feet can get there. The feet are taken as stretched out
        # for the landing, since the tuck would otherwise put it off
        j = jump_stage(current_stage)
        if j is not None and takeoff + 1 < j < touchdown:
            g = -GRAVITY[2]
            rise = (R @ state[19:22])[2]
            reach = np.array([0.0, 0.0, JUMP_TAKEOFF_HEIGHT])
            drop = max(
                min(
                    state[2] + (R @ (foot_offsets[name] - reach))[2]
                    for name in FOOT_NAMES
                ),
                0.0,
            )
            fall_time = (rise + np.sqrt(rise**2 + 2 * g * drop)) / g
            stages = round(fall_time / MPC_DT)
            stages = min(max(stages, 1), touchdown - takeoff - 2)
            est["jump_start"] = current_stage + stages - touchdown

        # Feet finishing their swing keep the spot they actually landed on
        for name in last_swing:
            if name not in swing_feet(current_stage):
                est["landed"][name] = state[:3] + R @ feet[name]
                est["landed"][name][2] = 0.0

        # The feet land from the jump away from where they took off, so the
        # base's reference moves to stand centered over them again
        if in_flight(current_stage - 1) and not in_flight(current_stage):
            x, y, yaw = est["poses"][current_stage - est["pose_stage"]]
            center = np.mean([est["landed"][n] for n in FOOT_NAMES], axis=0)
            offset = np.mean([foot_offsets[n] for n in FOOT_NAMES], axis=0)
            x, y, _ = center - rotation_z(yaw) @ offset
            est["poses"] = [np.array([x, y, yaw])]
            est["pose_stage"] = current_stage

            # The landing starts from the base's measured height and fall
            fall = min((R @ state[19:22])[2], 0.0)
            est["touchdown"] = (state[2], fall)

        # Once every foot has landed after the gait stops, the robot stands
        if est["gait_stop"] is not None and current_stage >= est["gait_stop"]:
            est["gait_start"] = None
            est["gait_stop"] = None

        # The filter runs until the next stage's update, while the robot may
        # still be in either stage, so it takes the feet in stance in both.
        # Right after a gait switch there are none, and it follows the
        # accelerometer alone
        est["stance_feet"] = [
            name
            for name in stance_feet
            if name not in swing_feet(current_stage)
        ]

        update_references()

        # Feet in stance over the stages the plan is played back for
        est["plan_stance"] = [
            [
                name
                for name in FOOT_NAMES
                if not foot_reference(name, current_stage + k)[1]
            ]
            for k in range(PLAN_STAGES)
        ]
        est["stage"] += 1

        return state

    def plan_commands(x, u):
        # The plan's states at the start of its first stages (and at the end of
        # the last one), its controls and stance feet over those stages, and
        # the joint torques of its first stage
        states = [np.array(x.stage(k)) for k in range(PLAN_STAGES + 1)]
        controls = [np.array(u.stage(k)) for k in range(PLAN_STAGES)]
        torques = model.getActuatedTorqueVector(states[0], controls[0], 0)
        torques += armature * controls[0][: nv - 6]
        return (
            states,
            controls,
            est["plan_stance"],
            np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS),
        )

    def command_torques(commands, elapsed, config, vel):
        states, controls, plan_stance, torques = commands
        if not TICK_TORQUES:
            return torques

        # Inverse dynamics at the measured orientation, joints and angular
        # velocity. The base position does not enter it, and the base linear
        # velocity estimate jumps at touchdown, so the plan's is kept
        planned = plan_state(states, elapsed)
        state = planned.copy()
        state[3:7] = config[3:7]
        state[7:nq] = config[7:nq]
        state[nq + 3 : nq + nv] = vel[3:]

        # The plan's forces, and its joint accelerations pulled toward its
        # joint positions and velocities
        k = min(int(max(elapsed, 0.0) / MPC_DT), PLAN_STAGES - 1)
        u = controls[k].copy()
        u[: nv - 6] += TICK_KP * (planned[7:nq] - state[7:nq]) + TICK_KD * (
            planned[nq + 6 : nq + nv] - state[nq + 6 : nq + nv]
        )

        # Swinging feet carry no force
        for i, name in enumerate(FOOT_NAMES):
            if name not in plan_stance[k]:
                u[nv - 6 + 3 * i : nv - 3 + 3 * i] = 0.0

        torques = torque_model.getActuatedTorqueVector(state, u, 0)
        torques += armature * u[: nv - 6]
        return np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)

    return (
        model,
        mpc,
        update_func,
        plan_commands,
        command_torques,
        filter_state,
    )


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
    states = commands[0]
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
    the plan started, interpolated along the plan
    """
    state = plan_state(commands[0], elapsed)
    return state[7:19], state[25:37]
