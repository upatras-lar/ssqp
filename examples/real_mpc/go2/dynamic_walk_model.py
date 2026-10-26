from collections import namedtuple

import ssqpy
import numpy as np

ssqpy.setSilentMode()

# Keep MPC_DT a multiple of the controller's command period, so that plan
# stages line up with commands
MPC_DT = 0.02
MPC_H = 14

T_WGT = 1.0
R_WGT = 1.0
C_WGT = 1e-2
V_WGT = 1e-3
U_WGT = 1e-5
F_WGT = 2e-4
# Weight on the base's roll and pitch rates, which keeps each plan from
# correcting the whole base tilt error within one stage
W_WGT = 0.1
# Weight on the base's height, which keeps its bounce in check while running
Z_WGT = 3.0

CONTACT_MU = 0.4
# Limit on each foot's normal force
MAX_FOOT_FORCE = 200.0

# Radius of the terrain patch around a foothold fit to the plane its friction
# cone stands on, and how near an edge the patch's points can be
CONTACT_PATCH = 0.04
CONTACT_EDGE_GAP = 0.02
# Limit on the tilt of a foot's contact plane, under the friction angle so
# that the foot's share of the weight stays inside its cone
MAX_CONTACT_TILT = 0.3

# Height of the model's foot points above the ground under stance feet. The
# feet are spheres of radius 0.022 and the points lie 0.0235 past their
# centres along the calves, so they rise as the calves tilt, to about 4 mm
# while walking. The terrain the walk takes is the surface they rest on, the
# ground raised by this
FOOT_POINT_HEIGHT = 0.004

# Speed limit on stance feet, keeping the plan from swinging planted legs
STANCE_VEL_TOL = 0.2

# While stepping, the base's reference starts from its measured position, and
# each foothold moves by FOOTHOLD_VEL_GAIN (in seconds) times the difference
# between the measured and commanded base velocity, up to FOOTHOLD_MAX_SHIFT.
# A foot landing ahead of where the base would pass over it slows the base
# down, and one landing behind speeds it up. Footholds stop moving
# FOOTHOLD_FREEZE_STAGES before touchdown, since the swing can no longer
# follow them
FOOTHOLD_VEL_GAIN = 0.19
FOOTHOLD_MAX_SHIFT = 0.2
FOOTHOLD_FREEZE_STAGES = 4

# Footholds stay FOOTHOLD_EDGE_MARGIN away from edges in the terrain, which
# covers the foot's radius and errors in the terrain and the base estimate.
# One that does not moves to the nearest spot that does, up to FOOTHOLD_SEARCH
# forward or back and FOOTHOLD_SIDE_SEARCH to the side, trying spots
# FOOTHOLD_SEARCH_STEP apart
FOOTHOLD_EDGE_MARGIN = 0.05
# The margin is measured from the foot's sphere centre, which sits this far
# from the foot frame along the heading when the foot lands
FOOT_SPHERE_OFFSET = np.array([-0.017, 0.0, 0.0])
# The knees point back, so the lower leg rises behind the foot by about
# CALF_SLOPE per unit of distance, up to CALF_REACH behind it. Footholds near
# edges also keep the terrain behind them LEG_CLEARANCE below that line
CALF_SLOPE = 1.5
CALF_REACH = 0.12
LEG_CLEARANCE = 0.02
FOOTHOLD_SEARCH = 0.14
FOOTHOLD_SIDE_SEARCH = 0.08
FOOTHOLD_SEARCH_STEP = 0.01
# Footholds keep the height they were first placed at, within
# FOOTHOLD_LEVEL_CHANGE, while there is a spot at that height. Otherwise they
# could jump between the levels of a step as their nominal position moves,
# late in a swing. Once the foot has lifted, one with no spot left at its
# height keeps the spot it had while that is still clear of edges
FOOTHOLD_LEVEL_CHANGE = 0.02
# A foothold moved away from an edge keeps that correction while the spot it
# gives stays clear of edges at the same level, and drops it once its nominal
# spot is FOOTHOLD_EDGE_MARGIN + FOOTHOLD_KEEP_SLACK from them. Otherwise small
# moves of the nominal spot and flicker in the edges make it jump back and
# forth between spots. Once the correction no longer gives such a spot, the
# foothold stays where it was while that spot is still clear of edges at the
# same level and within FOOTHOLD_STICK_RANGE of its nominal spot. The nearest
# spot to the nominal one could otherwise be on the other side of an obstacle
FOOTHOLD_KEEP_SLACK = 0.03
FOOTHOLD_STICK_RANGE = 0.14
# For the rear feet, spots ahead of a foothold count as FOOTHOLD_AHEAD_COST
# times further than they are, so that their steps near edges get shorter
# rather than longer
FOOTHOLD_AHEAD_COST = 1.5

# Time from measuring the state until the plan solved from it is applied. The
# state is predicted this far ahead before solving, which makes up for the
# solve time. Solves should finish within it, since the walk tolerates little
# more delay than this
PREDICT_TIME = 0.015

# Solve time limit, leaving room within PREDICT_TIME for the work around the
# solve. SQP iterations that would end past it are skipped
MAX_SOLVE_TIME = 0.012

# The robot stands on all four feet until it is commanded to move, and goes
# back to standing once the command, and the base's reference velocity as it
# slows down, have been at rest for IDLE_TIME. Commands below MOVE_THRESHOLD
# count as at rest. Stepping starts GAIT_START_STAGES after the move command,
# which gives the plan time to shift its weight
IDLE_TIME = 1.0
IDLE_STAGES = round(IDLE_TIME / MPC_DT)
MOVE_THRESHOLD = 0.02
GAIT_START_STAGES = 3
# Standing, a push that moves the base faster than PUSH_SPEED starts stepping
# as a move command would, so that the feet can catch it. The velocity is
# left alone for PUSH_SETTLE_STAGES after the gait stops, while the last
# feet land
PUSH_SPEED = 0.15
PUSH_SETTLE_STAGES = 5

# Limits on the commanded velocity, and on how quickly the command may change
MAX_VEL_FRONT = 0.5
MAX_VEL_BACK = 0.5
MAX_VEL_SIDE = 0.3
MAX_VEL_YAW = 1.0
MAX_ACC = 0.6
MAX_YAW_ACC = 1.0
# A foot that lands within LAND_HOLD of the height it last landed at keeps
# that height, so that small errors in the terrain do not build up in the
# height estimate over many steps
LAND_HOLD = 0.01
# Limit on the base's sideways acceleration while turning, as its speed times
# its turning rate, which slows turning while running fast
MAX_TURN_ACC = 0.8

# The base position and linear velocity come from a Kalman filter, which
# integrates the accelerometer and corrects it with the leg odometry of every
# state message. Noise of the accelerometer (m/s^2), and of the leg odometry
# base velocity (m/s) and position (m) from each stance foot
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

# Where the terrain along a swing, sampled at SWING_PATH_SAMPLES points, rises
# more than SWING_ROUGH_HEIGHT above its lower foothold, the foot rises from
# its liftoff, crosses high enough to clear the terrain on the way by
# SWING_CLEARANCE, and comes down onto its foothold from above. Within
# SWING_END_REACH of its footholds the foot only has to clear them, and it
# crosses at most MAX_SWING_LIFT above the higher one
SWING_ROUGH_HEIGHT = 0.02
SWING_PATH_SAMPLES = 16
SWING_CLEARANCE = 0.05
SWING_END_REACH = 0.03
MAX_SWING_LIFT = 0.3

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

# The base tilts to lie parallel to the terrain between its feet, up to these
# angles, on top of the tilt commands
MAX_TERRAIN_PITCH = 0.6
MAX_TERRAIN_ROLL = 0.3

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
JUMP_TROT_DELAY = 20
# How many gait cycles to perform after landing
JUMP_TROT_CYCLES = 2
# Forward jump, started by the forward jump button in the same way as the jump
# in place, with the same crouch stages. The base leans back as it crouches to
# FORWARD_JUMP_CROUCH_HEIGHT, and pushes off up to FORWARD_JUMP_TAKEOFF_HEIGHT,
# speeding up forward as well. It keeps its speed in the air while the legs
# tuck in by FORWARD_JUMP_TUCK and swing forward, touches down at
# FORWARD_JUMP_LAND_HEIGHT and stops its fall by the crouch height, with each
# foot pushing with up to FORWARD_JUMP_LAND_FORCE. It stops its forward speed
# at FORWARD_JUMP_BRAKE, which friction allows, landing with the feet far
# enough ahead to do so. Where the terrain rises by more than
# FORWARD_JUMP_RISE ahead of the front feet, the jump lands on it with the rear
# feet FORWARD_JUMP_EDGE_MARGIN past its edge, as long as the feet move at most
# FORWARD_JUMP_MAX_DISTANCE. Otherwise the feet land FORWARD_JUMP_DISTANCE
# ahead of where they took off. The base rises at least FORWARD_JUMP_HEIGHT
# above its takeoff height, and FORWARD_JUMP_CLEARANCE above where it lands.
# The robot does not jump when that takes a takeoff speed above
# FORWARD_JUMP_MAX_SPEED, or when the feet would land on different levels
FORWARD_JUMP_DISTANCE = 0.6
FORWARD_JUMP_RISE = 0.04
FORWARD_JUMP_EDGE_MARGIN = 0.12
FORWARD_JUMP_MAX_DISTANCE = 1.0
FORWARD_JUMP_CROUCH_HEIGHT = 0.15
FORWARD_JUMP_TAKEOFF_HEIGHT = 0.38
FORWARD_JUMP_LAND_HEIGHT = 0.32
FORWARD_JUMP_LAND_FORCE = 200.0
FORWARD_JUMP_BRAKE = 3.0
FORWARD_JUMP_TUCK = 0.15
FORWARD_JUMP_HEIGHT = 0.2
FORWARD_JUMP_CLEARANCE = 0.1
FORWARD_JUMP_MAX_SPEED = 2.6
# Weight holding the stance feet at their measured height through the push of
# a forward jump. Their contacts leave it free, and the plan would otherwise
# stretch the legs further by pushing the feet into the ground, which the
# ground turns into forces that tip the base
FORWARD_JUMP_FOOT_WGT = 300.0
# The robot takes off slower than its plan, by a steady share forward and up,
# so a forward jump's references aim faster by these factors
FORWARD_JUMP_SPEED_GAIN = 1.3
FORWARD_JUMP_RISE_GAIN = 1.1
# Share of the way from the previous plan's legs to the measured ones that
# each plan in the air of a forward jump starts from. The legs follow the plan
# with some lag, and starting from the measured legs alone makes each plan
# jerk them back toward the last one
FORWARD_JUMP_LEG_PULL = 0.5

# Ground modes
GROUND_MODE_NORMAL = "normal"
GROUND_MODE_LIFTED = "lifted"
GROUND_MODE_AIRBORNE = "airborne"
GROUND_MODE_SET_DOWN = "set_down"
GROUND_MODE_SWITCH_DELAY = 0.24
# While walking, every foot can read no contact for a moment, so the robot
# then only counts as held up while the accelerometer's upward reading, in
# m/s^2 and averaged since the feet left the ground, is above LIFT_MIN_ACC,
# and the walk stops once no foot has touched for LIFT_PENDING_TIME. Feet that
# each read below the contact threshold carry at most about half its weight
LIFT_MIN_ACC = 7.0
LIFT_PENDING_TIME = 0.05

JUMP_SPEED = -GRAVITY[2] * JUMP_FLIGHT_STAGES * MPC_DT / 2

# Joint gains around the planned joint positions and velocities, on top of the
# planned torques
JOINT_KP = 6.0
JOINT_KD = 2 * np.sqrt(JOINT_KP)
# Joint gains in the unitree_mujoco simulator, which holds each command's PD
# torque until the next command, where the robot's gains make the legs shake
SIM_JOINT_KP = 0.1
SIM_JOINT_KD = 2 * np.sqrt(SIM_JOINT_KP)

# How far ahead the plan is read for each command, since the robot's motors
# act on commands several milliseconds after they are sent. The simulator acts
# on them at once
COMMAND_LEAD = 0.01
SIM_COMMAND_LEAD = 0.0

# Gains pulling the planned joint accelerations toward the planned joint
# positions and velocities, when computing each command's torques
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

# Gaits, as the groups of feet that take turns to start their swing a gait
# period apart, the factor on the velocity limits they move at, the duty
# factor, which is the share of the gait cycle each foot spends on the ground,
# and the gait period in plan stages. Above 1, the factor scales only the
# forward limit, so no gait moves backward, sideways or turns faster than the
# trot. Below 1 - 1 / groups the swings overlap. The run is a trot with all
# feet in the air between stances, at a higher speed and with quicker steps.
# The walk lifts one foot at a time, so it steps slower
Gait = namedtuple("Gait", ["groups", "speed", "duty", "period"])
GAITS = {
    "trot": Gait(
        [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]], 1.0, 0.5, 12
    ),
    "walk": Gait(
        [["RL_foot"], ["FL_foot"], ["RR_foot"], ["FR_foot"]], 0.5, 0.75, 12
    ),
    # "run": Gait(
    #         [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]], 3.0, 0.4, 10
    #     ),
}
# Gait at the start. The gait button cycles through them while standing
GAIT = "trot"
# Gaits that leave the LIDAR map out and take the terrain to be flat at its
# height when they start. The run steps further ahead than the map covers
# densely and is upset by its small errors
MAP_FREE_GAITS = ["run"]


def compute_nominal_leg_joint_angles(height):
    # Hip, thigh and calf angles placing the foot straight below the hip
    calf = -np.arccos(
        (height**2 - THIGH_LENGTH**2 - CALF_LENGTH**2)
        / (2 * THIGH_LENGTH * CALF_LENGTH)
    )
    thigh = np.arcsin(-CALF_LENGTH * np.sin(calf) / height)
    return np.array([0.0, thigh, calf])


def rotation_matrix_from_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrix_from_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_matrix_from_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def compute_base_rotation_matrix(pitch, roll, yaw):
    # Base orientation relative to the walking direction
    return (
        rotation_matrix_from_z(yaw)
        @ rotation_matrix_from_y(-pitch)
        @ rotation_matrix_from_x(roll)
    )


def quat_xyzw_to_rotation_matrix(xyzw):
    x, y, z, w = xyzw
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def rotation_matrix_to_quat_xyzw(R):
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


def build_go2_mpc_model(horizon):
    # The robot's model over the given number of stages, in the mode the walk
    # is solved in
    return ssqpy.model.Model(
        horizon,
        MPC_DT,
        mjcf_path="../../assets/mjcf/go2/go2.xml",
        free_flyer=True,
        solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
    )


def create_controller(
    init_config, simulation=False, terrain=None, get_edge_distance=None
):
    """
    Builds the dynamics walking MPC for the robot starting at init_config,
    or for the unitree_mujoco simulator with simulation set

    The robot only measures the base orientation, not its position, so the
    base position is estimated from the stance feet (leg odometry), assuming
    they stay where they landed on the terrain, which a Kalman filter fuses
    with the accelerometer
    The yaw is taken relative to the initial one
    The base position in the given configurations is ignored

    terrain is the ground under the robot raised by FOOT_POINT_HEIGHT, in the
    frame the base position is estimated in, which starts with the base above
    the origin and x along its initial heading. It defaults to flat ground at
    zero height. get_edge_distance gives the distance from a point (x and y) to the
    nearest edge in the terrain, and defaults to there being none

    The robot stands on all four feet until the command moves it, and stands
    again once the command has been at rest for IDLE_TIME

    Returns the model, the MPC, and ten functions:
     - update_controller, to call once per plan stage, before solving. It takes the
       commanded velocity (forward, sideways and yaw), the D-pad state (x and
       y), the base orientation commands (pitch, roll and yaw, each 1 or -1
       while tilting the front up, tilting to the right or turning the front
       to the left, or the other way, and 0 otherwise), whether the jump
       button is held, the measured configuration and velocity, and
       optionally whether each foot is in contact (in FOOT_NAMES order),
       whether the forward jump button is held and the time since the last
       call. The base linear velocity is
       estimated from the stance feet as well, so the one in the given
       velocity is ignored. It returns the estimated state.
       Predict it to when the plan will be applied before solving (see
       propagate_state_over_interval)
     - prepare_mpc_playback_plan, taking the solved states and controls and returning the
       commands to play the plan back with (see playback)
     - compute_joint_torque_commands, taking the commands, the time since the plan started
       and the measured configuration and velocity, and returning the joint
       torques to apply
     - update_state_estimator, to call with every state message. It takes the measured
       configuration and velocity, the accelerometer reading (in the base
       frame) and the time in seconds. It returns the base position and
       velocity estimate and the base rotation in the same frame
     - flat_ground_height, returning the height to take the terrain to be flat at,
       with no edges, while the walk leaves the map out (see
       MAP_FREE_GAITS), and None while it uses the map
     - is_jump_active, returning whether the robot is jumping, when the base
       estimate is too poor to map the ground
     - get_stance_foot_mask, returning whether each foot (in FOOT_NAMES order) is one
       the state filter takes to be on the ground
     - is_robot_lifted, returning whether the robot has been held up with no
       foot on the ground for a while, which stops the walk until all feet
       are down again
     - is_lift_pending, returning whether the robot is held up with no foot
       on the ground while it has not been taken to be lifted yet, outside of
       jumps
     - is_robot_set_down, returning whether all feet have been down for a
       while after a lift. The controller then plans nothing more, and the
       robot stands up and starts over with a new one
    """
    if terrain is None:
        terrain = ssqpy.model.terrain.HeightmapTerrain(np.zeros((2, 2)), 1.0)

    # The model keeps its default flat ground. Each stance foot's contact
    # takes a plane fit to the terrain around its foothold instead (see
    # contact_plane), since the terrain's own normals turn sharply at edges and
    # follow its noise
    model = build_go2_mpc_model(MPC_H)

    nq = model.getnq()
    armature = SIM_JOINT_ARMATURE if simulation else JOINT_ARMATURE

    # Foot positions relative to the base at the nominal configuration, on
    # the ground
    nominal_config = np.zeros(nq)
    nominal_config[6] = 1.0
    nominal_config[7:] = np.tile(
        compute_nominal_leg_joint_angles(BASE_HEIGHT), 4
    )
    foot_offsets = {
        name: model.forwardKinematics(nominal_config, name)[0]
        for name in FOOT_NAMES
    }
    for offset in foot_offsets.values():
        offset[2] = 0.0

    # Points on a grid between the feet's nominal positions, where the terrain
    # is sampled for its tilt, and the matrix fitting a plane to them
    corners = np.array(list(foot_offsets.values()))
    tilt_offsets = [
        np.array([dx, dy, 0.0])
        for dx in np.linspace(corners[:, 0].min(), corners[:, 0].max(), 5)
        for dy in np.linspace(corners[:, 1].min(), corners[:, 1].max(), 3)
    ]
    tilt_fit = np.array([[1.0, p[0], p[1]] for p in tilt_offsets])

    def project_point_to_terrain(p):
        # The point on the terrain below p
        return np.array([p[0], p[1], terrain.at(p[0], p[1]).height])

    if get_edge_distance is None:

        def get_edge_distance(x, y):
            return np.inf

    # Spots to try for each foot's foothold near an edge, relative to the
    # heading, nearest first
    def generate_foothold_search_offsets(size):
        return np.arange(-size, size + 1e-9, FOOTHOLD_SEARCH_STEP)

    def sort_foothold_candidates(ahead_cost):
        return sorted(
            (
                np.array([dx, dy, 0.0])
                for dx in generate_foothold_search_offsets(FOOTHOLD_SEARCH)
                for dy in generate_foothold_search_offsets(
                    FOOTHOLD_SIDE_SEARCH
                )
            ),
            key=lambda offset: np.hypot(
                offset[0] * (ahead_cost if offset[0] > 0 else 1.0),
                offset[1],
            ),
        )

    foothold_offsets = {
        name: sort_foothold_candidates(
            FOOTHOLD_AHEAD_COST if name[0] == "R" else 1.0
        )
        for name in FOOT_NAMES
    }

    def compute_foot_sphere_center_xy(p, yaw):
        # Where the foot's sphere centre is, horizontally, when its frame is
        # at p
        return (p + rotation_matrix_from_z(yaw) @ FOOT_SPHERE_OFFSET)[:2]

    calf_samples = np.arange(0.02, CALF_REACH + 1e-9, 0.02)
    slack_margin = FOOTHOLD_EDGE_MARGIN + FOOTHOLD_KEEP_SLACK

    def is_foothold_clear_of_edges(p, yaw, margin=FOOTHOLD_EDGE_MARGIN):
        # Whether a foothold at p is margin from any edge, with the terrain
        # behind it clear of the lower leg
        distance = get_edge_distance(*compute_foot_sphere_center_xy(p, yaw))
        if distance < margin:
            return False
        if distance > CALF_REACH:
            return True
        back = rotation_matrix_from_z(yaw)[:2, 0]
        foot = terrain.at(p[0], p[1]).height
        for s in calf_samples:
            x, y = p[:2] - s * back
            if terrain.at(x, y).height > foot + CALF_SLOPE * s - LEG_CLEARANCE:
                return False
        return True

    def is_foothold_safe_at_height(p, yaw, level, margin=FOOTHOLD_EDGE_MARGIN):
        # Whether a foothold at p is clear, and within FOOTHOLD_LEVEL_CHANGE
        # of the given level
        return (
            is_foothold_clear_of_edges(p, yaw, margin)
            and abs(project_point_to_terrain(p)[2] - level)
            <= FOOTHOLD_LEVEL_CHANGE
        )

    def find_safe_foothold(name, p, yaw, level=None):
        # The nearest clear spot to p, at the given level if there is one
        # there, or p if there is none within the search
        R = rotation_matrix_from_z(yaw)
        nearest = None
        for offset in foothold_offsets[name]:
            q = p + R @ offset
            if not is_foothold_clear_of_edges(q, yaw):
                continue
            change = (
                0.0
                if level is None
                else abs(project_point_to_terrain(q)[2] - level)
            )
            if change <= FOOTHOLD_LEVEL_CHANGE:
                return q
            if nearest is None:
                nearest = q
        return p if nearest is None else nearest

    def get_max_terrain_height_along_path(start, end):
        # The highest terrain on the way from start to end
        return max(
            project_point_to_terrain((1 - s) * start + s * end)[2]
            for s in np.linspace(0.0, 1.0, SWING_PATH_SAMPLES)
        )

    def compute_swing_apex_height(liftoff, landing):
        # Height of the middle control points of a swing curve over rough
        # terrain, which lie above the footholds, so that the foot clears the
        # terrain on the way. The curve's height is linear in it, so each
        # sample gives the least height it needs directly
        z0, z1 = liftoff[2], landing[2]
        apex = max(z0, z1) + SWING_CLEARANCE / 0.75
        span = np.linalg.norm((landing - liftoff)[:2])
        for t in np.linspace(0.0, 1.0, SWING_PATH_SAMPLES + 2)[1:-1]:
            s = t * t * (3 - 2 * t)
            if min(s, 1 - s) * span < SWING_END_REACH:
                continue
            p = project_point_to_terrain((1 - s) * liftoff + s * landing)
            ends = (1 - t) ** 3 * z0 + t**3 * z1
            needed = (p[2] + SWING_CLEARANCE - ends) / (3 * t * (1 - t))
            apex = max(apex, needed)
        return min(apex, max(z0, z1) + MAX_SWING_LIFT)

    def estimate_terrain_height_under_base(x, y, yaw):
        # Terrain height under the base with the given pose, averaged over
        # the feet's nominal positions so that it changes smoothly over edges
        base = np.array([x, y, 0.0])
        return np.mean(
            [
                project_point_to_terrain(
                    base + rotation_matrix_from_z(yaw) @ foot_offsets[name]
                )[2]
                for name in FOOT_NAMES
            ]
        )

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

    def compute_leg_ik(foot_pos_in_hip_frame, side):
        # Hip, thigh and calf angles placing the foot at the given position
        # relative to the hip joint, in the base frame. side is 1 for the left
        # legs and -1 for the right ones. Feet out of reach get a straight leg
        # pointing at them
        x, y, z = foot_pos_in_hip_frame
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

    def compute_stance_joint_configuration(
        height, pitch, roll, yaw, ahead=0.0
    ):
        # Joint angles keeping each foot on its nominal spot on the ground,
        # moved ahead by the given distance, with the base at the given height
        # and orientation
        R = compute_base_rotation_matrix(pitch, roll, yaw)
        base = np.array([-ahead, 0.0, height])
        return np.hstack(
            [
                compute_leg_ik(
                    R.T @ (foot_offsets[name] - base) - hip_offsets[name],
                    1 if foot_offsets[name][1] > 0 else -1,
                )
                for name in FOOT_NAMES
            ]
        )

    initial_yaw = np.arctan2(
        *quat_xyzw_to_rotation_matrix(init_config[3:7])[[1, 0], 0]
    )

    def compute_walk_frame_base_rotation(config):
        # The base rotation of a configuration in the walk's frame, whose x
        # axis lies along the initial heading
        return rotation_matrix_from_z(
            -initial_yaw
        ) @ quat_xyzw_to_rotation_matrix(config[3:7])

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
        # Where each foot last landed, and where it is measured to be while in
        # stance, which its next swing starts from
        "landed": {},
        "liftoff": {},
        # Each foot's measured height at the last update
        "foot_heights": {},
        # Shift of the footholds for the base velocity error, and each foot's
        # frozen foothold with the stage it lands at
        "foothold_shift": np.zeros(3),
        "frozen": {},
        # Footholds and the apex of the swing between them (None over smooth
        # terrain) by foot and gait period, and foot references by foot and
        # stage, computed since the references were last updated
        "footholds": {},
        "swing_tops": {},
        "foot_references": {},
        # Where each foothold was placed, by foot and landing stage
        "placed_footholds": {},
        # Contact planes by foothold, fit since the references were last
        # updated
        "contact_planes": {},
        # Current gait
        "gait": GAIT,
        "gait_held": False,
        # Stages where the gait starts and stops, None while standing or
        # while stepping without a planned stop
        "gait_start": None,
        "gait_stop": None,
        # Stage the robot last started standing at
        "stand_stage": 0,
        # Stages the command has been at rest for
        "idle_stages": 0,
        # First stage of the jump, None when not jumping, how far ahead its
        # feet land, its stages pushing off, in the air, landing and stopping
        # its forward speed, the stage
        # it takes off at, its takeoff height and speed, its height at
        # touchdown, the force limit on each foot in the landing, how far its
        # legs tuck in, the change to the legs' landing configuration for the
        # base's tilt in the air of a forward jump, the legs' configuration
        # through each stage of its flight, and for forward jumps the terrain
        # height it takes off from and where each foot lands
        "jump_start": None,
        "jump_held": False,
        "forward_jump_held": False,
        "jump_distance": 0.0,
        "jump_push": JUMP_PUSH_STAGES,
        "jump_flight": JUMP_FLIGHT_STAGES,
        "jump_land": JUMP_LAND_STAGES,
        "jump_brake": JUMP_LAND_STAGES,
        "takeoff_at": None,
        "jump_takeoff": JUMP_TAKEOFF_HEIGHT,
        "jump_rise": JUMP_SPEED,
        "jump_land_height": JUMP_TAKEOFF_HEIGHT,
        "jump_land_force": JUMP_LAND_FORCE,
        "jump_tuck": JUMP_TUCK,
        "landing_tilt": None,
        "flight_configs": None,
        "jump_ground": None,
        "jump_landing": None,
        # Height of the terrain under the feet when the walk stopped using
        # the map, None while it uses it
        "flat_ground_height": None,
        # Base height and vertical velocity at touchdown, which the landing
        # starts from. Until the feet touch down, the planned ones
        "touchdown": None,
        # Feet still in the air once the landing has started, which swing
        # until the stage before touch_stage
        "falling_feet": set(),
        "touch_stage": 0,
        "ground_mode": GROUND_MODE_NORMAL,
        "no_contact_time": 0.0,
        "stable_contact_time": 0.0,
        # Sum and count of the accelerometer's vertical readings since the
        # last stage, which the filter adds to, and since no foot has touched
        # the ground, and whether the robot is taken to be held up then
        "vertical_acc": (0.0, 0),
        "off_ground_acc": (0.0, 0),
        "held": False,
    }

    def get_foot_positions_in_base_frame(config, kinematics=model):
        # Foot positions relative to the base, in the base frame
        local = config[:nq].copy()
        local[:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return {
            name: kinematics.forwardKinematics(local, name)[0]
            for name in FOOT_NAMES
        }

    def estimate_base_motion_from_stance_feet(
        config, vel, stance_feet, kinematics=model
    ):
        # The base position and linear velocity (in the base frame) each
        # stance foot implies, if it stays where it landed
        R = compute_walk_frame_base_rotation(config)
        feet = get_foot_positions_in_base_frame(config, kinematics)

        positions = {
            name: est["landed"][name] - R @ feet[name] for name in stance_feet
        }

        # Stance feet stay still, so the base moves opposite to how they move
        # relative to it, through the joints and the base's rotation
        step = 1e-4
        moved = config[:nq].copy()
        moved[7:] += step * vel[6:]
        feet_moved = get_foot_positions_in_base_frame(moved, kinematics)
        velocities = {
            name: -(feet_moved[name] - feet[name]) / step
            - np.cross(vel[3:6], feet[name])
            for name in stance_feet
        }
        return R, positions, velocities

    def estimate_robot_state(config, vel):
        # Replaces the base pose and linear velocity with the filter's
        # estimate
        R = compute_walk_frame_base_rotation(config)
        feet = get_foot_positions_in_base_frame(config)
        base_pos = est["filter"][1][:3]
        base_vel = R.T @ est["filter"][1][3:]

        state = np.hstack(
            (
                base_pos,
                rotation_matrix_to_quat_xyzw(R),
                config[7:nq],
                base_vel,
                vel[3:],
            )
        )
        return state, R, feet

    # Start with every foot on the ground where it currently is
    R0 = compute_walk_frame_base_rotation(init_config)
    feet0 = get_foot_positions_in_base_frame(init_config)
    height0 = -np.mean([(R0 @ p)[2] for p in feet0.values()])
    for name in FOOT_NAMES:
        est["landed"][name] = project_point_to_terrain(R0 @ feet0[name])
        est["liftoff"][name] = est["landed"][name]
    height0 += np.mean([est["landed"][name][2] for name in FOOT_NAMES])

    init_state = np.zeros(nq + model.getnv())
    init_state[:nq] = init_config[:nq]
    init_state[:3] = [0.0, 0.0, height0]
    init_state[3:7] = rotation_matrix_to_quat_xyzw(R0)

    # Filter time, estimate (base position and linear velocity, in the
    # yaw-aligned frame) and covariance, the feet it takes as stance, and
    # whether it takes the base to be falling freely
    est["filter"] = (
        None,
        np.array([0.0, 0.0, height0, 0.0, 0.0, 0.0]),
        1e-4 * np.eye(6),
    )
    est["stance_feet"] = FOOT_NAMES
    est["falling_freely"] = False

    # The filter runs alongside the solver, so it uses its own model
    filter_model = build_go2_mpc_model(1)

    def update_state_estimator(config, vel, acc, t):
        last_t, x, P = est["filter"]
        reset = est.pop("filter_reset", None)
        if reset is not None:
            x = x.copy()
            x[2] = reset
        stance_feet = est["stance_feet"]
        R, positions, velocities = estimate_base_motion_from_stance_feet(
            config, vel, stance_feet, filter_model
        )
        total, count = est["vertical_acc"]
        est["vertical_acc"] = (total + (R @ acc)[2], count + 1)

        # Predict with the accelerometer, which measures the base
        # acceleration minus gravity. In the air of a forward jump the base
        # falls freely, while the accelerometer also picks up the legs swinging
        # around it and the base's turning, so it is left out
        if last_t is not None:
            dt = min(max(t - last_t, 0.0), 0.05)
            a = GRAVITY if est["falling_freely"] else R @ acc + GRAVITY
            x = x.copy()
            x[:3] += dt * x[3:] + 0.5 * dt**2 * a
            x[3:] += dt * a
            F = np.kron([[1.0, dt], [0.0, 1.0]], np.eye(3))
            Q = np.kron([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], np.eye(3))
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
        return x, R

    def predict_future_base_poses(pose, count):
        v_cmd = est["v_cmd"]
        yaw = np.cumsum(np.append(pose[2], np.full(count, v_cmd[2] * MPC_DT)))
        c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
        dx = (c * v_cmd[0] - s * v_cmd[1]) * MPC_DT
        dy = (s * v_cmd[0] + c * v_cmd[1]) * MPC_DT
        x = np.cumsum(np.append(pose[0], dx))
        y = np.cumsum(np.append(pose[1], dy))
        return np.column_stack((x, y, yaw))

    def get_jump_stage(k):
        # Stage k's index within the jump, or None outside of it
        if est["jump_start"] is None:
            return None
        j = k - est["jump_start"]
        return j if 0 <= j < JUMP_STAGES else None

    def get_takeoff_stage():
        # Stage of the jump where the flight starts
        return JUMP_CROUCH_STAGES + est["jump_push"]

    def get_landing_stage():
        # Stage of the jump where the landing starts
        return get_takeoff_stage() + est["jump_flight"]

    def compute_jump_crouch_height(speed):
        # Base height the push starts from to take off at the given speed
        return est["jump_takeoff"] - speed * est["jump_push"] * MPC_DT / 2

    def get_jump_base_height(j):
        # Base reference height at stage j of the jump. In the air, it is
        # taken from the terrain the jump took off from
        height = est["base_height"]
        push = JUMP_CROUCH_STAGES
        flight = get_takeoff_stage()
        land = get_landing_stage()
        recover = land + est["jump_land"]
        speed = est["jump_rise"]
        crouch = compute_jump_crouch_height(speed)

        if j < push:
            blend = 0.5 * (1 - np.cos(np.pi * j / JUMP_CROUCH_STAGES))
            return (1 - blend) * height + blend * crouch
        if j < flight:
            t = (j - push) * MPC_DT
            acc = speed / (est["jump_push"] * MPC_DT)
            return crouch + 0.5 * acc * t**2
        if j < land:
            t = (j - flight) * MPC_DT
            rise = speed * t + 0.5 * GRAVITY[2] * t**2
            return est["jump_takeoff"] + rise
        # The landing stops the base's fall evenly, from where it touched down
        land_height, land_speed = est["touchdown"]
        duration = est["jump_land"] * MPC_DT
        if j < recover:
            t = (j - land) * MPC_DT
            acc = -land_speed / duration
            return land_height + land_speed * t + 0.5 * acc * t**2
        phase = (j - recover) / (JUMP_STAGES - recover)
        blend = 0.5 * (1 - np.cos(np.pi * phase))
        landed = land_height + land_speed * duration / 2
        return (1 - blend) * landed + blend * height

    def compute_jump_forward_speed():
        # Forward speed of the jump in the air. The feet take off a quarter of
        # the push's travel behind the base, and land half of the travel to
        # stop ahead of it, so that the base comes to rest over them
        stages = (
            est["jump_flight"] + est["jump_push"] / 4 + est["jump_brake"] / 2
        )
        speed = est["jump_distance"] / (stages * MPC_DT)
        return FORWARD_JUMP_SPEED_GAIN * speed

    def compute_jump_forward_displacement(j, landed):
        # How far ahead the base's reference is at stage j of the jump, from
        # where it started the jump, or once the feet have landed, from where
        # it stands over them
        speed = compute_jump_forward_speed()
        push_time = est["jump_push"] * MPC_DT
        land_time = est["jump_brake"] * MPC_DT
        back = speed * push_time / 4
        takeoff = get_takeoff_stage()
        land = get_landing_stage()
        start = 0.0 if landed else est["jump_distance"]
        if j < JUMP_CROUCH_STAGES:
            blend = 0.5 * (1 - np.cos(np.pi * j / JUMP_CROUCH_STAGES))
            return -blend * back
        if j < takeoff:
            t = (j - JUMP_CROUCH_STAGES) * MPC_DT
            return -back + 0.5 * speed / push_time * t**2
        if j < land:
            t = (j - takeoff) * MPC_DT
            return back + speed * t
        if j < land + est["jump_brake"]:
            t = (j - land) * MPC_DT
            slow = speed * t - 0.5 * speed / land_time * t**2
            return start - speed * land_time / 2 + slow
        return start

    def is_robot_airborne(k):
        # Whether all feet are in the air at stage k
        j = get_jump_stage(k)
        return j is not None and get_takeoff_stage() <= j < get_landing_stage()

    def is_foot_descending_during_landing(name, k):
        # Whether the foot is still coming down at stage k of the landing
        return name in est["falling_feet"] and k < est["touch_stage"]

    def get_active_gait():
        # The gait the robot walks in
        return GAITS[est["gait"]]

    def get_gait_period_stages():
        # Stages between the swing starts of consecutive groups of feet
        return get_active_gait().period

    def get_gait_cycle_length_stages():
        # Stages in one cycle of the gait
        return len(get_active_gait().groups) * get_gait_period_stages()

    def get_swing_duration_stages():
        # Stages each foot swings for in the gait
        return round(
            (1 - get_active_gait().duty) * get_gait_cycle_length_stages()
        )

    def get_gait_stop_stage():
        # The stage where the last swing of the stopping gait lands
        return est["gait_stop"] + max(
            get_swing_duration_stages() - get_gait_period_stages(), 0
        )

    def get_gait_base_height(k):
        # The base's reference height at stage k. When the swings overlap,
        # each gait period starts with all feet in the air, and the base flies
        # on a ballistic arc, then falls and rises back on the ground with
        # constant acceleration in time for the next flight. The arc is
        # offset to keep the base at its height on average. Before the first
        # flight and after the last one, the base stays at its height
        height = est["base_height"]
        start, stop = est["gait_start"], est["gait_stop"]
        groups = len(get_active_gait().groups)
        flight = (
            get_swing_duration_stages()
            - (groups - 1) * get_gait_period_stages()
        )
        if start is None or k < start or flight <= 0:
            return height

        # Flights start every period from the one where every group has
        # started swinging until the last one to start a swing
        period, stage = divmod(k - start, get_gait_period_stages())
        last = (
            np.inf
            if stop is None
            else (stop - start) // get_gait_period_stages() - 1
        )
        if stage < flight:
            if not groups - 1 <= period <= last:
                return height
        elif not groups - 1 <= period < last:
            return height

        g = -GRAVITY[2]
        t_flight = flight * MPC_DT
        t_stance = (get_gait_period_stages() - flight) * MPC_DT
        liftoff = height - g * t_flight * (t_flight - t_stance) / 12
        speed = g * t_flight / 2
        if stage < flight:
            t = stage * MPC_DT
            return liftoff + speed * t - 0.5 * g * t**2
        t = (stage - flight) * MPC_DT
        acc = g * t_flight / t_stance
        return liftoff - speed * t + 0.5 * acc * t**2

    def get_last_swing_period(name, period):
        # The latest gait period up to the given one where the foot starts a
        # swing
        groups = get_active_gait().groups
        group = next(i for i, g in enumerate(groups) if name in g)
        return period - (period - group) % len(groups)

    def latest_swing(name, k):
        # The gait period of the foot's latest swing to start by stage k. No
        # swings start once the gait stops
        start, stop = est["gait_start"], est["gait_stop"]
        period = (k - start) // get_gait_period_stages()
        if stop is not None:
            period = min(
                period, (stop - start) // get_gait_period_stages() - 1
            )
        return get_last_swing_period(name, period)

    def get_foot_swing_phase(name, k):
        # The foot's swing period and how far it is through it at stage k, or
        # None while it stands
        start = est["gait_start"]
        if start is None or k < start:
            return None
        period = latest_swing(name, k)
        stage = k - start - period * get_gait_period_stages()
        if period < 0 or stage >= get_swing_duration_stages():
            return None
        return period, stage / get_swing_duration_stages()

    def get_swinging_feet(k):
        # The feet swinging at stage k
        if is_robot_airborne(k):
            return FOOT_NAMES
        landing = [
            name
            for name in FOOT_NAMES
            if is_foot_descending_during_landing(name, k)
        ]
        if landing:
            return landing
        return [name for name in FOOT_NAMES if get_foot_swing_phase(name, k)]

    def plan_foothold(name, period):
        # Where the foot lands at the end of its swing in the given period of
        # the gait, placed so that the base passes over the foot's nominal
        # position mid-stance, then shifted for the base velocity error. Feet
        # that are yet to swing, or have landed already, stay where they are
        stage = est["gait_start"] - est["pose_stage"]
        land = period * get_gait_period_stages() + get_swing_duration_stages()
        if period < 0 or stage + land <= 0:
            return est["landed"][name]
        if (name, period) in est["footholds"]:
            return est["footholds"][name, period]
        frozen = est["frozen"].get(name)
        if frozen is not None and frozen[0] == est["gait_start"] + land:
            return frozen[1]
        middle = int(
            land
            + (get_gait_cycle_length_stages() - get_swing_duration_stages())
            / 2
        )
        x, y, yaw = est["poses"][stage + middle]
        nominal = (
            np.array([x, y, 0.0])
            + rotation_matrix_from_z(yaw) @ foot_offsets[name]
        )
        nominal = project_point_to_terrain(nominal + est["foothold_shift"])
        key = (name, est["gait_start"] + land)
        placed = est["placed_footholds"].get(key)
        if placed is None:
            p = project_point_to_terrain(
                find_safe_foothold(name, nominal, yaw)
            )
        else:
            spot, correction = placed
            level = spot[2]
            kept = project_point_to_terrain(nominal + correction)
            near = np.linalg.norm((spot - nominal)[:2]) <= FOOTHOLD_STICK_RANGE
            if is_foothold_safe_at_height(nominal, yaw, level, slack_margin):
                p = nominal
            elif is_foothold_safe_at_height(kept, yaw, level):
                p = kept
            elif near and is_foothold_safe_at_height(spot, yaw, level):
                p = project_point_to_terrain(spot)
            else:
                p = project_point_to_terrain(
                    find_safe_foothold(name, nominal, yaw, level)
                )
                if (
                    stage + land <= get_swing_duration_stages()
                    and abs(p[2] - level) > FOOTHOLD_LEVEL_CHANGE
                    and is_foothold_clear_of_edges(spot, yaw)
                ):
                    p = project_point_to_terrain(spot)
        est["placed_footholds"][key] = (p, (p - nominal) * [1.0, 1.0, 0.0])
        for old in [k for k in est["placed_footholds"] if k[1] < est["stage"]]:
            del est["placed_footholds"][old]
        if stage + land <= FOOTHOLD_FREEZE_STAGES:
            est["frozen"][name] = (est["gait_start"] + land, p)
        est["footholds"][name, period] = p
        return p

    def get_foot_reference(name, k):
        # Returns the foot's reference position at stage k and whether it
        # swings, computing each once per update of the references
        if (name, k) not in est["foot_references"]:
            est["foot_references"][name, k] = compute_foot_reference(name, k)
        return est["foot_references"][name, k]

    def compute_foot_reference(name, k):
        if is_robot_airborne(k) or is_foot_descending_during_landing(name, k):
            return est["landed"][name], True
        start = est["gait_start"]
        if start is None or k < start:
            return est["landed"][name], False

        swing = get_foot_swing_phase(name, k)
        if swing is None:
            return plan_foothold(name, latest_swing(name, k)), False

        period, phase = swing
        # A swing that follows the foot's last landing starts from where the
        # foot is, rather than where it landed
        previous = period - len(get_active_gait().groups)
        lands = (
            est["gait_start"]
            + previous * get_gait_period_stages()
            + get_swing_duration_stages()
        )
        if previous < 0 or lands <= est["pose_stage"]:
            liftoff = est["liftoff"][name]
        else:
            liftoff = plan_foothold(name, previous)
        landing = plan_foothold(name, period)
        if (name, period) not in est["swing_tops"]:
            top = get_max_terrain_height_along_path(liftoff, landing)
            rough = top - min(liftoff[2], landing[2]) > SWING_ROUGH_HEIGHT
            apex = (
                compute_swing_apex_height(liftoff, landing) if rough else None
            )
            est["swing_tops"][name, period] = apex

        # Ease in and out of the swing, so that the foot lifts off and touches
        # down without sliding. Over smooth terrain, midway, the foot is
        # step_height above the higher of its footholds
        apex = est["swing_tops"][name, period]
        blend = 0.5 * (1 - np.cos(np.pi * phase))
        if apex is None:
            p = (1 - blend) * liftoff + blend * landing
            lift = est["step_height"] + abs(landing[2] - liftoff[2]) / 2
            p[2] += lift * np.sin(np.pi * phase) ** 2
            return p, True

        # Over rough terrain, the foot follows a Bezier curve whose middle
        # control points are at the apex height above the footholds, so that
        # it moves across high and comes down onto its foothold from above
        s = blend * blend * (3 - 2 * blend)
        p = (1 - s) * liftoff + s * landing
        p[2] = (
            (1 - blend) ** 3 * liftoff[2]
            + 3 * blend * (1 - blend) * apex
            + blend**3 * landing[2]
        )
        return p, True

    def estimate_terrain_pitch_roll(x, y, yaw):
        # Pitch and roll of the plane fit to the terrain between the feet's
        # nominal positions, for the base pose with the given heading yaw
        R = rotation_matrix_from_z(yaw)
        heights = [
            project_point_to_terrain(np.array([x, y, 0.0]) + R @ offset)[2]
            for offset in tilt_offsets
        ]
        slope = np.linalg.lstsq(tilt_fit, heights, rcond=None)[0]
        pitch, roll = np.arctan(slope[1:])
        return (
            np.clip(pitch, -MAX_TERRAIN_PITCH, MAX_TERRAIN_PITCH),
            np.clip(roll, -MAX_TERRAIN_ROLL, MAX_TERRAIN_ROLL),
        )

    # Points of the patch around a foothold, relative to it
    patch_steps = np.arange(-CONTACT_PATCH, CONTACT_PATCH + 1e-9, 0.02)
    patch_offsets = [
        np.array([dx, dy])
        for dx in patch_steps
        for dy in patch_steps
        if np.hypot(dx, dy) <= CONTACT_PATCH + 1e-9
    ]
    max_contact_slope = np.tan(MAX_CONTACT_TILT)

    def fit_terrain_plane_at_foothold(p):
        # The plane through foothold p fit to the terrain around it, leaving
        # out points near edges, as the terrain for the foot's contact
        key = tuple(p)
        if key not in est["contact_planes"]:
            points = [
                p[:2] + offset
                for offset in patch_offsets
                if get_edge_distance(*(p[:2] + offset)) >= CONTACT_EDGE_GAP
            ]
            slope = np.zeros(2)
            if len(points) >= 3:
                fit = np.array([[1.0, *(q - p[:2])] for q in points])
                heights = [terrain.at(q[0], q[1]).height for q in points]
                slope = np.linalg.lstsq(fit, heights, rcond=None)[0][1:]
            size = np.linalg.norm(slope)
            if size > max_contact_slope:
                slope *= max_contact_slope / size
            est["contact_planes"][key] = ssqpy.model.terrain.SlopeTerrain(
                slope[0], slope[1], p[2] - slope @ p[:2]
            )
        return est["contact_planes"][key]

    def update_ground_mode(contacts, dt):
        # Updates the ground mode of the robot
        total, count = est["vertical_acc"]
        est["vertical_acc"] = (0.0, 0)
        if contacts is None:
            return est["ground_mode"]

        if is_robot_airborne(est["stage"]):
            est["ground_mode"] = GROUND_MODE_AIRBORNE
            return est["ground_mode"]

        if est["ground_mode"] == GROUND_MODE_AIRBORNE:
            est["ground_mode"] = GROUND_MODE_NORMAL
            est["no_contact_time"] = 0.0
            return est["ground_mode"]

        # With no foot on the ground, a standing robot is held up. While
        # walking, it is held up only if the accelerometer has read close to
        # gravity since the feet left the ground, and is falling or stepping
        # down otherwise. The legs swinging in the air shake the base, but
        # cannot move it on average
        if est["ground_mode"] == GROUND_MODE_NORMAL:
            if not any(contacts):
                off_total, off_count = est["off_ground_acc"]
                off_total, off_count = off_total + total, off_count + count
                est["off_ground_acc"] = (off_total, off_count)
                est["held"] = (
                    est["gait_start"] is None
                    or off_count > 0
                    and off_total / off_count > LIFT_MIN_ACC
                )
                est["no_contact_time"] += dt
                if (
                    est["no_contact_time"] >= GROUND_MODE_SWITCH_DELAY
                    and est["held"]
                ):
                    est["ground_mode"] = GROUND_MODE_LIFTED
                    est["stable_contact_time"] = 0.0
                    print("The robot was lifted")

                    # Stop any gait or jump, so that nothing resumes when the
                    # robot is set down
                    est["gait_start"] = None
                    est["gait_stop"] = None
                    est["jump_start"] = None
                    est["jump_ground"] = None
                    est["falling_feet"] = set()
                    est["stand_stage"] = est["stage"]
            else:
                est["no_contact_time"] = 0.0
                est["off_ground_acc"] = (0.0, 0)
                est["held"] = False

        elif est["ground_mode"] == GROUND_MODE_LIFTED:
            if all(contacts):
                est["stable_contact_time"] += dt
                if est["stable_contact_time"] >= GROUND_MODE_SWITCH_DELAY:
                    est["ground_mode"] = GROUND_MODE_SET_DOWN
                    print("The robot was set down")
            else:
                est["stable_contact_time"] = 0.0

        return est["ground_mode"]

    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, nominal_config[7:], C_WGT
    )
    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
    f_cost = ssqpy.model.costs.SquaredForceCost(model, F_WGT)
    w_cost = ssqpy.model.costs.FloatingBaseSquaredVelocityErrorCost(
        model, np.zeros(6), np.array([0.0, 0.0, 0.0, W_WGT, W_WGT, 0.0])
    )

    def compute_jump_flight_configurations():
        # Leg configuration through each stage of the flight and at its end,
        # pulling the feet in and out smoothly, and moving them from where
        # they take off to where they land relative to the base
        speed = compute_jump_forward_speed()
        takeoff = -speed * est["jump_push"] * MPC_DT / 4
        landing = speed * est["jump_brake"] * MPC_DT / 2
        stretch = est["jump_land_height"] - est["jump_takeoff"]
        configs = []
        stages = est["jump_flight"]
        for i in range(stages + 1):
            blend = 0.5 * (1 - np.cos(np.pi * i / stages))
            configs.append(
                compute_stance_joint_configuration(
                    est["jump_takeoff"]
                    + blend * stretch
                    - est["jump_tuck"] * np.sin(np.pi * i / stages) ** 2,
                    0.0,
                    0.0,
                    0.0,
                    (1 - blend) * takeoff + blend * landing,
                )
            )
        return configs

    est["flight_configs"] = compute_jump_flight_configurations()

    def get_jump_flight_configuration(k):
        # The legs' configuration at stage k of the flight, following how far
        # it is from takeoff to the landing, so that the legs move smoothly
        # when the landing moves
        takeoff = est["takeoff_at"]
        land = est["jump_start"] + get_landing_stage()
        i = (k - takeoff) * est["jump_flight"] / (land - takeoff)
        low = int(i)
        share = i - low
        configs = est["flight_configs"]
        if share == 0.0:
            config = configs[low]
        else:
            config = (1 - share) * configs[low] + share * configs[low + 1]
        if est["landing_tilt"] is not None:
            blend = 0.5 * (1 - np.cos(np.pi * i / est["jump_flight"]))
            config = config + blend * est["landing_tilt"]
        return config

    def compute_landing_tilt_correction(R):
        # Change to the legs' landing configuration that keeps the feet
        # landing level and where planned relative to the base, with the
        # base tilted as R
        yaw = np.arctan2(R[1, 0], R[0, 0])
        local = rotation_matrix_from_z(-yaw) @ R
        pitch = np.arcsin(np.clip(local[2, 0], -1.0, 1.0))
        roll = np.arctan2(local[2, 1], local[2, 2])
        ahead = compute_jump_forward_speed() * est["jump_brake"] * MPC_DT / 2
        height = est["jump_land_height"]
        return compute_stance_joint_configuration(
            height, pitch, roll, 0.0, ahead
        ) - compute_stance_joint_configuration(height, 0.0, 0.0, 0.0, ahead)

    def plan_forward_jump_trajectory():
        # Plans a forward jump from where the feet stand, returning whether
        # the robot can make it. With the terrain rising ahead of the front
        # feet within reach, the jump lands on it, and otherwise
        # FORWARD_JUMP_DISTANCE ahead. Each foot lands on the clear spot
        # nearest to where the jump would take it
        yaw = est["poses"][est["stage"] - est["pose_stage"]][2]
        heading = rotation_matrix_from_z(yaw)[:, 0]
        feet = est["landed"]
        start = np.mean([feet[name][2] for name in FOOT_NAMES])
        front = np.mean([feet[name] for name in FOOT_NAMES[:2]], axis=0)
        rear = np.mean([feet[name] for name in FOOT_NAMES[2:]], axis=0)
        length = (front - rear) @ heading
        reach = FORWARD_JUMP_MAX_DISTANCE - length - FORWARD_JUMP_EDGE_MARGIN
        distance = FORWARD_JUMP_DISTANCE
        for ahead in np.arange(0.02, reach + 1e-9, 0.02):
            if (
                project_point_to_terrain(front + ahead * heading)[2]
                > start + FORWARD_JUMP_RISE
            ):
                distance = ahead + length + FORWARD_JUMP_EDGE_MARGIN
                break
        landing = {
            name: project_point_to_terrain(
                find_safe_foothold(name, feet[name] + distance * heading, yaw)
            )
            for name in FOOT_NAMES
        }
        heights = [p[2] for p in landing.values()]
        if max(heights) - min(heights) > FORWARD_JUMP_RISE:
            print(
                "Not jumping forward, the feet would land on different levels"
            )
            return False

        # The base rises at least FORWARD_JUMP_HEIGHT, and
        # FORWARD_JUMP_CLEARANCE above where it lands, over a whole number of
        # stages in the air until it touches down. The push takes it from the
        # crouch to its takeoff height at a constant acceleration
        g = -GRAVITY[2]
        rise = np.mean(heights) - start
        apex = max(FORWARD_JUMP_HEIGHT, rise + FORWARD_JUMP_CLEARANCE)
        speed = np.sqrt(2 * g * apex)
        rise += FORWARD_JUMP_LAND_HEIGHT - FORWARD_JUMP_TAKEOFF_HEIGHT
        flight = (speed + np.sqrt(speed**2 - 2 * g * rise)) / g
        stages = round(flight / MPC_DT)
        flight = stages * MPC_DT
        speed = (rise + g * flight**2 / 2) / flight
        if speed > FORWARD_JUMP_MAX_SPEED:
            print(f"Not jumping forward, {rise:.2f} m up is too high")
            return False
        travel = FORWARD_JUMP_TAKEOFF_HEIGHT - FORWARD_JUMP_CROUCH_HEIGHT
        speed *= FORWARD_JUMP_RISE_GAIN

        est["jump_distance"] = np.mean(
            [(landing[name] - feet[name]) @ heading for name in FOOT_NAMES]
        )
        est["jump_push"] = max(round(2 * travel / (speed * MPC_DT)), 1)
        est["jump_flight"] = stages
        est["jump_takeoff"] = FORWARD_JUMP_TAKEOFF_HEIGHT
        est["jump_rise"] = speed
        est["jump_land_height"] = FORWARD_JUMP_LAND_HEIGHT
        est["jump_land_force"] = FORWARD_JUMP_LAND_FORCE
        est["jump_tuck"] = FORWARD_JUMP_TUCK
        est["jump_ground"] = start
        est["jump_landing"] = landing
        est["touchdown"] = (FORWARD_JUMP_LAND_HEIGHT, speed - g * flight)

        # The landing stops the fall evenly by the crouch height, taking at
        # most as long as the jump in place
        depth = FORWARD_JUMP_LAND_HEIGHT - FORWARD_JUMP_CROUCH_HEIGHT
        land = np.ceil(2 * depth / (g * flight - speed) / MPC_DT)
        est["jump_land"] = min(int(land), JUMP_LAND_STAGES)

        # Stopping the forward speed at FORWARD_JUMP_BRAKE takes longer than
        # stopping the fall. The speed depends on how long that takes, since
        # the feet land half of its travel ahead
        start = (stages + est["jump_push"] / 4) * MPC_DT
        brake = FORWARD_JUMP_BRAKE
        forward = brake * (
            np.sqrt(start**2 + 2 * est["jump_distance"] / brake) - start
        )
        est["jump_brake"] = max(int(np.ceil(forward / (brake * MPC_DT))), 1)
        return True

    base_costs = []
    flight_costs = []
    flight_vel_costs = []
    foot_funcs = {name: [] for name in FOOT_NAMES}

    for k in range(MPC_H):
        base_costs.append(
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                model, "base_link", init_state[:3], np.eye(3), T_WGT, R_WGT
            )
        )
        flight_costs.append(
            ssqpy.model.costs.SquaredConfigurationErrorCost(
                model, est["flight_configs"][0], JUMP_CONFIG_WGT
            )
        )
        flight_vel_costs.append(
            ssqpy.model.costs.SquaredJointVelocityCost(
                model, JUMP_FLIGHT_VEL_WGT
            )
        )
        for cost in [config_cost, v_cost, u_cost, f_cost, w_cost]:
            model.addCost(k, cost)
        model.addCost(k, base_costs[-1])
        model.addCost(k, flight_costs[-1])
        model.addCost(k, flight_vel_costs[-1])

        for name in FOOT_NAMES:
            foot_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
                model, name, est["landed"][name], T_WGT
            )
            # The contact constraint leaves the foot's height free, so that a
            # foot the plan has in stance while it is off the terrain, like
            # one landing late, cannot make the plan infeasible
            foot_constraint = model.addContactConstraint(
                k,
                name,
                CONTACT_MU,
                eq_tol=np.inf,
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

    def update_motion_references():
        current_stage = est["stage"]

        # Look far enough ahead for the footholds of every swing in the
        # horizon. While stepping, start from the measured base position
        pose = est["poses"][current_stage - est["pose_stage"]].copy()
        start = est["gait_start"]
        stepping = start is not None and current_stage >= start
        if stepping:
            pose[:2] = est["filter"][1][:2]
        est["poses"] = predict_future_base_poses(
            pose,
            MPC_H
            + (len(get_active_gait().groups) + 1) * get_gait_period_stages(),
        )
        est["pose_stage"] = current_stage

        shift = np.zeros(3)
        if stepping:
            v_ref = rotation_matrix_from_z(pose[2]) @ np.append(
                est["v_cmd"][:2], 0.0
            )
            shift[:2] = FOOTHOLD_VEL_GAIN * (est["filter"][1][3:5] - v_ref[:2])

            # While turning, the feet land outside the turn, so that the base
            # leans into it
            turn = np.cross(v_ref, [0.0, 0.0, est["v_cmd"][2]])
            shift[:2] += 0.5 * est["base_height"] / -GRAVITY[2] * turn[:2]
            size = np.linalg.norm(shift)
            if size > FOOTHOLD_MAX_SHIFT:
                shift *= FOOTHOLD_MAX_SHIFT / size
        est["foothold_shift"] = shift
        est["footholds"] = {}
        est["foot_references"] = {}
        est["swing_tops"] = {}
        est["contact_planes"] = {}

        orientation = est["base_pitch"], est["base_roll"], est["base_yaw"]
        config_cost.set_desired_configuration(
            compute_stance_joint_configuration(
                est["base_height"], *orientation
            )
        )

        tilt = compute_base_rotation_matrix(*orientation)
        current = get_jump_stage(current_stage)
        landed = current is not None and current >= get_landing_stage()
        for idx in range(MPC_H):
            k = current_stage + idx
            x, y, yaw = est["poses"][idx]

            # Through a forward jump, the base moves ahead of its pose
            j = get_jump_stage(k)
            if j is not None:
                ahead = compute_jump_forward_displacement(j, landed)
                x += ahead * np.cos(yaw)
                y += ahead * np.sin(yaw)

            # Outside of jumps, the base follows the terrain's tilt, keeping
            # its height along the tilted base's up axis
            height = (
                get_gait_base_height(k)
                if j is None
                else get_jump_base_height(j)
            )
            pitch, roll = 0.0, 0.0
            if j is None:
                pitch, roll = estimate_terrain_pitch_roll(x, y, yaw)
            height /= np.cos(pitch) * np.cos(roll)
            if (
                j is not None
                and est["jump_ground"] is not None
                and j < get_landing_stage()
            ):
                height += est["jump_ground"]
            else:
                height += estimate_terrain_height_under_base(x, y, yaw)
            terrain_rotation = rotation_matrix_from_y(
                -pitch
            ) @ rotation_matrix_from_x(roll)
            base_costs[idx].set_desired_translation(np.array([x, y, height]))
            base_costs[idx].set_desired_rotation(
                rotation_matrix_from_z(yaw) @ terrain_rotation @ tilt
            )

            # In the air, the legs hold their configuration rather than
            # following the feet's references. Before takeoff, the base's
            # reference through the flight sets the takeoff speed, but once in
            # the air its path can no longer be steered
            flight = is_robot_airborne(k)
            airborne = flight and is_robot_airborne(current_stage)
            base_costs[idx].set_weight(
                np.zeros(6)
                if airborne
                else np.array([T_WGT, T_WGT, Z_WGT, R_WGT, R_WGT, R_WGT])
            )
            flight_costs[idx].set_enabled(flight)
            if flight:
                flight_costs[idx].set_desired_configuration(
                    get_jump_flight_configuration(k)
                )
            flight_vel_costs[idx].set_enabled(flight)

            land = get_landing_stage()
            landing = j is not None and land <= j < land + est["jump_land"]
            pushing = (
                j is not None
                and est["jump_ground"] is not None
                and JUMP_CROUCH_STAGES <= j < get_takeoff_stage()
            )
            foot_weight = np.array(
                [1.0, 1.0, FORWARD_JUMP_FOOT_WGT if pushing else 1.0]
            )
            max_force = est["jump_land_force"] if landing else MAX_FOOT_FORCE

            for name in FOOT_NAMES:
                foot_pos, swinging = get_foot_reference(name, k)
                foot_cost, foot_constraint, vel_bound = foot_funcs[name][idx]
                foot_constraint.set_max_normal_force(max_force)

                # Swinging feet track their swing trajectory, while stance
                # feet stay in contact. The contact constraint only bounds the
                # foot's force, so the tracking cost and the velocity bound
                # keep stance feet in place
                foot_constraint.set_enabled(not swinging)
                if not swinging:
                    foot_constraint.set_terrain(
                        fit_terrain_plane_at_foothold(foot_pos)
                    )
                if vel_bound is not None:
                    vel_bound.set_enabled(not swinging)
                foot_cost.set_enabled(not flight)
                if pushing and not swinging:
                    foot_pos = foot_pos.copy()
                    foot_pos[2] = est["foot_heights"][name]
                foot_cost.set_desired_translation(foot_pos)
                foot_cost.set_weight(foot_weight)

    update_motion_references()

    model.addCost(MPC_H, v_cost)

    model.finalize(init_states=[init_state])

    # The torques are computed alongside the solver, so they use their own
    # model, which is never solved. Its contacts give its controls the feet's
    # forces, as the plan's have, and it only has costs since a model cannot
    # be finalized without them
    torque_model = build_go2_mpc_model(1)
    for name in FOOT_NAMES:
        torque_model.addContactConstraint(0, name, CONTACT_MU)
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
        qp_iters=25,
        max_time=MAX_SOLVE_TIME,
    )

    def update_controller(
        v_cmd,
        dpad_x,
        dpad_y,
        orientation,
        jump,
        switch_gait,
        config,
        vel,
        contacts=None,
        forward_jump=False,
        dt=MPC_DT,
    ):
        current_stage = est["stage"]
        moving = np.max(np.abs(v_cmd)) > MOVE_THRESHOLD

        if update_ground_mode(contacts, dt) in [
            GROUND_MODE_LIFTED,
            GROUND_MODE_SET_DOWN,
        ]:
            moving = False
            v_cmd = np.zeros(3)
            dpad_x = 0
            dpad_y = 0
            orientation = [0, 0, 0]
            jump = False
            forward_jump = False
            est["stance_feet"] = []
            # The velocity estimate drifts in the air, which must not count as
            # a push that starts the gait
            est["stand_stage"] = current_stage

        # Jump in place or forward on each new press of the jump or forward
        # jump button while standing still with the base level and facing
        # straight, and ignore move and tilt commands until the jump is over
        level = not np.any(orientation) and not any(
            est[key] for key in ["base_pitch", "base_roll", "base_yaw"]
        )
        if (
            est["jump_start"] is not None
            and current_stage >= est["jump_start"] + JUMP_STAGES
        ):
            est["jump_start"] = None
        pressed = jump and not est["jump_held"]
        forward = forward_jump and not est["forward_jump_held"]
        if (
            JUMP_ENABLED
            and (pressed or forward)
            and est["jump_start"] is None
            and est["gait_start"] is None
            and not moving
            and not np.any(est["v_cmd"])
            and level
        ):
            if pressed:
                est["jump_distance"] = 0.0
                est["jump_push"] = JUMP_PUSH_STAGES
                est["jump_flight"] = JUMP_FLIGHT_STAGES
                est["jump_land"] = JUMP_LAND_STAGES
                est["jump_brake"] = JUMP_LAND_STAGES
                est["jump_takeoff"] = JUMP_TAKEOFF_HEIGHT
                est["jump_rise"] = JUMP_SPEED
                est["jump_land_height"] = JUMP_TAKEOFF_HEIGHT
                est["jump_land_force"] = JUMP_LAND_FORCE
                est["jump_tuck"] = JUMP_TUCK
                est["jump_ground"] = None
                est["jump_landing"] = None
                est["touchdown"] = (JUMP_TAKEOFF_HEIGHT, -JUMP_SPEED)
            if pressed or plan_forward_jump_trajectory():
                est["jump_start"] = current_stage + 1
                est["takeoff_at"] = est["jump_start"] + get_takeoff_stage()
                est["falling_feet"] = set()
                est["flight_configs"] = compute_jump_flight_configurations()
        est["jump_held"] = jump
        est["forward_jump_held"] = forward_jump

        # Start the landing as soon as a foot touches down, and follow the
        # other feet down as they touch down too. Nothing measures the base's
        # height in the air, so it is reset from the feet that touched down.
        # Their joints have yet to move on impact, so they are no good for
        # the velocity
        j = get_jump_stage(current_stage)
        takeoff = get_takeoff_stage()
        touchdown = get_landing_stage()
        if contacts is not None and j is not None:
            touching = [n for n, c in zip(FOOT_NAMES, contacts) if c]
            if takeoff + 1 < j < touchdown and touching:
                est["jump_start"] = current_stage - touchdown
                est["falling_feet"] = set(FOOT_NAMES) - set(touching)
                _, positions, _ = estimate_base_motion_from_stance_feet(
                    config, vel, touching
                )
                reset = np.mean([positions[n][2] for n in touching])
                # Forward jumps land away from where the feet took off, so
                # the height comes from the terrain where they touch down
                if est["jump_landing"] is not None:
                    _, R, feet = estimate_robot_state(config, vel)
                    reset = np.mean(
                        [
                            project_point_to_terrain(
                                est["filter"][1][:3] + R @ feet[n]
                            )[2]
                            - (R @ feet[n])[2]
                            for n in touching
                        ]
                    )
                # The filter runs alongside, so it takes the reset as well
                filter_time, x, P = est["filter"]
                x = x.copy()
                x[2] = reset
                est["filter"] = (filter_time, x, P)
                est["filter_reset"] = reset
            elif touchdown <= j < touchdown + est["jump_land"]:
                est["falling_feet"] -= set(touching)
            else:
                est["falling_feet"] = set()
            est["touch_stage"] = current_stage + JUMP_TOUCH_STAGES

        # The feet land out of place, so they step back under the hips once
        # they are all down, which also catches the robot if it landed
        # unsteadily. Without foot contacts, they are down on schedule
        j = get_jump_stage(current_stage)
        if (
            j is not None
            and touchdown <= j <= touchdown + est["jump_land"]
            and not est["falling_feet"]
            and est["gait_start"] is None
        ):
            est["gait_start"] = current_stage + 1 + JUMP_TROT_DELAY
            est["gait_stop"] = (
                est["gait_start"]
                + JUMP_TROT_CYCLES * get_gait_cycle_length_stages()
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

        # The map is left out while jumping in place and in MAP_FREE_GAITS,
        # taking the terrain to be flat at its height under the feet from
        # then on
        in_place = est["jump_start"] is not None and est["jump_ground"] is None
        if not in_place and est["gait"] not in MAP_FREE_GAITS:
            est["flat_ground_height"] = None
        elif est["flat_ground_height"] is None:
            est["flat_ground_height"] = np.mean(
                [est["landed"][name][2] for name in FOOT_NAMES]
            )

        # Start stepping on a move command or a push, and stop at the end of
        # the current period once the command has been at rest for a while,
        # counting from when the base's reference has slowed to rest or from
        # the push. With no foot on the ground, the velocity estimate follows
        # the accelerometer alone, so it takes no push then
        pushed = (
            est["gait_start"] is None
            and est["jump_start"] is None
            and current_stage >= est["stand_stage"] + PUSH_SETTLE_STAGES
            and np.linalg.norm(est["filter"][1][3:5]) > PUSH_SPEED
            and (contacts is None or any(contacts))
        )
        slowing = np.max(np.abs(est["v_cmd"])) > MOVE_THRESHOLD
        est["idle_stages"] = (
            0 if moving or slowing or pushed else est["idle_stages"] + 1
        )
        if moving or pushed:
            if est["gait_start"] is None:
                est["gait_start"] = current_stage + GAIT_START_STAGES
            est["gait_stop"] = None
        elif (
            est["idle_stages"] >= IDLE_STAGES
            and est["gait_start"] is not None
            and est["gait_stop"] is None
        ):
            periods = -(
                -(current_stage - est["gait_start"])
                // get_gait_period_stages()
            )
            est["gait_stop"] = (
                est["gait_start"] + max(periods, 0) * get_gait_period_stages()
            )

        # Stay in place while not stepping
        stepping = (
            est["gait_start"] is not None
            and current_stage >= est["gait_start"]
            and est["gait_stop"] is None
        )
        if not stepping:
            v_cmd = np.zeros(3)
        factor = get_active_gait().speed
        capped = min(factor, 1.0)
        forward = factor if v_cmd[0] > 0 else capped
        v_cmd = np.array(v_cmd) * [forward, capped, capped]
        speed = np.linalg.norm(est["v_cmd"][:2])
        if speed > 0.0:
            max_yaw = MAX_TURN_ACC / speed
            v_cmd[2] = np.clip(v_cmd[2], -max_yaw, max_yaw)

        max_change = np.array([MAX_ACC, MAX_ACC, MAX_YAW_ACC]) * MPC_DT
        est["v_cmd"] += np.clip(v_cmd - est["v_cmd"], -max_change, max_change)

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

        # The feet that were in stance during the last stage, which are all
        # still on the ground
        last_swing = get_swinging_feet(current_stage - 1)
        stance_feet = [name for name in FOOT_NAMES if name not in last_swing]
        state, R, feet = estimate_robot_state(config, vel)
        est["foot_heights"] = {
            name: (state[:3] + R @ feet[name])[2] for name in FOOT_NAMES
        }

        # In the air of a forward jump, the legs land level for however the
        # base has tilted
        est["landing_tilt"] = None
        if est["jump_ground"] is not None and is_robot_airborne(current_stage):
            est["landing_tilt"] = compute_landing_tilt_correction(R)

        # In the air, the landing moves to when the lowest foot is predicted
        # to reach the ground, so that the plan never reaches for the ground
        # before the feet can get there. The feet are taken as stretched out
        # for the landing, since the tuck would otherwise put it off
        j = get_jump_stage(current_stage)
        if j is not None and takeoff + 1 < j < touchdown:
            g = -GRAVITY[2]
            rise = (R @ state[19:22])[2]
            reach = np.array([0.0, 0.0, est["jump_land_height"]])
            feet_down = [
                state[:3] + R @ (foot_offsets[name] - reach)
                for name in FOOT_NAMES
            ]
            # A forward jump tilts the legs to keep the feet level below the
            # base, at the landing height
            if est["landing_tilt"] is not None:
                for p in feet_down:
                    p[2] = state[2] - est["jump_land_height"]
            # Forward jumps land on the terrain at their landing spots,
            # rather than on the terrain below the feet
            if est["jump_landing"] is None:
                below = [project_point_to_terrain(p)[2] for p in feet_down]
            else:
                below = [est["jump_landing"][name][2] for name in FOOT_NAMES]
            drop = max(min(p[2] - z for p, z in zip(feet_down, below)), 0.0)
            fall_time = (rise + np.sqrt(rise**2 + 2 * g * drop)) / g
            stages = round(fall_time / MPC_DT)
            # With foot contacts measured, a forward jump's landing starts at
            # the first stage after the feet are predicted to touch down,
            # since a landing that starts early stops the feet in the air and
            # folds the legs, while touching down early starts it anyway
            if est["landing_tilt"] is not None and contacts is not None:
                stages = int(np.ceil(fall_time / MPC_DT))
            stages = min(max(stages, 1), touchdown - takeoff - 2)
            est["jump_start"] = current_stage + stages - touchdown

        # Feet finishing their swing keep the spot they actually landed on
        for name in last_swing:
            if name not in get_swinging_feet(current_stage):
                p = project_point_to_terrain(state[:3] + R @ feet[name])
                if abs(p[2] - est["landed"][name][2]) < LAND_HOLD:
                    p[2] = est["landed"][name][2]
                est["landed"][name] = p

        # Feet in stance may have slipped from where they landed, so their
        # next swings start from where they are measured to be. Swinging feet
        # keep the spot their swing started from
        for name in FOOT_NAMES:
            if name not in get_swinging_feet(current_stage):
                est["liftoff"][name] = state[:3] + R @ feet[name]

        # The feet land from the jump away from where they took off, so the
        # base's reference moves to stand centered over them again
        if is_robot_airborne(current_stage - 1) and not is_robot_airborne(
            current_stage
        ):
            x, y, yaw = est["poses"][current_stage - est["pose_stage"]]
            center = np.mean([est["landed"][n] for n in FOOT_NAMES], axis=0)
            offset = np.mean([foot_offsets[n] for n in FOOT_NAMES], axis=0)
            x, y, _ = center - rotation_matrix_from_z(yaw) @ offset
            est["poses"] = [np.array([x, y, yaw])]
            est["pose_stage"] = current_stage

            # The landing starts from the base's measured height above the
            # terrain and its fall
            fall = min((R @ state[19:22])[2], 0.0)
            est["touchdown"] = (
                state[2] - estimate_terrain_height_under_base(x, y, yaw),
                fall,
            )

        # Once every foot has landed after the gait stops, the robot stands
        if (
            est["gait_stop"] is not None
            and current_stage >= get_gait_stop_stage()
        ):
            est["gait_start"] = None
            est["gait_stop"] = None
            est["stand_stage"] = current_stage

        # The filter runs until the next stage's update, while the robot may
        # still be in either stage, so it takes the feet in stance in both.
        # Right after a gait switch there are none, and it follows the
        # accelerometer alone. With foot contacts measured, feet that have
        # yet to touch down are left out as well
        est["stance_feet"] = [
            name
            for i, name in enumerate(FOOT_NAMES)
            if name in stance_feet
            and name not in get_swinging_feet(current_stage)
            and (contacts is None or contacts[i])
        ]
        est["falling_freely"] = (
            est["jump_ground"] is not None
            and is_robot_airborne(current_stage)
            and (contacts is None or not any(contacts))
        )

        update_motion_references()

        # Feet in stance over the stages the plan is played back for
        est["plan_stance"] = [
            [
                name
                for name in FOOT_NAMES
                if not get_foot_reference(name, current_stage + k)[1]
            ]
            for k in range(PLAN_STAGES)
        ]
        est["stage"] += 1

        return state

    def prepare_mpc_playback_plan(x, u):
        # The plan's states at the start of its first stages (and at the end of
        # the last one), its controls and stance feet over those stages, and
        # whether the next plan starts partly from this one's legs, which it
        # does in the air of a forward jump
        states = [np.array(x.stage(k)) for k in range(PLAN_STAGES + 1)]
        controls = [np.array(u.stage(k)) for k in range(PLAN_STAGES)]
        airborne = est["jump_ground"] is not None and is_robot_airborne(
            est["stage"]
        )
        return states, controls, est["plan_stance"], airborne

    def compute_joint_torque_commands(commands, elapsed, config, vel):
        states, controls, plan_stance, _ = commands

        # Inverse dynamics at the measured orientation, joints and angular
        # velocity. The base position does not enter it, and the base linear
        # velocity estimate jumps at touchdown, so the plan's is kept
        planned = interpolate_planned_state(states, elapsed)
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

    def flat_ground_height():
        return est["flat_ground_height"]

    def is_jump_active():
        return est["jump_start"] is not None

    def get_stance_foot_mask():
        return [name in est["stance_feet"] for name in FOOT_NAMES]

    def is_robot_lifted():
        return est["ground_mode"] == GROUND_MODE_LIFTED

    def is_robot_set_down():
        return est["ground_mode"] == GROUND_MODE_SET_DOWN

    def is_lift_pending():
        # Whether the robot is held up with no foot touching, for
        # LIFT_PENDING_TIME while walking, while it has not been taken to be
        # lifted yet, outside of jumps, whose feet leave the ground anyway
        wait = 0.0 if est["gait_start"] is None else LIFT_PENDING_TIME
        return (
            est["ground_mode"] == GROUND_MODE_NORMAL
            and est["held"]
            and est["no_contact_time"] > wait
            and est["jump_start"] is None
        )

    return (
        model,
        mpc,
        update_controller,
        prepare_mpc_playback_plan,
        compute_joint_torque_commands,
        update_state_estimator,
        flat_ground_height,
        is_jump_active,
        get_stance_foot_mask,
        is_robot_lifted,
        is_lift_pending,
        is_robot_set_down,
    )


def interpolate_planned_state(states, elapsed):
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


def propagate_state_over_interval(state, commands, start, end):
    """
    Predicts how the given state changes between two times since the plan in
    commands started, by applying the change the plan predicts over them
    """
    states = commands[0]
    before = interpolate_planned_state(states, start)
    after = interpolate_planned_state(states, end)

    predicted = state + after - before

    # In the air, the legs start partly from the plan's
    if commands[3]:
        for part in [slice(7, 19), slice(25, 37)]:
            predicted[part] = after[part] + FORWARD_JUMP_LEG_PULL * (
                predicted[part] - after[part]
            )

    # Rotate the base by the plan's rotation in between, in the base frame
    rotation = quat_xyzw_to_rotation_matrix(
        before[3:7]
    ).T @ quat_xyzw_to_rotation_matrix(after[3:7])
    predicted[3:7] = rotation_matrix_to_quat_xyzw(
        quat_xyzw_to_rotation_matrix(state[3:7]) @ rotation
    )
    return predicted


def get_joint_references_from_plan(commands, elapsed):
    """
    Returns the joint position and velocity targets for the given time since
    the plan started, interpolated along the plan
    """
    state = interpolate_planned_state(commands[0], elapsed)
    return state[7:19], state[25:37]
