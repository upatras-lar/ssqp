import mujoco
import mujoco.viewer
import glfw
import ssqpy
import numpy as np
from time import time
from go2_controllers.joystick import Controller
from utils_go2_mujoco import (
    ssqp_to_mujoco_state,
    ssqp_to_mujoco_control,
    ssqp_to_mujoco_ctrl_kin,
    mujoco_to_ssqp_state,
    clip_joints_to_limits,
)

ssqpy.setSilentMode()
np.set_printoptions(suppress=True)


dt = 0.002
mpc_H = 20
# Keep mpc_dt a multiple of dt, so that plan stages line up with sim steps
mpc_dt = 0.024

# stage indexes the plan (mpc_dt per step), so only advance it once per
# planning step, not every simulation step
STAGE_EVERY = round(mpc_dt / dt)

# Print and sync the viewer once per plan stage, so that each measured window
# holds exactly one MPC solve
DISPLAY_EVERY = STAGE_EVERY

dynamics_mode = ssqpy.model.SolverMode.ReducedInverseDynamics

T_WGT = 1.0
R_WGT = 1.0
C_WGT = 1e-2
V_WGT = 1e-3

# The controls are joint torques in forward dynamics, but joint accelerations
# in reduced inverse dynamics, so the control and force weights depend on the
# mode
if dynamics_mode == ssqpy.model.SolverMode.ForwardDynamics:
    U_WGT = 1e-3
    F_WGT = 1e-4
elif dynamics_mode == ssqpy.model.SolverMode.ReducedInverseDynamics:
    U_WGT = 1e-5
    F_WGT = 2e-4

CONTACT_MU = 0.6

# Plan stages per swing phase
GAIT_PERIOD = 10

# Default step height, and the range and rate of change for adjusting it with
# the D-pad
STEP_HEIGHT = 0.05
MIN_STEP_HEIGHT = 0.03
MAX_STEP_HEIGHT = 0.1
STEP_HEIGHT_RATE = 0.05

# Limits on the commanded velocity, and on how quickly the command may change
MAX_VEL_FRONT = 0.5
MAX_VEL_BACK = 0.5
MAX_VEL_SIDE = 0.3
MAX_VEL_YAW = 0.8
MAX_ACC = 1.0
MAX_YAW_ACC = 2.0

# Default base height, and the range and rate of change for adjusting it with
# the D-pad
BASE_HEIGHT = 0.34
MIN_BASE_HEIGHT = 0.28
MAX_BASE_HEIGHT = 0.38
HEIGHT_RATE = 0.05

# Base pitch while holding the buttons that tilt the front up or down, and the
# rate at which it changes, including back to level on release
MAX_BASE_PITCH = 0.25
PITCH_RATE = 0.3
PITCH_UP_BUTTON = "BTN_Z"
PITCH_DOWN_BUTTON = "BTN_TR"

# Base roll while holding the buttons that tilt the base to the left or right,
# and the rate at which it changes, including back to level on release
MAX_BASE_ROLL = 0.5
ROLL_RATE = 0.5
ROLL_LEFT_BUTTON = "BTN_NORTH"
ROLL_RIGHT_BUTTON = "BTN_EAST"

# Base yaw relative to the walking direction while holding the buttons that
# turn the front to the left or right, with the feet staying in place, and the
# rate at which it changes, including back to straight on release
MAX_BASE_YAW = 0.3
YAW_RATE = 0.5
YAW_LEFT_BUTTON = "BTN_WEST"
YAW_RIGHT_BUTTON = "BTN_TL"

THIGH_LENGTH = 0.213
# Distance from the knee to the foot frame
CALF_LENGTH = 0.23648

model = ssqpy.model.Model(
    mpc_H,
    mpc_dt,
    mjcf_path="../../assets/mjcf/go2/go2.xml",
    # urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    solver_mode=dynamics_mode,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()


def leg_config(height):
    # Hip, thigh and calf angles placing the foot straight below the hip
    calf = -np.arccos(
        (height**2 - THIGH_LENGTH**2 - CALF_LENGTH**2)
        / (2 * THIGH_LENGTH * CALF_LENGTH)
    )
    thigh = np.arcsin(-CALF_LENGTH * np.sin(calf) / height)
    return np.array([0.0, thigh, calf])


base_pos = np.array([0.0, 0.0, BASE_HEIGHT])

base_config = np.tile(leg_config(BASE_HEIGHT), 4)

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[6] = 1.0
init_state[7:19] = base_config


foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Trot pairs, swinging in alternate gait periods
SWING_PAIRS = [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]]

# The base starts facing along the world x axis, away from the scene's
# obstacles
R = np.eye(3)

# Foot positions relative to the base, on the ground
foot_offsets = {
    name: model.forwardKinematics(init_state[:nq], name)[0] - base_pos
    for name in foot_names
}
for offset in foot_offsets.values():
    offset[2] = 0.0

# Hip joint positions relative to the base, and the sideways offset from each
# hip joint to its thigh joint
hip_offsets = {
    name: model.forwardKinematics(
        init_state[:nq], name.replace("foot", "hip")
    )[0]
    - base_pos
    for name in foot_names
}
HIP_LENGTH = abs(
    model.forwardKinematics(init_state[:nq], "FL_thigh")[0][1]
    - model.forwardKinematics(init_state[:nq], "FL_hip")[0][1]
)

# Commanded velocity (forward, sideways and yaw) in the base frame
v_cmd = np.zeros(3)
base_height = BASE_HEIGHT
step_height = STEP_HEIGHT
# Positive pitch tilts the front of the base up, positive roll tilts the base
# to the right, and positive yaw turns the front to the left
base_pitch = 0.0
base_roll = 0.0
base_yaw = 0.0

# Reference base poses (x, y and yaw) for the stages from pose_stage onwards,
# extrapolated from the current command
pose_stage = 0
poses = [np.zeros(3)]

# Where each foot last landed
landed = {name: foot_offsets[name].copy() for name in foot_names}


def rotation_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def leg_ik(foot, side):
    # Hip, thigh and calf angles placing the foot at the given position
    # relative to the hip joint, in the base frame. side is 1 for the left
    # legs and -1 for the right ones. Feet out of reach get a straight leg
    # pointing at them
    x, y, z = foot
    length = np.sqrt(y**2 + z**2 - HIP_LENGTH**2)
    hip = np.arctan2(z, y) - np.arctan2(-length, side * HIP_LENGTH)
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
        CALF_LENGTH * np.sin(calf), THIGH_LENGTH + CALF_LENGTH * np.cos(calf)
    )
    return np.array([hip, thigh, calf])


def base_rotation(pitch, roll, yaw):
    # Base orientation relative to the walking direction
    return rotation_z(yaw) @ rotation_y(-pitch) @ rotation_x(roll)


def posture_config(height, pitch, roll, yaw):
    # Joint angles keeping each foot on its nominal spot on the ground, with
    # the base at the given height and orientation
    R = base_rotation(pitch, roll, yaw)
    base = np.array([0.0, 0.0, height])
    return np.hstack(
        [
            leg_ik(
                R.T @ (foot_offsets[name] - base) - hip_offsets[name],
                1 if foot_offsets[name][1] > 0 else -1,
            )
            for name in foot_names
        ]
    )


def predict_poses(pose, count):
    predicted = [pose]
    for _ in range(count):
        x, y, yaw = predicted[-1]
        c, s = np.cos(yaw), np.sin(yaw)
        predicted.append(
            np.array(
                [
                    x + (c * v_cmd[0] - s * v_cmd[1]) * mpc_dt,
                    y + (s * v_cmd[0] + c * v_cmd[1]) * mpc_dt,
                    yaw + v_cmd[2] * mpc_dt,
                ]
            )
        )
    return predicted


def foothold(name, period):
    # Where the foot lands at the end of its swing in the given period, placed
    # so that the base passes over the foot's nominal position mid-stance
    if (period + 1) * GAIT_PERIOD <= pose_stage:
        return landed[name]
    x, y, yaw = poses[int((period + 1.5) * GAIT_PERIOD) - pose_stage]
    return np.array([x, y, 0.0]) + rotation_z(yaw) @ foot_offsets[name]


def foot_reference(name, k):
    # Returns the foot's reference position at stage k and whether it swings
    period, phase = divmod(k, GAIT_PERIOD)
    phase /= GAIT_PERIOD

    if name not in SWING_PAIRS[period % 2]:
        return foothold(name, period - 1), False

    # Ease in and out of the swing, so that the foot lifts off and touches
    # down without sliding
    blend = 0.5 * (1 - np.cos(np.pi * phase))
    p = (1 - blend) * foothold(name, period - 2) + blend * foothold(
        name, period
    )
    p[2] += step_height * np.sin(np.pi * phase) ** 2
    return p, True


config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[7:nq], C_WGT
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
f_cost = ssqpy.model.costs.SquaredForceCost(model, F_WGT)

base_costs = []
foot_funcs = {name: [] for name in foot_names}

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, f_cost)

    base_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, "base_link", base_pos, R, T_WGT, R_WGT
        )
    )
    model.addCost(k, base_costs[-1])

    for name in foot_names:
        foot_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
            model, name, landed[name], T_WGT
        )
        foot_constraint = model.addContactConstraint(
            k, name, CONTACT_MU, eq_tol=0.02, max_normal_force=150
        )

        model.addCost(k, foot_cost)

        foot_funcs[name].append((foot_cost, foot_constraint))


def update_command(joystick):
    global base_height, step_height, base_pitch, base_roll, base_yaw

    target = np.array(
        joystick.read(MAX_VEL_FRONT, MAX_VEL_BACK, MAX_VEL_SIDE, MAX_VEL_YAW)
    )
    max_change = np.array([MAX_ACC, MAX_ACC, MAX_YAW_ACC]) * mpc_dt
    v_cmd[:] += np.clip(target - v_cmd, -max_change, max_change)

    base_height = np.clip(
        base_height - joystick.dpad_y * HEIGHT_RATE * mpc_dt,
        MIN_BASE_HEIGHT,
        MAX_BASE_HEIGHT,
    )

    step_height = np.clip(
        step_height + joystick.dpad_x * STEP_HEIGHT_RATE * mpc_dt,
        MIN_STEP_HEIGHT,
        MAX_STEP_HEIGHT,
    )

    tilt = (PITCH_UP_BUTTON in joystick.buttons) - (
        PITCH_DOWN_BUTTON in joystick.buttons
    )
    max_change = PITCH_RATE * mpc_dt
    base_pitch += np.clip(
        tilt * MAX_BASE_PITCH - base_pitch, -max_change, max_change
    )

    tilt = (ROLL_RIGHT_BUTTON in joystick.buttons) - (
        ROLL_LEFT_BUTTON in joystick.buttons
    )
    max_change = ROLL_RATE * mpc_dt
    base_roll += np.clip(
        tilt * MAX_BASE_ROLL - base_roll, -max_change, max_change
    )

    turn = (YAW_LEFT_BUTTON in joystick.buttons) - (
        YAW_RIGHT_BUTTON in joystick.buttons
    )
    max_change = YAW_RATE * mpc_dt
    base_yaw += np.clip(turn * MAX_BASE_YAW - base_yaw, -max_change, max_change)


def update_model(current_stage):
    global pose_stage, poses

    # Feet finishing their swing keep the foothold they were aiming for
    period = current_stage // GAIT_PERIOD
    if current_stage > 0 and current_stage % GAIT_PERIOD == 0:
        for name in SWING_PAIRS[(period - 1) % 2]:
            landed[name] = foothold(name, period - 1)

    # Look far enough ahead for the footholds of every swing in the horizon
    poses = predict_poses(
        poses[current_stage - pose_stage], mpc_H + 3 * GAIT_PERIOD
    )
    pose_stage = current_stage

    config_cost.set_desired_configuration(
        posture_config(base_height, base_pitch, base_roll, base_yaw)
    )

    for idx in range(mpc_H):
        k = current_stage + idx
        x, y, yaw = poses[idx]

        base_costs[idx].set_desired_translation(np.array([x, y, base_height]))
        base_costs[idx].set_desired_rotation(
            rotation_z(yaw) @ base_rotation(base_pitch, base_roll, base_yaw)
        )

        for name in foot_names:
            foot_pos, swinging = foot_reference(name, k)
            foot_cost, foot_constraint = foot_funcs[name][idx]

            # Swinging feet track their swing trajectory, while stance feet
            # stay in contact. The contact constraint only bounds the foot's
            # height, so the tracking cost also keeps stance feet from sliding
            foot_constraint.set_enabled(not swinging)
            foot_cost.set_desired_translation(foot_pos)


update_model(0)

model.addCost(mpc_H, v_cost)

model.finalize(init_states=[init_state])


ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=2, qp_iters=100)

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state(init_state)[:nq]

np.set_printoptions(suppress=True)

Kp = 0.01
Kd = 2 * np.sqrt(Kp)

stage = 0
step = 0

joystick = Controller()

# Start the viewer maximized
glfw.init()
glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        display = step % DISPLAY_EVERY == 0

        """
        if display:
            print(f"\rSimulation time: {mjdata.time:.2f}s")

            if step > 0:
                loop_rate = DISPLAY_EVERY / (time() - display_time)
                print(
                    f"\rLoop frequency: {loop_rate:.2f}Hz\t"
                    f"MPC frequency: {(loop_rate / STAGE_EVERY):.2f}Hz"
                )

            display_time = time()
        """

        state = mujoco_to_ssqp_state(np.hstack((mjdata.qpos, mjdata.qvel)))
        state = clip_joints_to_limits(state, mjmodel)

        # Solve once per plan stage and hold the stage's torques in between,
        # as the planner assumes
        if step % STAGE_EVERY == 0:
            try:
                x, u = mpc.step(state)
                xk = ssqp_to_mujoco_state(x.stage(1))

                if dynamics_mode == ssqpy.model.SolverMode.ForwardDynamics:
                    uk = ssqp_to_mujoco_control(u.stage(0))
                elif (
                    dynamics_mode
                    == ssqpy.model.SolverMode.ReducedInverseDynamics
                ):
                    uk = ssqp_to_mujoco_control(
                        np.hstack(
                            (
                                np.zeros(6),
                                model.getActuatedTorqueVector(
                                    state, u.stage(0)
                                ),
                            )
                        )
                    )

            except RuntimeError:
                print("Exiting after regularization failure")
                break

        mjdata.ctrl = (
            ssqp_to_mujoco_ctrl_kin(Kp * (xk[7:19] - state[7:19]))
            + ssqp_to_mujoco_ctrl_kin(Kd * (xk[25:] - state[25:]))
            + uk[6:18]
        )

        step += 1
        if step % STAGE_EVERY == 0:
            stage += 1
            update_command(joystick)
            update_model(stage)

        mujoco.mj_step(mjmodel, mjdata)

        if display:
            viewer.sync()

print()
