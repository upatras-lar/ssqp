import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep
from go2_controllers.JointSpacePD import JointSpacePD

ssqpy.setSilentMode()


def quat_wxyz_to_xyzw(wxyz):
    xyzw = np.zeros(4)
    xyzw[:3] = wxyz[1:]
    xyzw[3] = wxyz[0]
    return xyzw


def quat_xyzw_to_wxyz(xyzw):
    wxyz = np.zeros(4)
    wxyz[0] = xyzw[3]
    wxyz[1:] = xyzw[:3]
    return wxyz


def ssqp_to_mujoco_state(ssqp_state):
    mujoco_state = ssqp_state.copy()
    mujoco_state[3:7] = quat_xyzw_to_wxyz(ssqp_state[3:7])
    mujoco_state[7:10] = ssqp_state[10:13]
    mujoco_state[10:13] = ssqp_state[7:10]
    mujoco_state[13:16] = ssqp_state[16:19]
    mujoco_state[16:19] = ssqp_state[13:16]
    mujoco_state[25:28] = ssqp_state[28:31]
    mujoco_state[28:31] = ssqp_state[25:28]
    mujoco_state[31:34] = ssqp_state[34:37]
    mujoco_state[34:37] = ssqp_state[31:34]
    return mujoco_state


def ssqp_to_mujoco_control(ssqp_control):
    mujoco_control = ssqp_control.copy()
    mujoco_control[6:9] = ssqp_control[9:12]
    mujoco_control[9:12] = ssqp_control[6:9]
    mujoco_control[12:15] = ssqp_control[15:18]
    mujoco_control[15:18] = ssqp_control[12:15]
    return mujoco_control


def mujoco_to_ssqp_state(mujoco_state):
    ssqp_state = mujoco_state.copy()
    ssqp_state[3:7] = quat_wxyz_to_xyzw(mujoco_state[3:7])
    ssqp_state[7:10] = mujoco_state[10:13]
    ssqp_state[10:13] = mujoco_state[7:10]
    ssqp_state[13:16] = mujoco_state[16:19]
    ssqp_state[16:19] = mujoco_state[13:16]
    ssqp_state[25:28] = mujoco_state[28:31]
    ssqp_state[28:31] = mujoco_state[25:28]
    ssqp_state[31:34] = mujoco_state[34:37]
    ssqp_state[34:37] = mujoco_state[31:34]
    return ssqp_state


dt = 0.002
mpc_H = 20
mpc_dt = 0.1

T_WGT = 20.0
R_WGT = 1.0
C_WGT = 1e-2
V_WGT = 1e-1
U_WGT = 1e-3

CONTACT_MU = 0.6
GAIT_PERIOD = 60

dynamics_mode = ssqpy.model.DynamicsMode.Forward

model = ssqpy.model.Model(
    mpc_H,
    mpc_dt,
    mjcf_path="../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    dynamics_mode=dynamics_mode,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

base_pos = np.array([0.0, 0.0, 0.389])
target_pos = np.array([-0.5, 0.0, 0.389])

base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[5] = 1.0
init_state[7:19] = base_config


# trot pairs
# phase 0: FL + RR swing
# phase 1: FR + RL swing
def get_swing_feet(k):
    phase = (k // GAIT_PERIOD) % 2
    if phase == 0:
        return ["FL_foot", "RR_foot"]
    else:
        return ["FR_foot", "RL_foot"]


def swing_trajectory(p0, pf, phase, height=0.05):
    """
    p0: lift-off position
    pf: landing position
    phase: [0,1]
    """
    p = (1 - phase) * p0 + phase * pf
    p[2] += height * np.sin(np.pi * phase)
    return p


STEP_LENGTH = 0.1


def compute_next_foothold(current_pos):
    foothold = current_pos.copy()
    foothold[0] += STEP_LENGTH
    return foothold


H = 500
traj = []
for i in range(H + 1):
    traj.append(base_pos + i * (target_pos - base_pos) / H)

foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# initialize from initial state
foot_positions = {
    name: model.forwardKinematics(init_state[:nq], name)[0]
    for name in foot_names
}

foot_trajs = {name: [foot_positions[name].copy()] for name in foot_names}

# keep track of swing starts
swing_start = {name: foot_positions[name].copy() for name in foot_names}
swing_target = {name: foot_positions[name].copy() for name in foot_names}
swing_phase = {name: 0.0 for name in foot_names}

for k in range(1, H + 1):
    swing_feet = get_swing_feet(k)

    for name in foot_names:
        if name in swing_feet:
            # start swing if just entered
            if swing_phase[name] == 0.0:
                swing_start[name] = foot_positions[name].copy()
                swing_target[name] = compute_next_foothold(swing_start[name])

            # update phase
            swing_phase[name] += 1.0 / GAIT_PERIOD
            swing_phase[name] = min(swing_phase[name], 1.0)

            # compute swing position
            pos = swing_trajectory(
                swing_start[name],
                swing_target[name],
                swing_phase[name],
            )

            foot_positions[name] = pos

            # reset when finished
            if swing_phase[name] >= 1.0:
                swing_phase[name] = 0.0

        else:
            # stance: keep fixed
            swing_phase[name] = 0.0

        foot_trajs[name].append(foot_positions[name].copy())

config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[7:nq], nq, nv, C_WGT, True
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT, True)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)

base_costs = []
foot_funcs = {name: [] for name in foot_names}

R = np.eye(3)
R[0, 0] = -1.0
R[1, 1] = -1.0

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)

    base_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, "base_link", traj[k], R, nq, nv, T_WGT, R_WGT
        )
    )
    model.addCost(k, base_costs[-1])

    for name in foot_names:
        # add swing tracking
        foot_pos = foot_trajs[name][k]
        foot_funcs[name].append(
            (
                ssqpy.model.costs.FrameSquaredTranslationErrorCost(
                    model, name, foot_pos, nq, nv, T_WGT
                ),
                model.addContactConstraint(k, name, CONTACT_MU),
            )
        )

    # Feet translation costs
    foot_funcs["FR_foot"][-1][0].set_enabled(False)
    foot_funcs["RL_foot"][-1][0].set_enabled(False)

    # Feet contact constraints
    foot_funcs["FL_foot"][-1][1].set_enabled(False)
    foot_funcs["RR_foot"][-1][1].set_enabled(False)

model.addCost(mpc_H, v_cost)

model.finalize(init_states=[init_state])


def update_model(current_stage):
    for k in range(current_stage, current_stage + mpc_H):
        idx = k - current_stage

        # base tracking
        base_costs[idx].set_desired_translation(
            traj[k] if k <= H else traj[-1]
        )

        swing_feet = get_swing_feet(k)

        for name in foot_names:
            if name in swing_feet:
                # disable contact
                foot_funcs[name][idx][1].set_enabled(False)

                # update swing tracking point
                foot_funcs[name][idx][0].set_enabled(True)
                foot_funcs[name][idx][0].set_desired_translation(
                    foot_trajs[name][k]
                )
            else:
                # enable contact
                foot_funcs[name][idx][1].set_enabled(True)
                foot_funcs[name][idx][0].set_enabled(False)


ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
# ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
# ssqp_params.line_search_method = ssqpy.solvers.LineSearchMethod.Merit
# ssqp_params.stop_after_constraint_satisfaction = True
# ssqp_params.skip_regularization = True

admm_params = ssqpy.solvers.admmParams()
# admm_params.init_rho = 1.0
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=400)

# joint_space_pd_controller = JointSpacePD(mpc, init_state, Kp=2.0, Kd=0.05)

mjmodel = mujoco.MjModel.from_xml_path("../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state(init_state)[:nq]

np.set_printoptions(suppress=True)

Kp = 0.0
Kd = 0.0

states = []

stage = 0
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        print(f"\rSimulation time: {mjdata.time:.2f}s")
        frame_start = time()

        state = mujoco_to_ssqp_state(np.hstack((mjdata.qpos, mjdata.qvel)))
        states.append(state)

        # try:
        #     mjdata.ctrl = joint_space_pd_controller.step(state)
        # except RuntimeError:
        #     print("Exiting after regularization failure")
        #     break

        try:
            x, u = mpc.step(state)
            xk, uk = ssqp_to_mujoco_state(x.stage(1)), ssqp_to_mujoco_control(
                u.stage(0)
            )
        except RuntimeError:
            print("Exiting after regularization failure")
            break

        # mjdata.qpos = xk[:nq]

        # mjdata.qpos[:3] = np.array((0.0, 0.0, 0.5))
        mjdata.ctrl = (
            Kp * (xk[7:19] - state[7:19])
            + Kd * (xk[25:] - state[25:])
            + uk[6:18]
        )

        stage += 1
        update_model(stage)

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

        # sleep(0.1)

print()
