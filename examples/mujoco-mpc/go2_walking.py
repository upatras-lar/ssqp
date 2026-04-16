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


dt = 0.002
mpc_H = 50
mpc_dt = 0.1

T_WGT = 1.0
C_WGT = 0.1
V_WGT = 1e-2
U_WGT = 1e-4

CONTACT_MU = 0.6
GAIT_SWAP_FREQ = 10

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

base_pos = np.array([0.0, 0.0, 0.392])
base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

target_pos = np.array([1.0, 0.0, 0.392])

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[5] = 1.0
init_state[7:19] = base_config


H = 500
traj = []
for i in range(H + 1):
    traj.append(base_pos + i * (target_pos - base_pos) / H)

config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[7:nq], nq, nv, C_WGT, True
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT, True)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)

pos_costs = []
contact_constraints = []

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)

    pos_costs.append(
        ssqpy.model.costs.FrameSquaredTranslationErrorCost(
            model, "base_link", traj[k], nq, nv, T_WGT
        )
    )
    model.addCost(k, pos_costs[-1])

    stage_contact_constraints = []

    stage_contact_constraints.append(
        model.addContactConstraint(k, "FL_foot", CONTACT_MU)
    )

    stage_contact_constraints.append(
        model.addContactConstraint(k, "FR_foot", CONTACT_MU)
    )
    stage_contact_constraints[-1].set_enabled(False)

    stage_contact_constraints.append(
        model.addContactConstraint(k, "RL_foot", CONTACT_MU)
    )
    stage_contact_constraints[-1].set_enabled(False)

    stage_contact_constraints.append(
        model.addContactConstraint(k, "RR_foot", CONTACT_MU)
    )

    contact_constraints.append(stage_contact_constraints)

model.addCost(mpc_H, v_cost)

model.finalize(init_states=[init_state])


def update_model(current_stage):
    for k in range(current_stage, current_stage + mpc_H):
        pos_costs[k - current_stage].set_desired_translation(
            traj[k] if k <= H else traj[-1]
        )
        stage_contact_constraints = contact_constraints[k - current_stage]
        if (k // GAIT_SWAP_FREQ) % 2:
            # print(f"Stage {k}: Enabling 1,2")
            stage_contact_constraints[0].set_enabled(False)
            stage_contact_constraints[1].set_enabled(True)
            stage_contact_constraints[2].set_enabled(True)
            stage_contact_constraints[3].set_enabled(False)
        else:
            # print(f"Stage {k}: Enabling 0,3")
            stage_contact_constraints[0].set_enabled(True)
            stage_contact_constraints[1].set_enabled(False)
            stage_contact_constraints[2].set_enabled(False)
            stage_contact_constraints[3].set_enabled(True)


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

joint_space_pd_controller = JointSpacePD(mpc, init_state, Kp=2.0, Kd=0.05)

mjmodel = mujoco.MjModel.from_xml_path("../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:nq].copy()
mjdata.qpos[3:7] = quat_xyzw_to_wxyz(mjdata.qpos[3:7])

np.set_printoptions(suppress=True)

states = []

STEPS = 125
stage = 0
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        print(f"\rSimulation time: {mjdata.time:.2f}s")
        frame_start = time()

        pos = mjdata.qpos.copy()
        pos[3:7] = quat_wxyz_to_xyzw(pos[3:7])
        state = np.hstack((pos, mjdata.qvel))

        states.append(state)

        pre = time()

        try:
            mjdata.ctrl = joint_space_pd_controller.step(state)
        except RuntimeError:
            print("Exiting after regularization failure")
            break
        # try:
        #     u = mpc.step(state)[1].stage(0)
        # except RuntimeError:
        #     print("Exiting after regularization failure")
        #     break

        # mjdata.ctrl = u[6:18]

        elapsed = time() - pre

        print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        stage += 1
        update_model(stage)

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

print()

model = ssqpy.model.Model(
    len(states) - 1,
    mpc_dt,
    mjcf_path="../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    dynamics_mode=dynamics_mode,
)
for k in range(len(states)):
    model.addCost(k, v_cost)

model.finalize(init_states=states)
sqp = ssqpy.solvers.ssqp(model, line_search_method=ssqpy.solvers.LineSearchMethod.Merit)
sqp.write("test.json")
