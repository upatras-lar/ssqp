import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep
from go2_controllers.JointSpacePD import JointSpacePD
from utils_go2_mujoco import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

ssqpy.setSilentMode()


dt = 0.002
mpc_H = 10
mpc_dt = 0.1

C_WGT = 10.0
V_WGT = 1e-2
U_WGT = 1e-4

CONTACT_MU = 0.6

dynamics_mode = ssqpy.model.SolverMode.ForwardDynamics

model = ssqpy.model.Model(
    mpc_H,
    mpc_dt,
    mjcf_path="../../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    solver_mode=dynamics_mode,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

base_pos = np.array([0.0, 0.0, 0.392])
base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[5] = 1.0
init_state[7:19] = base_config

config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[7:nq], C_WGT
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)

    model.addContactConstraint(k, "FL_foot", CONTACT_MU, max_normal_force=100)
    model.addContactConstraint(k, "FR_foot", CONTACT_MU, max_normal_force=100)
    model.addContactConstraint(k, "RL_foot", CONTACT_MU, max_normal_force=100)
    model.addContactConstraint(k, "RR_foot", CONTACT_MU, max_normal_force=100)

model.addCost(mpc_H, v_cost)

model.finalize(init_states=[init_state])

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

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:nq].copy()
mjdata.qpos[3:7] = quat_xyzw_to_wxyz(mjdata.qpos[3:7])

np.set_printoptions(suppress=True)

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        print(f"\rSimulation time: {mjdata.time:.2f}s")
        frame_start = time()

        pos = mjdata.qpos.copy()
        pos[3:7] = quat_wxyz_to_xyzw(pos[3:7])
        state = np.hstack((pos, mjdata.qvel))

        pre = time()

        mjdata.ctrl = joint_space_pd_controller.step(state)

        elapsed = time() - pre

        print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

print()
