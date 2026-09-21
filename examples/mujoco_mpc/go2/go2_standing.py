import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep
from utils_go2_mujoco import (
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    ssqp_to_mujoco_control,
)

ssqpy.setSilentMode()


dt = 0.002
mpc_H = 10
mpc_dt = 0.1

# Solve the MPC every few simulation steps, and hold the plan in between
MPC_EVERY = 4

# Print and sync the viewer every few steps, a multiple of the MPC rate
DISPLAY_EVERY = 8

C_WGT = 1.0
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-4

CONTACT_MU = 0.6
CONTACT_EQ_TOL = 0.01

dynamics_mode = ssqpy.model.SolverMode.ReducedInverseDynamics

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

base_pos = np.array([0.0, 0.0, 0.389])
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
f_cost = ssqpy.model.costs.SquaredForceCost(model, F_WGT)

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, f_cost)

    model.addContactConstraint(k, "FL_foot", CONTACT_MU, eq_tol=CONTACT_EQ_TOL)
    model.addContactConstraint(k, "FR_foot", CONTACT_MU, eq_tol=CONTACT_EQ_TOL)
    model.addContactConstraint(k, "RL_foot", CONTACT_MU, eq_tol=CONTACT_EQ_TOL)
    model.addContactConstraint(k, "RR_foot", CONTACT_MU, eq_tol=CONTACT_EQ_TOL)

model.addCost(mpc_H, v_cost)

model.finalize(init_states=[init_state])

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:nq].copy()
mjdata.qpos[3:7] = quat_xyzw_to_wxyz(mjdata.qpos[3:7])

np.set_printoptions(suppress=True)

step = 0

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        display = step % DISPLAY_EVERY == 0

        if display:
            print(f"\rSimulation time: {mjdata.time:.2f}s")

            if step > 0:
                loop_rate = DISPLAY_EVERY / (time() - display_time)
                print(
                    f"\rLoop frequency: {loop_rate:.2f}Hz\t"
                    f"MPC frequency: {(loop_rate / MPC_EVERY):.2f}Hz"
                )

            display_time = time()
        frame_start = time()

        pos = mjdata.qpos.copy()
        pos[3:7] = quat_wxyz_to_xyzw(pos[3:7])
        state = np.hstack((pos, mjdata.qvel))

        if step % MPC_EVERY == 0:
            u = mpc.step(state)[1].stage(0)

        if dynamics_mode == ssqpy.model.SolverMode.ForwardDynamics:
            mjdata.ctrl = ssqp_to_mujoco_control(u)[6:18]
        else:
            forces = {
                "FL_foot": [u[nu : nu + 3]],
                "FR_foot": [u[nu + 3 : nu + 6]],
                "RL_foot": [u[nu + 6 : nu + 9]],
                "RR_foot": [u[nu + 9 : nu + 12]],
            }

            mjdata.ctrl = ssqp_to_mujoco_control(
                model.inverseDynamics(
                    state[:nq],
                    state[nq:],
                    model.getFullAccelerationVector(state, u, 0),
                    forces,
                )
            )[6:]

        mujoco.mj_step(mjmodel, mjdata)

        if display:
            viewer.sync()

        step += 1

        # to_sleep = dt - (time() - frame_start)
        # if to_sleep > 0:
        #     sleep(to_sleep)

print()
