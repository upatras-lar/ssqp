import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep

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
mpc_H = 5
mpc_dt = 0.1

C_WGT = 1.0
V_WGT = 1e-2
U_WGT = 1e-4

CONTACT_MU = 0.6

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
base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[5] = 1.0
init_state[7:19] = base_config

config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[7:nq], nq, nv, C_WGT, True
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT, True)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)

    model.addContactConstraint(k, "FL_foot", CONTACT_MU)
    model.addContactConstraint(k, "FR_foot", CONTACT_MU)
    model.addContactConstraint(k, "RL_foot", CONTACT_MU)
    model.addContactConstraint(k, "RR_foot", CONTACT_MU)

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

mjmodel = mujoco.MjModel.from_xml_path("../assets/mjcf/go2/scene.xml")
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

        if dynamics_mode == ssqpy.model.DynamicsMode.Forward:
            u = mpc.step(state)[1].stage(0)
            mjdata.ctrl = u[6:18]
        else:
            u = mpc.step(state)[1].stage(0)

            forces = {
                "FL_foot": [u[nu : nu + 3]],
                "FR_foot": [u[nu + 3 : nu + 6]],
                "RL_foot": [u[nu + 6 : nu + 9]],
                "RR_foot": [u[nu + 9 : nu + 12]],
            }

            mjdata.ctrl = model.inverseDynamics(
                state[:nq],
                state[nq:],
                model.getFullAccelerationVector(state, u, 0),
                forces,
            )[6:]

        elapsed = time() - pre

        print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        # to_sleep = dt - (time() - frame_start)
        # if to_sleep > 0:
        #     sleep(to_sleep)

print()
