import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos, sqrt
from time import time, sleep

dt = 0.01
mpc_H = 10
mpc_dt = 0.05

V_WGT = 1e-3
U_WGT = 1e-4
C_WGT = 1.0

CONTACT_MU = 0.6

model = ssqpy.model.Model(
    mpc_H,
    mpc_dt,
    # urdf_path="../assets/urdf/go2/go2.urdf",
    mjcf_path="../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    dynamics_mode=ssqpy.model.DynamicsMode.ReducedInverse,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

base_pos = np.array([0.0, 0.0, 0.3689268220122])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[6] = 1.0
init_state[7:19] = base_config


config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[7:nq], nq, nv, C_WGT, True
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT, True)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)

fl_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3), nq, nv
)

fr_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
    model, "FR_foot", np.zeros(3), np.zeros(3), nq, nv
)

rl_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
    model, "RL_foot", np.zeros(3), np.zeros(3), nq, nv
)

rr_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
    model, "RR_foot", np.zeros(3), np.zeros(3), nq, nv
)

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)

    model.addContactConstraint(k, "FL_foot", CONTACT_MU)
    model.addConstraints(k, fl_foot_vel_bound)

    model.addContactConstraint(k, "FR_foot", CONTACT_MU)
    model.addConstraints(k, fr_foot_vel_bound)

    model.addContactConstraint(k, "RL_foot", CONTACT_MU)
    model.addConstraints(k, rl_foot_vel_bound)

    model.addContactConstraint(k, "RR_foot", CONTACT_MU)
    model.addConstraints(k, rr_foot_vel_bound)

model.addCost(mpc_H, v_cost)

model.finalize(init_states=[init_state])


ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
# admm_params.adaptive_rho_tolerance = 1.0
# admm_params.rho_update_iter_freq = 15
# admm_params.abs_tolerance = 1e-3
# admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=5, qp_iters=200)

mjmodel = mujoco.MjModel.from_xml_path("../assets/mjcf/go2/go2.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:nq]

elapsed = 0
steps = 0
total_steps = int(1 / dt)

stage = 0

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        pre = time()
        # mjdata.ctrl = mpc.step(state)[6:nv]
        mjdata.ctrl = model.inverseDynamics(
            mjdata.qpos,
            mjdata.qvel,
            model.getFullAccelerationVector(state, mpc.step(state), 0),
        )[6:]
        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz\t", end="")
            elapsed = 0
            steps = 0

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        stage += 1

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

print()
