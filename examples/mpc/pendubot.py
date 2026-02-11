import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi
from time import time, sleep

nq = 2
nv = 2
MPC_H = 60
dt = 0.01

TIP_WGT = 5.0
LINK_WGT = 1.4
V_WGT = 1e-3
U_WGT = 1e-4

REAL_TIME = True
NOISE = True


model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../assets/urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    # dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, U_WGT)
link2_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, np.array((0.0, 0.0, 0.05)), "link2", nq, nv, LINK_WGT
)
tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, np.array((0.0, 0.0, 0.1)), "tip", nq, nv, TIP_WGT
)
config_cost = ssqpy.model.costs.SquaredConfigurationDifferenceCost(
    np.array((pi, 0.0)),
    nq,
    nv,
    np.array((1e-5, 0.0)),
    False,
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, tip_cost)

model.addCost(MPC_H, link2_cost)
model.addCost(MPC_H, tip_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.adaptive_rho_tolerance = 20
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

mpc = ssqpy.solvers.MPC(
    model, ssqp_params, admm_params, sqp_iters=3, admm_iters=100
)

mjmodel = mujoco.MjModel.from_xml_path(
    "../assets/mjcf/double_pendulum/double_pendulum.xml"
)
mjdata = mujoco.MjData(mjmodel)

elapsed = 0
steps = 0
total_steps = int(1 / dt)

reg_fails = 0

np.set_printoptions(suppress=True)

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        pre = time()
        try:
            control = mpc.step(state)
        except RuntimeError:
            reg_fails += 1
            control = np.zeros(nv)

        # torques = model.inverseDynamics(mjdata.qpos, mjdata.qvel, control)
        # mjdata.ctrl[0] = torques[0]

        mjdata.ctrl = control
        if NOISE:
            mjdata.ctrl[0] += (
                np.random.uniform(0, 0.1)
                if np.random.uniform(0, 10) > 8
                else 0.0
            )

        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(
                f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz, regularization fails: {reg_fails}\t",
                end="",
            )
            elapsed = 0
            steps = 0

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        if REAL_TIME:
            to_sleep = dt - (time() - frame_start)
            if to_sleep > 0:
                sleep(to_sleep)
