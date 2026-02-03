import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi
from time import time, sleep

nq = 2
nv = 2
MPC_H = 15
dt = 0.01

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../urdf/double_pendulum/double_pendulum_2.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, 1e-9)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, 1e-9)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

config_cost = ssqpy.model.costs.SquaredConfigurationDifferenceCost(
    np.array((pi, 0.0)), nq, nv, 10.0, False
)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, config_cost)

mpc = ssqpy.solvers.MPC(model, sqp_iters=4, admm_iters=50)

mjmodel = mujoco.MjModel.from_xml_path(
    "../mjcf/double_pendulum/double_pendulum.xml"
)
mjdata = mujoco.MjData(mjmodel)

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))
        accel = mpc.step(state)

        mjdata.ctrl[0] = model.inverseDynamics(
            mjdata.qpos, mjdata.qvel, accel
        )[0]

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)
