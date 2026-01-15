import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos
from time import time

nq = 9
nv = 9
MPC_H = 20
dt = 0.01
P_WGT = 10
R_WGT = 10
V_WGT = 1e-5
U_WGT = 1e-5

pd = np.array((0.2, 0.2, 0.4))

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

desired_pose = np.zeros((4, 4))
desired_pose[:3, :3] = Rd
desired_pose[:3, 3] = pd
desired_pose[3, 3] = 1

model = ssqpy.model.Model(
    MPC_H, dt, urdf_path="../examples/urdf/panda/panda.urdf"
)

init_state = np.zeros(2 * nq)
init_state[:nq] = [
    0.0013,
    -0.9115,
    -0.0010,
    -2.6316,
    -0.0008,
    1.7431,
    0.7862,
    0.0,
    0.0,
]

for k in range(MPC_H):
    model.addCost(
        k,
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, desired_pose, "panda_ee", nq, nq, P_WGT, R_WGT
        ),
    )
    model.addCost(k, ssqpy.model.costs.SquaredControlCost(nq, U_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

model.addCost(
    MPC_H,
    ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, desired_pose, "panda_ee", nq, nq, P_WGT, R_WGT
    ),
)
model.addCost(MPC_H, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

mpc = ssqpy.solvers.MPC(model, sqp_iters=5, admm_iters=50)

model = mujoco.MjModel.from_xml_path("mjcf/panda/panda.xml")
data = mujoco.MjData(model)

data.qpos = init_state[:9]

elapsed = 0
steps = 0
total_steps = int(1 / dt)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        state = np.hstack((data.qpos, data.qvel))

        pre = time()
        data.ctrl = mpc.step(state)[:8]
        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz", end="")
            elapsed = 0
            steps = 0

        mujoco.mj_step(model, data)
        viewer.sync()
