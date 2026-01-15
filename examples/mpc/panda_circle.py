import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos

nq = 9
nv = 9
MPC_H = 15
H = 200
dt = 0.01
P_WGT = 10
R_WGT = 1
V_WGT = 1e-5
U_WGT = 1e-5

cx = 0.3
cy = 0
cz = 0.26
cr = 0.22

DISTANCE = 2 * pi
ω = DISTANCE / (H * dt)
φ0 = 0

traj = []

for k in range(H + 1):
    φk = φ0 + ω * k * dt
    traj.append(np.array([cx, cy + cr * sin(φk), cz + cr * cos(φk)]))

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

model = ssqpy.model.Model(MPC_H, dt, urdf_path="../urdf/panda/panda.urdf")

for k in range(H):
    desired_pose = np.zeros((4, 4))
    desired_pose[:3, :3] = Rd
    desired_pose[:3, 3] = traj[k]
    desired_pose[3, 3] = 1

    model.addCost(
        k,
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, desired_pose, "panda_ee", nq, nq, P_WGT, R_WGT
        ),
    )
    model.addCost(k, ssqpy.model.costs.SquaredControlCost(nq, U_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

desired_pose = np.zeros((4, 4))
desired_pose[:3, :3] = Rd
desired_pose[:3, 3] = traj[k]
desired_pose[3, 3] = 1

model.addCost(
    H,
    ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, desired_pose, "panda_ee", nq, nq, P_WGT, R_WGT
    ),
)
model.addCost(H, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

mpc = ssqpy.solvers.MPC(model)

model = mujoco.MjModel.from_xml_path("../mjcf/panda/panda.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        state = np.hstack((data.qpos, data.qvel))
        # data.ctrl = MPC.step(state)
        print(mpc.step(state))
        mujoco.mj_step(model, data)
        viewer.sync()
