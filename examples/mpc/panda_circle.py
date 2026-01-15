import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos
from time import time

nq = 9
nv = 9
MPC_H = 15
H = 200
dt = 0.01
P_WGT = 10
R_WGT = 10
A_WGT = 1e-1
V_WGT = 1e-3
U_WGT = 1e-5

cx = 0.3
cy = 0
cz = 0.26
cr = 0.3

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

pose_costs = []

for k in range(MPC_H):
    pose_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, traj[k], Rd, "panda_ee", nq, nq, P_WGT, R_WGT
        )
    )

    model.addCost(k, pose_costs[-1])
    model.addCost(k, ssqpy.model.costs.SquaredJointAngleCost(nq, nv, A_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredControlCost(nq, U_WGT))

pose_costs.append(
    ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, traj[MPC_H], Rd, "panda_ee", nq, nq, P_WGT, R_WGT
    )
)

model.addCost(MPC_H, pose_costs[-1])
model.addCost(MPC_H, ssqpy.model.costs.SquaredJointAngleCost(nq, nv, A_WGT))
model.addCost(MPC_H, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))


def update_pose_costs(current_stage):
    for k in range(current_stage, current_stage + MPC_H):
        pose_costs[k - current_stage].set_desired_translation(
            traj[current_stage % H]
        )


mpc = ssqpy.solvers.MPC(model, sqp_iters=4, admm_iters=50)

mjmodel = mujoco.MjModel.from_xml_path("../mjcf/panda/panda.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:9]

elapsed = 0
steps = 0
total_steps = int(1 / dt)

states = np.tile(init_state, (MPC_H + 1, 1))
controls = np.zeros((MPC_H, nv))

stage = 0

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        state = np.hstack((mjdata.qpos, mjdata.qvel))

        states = np.delete(states, (0), axis=0)
        states[0, :] = state
        states = np.vstack([states, states[MPC_H - 1, :]])

        controls = np.delete(controls, (0), axis=0)
        controls = np.vstack([controls, controls[MPC_H - 2, :]])

        pre = time()
        states, controls = mpc.step(states, controls)
        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz\t", end="")
            elapsed = 0
            steps = 0

        mjdata.ctrl = controls.stage(0)[:8]
        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        states, controls = states.toarray(), controls.toarray()

        stage += 1
        update_pose_costs(stage)
