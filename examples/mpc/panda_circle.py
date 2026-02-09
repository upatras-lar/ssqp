import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos
from time import time, sleep

PLOT = True

if PLOT:
    import matplotlib.pyplot as plt

nq = 9
nv = 9
MPC_H = 10
H = 200
dt = 0.01
P_WGT = 35
R_WGT = 35
A_WGT = 0
V_WGT = 1e-4
U_WGT = 1e-5

CONSTRAIN = False
constraint_hlines = [0.1, 0.41]
constraint_vlines = [-0.15, 0.15]

INVERSE_MODE = True

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

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../assets/urdf/panda/panda.urdf",
    dynamics_mode=(
        ssqpy.model.DynamicsMode.Inverse
        if INVERSE_MODE
        else ssqpy.model.DynamicsMode.Forward
    ),
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

pose_costs = []

lb = np.array((-np.inf, constraint_vlines[0], constraint_hlines[0]))
ub = np.array((np.inf, constraint_vlines[1], constraint_hlines[1]))
box_constraint = ssqpy.model.constraints.FramePositionBound(
    model, "panda_ee", lb, ub, nq, nv
)

for k in range(MPC_H):
    pose_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, traj[k], Rd, "panda_ee", nq, nv, P_WGT, R_WGT
        )
    )

    model.addCost(k, pose_costs[-1])
    model.addCost(k, ssqpy.model.costs.SquaredJointAngleCost(nq, nv, A_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredControlCost(nq, U_WGT))

    if CONSTRAIN:
        model.addConstraints(k, box_constraint)

pose_costs.append(
    ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, traj[MPC_H], Rd, "panda_ee", nq, nv, P_WGT, R_WGT
    )
)

model.addCost(MPC_H, pose_costs[-1])
model.addCost(MPC_H, ssqpy.model.costs.SquaredJointAngleCost(nq, nv, A_WGT))
model.addCost(MPC_H, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

if CONSTRAIN:
    model.addConstraints(MPC_H, box_constraint)


def update_pose_costs(current_stage):
    for k in range(current_stage, current_stage + MPC_H):
        pose_costs[k - current_stage].set_desired_translation(traj[k % H])


mpc = ssqpy.solvers.MPC(model, sqp_iters=5, admm_iters=75)

mjmodel = mujoco.MjModel.from_xml_path("../assets/mjcf/panda/panda.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:9]

elapsed = 0
steps = 0
total_steps = int(1 / dt)

stage = 0

state_data = None
if PLOT:
    state_data = []

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        if PLOT:
            # print(
            #     f"Stage: {stage}, Mujoco: {mjdata.xpos[9]}, Ours: {model.forwardKinematics(mjdata.qpos, "panda_ee")[0]}"
            # )
            # state_data.append(mjdata.xpos[9].copy())
            state_data.append(
                model.forwardKinematics(mjdata.qpos, "panda_ee")[0]
            )

            if stage == H:
                break

        pre = time()
        if INVERSE_MODE:
            mjdata.ctrl = model.inverseDynamics(
                mjdata.qpos, mjdata.qvel, mpc.step(state)
            )[:8]
        else:
            mjdata.ctrl = mpc.step(state)[:8]
        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz\t", end="")
            elapsed = 0
            steps = 0

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        stage += 1
        update_pose_costs(stage)

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

if PLOT:
    state_data = np.array(state_data)
    traj = np.array(traj)

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Total Error")
    axs[0].plot(
        np.arange(H + 1),
        np.linalg.norm(state_data - traj, axis=1),
    )
    axs[0].set(xlabel="Step", ylabel="End effector position error")
    axs[0].grid(True)

    axs[1].set_title("Trajectory")

    traj = np.delete(traj, 0, axis=1)

    state_data = np.delete(state_data, 0, axis=1)

    axs[1].scatter(*zip(*traj), label="Cost trajectory")

    if CONSTRAIN:
        axs[1].hlines(
            y=constraint_hlines,
            xmin=constraint_vlines[0],
            xmax=constraint_vlines[1],
        )
        axs[1].vlines(
            x=constraint_vlines,
            ymin=constraint_hlines[0],
            ymax=constraint_hlines[1],
        )

    axs[1].scatter(*zip(*state_data), label="Actual trajectory")
    axs[1].legend()

    plt.show()
