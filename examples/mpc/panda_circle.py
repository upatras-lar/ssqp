import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos, atan2, sqrt
from time import time, sleep

PLOT = True

if PLOT:
    import matplotlib.pyplot as plt

nq = 9
nv = 9
MPC_H = 15
H = 200
dt = 0.01
P_WGT = 100
R_WGT = 10
A_WGT = 1e-1
V_WGT = 1e-4
U_WGT = 1e-5

CONSTRAIN = True

cx = 0.3
cy = 0
cz = 0.26
cr = 0.22

DISTANCE = 2 * pi
ω = DISTANCE / (H * dt)
φ0 = 0


def circle_to_square_point(p):
    θ = atan2(p[2] - cz, p[1] - cy)
    c = cos(θ)
    s = sin(θ)
    a = cr / sqrt(2.0)
    t = a / max(abs(c), abs(s))

    return np.array((cx, cy + t * c, cz + t * s))


def square_point_bounds(p):
    lb = np.zeros(3)
    ub = np.zeros(3)

    lb[0] = -np.inf
    ub[0] = np.inf

    if p[1] > cy:
        lb[1] = -np.inf
        ub[1] = p[1]
    else:
        lb[1] = p[1]
        ub[1] = np.inf

    if p[2] > cz:
        lb[2] = -np.inf
        ub[2] = p[2]
    else:
        lb[2] = p[2]
        ub[2] = np.inf

    return lb, ub


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
box_constraints = []

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

    if CONSTRAIN:
        lb, ub = square_point_bounds(circle_to_square_point(traj[k]))

        box_constraints.append(
            ssqpy.model.constraints.FramePositionBound(
                model, "panda_ee", lb, ub, nq, nv
            )
        )

        model.addConstraints(k, box_constraints[-1])

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
        pose_costs[k - current_stage].set_desired_translation(traj[k % H])


def update_box_constraints(current_stage):
    for k in range(current_stage, current_stage + MPC_H):
        lb, ub = square_point_bounds(circle_to_square_point(traj[k % H]))

        box_constraints[k - current_stage].set_lower_bound(lb)
        box_constraints[k - current_stage].set_upper_bound(ub)


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
        if CONSTRAIN:
            update_box_constraints(stage)

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

if PLOT:
    if CONSTRAIN:
        traj = [circle_to_square_point(p) for p in traj]

    for i, p in enumerate(zip(traj, state_data)):
        print(f"{i}. Expected: {p[0]}, Actual: {p[1]}")

    plt.figure()
    plt.plot(
        np.arange(H + 1),
        np.linalg.norm(np.array(state_data) - np.array(traj), axis=1),
    )
    plt.xlabel("Step")
    plt.ylabel("End effector position error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
