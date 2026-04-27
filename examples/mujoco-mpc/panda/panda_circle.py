import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos, sqrt
from time import time, sleep

ssqpy.setSilentMode()

PLOT = True

if PLOT:
    import matplotlib.pyplot as plt

H = 400
dt = 0.01
mpc_H = 10
mpc_dt = 0.05

T_WGT = 10.0
R_WGT = 1.0
V_WGT = 1e-1
U_WGT = 5e-3
C_WGT = 0.18

INVERSE_MODE = False

cx = 0.3
cy = 0
cz = 0.26
cr = 0.22

DISTANCE = 2 * pi
ω = DISTANCE / (H * dt)
φ0 = 0

CONSTRAIN = True
square_side_hlf = cr / sqrt(2)
constraint_hlines = [cz - square_side_hlf, cz + square_side_hlf]
constraint_vlines = [cy - square_side_hlf, cy + square_side_hlf]


traj = []

for k in range(H + 1):
    φk = φ0 + ω * k * dt
    traj.append(np.array([cx, cy + cr * sin(φk), cz + cr * cos(φk)]))

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

model = ssqpy.model.Model(
    mpc_H,
    mpc_dt,
    urdf_path="../../assets/urdf/panda/panda.urdf",
    dynamics_mode=(
        ssqpy.model.DynamicsMode.Inverse
        if INVERSE_MODE
        else ssqpy.model.DynamicsMode.Forward
    ),
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

init_state = np.zeros(nq + nv)
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

config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[:nq], nq, nv, C_WGT, False
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)

for k in range(mpc_H):
    pose_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, "panda_ee", traj[k], Rd, nq, nv, T_WGT, R_WGT
        )
    )

    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, config_cost)
    model.addCost(k, pose_costs[-1])

    if CONSTRAIN:
        model.addConstraints(k, box_constraint)

pose_costs.append(
    ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, "panda_ee", traj[mpc_H], Rd, nq, nv, T_WGT, R_WGT
    )
)

model.addCost(mpc_H, v_cost)
model.addCost(mpc_H, pose_costs[-1])

if CONSTRAIN:
    model.addConstraints(mpc_H, box_constraint)


def update_pose_costs(current_stage):
    for k in range(current_stage, current_stage + mpc_H):
        pose_costs[k - current_stage].set_desired_translation(traj[k % H])


ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
# admm_params.init_rho = 0.5
admm_params.adaptive_rho_tolerance = 1.0
admm_params.rho_update_iter_freq = 15
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

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
                mjdata.qpos, mjdata.qvel, mpc.step(state)[1].stage(0)
            )[:8]
        else:
            mjdata.ctrl = mpc.step(state)[1].stage(0)[:8]
        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz\t", end="")
            elapsed = 0
            steps = 0

        mujoco.mj_step(mjmodel, mjdata)
        # viewer.sync()

        stage += 1
        update_pose_costs(stage)

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

print()

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
