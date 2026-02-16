import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep

np.random.seed(int(time()))


def get_random_pd():
    return np.array(
        (
            0.3,
            np.random.random() - 0.5,
            np.random.random() / 3 + 0.2,
        )
    )


nq = 9
nv = 9
MPC_H = 20
dt = 0.01

T_WGT = 4.0
R_WGT = 3.0
V_WGT = 1e-1
U_WGT = 1e-3

REAL_TIME = True
INVERSE_MODE = True
PD_CHANGE_FREQ = 2

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

desired_pose = np.zeros((4, 4))
desired_pose[:3, :3] = Rd
desired_pose[:3, 3] = get_random_pd()
desired_pose[3, 3] = 1

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

v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nq, U_WGT)
pose_error = ssqpy.model.costs.FrameSquaredPoseErrorCost(
    model, desired_pose, "panda_ee", nq, nq, T_WGT, R_WGT
)

for k in range(MPC_H):
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, pose_error)

model.addCost(MPC_H, pose_error)
model.addCost(MPC_H, v_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

mpc = ssqpy.solvers.MPC(
    model, ssqp_params, admm_params, sqp_iters=5, admm_iters=75
)

mjmodel = mujoco.MjModel.from_xml_path("../assets/mjcf/panda/panda.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:9]

elapsed = 0
steps = 0
total_steps = PD_CHANGE_FREQ * int(1 / dt)

states = np.tile(init_state, (MPC_H + 1, 1))
controls = np.zeros((MPC_H, nv))

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

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
            print(f"Control frequency: {(total_steps / elapsed):.2f}Hz\t")
            elapsed = 0
            steps = 0

            new_pd = get_random_pd()
            print()
            print("Setting new goal:", new_pd)

            pose_error.set_desired_translation(new_pd)

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        if REAL_TIME:
            to_sleep = dt - (time() - frame_start)
            if to_sleep > 0:
                sleep(to_sleep)

print()
