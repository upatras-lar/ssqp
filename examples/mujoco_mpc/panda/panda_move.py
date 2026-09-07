import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep

ssqpy.setSilentMode()

np.random.seed(int(time()))


def get_random_pd():
    return np.array(
        (
            0.3,
            np.random.random() - 0.5,
            np.random.random() / 3 + 0.2,
        )
    )


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

target_pos = get_random_pd()

desired_pose = np.zeros((4, 4))
desired_pose[:3, :3] = Rd
desired_pose[:3, 3] = target_pos
desired_pose[3, 3] = 1

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../assets/urdf/panda/panda.urdf",
    solver_mode=(
        ssqpy.model.SolverMode.InverseDynamics
        if INVERSE_MODE
        else ssqpy.model.SolverMode.ForwardDynamics
    ),
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

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

start_pos = model.forwardKinematics(init_state[:nq], "panda_ee")[0]

v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
pose_error = ssqpy.model.costs.FrameSquaredPoseErrorCost(
    model, "panda_ee", desired_pose, T_WGT, R_WGT
)


def get_constraint_bounds(current_pos, current_target):
    alpha = np.random.uniform(0.3, 0.7)
    obstacle_center = (1 - alpha) * current_pos + alpha * current_target

    half_size = np.random.uniform(0.05, 0.1)

    obstacle_lower = obstacle_center - half_size
    obstacle_upper = obstacle_center + half_size

    return obstacle_lower, obstacle_upper


obstacle_box = ssqpy.model.constraints.FrameTranslationBound(
    model, "panda_ee", *get_constraint_bounds(start_pos, target_pos)
)


for k in range(MPC_H):
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, pose_error)

    # model.addConstraints(k, obstacle_box)

model.addCost(MPC_H, pose_error)
model.addCost(MPC_H, v_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=5, qp_iters=75)

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/panda/panda.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:9]

elapsed = 0
steps = 0
total_steps = PD_CHANGE_FREQ * int(1 / dt)

states = np.tile(init_state, (MPC_H + 1, 1))
controls = np.zeros((MPC_H, nv))

current_target = target_pos

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

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
            print(f"Control frequency: {(total_steps / elapsed):.2f}Hz\t")
            elapsed = 0
            steps = 0

            current_pos = current_target
            current_target = get_random_pd()
            print()
            print("Setting new goal:", current_target)

            pose_error.set_desired_translation(current_target)
            lb, ub = get_constraint_bounds(current_pos, current_target)
            obstacle_box.set_lower_bound(lb)
            obstacle_box.set_upper_bound(ub)

            print("Setting box:")
            print("Lower:", lb)
            print("Upper:", ub)

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        if REAL_TIME:
            to_sleep = dt - (time() - frame_start)
            if to_sleep > 0:
                sleep(to_sleep)

print()
