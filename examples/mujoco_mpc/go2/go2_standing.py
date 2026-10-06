import argparse
import mujoco
import mujoco.viewer
import glfw
import ssqpy
import numpy as np
from time import time
from utils_go2_mujoco import (
    clip_joints_to_limits,
    mujoco_to_ssqp_state,
    quat_xyzw_to_wxyz,
    ssqp_to_mujoco_control,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 standing MPC in MuJoCo using SSQP, on four, three \
            or two legs"
    )

    parser.add_argument(
        "-l",
        "--legs",
        type=int,
        default=4,
        choices=[4, 3, 2],
        help="Number of legs to stand on. With two legs, the diagonal pair \
            FL and RR stays on the ground",
    )

    parser.add_argument(
        "--lifted-foot",
        default="FR_foot",
        choices=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        help="Foot to lift off the ground (only used when standing on three \
            legs)",
    )

    return parser.parse_args()


args = parse_args()

ssqpy.setSilentMode()

LEGS = args.legs

# Per number of legs: horizon, MPC rate and cost weights
# Solve the MPC every few simulation steps, and hold the plan in between
if LEGS == 4:
    mpc_H, MPC_EVERY = 10, 4
    C_WGT, V_WGT, F_WGT, R_BASE_WGT = 1.0, 1e-2, 1e-4, 10.0
elif LEGS == 3:
    mpc_H, MPC_EVERY = 5, 1
    C_WGT, V_WGT, F_WGT, R_BASE_WGT = 5.0, 1e-4, 1e-3, 10.0
else:
    mpc_H, MPC_EVERY = 5, 1
    C_WGT, V_WGT, F_WGT, R_BASE_WGT = 1.0, 1e-2, 1e-4, 1.0

mpc_dt = 0.1

# Print and sync the viewer every few steps, a multiple of the MPC rate
DISPLAY_EVERY = 8

U_WGT = 1e-4
T_BASE_WGT = 1.0

CONTACT_MU = 0.6
CONTACT_EQ_TOL = 0.01

dynamics_mode = ssqpy.model.SolverMode.ReducedInverseDynamics

model = ssqpy.model.Model(
    mpc_H,
    mpc_dt,
    from_preset=ssqpy.model.Preset.Go2,
    solver_mode=dynamics_mode,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

all_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

if LEGS == 4:
    lifted_feet = []
elif LEGS == 3:
    lifted_feet = [args.lifted_foot]
else:
    # FL and RR are the stance legs, FR and RL are lifted
    lifted_feet = ["FR_foot", "RL_foot"]

stance_feet = [foot for foot in all_feet if foot not in lifted_feet]

base_pos = np.array([0.0, 0.0, 0.389 if LEGS == 4 else 0.3905])
base_rot = np.diag([-1.0, -1.0, 1.0])

stance_leg = np.array([0.0, np.pi / 6, -np.pi / 3])
lifted_leg = np.array([0.0, 0.8, -1.7])

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[5] = 1.0
init_state[7:19] = np.concatenate(
    [lifted_leg if foot in lifted_feet else stance_leg for foot in all_feet]
)

if LEGS == 4:
    mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
else:
    # The planner contacts the ground at the foot frames, while the simulator
    # feet are spheres whose contact point sits away from the frame, so use
    # point feet
    mjspec = mujoco.MjSpec.from_file("../../assets/mjcf/go2/scene.xml")
    for foot in ["FL", "FR", "RL", "RR"]:
        geom = mjspec.geom(foot)
        geom.pos = [0.0, 0.0, -0.23648]
        geom.size[0] = 0.001
    mjmodel = mjspec.compile()

if LEGS == 3:
    # With one foot lifted the CoM sits on the edge of the support triangle, so
    # move the base toward the triangle's centroid and solve each stance leg's
    # joints to keep its foot in place. The base is also lowered, since
    # reaching the far stance foot otherwise straightens its calf close to the
    # joint limit
    ik_data = mujoco.MjData(mjmodel)
    ik_data.qpos = init_state[:nq].copy()
    ik_data.qpos[3:7] = quat_xyzw_to_wxyz(ik_data.qpos[3:7])
    mujoco.mj_kinematics(mjmodel, ik_data)

    foot_pos = {
        foot: ik_data.xpos[mjmodel.body(foot).id].copy()
        for foot in stance_feet
    }
    ik_data.qpos[:2] = 0.75 * np.mean(
        [pos[:2] for pos in foot_pos.values()], axis=0
    )
    ik_data.qpos[2] = 0.37

    for foot in stance_feet:
        body = mjmodel.body(foot).id
        leg = all_feet.index(foot)
        jac = np.zeros((3, nv))

        for _ in range(20):
            mujoco.mj_kinematics(mjmodel, ik_data)
            mujoco.mj_comPos(mjmodel, ik_data)
            mujoco.mj_jacBody(mjmodel, ik_data, jac, None, body)
            ik_data.qpos[7 + 3 * leg : 10 + 3 * leg] += np.linalg.lstsq(
                jac[:, 6 + 3 * leg : 9 + 3 * leg],
                foot_pos[foot] - ik_data.xpos[body],
                rcond=None,
            )[0]

    init_state[:3] = ik_data.qpos[:3]
    init_state[7:nq] = ik_data.qpos[7:]
    base_pos = init_state[:3].copy()

config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, init_state[7:nq], C_WGT
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
f_cost = ssqpy.model.costs.SquaredForceCost(model, F_WGT)
base_cost = ssqpy.model.costs.FloatingBaseSquaredPoseErrorCost(
    model, base_pos, base_rot, T_BASE_WGT, R_BASE_WGT
)

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, f_cost)
    model.addCost(k, base_cost)

    for foot in stance_feet:
        model.addContactConstraint(
            k, foot, CONTACT_MU, eq_tol=CONTACT_EQ_TOL, no_slip=False
        )

if LEGS == 3:
    model.addCost(mpc_H, config_cost)
model.addCost(mpc_H, v_cost)
if LEGS != 2:
    model.addCost(mpc_H, base_cost)

model.finalize(init_states=[init_state])

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

hpipm_params = ssqpy.solvers.hpipmParams()
if LEGS != 4:
    hpipm_params.mode = ssqpy.solvers.HPIPMMode.Balance
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
if LEGS != 4:
    hpipm_params.warm_start = True
ssqp_params.hpipmParams = hpipm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:nq].copy()
mjdata.qpos[3:7] = quat_xyzw_to_wxyz(mjdata.qpos[3:7])

np.set_printoptions(suppress=True)

step = 0

# Start the viewer maximized
glfw.init()
glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        display = step % DISPLAY_EVERY == 0

        if display:
            print(f"\rSimulation time: {mjdata.time:.2f}s")

            if step > 0:
                loop_rate = DISPLAY_EVERY / (time() - display_time)
                print(
                    f"\rLoop frequency: {loop_rate:.2f}Hz\t"
                    f"MPC frequency: {(loop_rate / MPC_EVERY):.2f}Hz"
                )

            display_time = time()

        state = mujoco_to_ssqp_state(np.hstack((mjdata.qpos, mjdata.qvel)))
        state = clip_joints_to_limits(state, mjmodel)

        if step % MPC_EVERY == 0:
            u = mpc.step(state)[1].stage(0)

        # Forces are ordered by the stance feet, following the accelerations
        forces = {
            foot: [u[nu + 3 * i : nu + 3 * i + 3]]
            for i, foot in enumerate(stance_feet)
        }

        mjdata.ctrl = ssqp_to_mujoco_control(
            model.inverseDynamics(
                state[:nq],
                state[nq:],
                model.getFullAccelerationVector(state, u, 0),
                forces,
            )
        )[6:]

        mujoco.mj_step(mjmodel, mjdata)

        if display:
            viewer.sync()

        step += 1

print()
