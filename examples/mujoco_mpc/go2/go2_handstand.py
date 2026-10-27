import mujoco
import mujoco.viewer
import glfw
import ssqpy
import argparse
import numpy as np
from time import time
from utils_go2_mujoco import (
    clip_joints_to_limits,
    mujoco_to_ssqp_state,
    quat_wxyz_to_rotation,
    quat_xyzw_to_wxyz,
    ssqp_to_mujoco_control,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 handstand MPC in MuJoCo using SSQP"
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm", "piqp"],
        default="hpipm",
        help="QP solver used at each SSQP iteration",
    )

    return parser.parse_args()


args = parse_args()

ssqpy.setSilentMode()


dt = 0.002
mpc_H = 10
mpc_dt = 0.05

# Solve the MPC every few simulation steps, and hold the plan in between
MPC_EVERY = 1

# Print and sync the viewer every few steps, a multiple of the MPC rate
DISPLAY_EVERY = 8

C_WGT = 0.1
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-4
COM_WGT = 1.0
FOOT_WGT = 100.0
R_BASE_WGT = 10.0

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

# Handstand on the front left foot, with the center of mass directly above it
# The base is pitched steeply nose down and rolled to the side, the front left
# leg reaches forward under the body and the other three legs stay folded
base_pos = np.array([0.02724554, 0.01246858, 0.58248232])
base_quat = np.array([-0.20712838, 0.53766406, 0.13639596, 0.80586062])
base_config = np.array(
    [-0.17163151, -0.49722650, -1.00921683]
    + [0.06798100, 0.89222896, -1.78238015]
    + [-0.03464934, 0.89292408, -1.78283870]
    + [0.06798198, 0.89222698, -1.78238015]
)

# Initial configuration
init_state = np.zeros(37)
init_state[:3] = base_pos
init_state[3:7] = base_quat
init_state[7:19] = base_config

# Keep the center of mass above the front left foot, at the pose's height
# above it
foot_pos = model.forwardKinematics(init_state[:nq], "FL_foot")[0]
com_height = model.centerOfMass(init_state[:nq])[2] - foot_pos[2]

config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, base_config, C_WGT
)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
f_cost = ssqpy.model.costs.SquaredForceCost(model, F_WGT)
com_cost = ssqpy.model.costs.CenterOfMassTrackingCost(
    model, foot_pos + [0.0, 0.0, com_height], COM_WGT
)

base_rot_cost = ssqpy.model.costs.FloatingBaseSquaredRotationErrorCost(
    model, quat_wxyz_to_rotation(quat_xyzw_to_wxyz(base_quat)), R_BASE_WGT
)

# The contact constraint only fixes the foot's height, so also keep the foot
# from sliding
foot_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, "FL_foot", foot_pos, FOOT_WGT
)

for k in range(mpc_H):
    model.addCost(k, config_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, f_cost)
    model.addCost(k, com_cost)
    model.addCost(k, foot_cost)
    model.addCost(k, base_rot_cost)

    model.addContactConstraint(
        k, "FL_foot", CONTACT_MU, eq_tol=CONTACT_EQ_TOL, no_slip=False
    )

model.addCost(mpc_H, v_cost)
model.addCost(mpc_H, com_cost)
model.addCost(mpc_H, foot_cost)
model.addCost(mpc_H, base_rot_cost)

model.finalize(init_states=[init_state])

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

# The planner contacts the ground at the foot frames, while the simulator feet
# are spheres whose contact point sits away from the frame, so use point feet
mjspec = mujoco.MjSpec.from_file("../../assets/mjcf/go2/scene.xml")
for foot in ["FL", "FR", "RL", "RR"]:
    geom = mjspec.geom(foot)
    geom.pos = [0.0, 0.0, -0.23648]
    geom.size[0] = 0.001
mjmodel = mjspec.compile()
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
        frame_start = time()

        state = mujoco_to_ssqp_state(np.hstack((mjdata.qpos, mjdata.qvel)))
        state = clip_joints_to_limits(state, mjmodel)

        if step % MPC_EVERY == 0:
            foot_pos = model.forwardKinematics(state[:nq], "FL_foot")[0]
            com_cost.set_cref(foot_pos + [0.0, 0.0, com_height])
            foot_cost.set_desired_translation(foot_pos)

            u = mpc.step(state)[1].stage(0)

        mjdata.ctrl = ssqp_to_mujoco_control(
            model.inverseDynamics(
                state[:nq],
                state[nq:],
                model.getFullAccelerationVector(state, u, 0),
                {"FL_foot": [u[nu : nu + 3]]},
            )
        )[6:]

        mujoco.mj_step(mjmodel, mjdata)

        if display:
            viewer.sync()

        step += 1

print()
