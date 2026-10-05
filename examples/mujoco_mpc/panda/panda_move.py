import mujoco
import mujoco.viewer
import glfw
import ssqpy
import numpy as np
import argparse
from time import time, sleep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Panda random goal reaching MPC in MuJoCo using SSQP"
    )

    parser.add_argument(
        "-i",
        "--inverse-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use the inverse dynamics formulation instead of \
            the forward dynamics one",
    )

    parser.add_argument(
        "-c",
        "--constrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to add a random pillar blocking the path to the goal, \
            which the end effector has to pass over, and to draw it",
    )

    parser.add_argument(
        "-f",
        "--pd-change-freq",
        type=float,
        default=5,
        help="Time in seconds between changes of the desired position \
            (and obstacle box)",
    )

    return parser.parse_args()


args = parse_args()

ssqpy.setSilentMode()

np.random.seed(int(time()))


# Random desired position in front of the robot, far enough from the current
# one horizontally to place an obstacle between them
def get_random_pd(current_pos):
    while True:
        pd = np.array(
            (
                0.3,
                np.random.random() - 0.5,
                np.random.random() / 3 + 0.2,
            )
        )

        if np.linalg.norm(pd[:2] - current_pos[:2]) > 0.45:
            return pd


MPC_H = 40
dt = 0.02

T_WGT = 3.0
R_WGT = 6.0
A_WGT = 1e-1
V_WGT = 2e-1
U_WGT = 1e-3

# Desired end effector orientation, pointing down
Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

model = ssqpy.model.Model(
    MPC_H,
    dt,
    from_preset=ssqpy.model.Preset.Panda,
    solver_mode=(
        ssqpy.model.SolverMode.InverseDynamics
        if args.inverse_mode
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
target_pos = get_random_pd(start_pos)

desired_pose = np.zeros((4, 4))
desired_pose[:3, :3] = Rd
desired_pose[:3, 3] = target_pos
desired_pose[3, 3] = 1

angle_cost = ssqpy.model.costs.SquaredJointAngleCost(model, A_WGT)
v_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
pose_error = ssqpy.model.costs.FrameSquaredPoseErrorCost(
    model, "panda_ee", desired_pose, T_WGT, R_WGT
)


# Random pillar on the floor, tall enough to block the straight path from the
# current position to the target
def get_obstacle(current_pos, current_target):
    # Position along the path is measured horizontally
    direction = (current_target - current_pos)[:2]
    length = np.linalg.norm(direction)
    direction /= length
    slope = (current_target[2] - current_pos[2]) / length

    half_size = np.random.uniform(0.05, 0.1)
    half_len = half_size * np.abs(direction).sum()
    center_s = np.random.uniform(half_len + 0.1, length - half_len - 0.1)

    center = current_pos[:2] + center_s * direction
    top = current_pos[2] + slope * center_s + np.random.uniform(0.06, 0.1)

    obstacle_lower = np.array(
        (center[0] - half_size, center[1] - half_size, 0)
    )
    obstacle_upper = np.array(
        (center[0] + half_size, center[1] + half_size, top)
    )

    return obstacle_lower, obstacle_upper


# Obstacles are non-convex, so instead of excluding the pillar, stages that
# the previous plan puts above the pillar (with a margin) must stay above its
# top face
def update_height_bounds(plan_ee_pos, bounds):
    lb, ub = bounds
    margin = 0.08
    clearance = 0.02

    for bound, pos in zip(height_bounds, plan_ee_pos):
        above = np.all(pos[:2] > lb[:2] - margin) and np.all(
            pos[:2] < ub[:2] + margin
        )
        z = ub[2] + clearance if above else -np.inf
        bound.set_lower_bound(np.array((-np.inf, -np.inf, z)))


box_bounds = None
height_bounds = []
if args.constrain:
    box_bounds = get_obstacle(start_pos, target_pos)

    for k in range(MPC_H):
        height_bounds.append(
            ssqpy.model.constraints.FrameTranslationBound(
                model,
                "panda_ee",
                np.array((-np.inf, -np.inf, -np.inf)),
                np.full(3, np.inf),
            )
        )
        model.addConstraints(k + 1, height_bounds[-1])

for k in range(MPC_H):
    model.addCost(k, angle_cost)
    model.addCost(k, v_cost)
    model.addCost(k, u_cost)
    model.addCost(k, pose_error)

model.addCost(MPC_H, angle_cost)
model.addCost(MPC_H, pose_error)
model.addCost(MPC_H, v_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

ssqp_params.hpipmParams.tol_comp = 1e-3
ssqp_params.hpipmParams.tol_stat = 1e-3
ssqp_params.hpipmParams.tol_eq = 1e-3
ssqp_params.hpipmParams.tol_ineq = 1e-3

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=20)

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/panda/panda.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = init_state[:9]

elapsed = 0
steps = 0
total_steps = int(args.pd_change_freq / dt)

states = np.tile(init_state, (MPC_H + 1, 1))
controls = np.zeros((MPC_H, nv))

current_target = target_pos


# Draw the goal (and the box, if any) as extra geoms in the viewer scene
def draw_markers(scene, target, bounds):
    identity = np.eye(3).flatten()

    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((0.02, 0, 0)),
        target,
        identity,
        np.array((1, 0, 0, 1), dtype=np.float32),
    )
    scene.ngeom = 1

    if bounds is not None:
        lb, ub = bounds
        mujoco.mjv_initGeom(
            scene.geoms[1],
            mujoco.mjtGeom.mjGEOM_BOX,
            (ub - lb) / 2,
            (lb + ub) / 2,
            identity,
            np.array((0, 0.5, 1, 0.25), dtype=np.float32),
        )
        scene.ngeom = 2


# Start the viewer maximized
glfw.init()
glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    draw_markers(viewer.user_scn, current_target, box_bounds)

    # End effector positions of stages 1 to MPC_H in the next solve
    plan_ee_pos = [start_pos] * MPC_H

    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        if args.constrain:
            update_height_bounds(plan_ee_pos, box_bounds)

        pre = time()
        plan_states, plan_controls = mpc.step(state)
        # MuJoCo has 8 actuators (7 arm joints and one finger), the model has 9 DoF
        if args.inverse_mode:
            mjdata.ctrl = model.inverseDynamics(
                mjdata.qpos, mjdata.qvel, plan_controls.stage(0)
            )[:8]
        else:
            mjdata.ctrl = plan_controls.stage(0)[:8]

        if args.constrain:
            # The plan is shifted by one stage for the next solve
            plan_ee_pos = [
                model.forwardKinematics(
                    plan_states.stage(min(k + 2, MPC_H))[:nq], "panda_ee"
                )[0]
                for k in range(MPC_H)
            ]
        elapsed += time() - pre
        steps += 1

        # Periodically pick a new goal and move the obstacle box
        if steps == total_steps:
            print(f"Control frequency: {(total_steps / elapsed):.2f}Hz\t")
            elapsed = 0
            steps = 0

            current_pos = current_target
            current_target = get_random_pd(current_pos)
            print()
            print("Setting new goal:", current_target)

            pose_error.set_desired_translation(current_target)

            if args.constrain:
                box_bounds = get_obstacle(current_pos, current_target)

                print("Setting obstacle:")
                print("Lower:", box_bounds[0])
                print("Upper:", box_bounds[1])

            with viewer.lock():
                draw_markers(viewer.user_scn, current_target, box_bounds)

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

print()
