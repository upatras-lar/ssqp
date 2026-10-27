import mujoco
import mujoco.viewer
import glfw
import ssqpy
import numpy as np
import argparse
from math import pi
from time import time, sleep

MPC_H = 45
dt = 0.01

V_WGT = 1e-3
U_WGT = 1e-3
TIP_WGT = 5.0

torque_limit = 0.15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pendubot MPC in MuJoCo using SSQP"
    )

    parser.add_argument(
        "-c",
        "--control-delay",
        type=int,
        default=1,
        help="Number of steps before a computed control is applied \
            (1 means no delay)",
    )

    parser.add_argument(
        "-n",
        "--noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to add random torque disturbances",
    )

    parser.add_argument(
        "-i",
        "--inverse-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use the inverse dynamics formulation instead of \
            the forward dynamics one",
    )

    parser.add_argument(
        "--identification",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to identify the parameters of the second link online \
            and update the model once enough data has been collected",
    )

    parser.add_argument(
        "--identification-steps",
        type=int,
        default=1000,
        help="Number of simulation steps to collect data for before \
            identification",
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm", "piqp"],
        default=None,
        help="QP solver used at each SSQP iteration, HPIPM in inverse \
            dynamics mode and ADMM otherwise by default",
    )

    return parser.parse_args()


args = parse_args()

INVERSE_MODE = args.inverse_mode

ssqpy.setSilentMode()


model = ssqpy.model.Model(
    MPC_H,
    dt,
    from_preset=ssqpy.model.Preset.DoublePendulum,
    actuated_joints=[0],
    solver_mode=(
        ssqpy.model.SolverMode.InverseDynamics
        if INVERSE_MODE
        else ssqpy.model.SolverMode.ForwardDynamics
    ),
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, "tip", np.array((0.0, 0.0, 0.1)), TIP_WGT
)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model,
    np.array((pi, 0.0)),
    np.array((1.0, 0.0)),
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, tip_cost)

model.addCost(MPC_H, tip_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()

mpc = None

if INVERSE_MODE:
    ssqp_params.tolerance = 1e-3
    ssqp_params.qp_solver_type = getattr(
        ssqpy.solvers.QPSolverType, (args.qp_solver or "hpipm").upper()
    )

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)
else:
    ssqp_params.tolerance = 1e-2
    ssqp_params.qp_solver_type = getattr(
        ssqpy.solvers.QPSolverType, (args.qp_solver or "admm").upper()
    )
    ssqp_params.admmParams.abs_tolerance = 1e-3
    ssqp_params.admmParams.rel_tolerance = 1e-3
    ssqp_params.admmParams.warm_start = False

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=3, qp_iters=100)

mjmodel = mujoco.MjModel.from_xml_path(
    "../assets/mjcf/double_pendulum/double_pendulum.xml"
)
mjdata = mujoco.MjData(mjmodel)

elapsed = 0
steps = 0
total_steps = int(1 / dt)

reg_fails = 0

np.set_printoptions(suppress=True)

delayed_control = []

Ys = []
taus = []
i = 0

# Start the viewer maximized
glfw.init()
glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        pre = time()
        # Fall back to zero control if the solver fails
        try:
            control = mpc.step(state)[1].stage(0)
        except RuntimeError:
            reg_fails += 1
            control = np.zeros(nv)

        if INVERSE_MODE:
            # The solver outputs accelerations, convert them to torques
            torques = model.inverseDynamics(mjdata.qpos, mjdata.qvel, control)
            delayed_control.append(torques[0])
            if len(delayed_control) == args.control_delay:
                mjdata.ctrl[0] = delayed_control.pop(0)

        else:
            delayed_control.append(control[0])
            if len(delayed_control) == args.control_delay:
                mjdata.ctrl[0] = delayed_control.pop(0)

        # Occasional random torque spikes
        if args.noise:
            mjdata.ctrl[0] += (
                np.random.uniform(0, 0.1)
                if np.random.uniform(0, 10) > 8
                else 0.0
            )

        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(
                f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz, regularization fails: {reg_fails}\t",
                end="",
            )
            elapsed = 0
            steps = 0

        mjdata.ctrl = np.clip(mjdata.ctrl, -torque_limit, torque_limit)
        mujoco.mj_step(mjmodel, mjdata)
        mujoco.mj_forward(mjmodel, mjdata)
        viewer.sync()

        # Collect regressor and torque data for identification
        if args.identification and i < args.identification_steps:
            q = mjdata.qpos.copy()
            v = mjdata.qvel.copy()
            a = mjdata.qacc.copy()

            Ys.append(model.getJointTorqueRegressor(q, v, a))

            # Actual torque measurement
            mujoco.mj_inverse(mjmodel, mjdata)
            taus.append(mjdata.qfrc_inverse.copy())

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

        i += 1
        if args.identification and i == args.identification_steps:
            Y_stack = np.vstack(Ys)
            tau_stack = np.hstack(taus)

            # Least squares on tau = Y * params
            params = model.getAllDynamicParameters()

            residual = tau_stack - Y_stack @ params
            print("Res:", np.abs(residual).max(), np.abs(residual).mean())

            # Only the second link's parameters are estimated
            # Corresponds to: [mass_2, com_x_2, com_y_2, com_z_2, I_xx]
            cols = [10, 11, 12, 13, 14]
            known_cols = np.setdiff1d(np.arange(Y_stack.shape[1]), cols)

            Y_known = Y_stack[:, known_cols]
            Y_unknown = Y_stack[:, cols]

            tau_residual = tau_stack - Y_known @ params[known_cols]

            pi_hat, *_ = np.linalg.lstsq(
                Y_unknown,
                tau_residual,
                rcond=None,
            )
            print("Previous parameters:", model.getAllDynamicParameters())
            print("Calculated parameters:", pi_hat)

            params[cols] = pi_hat

            print("Final parameters:", params)

            model.setAllDynamicParameters(params)

            mpc.reinitialize(np.hstack((q, v)))
