import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi
from time import time, sleep

ssqpy.setSilentMode()

MPC_H = 45
dt = 0.01

V_WGT = 1e-3
U_WGT = 1e-3
TIP_WGT = 5.0

REAL_TIME = True
NOISE = False
INVERSE_MODE = False
CONTROL_DELAY = 1


model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../assets/urdf/double_pendulum/pendubot.urdf",
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
# tau_cost = ssqpy.model.costs.SquaredIDTorqueCost(
#     model, np.array((0.0, 1.0))
# )
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
    # model.addCost(k, tau_cost)
    model.addCost(k, tip_cost)

model.addCost(MPC_H, tip_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()

mpc = None

if INVERSE_MODE:
    ssqp_params.tolerance = 1e-3
    ssqp_params.admmParams.init_rho = 5.0
    ssqp_params.admmParams.adaptive_rho_tolerance = 5.0
    ssqp_params.admmParams.rho_update_iter_freq = 25
    ssqp_params.admmParams.abs_tolerance = 1e-3
    ssqp_params.admmParams.rel_tolerance = 1e-3
    ssqp_params.admmParams.warm_start = False
    # ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)
else:
    ssqp_params.tolerance = 1e-2
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

i = 0

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        pre = time()
        try:
            control = mpc.step(state)[1].stage(0)
        except RuntimeError:
            reg_fails += 1
            control = np.zeros(nv)

        if INVERSE_MODE:
            torques = model.inverseDynamics(mjdata.qpos, mjdata.qvel, control)
            delayed_control.append(torques[0])
            if len(delayed_control) == CONTROL_DELAY:
                mjdata.ctrl[0] = delayed_control.pop(0)

        else:
            delayed_control.append(control[0])
            if len(delayed_control) == CONTROL_DELAY:
                mjdata.ctrl[0] = delayed_control.pop(0)

        if NOISE:
            mjdata.ctrl[0] += (
                np.random.uniform(0, 0.1) if np.random.uniform(0, 10) > 8 else 0.0
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

        mjdata.ctrl = np.clip(mjdata.ctrl, -0.15, 0.15)
        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        if REAL_TIME:
            to_sleep = dt - (time() - frame_start)
            if to_sleep > 0:
                sleep(to_sleep)
