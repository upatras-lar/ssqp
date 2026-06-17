import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep

MPC_H = 150
dt = 0.01

USE_ACTUAL_PARAMS = False

actual_params = np.array(
    [
        0.10548177618443695,
        0.07619744360415454,
        0.05,
        0.05,
        0.00046166221821039165,
        0.00023702395072092597,
    ]
)

params = np.array([0.4, 0.4, 0.1, 0.1, 0.01, 0.01])


dyn = ssqpy.model.custom.double_pendulum.DoublePendulumAmbientDynamics(
    params,
    actuation_mode=ssqpy.model.custom.double_pendulum.ActuationMode.Acrobot,
    dt=dt,
)

window_size = 2000

search = ssqpy.solvers.search.DynamicsParamGaussNewton(
    dyn, window_size, armijo_beta=0.5
)

# ssqpy.setSilentMode()
# ssqpy.setLogLevel(ssqpy.LogLevel.Warning)
# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)
ssqpy.unsetLogLevel(ssqpy.LogLevel.Info)

V_WGT = 1e-2
U_WGT = 1e-3
C_WGT = np.array((100.0, 10.0, 0.01, 0.01))

INIT_STATE = np.array((0.0, 1.0, 0.0, 1.0, 0.0, 0.0))

model = ssqpy.model.Model(MPC_H, dt)

custom_properties = ssqpy.model.CustomProperties()
custom_properties.nq = 4
custom_properties.nv = 2
custom_properties.nu = 1
custom_properties.njoints = 2
custom_properties.floating_base = False

model.setCustomProperties(custom_properties)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)

terminal_cost = (
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientTrackingCost(
        np.array((0.0, -1.0, 0.0, 1.0, 0.0, 0.0)), 10.0 * C_WGT
    )
)

torque_limits = np.array((0.15,))
torque_constraint = ssqpy.model.constraints.JointControlConstraint(
    model, -torque_limits, torque_limits
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    model.addConstraints(k, torque_constraint)

model.addCost(MPC_H, terminal_cost)

dynamics = [
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientDynamics(
        actual_params if USE_ACTUAL_PARAMS else params,
        actuation_mode=ssqpy.model.custom.double_pendulum.ActuationMode.Acrobot,
        dt=dt,
    )
    for _ in range(MPC_H + 1)
]

model.finalize(dynamics=dynamics, init_states=[INIT_STATE])

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

ssqp_params.tolerance = 1e-2
ssqp_params.admmParams.abs_tolerance = 1e-3
ssqp_params.admmParams.rel_tolerance = 1e-3
ssqp_params.admmParams.reset_rho = False
ssqp_params.admmParams.reset_y = False

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

mjmodel = mujoco.MjModel.from_xml_path(
    "../assets/mjcf/double_pendulum/double_pendulum.xml"
)
mjdata = mujoco.MjData(mjmodel)

elapsed = 0
steps = 0
total_steps = int(1 / dt)

np.set_printoptions(suppress=True)

i = 0

prev_state = None
last_update = time()

# mjdata.qpos = np.array((0.1, 0.0))

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack(
            (
                np.sin(mjdata.qpos[0]),
                np.cos(mjdata.qpos[0]),
                np.sin(mjdata.qpos[1]),
                np.cos(mjdata.qpos[1]),
                mjdata.qvel,
            )
        )

        if (
            not USE_ACTUAL_PARAMS
            and time() - last_update > 0.0
            and prev_state is not None
            and search.addData(prev_state, control, state)
        ):
            new_params = dyn.parameters()

            print("New parameters:", new_params.tolist())

            for d in dynamics:
                d.update_parameters(new_params)

            window_size += 1000

            if window_size > 5000:
                window_size = 1000
            search.setWindowSize(window_size)

            # search.setMinimumDataSize(window_size)
            mpc.reinitialize(state)

            last_update = time()

        prev_state = state

        pre = time()
        try:
            states, controls = mpc.step(state)
            control = controls.stage(0)
        except RuntimeError as exc:
            print(exc)
            control = np.zeros(2)

        control = np.clip(control, -0.15, 0.15)
        mjdata.ctrl[1] = control[0]

        elapsed = time() - pre
        steps += 1

        if steps % 100 == 0:
            print(f"\rFrequency: {1 / elapsed:.2f}Hz\t", end="")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        to_sleep = mjmodel.opt.timestep - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)
