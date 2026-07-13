import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep

MPC_H = 45
dt = 0.01

DATA_COLLECTION_STEPS = 500

# Initial Parameter Guess
m1 = 0.100
m2 = 0.070
I1 = 0.0005
I2 = 0.0001
r1 = 0.012
r2 = 0.02
b1 = 0.0004
b2 = 0.0004
cf1 = 0.00305
cf2 = 0.000777

initial_params = np.array((r1 * m1, I1, cf1, b1, r2 * m2, m2, I2, cf2, b2))

bounds = np.array(
    [
        [0.0001, 0.5],  # r1*m1
        [0.00001, 0.2],  # I1
        [0.0, 0.5],  # cf1
        [0.0, 0.5],  # b1
        [0.0001, 0.5],  # r2*m2
        [0.06, 1.0],  # m2
        [0.00001, 0.2],  # I2
        [0.0, 0.5],  # cf2
        [0.0, 0.5],  # b2
    ]
).T

print("Initial parameters:", initial_params)

actual = [
    5.246e-03,
    2.545e-04,
    1.829e-03,
    1.817e-04,
    2.978e-03,
    8.029e-02,
    1.626e-04,
    1.992e-03,
    4.228e-04,
]

np.set_printoptions(suppress=True)
print("Actual:", actual)

# ssqpy.setSilentMode()
# ssqpy.setLogLevel(ssqpy.LogLevel.Warning)
# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)
ssqpy.unsetLogLevel(ssqpy.LogLevel.Info)

V_WGT = 1e-3
U_WGT = 1e-3
C_WGT = np.array((5.0, 5.0, 0.0, 0.0))

INIT_STATE = np.array((0.0, 1.0, 0.0, 1.0, 0.0, 0.0))
FINAL_STATE = np.array((0.0, -1.0, 0.0, 1.0, 0.0, 0.0))

model = ssqpy.model.Model(MPC_H, dt)

custom_properties = ssqpy.model.CustomProperties()
custom_properties.nq = 4
custom_properties.nv = 2
custom_properties.nu = 2
custom_properties.njoints = 2
custom_properties.floating_base = False

model.setCustomProperties(custom_properties)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)

config_cost = (
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientTrackingCost(
        FINAL_STATE, C_WGT
    )
)

torque_limits = np.array((0.15, 0.15))
torque_constraint = ssqpy.model.constraints.JointControlConstraint(
    model, -torque_limits, torque_limits
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    model.addCost(k, config_cost)
    model.addConstraints(k, torque_constraint)

model.addCost(MPC_H, config_cost)

dynamics = [
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientDynamics(
        initial_params, dt=dt
    )
    for _ in range(MPC_H + 1)
]

model.finalize(dynamics=dynamics, init_states=[INIT_STATE])

ssqp_params = ssqpy.solvers.ssqpParams()

ssqp_params.tolerance = 1e-2
ssqp_params.admmParams.abs_tolerance = 1e-3
ssqp_params.admmParams.rel_tolerance = 1e-3
ssqp_params.admmParams.warm_start = False

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

mjmodel = mujoco.MjModel.from_xml_path(
    "../assets/mjcf/double_pendulum/double_pendulum.xml"
)
mjdata = mujoco.MjData(mjmodel)

elapsed = 0
steps = 0
total_steps = int(1 / dt)

np.set_printoptions(suppress=True)

data_states = []
data_controls = []
data_accelerations = []

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

        pre = time()
        try:
            states, controls = mpc.step(state)
            control = controls.stage(0)
        except RuntimeError as exc:
            print(exc)
            control = np.zeros(2)

        control = np.clip(control, -0.15, 0.15)
        mjdata.ctrl = control

        elapsed += time() - pre
        steps += 1

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        data_states.append(state)
        data_controls.append(control)
        data_accelerations.append(mjdata.qacc)

        to_sleep = mjmodel.opt.timestep - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

        if steps == DATA_COLLECTION_STEPS:
            Y_stacked = np.zeros((2 * DATA_COLLECTION_STEPS, 9))
            tau_stacked = np.zeros((2 * DATA_COLLECTION_STEPS))

            for i, z in enumerate(
                zip(data_states, data_controls, data_accelerations)
            ):
                state, control, accel = z

                Y, tau = dynamics[0].computeRegressorAndTorque(
                    state, accel, control
                )

                Y_stacked[2 * i : 2 * i + 2, :] = Y
                tau_stacked[2 * i : 2 * i + 2] = tau

            phi = np.linalg.lstsq(Y_stacked, tau_stacked)[0]
            phi = np.clip(phi, bounds[0, :], bounds[1, :])

            print("Optimal parameters:", phi)

            for d in dynamics:
                d.update_parameters(phi)
