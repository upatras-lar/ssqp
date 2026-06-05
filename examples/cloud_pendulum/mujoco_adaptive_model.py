import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi
from time import time, sleep

MPC_H = 45
dt = 0.01

params = np.array([0.1, 0.07, 0.05, 0.05, 0.004, 0.002])
dyn = ssqpy.model.custom.double_pendulum.DoublePendulumDynamics(params, dt=dt)
gn = ssqpy.solvers.DynamicsParamGaussNewton(dyn, window_size=500)

# ssqpy.setSilentMode()
# ssqpy.setLogLevel(ssqpy.LogLevel.Warning)
# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)
ssqpy.unsetLogLevel(ssqpy.LogLevel.Info)

V_WGT = 1e-5
U_WGT = 1e-5
TIP_WGT = 5.0

model = ssqpy.model.Model(MPC_H, dt)

custom_properties = ssqpy.model.CustomProperties()
custom_properties.nq = 2
custom_properties.nv = 2
custom_properties.nu = 2
custom_properties.njoints = 2
custom_properties.floating_base = False

model.setCustomProperties(custom_properties)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
target = np.array((0.0, params[2] + params[3] + 0.01))
tip_cost = ssqpy.model.custom.double_pendulum.DoublePendulumTipCost(
    target, params[2], params[3], TIP_WGT
)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model,
    np.array((pi, 0.0)),
    np.array((0.0001, 0.0)),
)

torque_limits = np.array((0.15, 0.0))
torque_constraint = ssqpy.model.constraints.JointControlConstraint(
    model, -torque_limits, torque_limits
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    model.addCost(k, tip_cost)
    model.addConstraints(k, torque_constraint)

# model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, tip_cost)
model.addCost(MPC_H, config_cost)

dynamics = [
    ssqpy.model.custom.double_pendulum.DoublePendulumDynamics(params, dt=dt)
    for _ in range(MPC_H + 1)
]

model.finalize(dynamics=dynamics)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

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

i = 0

prev_state = None

# mjdata.qpos = np.array((0.1, 0.0))

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        # if prev_state is not None:
        #     gn.addData(prev_state, control, state)
        # print("Adding data...")
        # print("x:", prev_state.tolist())
        # print("u:", control.tolist())
        # print("Next x:", state.tolist())

        # new_params = dyn.parameters()
        # print("New parameters:", new_params.tolist())

        # for d in dynamics:
        #     d.update_parameters(new_params)

        prev_state = state

        pre = time()
        states, controls = mpc.step(state)
        # print("State:")
        # states.print()
        # print("Control:")
        # controls.print()

        control = controls.stage(0)

        control = np.clip(control, -0.15, 0.15)
        # control[1] = 0.0
        mjdata.ctrl = control

        elapsed += time() - pre
        steps += 1

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)
