import socket
import os
import numpy as np
import ssqpy
from time import time

from real_mpc.socket_utils import send, recv

SOCK_PATH = "/tmp/mpc_socket"

if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)


def convert_state(mq, mv):
    return np.hstack(
        (
            np.sin(mq[0]),
            np.cos(mq[0]),
            np.sin(mq[1]),
            np.cos(mq[1]),
            mv,
        )
    )


MPC_H = 125
dt = 0.01

USE_ACTUAL_PARAMS = True

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

params = actual_params


dyn = ssqpy.model.custom.double_pendulum.DoublePendulumAmbientDynamics(
    params,
    actuation_mode=ssqpy.model.custom.double_pendulum.ActuationMode.Pendubot,
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

A_WGT = np.array((1e-2, 10.0, 0.0, 0.0))
V_WGT = 1e-1
U_WGT = 1e-3
C_WGT = np.array((100.0, 50.0, 1.0, 1.0))

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

stage_cost = (
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientTrackingCost(
        np.array((0.0, -1.0, 0.0, 1.0, 0.0, 0.0)), A_WGT
    )
)
terminal_cost = (
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientTrackingCost(
        np.array((0.0, -1.0, 0.0, 1.0, 0.0, 0.0)), C_WGT
    )
)

velocity_limits = np.array((25.0, 25.0))
velocity_constraint = ssqpy.model.constraints.JointVelocityConstraint(
    model, -velocity_limits, velocity_limits
)

torque_limits = np.array((0.15,))
torque_constraint = ssqpy.model.constraints.JointControlConstraint(
    model, -torque_limits, torque_limits
)

for k in range(MPC_H):
    model.addCost(k, stage_cost)
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    model.addConstraints(k, velocity_constraint)
    model.addConstraints(k, torque_constraint)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, terminal_cost)

model.addConstraints(MPC_H, velocity_constraint)

dynamics = [
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientDynamics(
        actual_params if USE_ACTUAL_PARAMS else params,
        actuation_mode=ssqpy.model.custom.double_pendulum.ActuationMode.Pendubot,
        damping=np.array((7.634058385430087e-12, 0.0005106535523065844)),
        dt=dt,
    )
    for _ in range(MPC_H + 1)
]

model.finalize(dynamics=dynamics, init_states=[INIT_STATE])

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

ssqp_params.admmParams.abs_tolerance = 1e-3
ssqp_params.admmParams.rel_tolerance = 1e-3

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=3, qp_iters=100)

# Socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

print("MPC ready, waiting for connection…")

conn, _ = server.accept()
print("Hardware connected")

prev_state = None
last_update = 0.0

while True:
    state = recv(conn)
    if state is None:
        break

    pos = state[:2]
    vel = state[2:]

    state = convert_state(pos, vel)

    mpc.reinitialize(state)

    try:
        states, controls = mpc.step(state)
        u = controls.stage(0)

        # print("States:")
        # states.print()
        # print("Controls:")
        # controls.print()
    except RuntimeError as exc:
        print(exc)
        u = np.zeros(custom_properties.nu)

    u = np.clip(u, -torque_limits, torque_limits)
    u = np.array((u[0], 0.0))

    if (
        not USE_ACTUAL_PARAMS
        and time() - last_update > 5.0
        and prev_state is not None
        and search.addData(prev_state, u, state)
    ):
        new_params = dyn.parameters()

        print("New parameters:", new_params.tolist())

        for d in dynamics:
            d.update_parameters(new_params)

        window_size += 500

        if window_size > 5000:
            window_size = 1000
        search.setWindowSize(window_size)

        mpc.reinitialize(state)

        last_update = time()

    prev_state = state

    send(conn, u)
