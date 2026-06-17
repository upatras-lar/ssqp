import socket
import os
import numpy as np
import ssqpy

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


# Build MPC
MPC_H = 45
dt = 0.01

params = np.array(
    [
        0.10548177618443695,
        0.07619744360415454,
        0.05,
        0.05,
        0.00046166221821039165,
        0.00023702395072092597,
    ]
)

ssqpy.unsetLogLevel(ssqpy.LogLevel.Info)

V_WGT = 1e-3
U_WGT = 5.0
C_WGT = np.array((0.1, 0.1, 0.0, 0.0))

INIT_STATE = np.array((0.0, 1.0, 0.0, 1.0, 0.0, 0.0))

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
        np.array((0.0, -1.0, 0.0, 1.0, 0.0, 0.0)), C_WGT
    )
)
terminal_cost = (
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientTrackingCost(
        np.array((0.0, -1.0, 0.0, 1.0, 0.0, 0.0)), 100.0 * C_WGT
    )
)

torque_limits = np.array((0.1, 0.1))
torque_constraint = ssqpy.model.constraints.JointControlConstraint(
    model, -torque_limits, torque_limits
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    model.addCost(k, config_cost)
    model.addConstraints(k, torque_constraint)

model.addCost(MPC_H, terminal_cost)

dynamics = [
    ssqpy.model.custom.double_pendulum.DoublePendulumAmbientDynamics(
        params, dt=dt
    )
    for _ in range(MPC_H + 1)
]

model.finalize(dynamics=dynamics, init_states=[INIT_STATE])

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

# Socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

print("MPC ready, waiting for connection…")

conn, _ = server.accept()
print("Hardware connected")

while True:
    state = recv(conn)
    if state is None:
        break

    pos = state[:2]
    vel = state[2:]

    state = convert_state(pos, vel)

    try:
        u = mpc.step(state)[1].stage(0)
    except RuntimeError as exc:
        print(exc)
        u = np.zeros(custom_properties.nv)

    u = np.clip(u, -torque_limits, torque_limits)
    send(conn, u)
