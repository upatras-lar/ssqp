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


MPC_H = 100
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

# ssqpy.setSilentMode()
# ssqpy.setLogLevel(ssqpy.LogLevel.Warning)
# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)
ssqpy.unsetLogLevel(ssqpy.LogLevel.Info)

np.set_printoptions(suppress=True)

A_WGT = np.array((1e-2, 1e-2, 0.0, 0.0))
V_WGT = 1e-1
U_WGT = 1e-3
C_WGT = np.array((100.0, 20.0, 1.0, 1.0))

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
        initial_params,
        actuation_mode=ssqpy.model.custom.double_pendulum.ActuationMode.Pendubot,
        dt=dt,
    )
    for _ in range(MPC_H + 1)
]

model.finalize(dynamics=dynamics, init_states=[INIT_STATE])

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=3, qp_iters=100)

# Socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

print("MPC ready, waiting for connection…")

conn, _ = server.accept()
print("Hardware connected")

steps = 0

data_states = []
data_controls = []
data_accelerations = []

prev_time = time()

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

    accel = np.zeros(2)
    if data_states:
        new_time = time()
        accel = (vel - data_states[-1][4:]) / (new_time - prev_time)
        prev_time = new_time

    send(conn, u)

    data_states.append(state)
    data_controls.append(u)
    data_accelerations.append(accel)

    steps += 1

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
