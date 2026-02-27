import socket
import os
import numpy as np
from math import atan2, sin, cos
import ssqpy
import sys

sys.path.insert(0, "..")
from socket_utils import send, recv

# Use -1 to to not plot
PLOT_STEPS = -1

if PLOT_STEPS > 0:
    import matplotlib.pyplot as plt

SOCK_PATH = "/tmp/mpc_socket"

if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)

# Build MPC
nq = 2
nv = 2
dt = 0.01
MPC_H = 45

V_WGT = 7e-4
U_WGT = 7e-4
TIP_WGT = 4.0

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, U_WGT)
tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, np.array((0.0, 0.0, 0.1)), "tip", nq, nv, TIP_WGT
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, tip_cost)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, tip_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.adaptive_rho_tolerance = 10
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

ssqp_params.admm_params = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=200)

# Socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

print("MPC ready, waiting for connection…")

conn, _ = server.accept()
print("Hardware connected")

data = []

normalize_angle = lambda a: atan2(sin(a), cos(a))

i = 0
while True if PLOT_STEPS <= 0 else i < PLOT_STEPS:
    state = recv(conn)
    if state is None:
        break

    pos = state[:nq]
    vel = state[nq:]

    try:
        u = mpc.step(state)
    except RuntimeError:
        u = np.zeros(nv)

    tau = model.inverseDynamics(pos, vel, u)
    tau[1] = 0.0

    if PLOT_STEPS > 0:
        data.append(
            (
                np.array(list(map(normalize_angle, pos))),
                vel,
                np.clip(tau, -0.15, 0.15),
            )
        )
        i += 1

    send(conn, tau)

conn.close()

if PLOT_STEPS > 0:
    # Unpack data
    positions = np.array([item[0] for item in data])  # (N, 2)
    velocities = np.array([item[1] for item in data])  # (N, 2)
    torques = np.array([item[2] for item in data])  # (N, 2)

    # Time axis (sample index)
    t = np.arange(len(data))

    # Torque first component
    torque_0 = torques[:, 0]

    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    # --- Position (x & y together) ---
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_pos.plot(t, positions[:, 0], label="q₀")
    ax_pos.plot(t, positions[:, 1], label="q₁")
    ax_pos.set_title("Position")
    ax_pos.set_ylabel("Position")
    ax_pos.legend()
    ax_pos.grid(True)

    # --- Velocity (vx & vy together) ---
    ax_vel = fig.add_subplot(gs[1, 0], sharex=ax_pos)
    ax_vel.plot(t, velocities[:, 0], label="v₀")
    ax_vel.plot(t, velocities[:, 1], label="v₁")
    ax_vel.set_title("Velocity")
    ax_vel.set_ylabel("Velocity")
    ax_vel.legend()
    ax_vel.grid(True)

    # --- Torque (first component) ---
    ax_tau = fig.add_subplot(gs[2, 0], sharex=ax_pos)
    ax_tau.plot(t, torque_0, label="τ₀")
    ax_tau.set_title("Torque")
    ax_tau.set_xlabel("Time")
    ax_tau.set_ylabel("τ₀")
    ax_tau.grid(True)

    plt.tight_layout()
    plt.show()
