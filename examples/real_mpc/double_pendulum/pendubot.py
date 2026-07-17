import socket
import os
import numpy as np
import ssqpy
import argparse
from time import time

from real_mpc.socket_utils import send, recv

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Pendubot controller server")

    parser.add_argument(
        "-p",
        "--plot",
        action=argparse.BooleanOptionalAction,
        help="Whether to run for a certain number of steps and then plot",
    )

    parser.add_argument(
        "-s",
        "--steps-to-plot",
        type=float,
        default=1000,
        help="Number of steps to run for before plotting",
    )

    parser.add_argument(
        "-a",
        "--adapt-model",
        action=argparse.BooleanOptionalAction,
        help="Whether to run with model adaptation enabled",
    )

    return parser.parse_args()


args = parse_args()

if args.plot:
    import matplotlib.pyplot as plt

SOCK_PATH = "/tmp/mpc_socket"

if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)

# Build MPC
ssqpy.setSilentMode()

dt = 0.01
MPC_H = 60

V_WGT = 2e-2
U_WGT = 7e-4
TIP_WGT = 4.2

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    solver_mode=ssqpy.model.SolverMode.InverseDynamics,
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
    model, np.array((0.0, 0.0)), np.array((0.0, 2.3e-1))
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, tip_cost)
    model.addCost(k, config_cost)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, tip_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1
# ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.adaptive_rho_tolerance = 10
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_comp = 1e-3
hpipm_params.tol_stat = 1e-3
hpipm_params.tol_eq = 1e-3
hpipm_params.tol_ineq = 1e-3
ssqp_params.hpipmParams = hpipm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=8, qp_iters=200)

# Socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

print("MPC ready, waiting for connection…")

conn, _ = server.accept()
print("Hardware connected")

data = []


def wrap_to_reference(theta, reference=np.pi):
    return reference + np.arctan2(
        np.sin(theta - reference), np.cos(theta - reference)
    )


t0 = time()
timestamps = []
frequencies = []

Ys = []
taus = []
j = 0
DATA_COLLECTION_STEPS = 2000
previous_filtered_vel = None

i = 0
while True if not args.plot else i < args.steps_to_plot:
    state = recv(conn)
    if state is None:
        break

    pos = state[:nq]
    vel = state[nq : nq + nv]

    if args.adapt_model and j <= DATA_COLLECTION_STEPS:
        measured_tau = state[nq + nv :]
        # print("Tau:", measured_tau)
        new_measurement = time()

        alpha = 0.1  # smaller = more smoothing

        if previous_filtered_vel is None:
            filtered_vel = vel.copy()
        else:
            filtered_vel = alpha * vel + (1 - alpha) * previous_filtered_vel

        if previous_filtered_vel is None:
            acc = np.zeros_like(vel)
        else:
            acc = (filtered_vel - previous_filtered_vel) / dt

        previous_filtered_vel = filtered_vel

        previous_vel = vel
        previous_measurement_t = new_measurement

        Y = model.getJointTorqueRegressor(pos, vel, acc)

        Ys.append(Y)
        taus.append(measured_tau)

        j += 1

        if j == DATA_COLLECTION_STEPS:
            Y_stack = np.vstack(Ys)
            tau_stack = np.hstack(taus)

            params = model.getAllDynamicParameters()

            # Indices of parameters to guess
            # Corresponds to: [mass_2, com_x_2, com_y_2, com_z_2, I_xx]
            cols = [10, 12, 13, 14]
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

            mpc.reinitialize(np.hstack((pos, vel)))

    start = time()
    try:
        u = mpc.step(state[: nq + nv])[1].stage(0)
    except RuntimeError:
        u = np.zeros(nv)

    tau = model.inverseDynamics(pos, vel, u)
    tau[1] = 0.0

    elapsed = time() - start

    if args.plot:
        timestamps.append(time() - t0)
        data.append(
            (
                np.array(
                    (wrap_to_reference(pos)[0], wrap_to_reference(pos[1], 0.0))
                ),
                vel,
                np.clip(tau, -0.1, 0.1),
            )
        )
        frequencies.append(1 / elapsed)
        i += 1

    send(conn, tau)

conn.close()

if args.plot:
    # Unpack data
    positions = np.array([item[0] for item in data])  # (N, 2)
    velocities = np.array([item[1] for item in data])  # (N, 2)
    torques = np.array([item[2] for item in data])[:, 0]  # (N, 2)
    frequencies = np.array(frequencies)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])

    ax_pos = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[1, 0], sharex=ax_pos)
    ax_tau = fig.add_subplot(gs[2, 0], sharex=ax_pos)
    ax_freq = fig.add_subplot(gs[3, 0], sharex=ax_pos)

    t = np.array(timestamps)

    # --- Position ---
    for i, label in enumerate(["q₀", "q₁"]):
        ax_pos.plot(t, positions[:, i], label=label)

    ax_pos.set_title("Position")
    ax_pos.set_ylabel("Position")
    ax_pos.legend()
    ax_pos.grid(True)

    # --- Velocity ---
    for i, label in enumerate(["v₀", "v₁"]):
        ax_vel.plot(t, velocities[:, i], label=label)

    ax_vel.set_title("Velocity")
    ax_vel.set_ylabel("Velocity")
    ax_vel.legend()
    ax_vel.grid(True)

    # --- Torque (τ₀) ---
    ax_tau.plot(t, torques, label="τ₀")

    ax_tau.set_title("Torque")
    ax_tau.set_ylabel("τ₀")
    ax_tau.grid(True)

    # --- Frequency ---
    ax_freq.plot(t, frequencies, label=f"f")

    ax_freq.set_title("Control Frequency")
    ax_freq.set_xlabel("Time (s)")
    ax_freq.set_ylabel("Frequency (Hz)")
    ax_freq.set_yscale("log")

    ax_freq.legend()
    ax_freq.grid(True)

    plt.tight_layout()
    plt.savefig(f"../plots/real_pendubot_{args.steps_to_plot}.pdf")
    plt.show()
