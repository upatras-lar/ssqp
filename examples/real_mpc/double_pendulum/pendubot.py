import socket
import os
import numpy as np
import ssqpy
import argparse
from time import time, sleep

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
        "--hpipm",
        action=argparse.BooleanOptionalAction,
        help="Whether to use HPIPM as the QP backend",
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

if args.hpipm:
    dt = 0.01
    MPC_H = 45

    V_WGT = np.array((1e-3, 5e-2))
    U_WGT = 1e-2
    TIP_WGT = 20.0
    J1_WGT = 0.0
    J2_WGT = 2.0

    torque_clip = 0.1

else:
    dt = 0.01
    MPC_H = 20

    V_WGT = 3e-2
    U_WGT = 2e-3
    TIP_WGT = 4.2
    J1_WGT = 0.0
    J2_WGT = 2.0

    torque_clip = 0.1

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
    model, "tip", np.array((0.0, 0.0, 0.2)), TIP_WGT
)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, np.array((np.pi, 0.0)), np.array((J1_WGT, J2_WGT))
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

if args.hpipm:
    ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
    hpipm_params = ssqpy.solvers.hpipmParams()
    hpipm_params.tol_comp = 1e-3
    hpipm_params.tol_stat = 1e-3
    hpipm_params.tol_eq = 1e-3
    hpipm_params.tol_ineq = 1e-3
    hpipm_params.warm_start = True
    ssqp_params.hpipmParams = hpipm_params
else:
    admm_params = ssqpy.solvers.admmParams()
    admm_params.abs_tolerance = 1e-2
    admm_params.rel_tolerance = 1e-2
    admm_params.warm_start = True
    ssqp_params.admmParams = admm_params


mpc = ssqpy.solvers.MPC(
    model, ssqp_params, sqp_iters=4, qp_iters=100, keep_convergence_data=True
)

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

previous_filtered_vel = None
previous_filtered_pos = None
alpha_pos = 0.9
alpha_vel = 0.5
filtered_elapsed = dt

i = 0
while True if not args.plot else i < args.steps_to_plot:
    state = recv(conn)
    if state is None:
        break

    pos = state[:nq]
    vel = state[nq : nq + nv]

    if previous_filtered_pos is None:
        previous_filtered_pos = np.copy(pos)
    if previous_filtered_vel is None:
        previous_filtered_vel = np.copy(vel)
    pos = alpha_pos * pos + (1.0 - alpha_pos) * previous_filtered_pos
    vel = (pos - previous_filtered_pos) / filtered_elapsed
    vel = alpha_vel * vel + (1.0 - alpha_vel) * previous_filtered_vel

    previous_filtered_pos = np.copy(pos)
    previous_filtered_vel = np.copy(vel)

    filtered_state = np.concatenate([pos, vel])

    start = time()
    try:
        u = mpc.step(state[: nq + nv])[1].stage(0)
    except RuntimeError:
        u = np.zeros(nv)

    tau = np.clip(
        model.inverseDynamics(pos, vel, u), -torque_clip, torque_clip
    )
    tau[1] = 0.0

    elapsed = time() - start
    alpha_dt = 0.5
    filtered_elapsed = alpha_dt * elapsed + (1.0 - alpha_dt) * filtered_elapsed

    if args.plot:
        timestamps.append(time() - t0)
        data.append(
            (
                np.array(
                    (wrap_to_reference(pos)[0], wrap_to_reference(pos[1], 0.0))
                ),
                vel,
                tau,
            )
        )
        frequencies.append(1 / elapsed)
        i += 1

    send(conn, tau)

    # sleep(max(0.025 - elapsed, 0.0))

conn.close()

cd = mpc.getConvergenceData()

for i, mpc_step in enumerate(cd):
    mpc_step.print()

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
