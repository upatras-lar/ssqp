import ssqpy
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from time import time
import argparse

from simulators.simulator import SSQPYSimulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmarking MPC script for Pendubot problem"
    )

    parser.add_argument(
        "-m",
        "--dynamics-mode",
        choices=["inverse", "forward"],
        default="forward",
        help="The dynamics mode to use",
    )

    parser.add_argument(
        "-d",
        "--control-delay",
        type=int,
        default=1,
        help="Number of steps to delay controls by",
    )

    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=1000,
        help="Number of steps to simulate",
    )

    parser.add_argument(
        "--disturbance-step",
        type=int,
        default=None,
        help="Step number at which to apply a disturbance",
    )

    parser.add_argument(
        "--disturbance-within",
        type=float,
        default=0.1,
        help="Absolute value bound for random disturbance",
    )

    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=1,
        help="Times to repeat experiment for averaging",
    )

    return parser.parse_args()


args = parse_args()

nq = 2
nv = 2
MPC_H = 45
dt = 0.01

V_WGT = 1e-3
U_WGT = 1e-3
TIP_WGT = 5.0

INVERSE_MODE = True if args.dynamics_mode == "inverse" else False

C_WGT = 1.0 if INVERSE_MODE else 0.01


def run():

    model = ssqpy.model.Model(
        MPC_H,
        dt,
        urdf_path="../../assets/urdf/double_pendulum/pendubot.urdf",
        actuated_joints=[0],
        dynamics_mode=(
            ssqpy.model.DynamicsMode.Inverse
            if INVERSE_MODE
            else ssqpy.model.DynamicsMode.Forward
        ),
    )

    vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(nv, U_WGT)
    tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
        model, np.array((0.0, 0.0, 0.1)), "tip", nq, nv, TIP_WGT
    )
    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        np.array((pi, 0.0)),
        nq,
        nv,
        np.array((C_WGT, 0.0)),
        False,
    )

    for k in range(MPC_H):
        model.addCost(k, vel_cost)
        model.addCost(k, u_cost)
        model.addCost(k, tip_cost)

    model.addCost(MPC_H, tip_cost)
    model.addCost(MPC_H, config_cost)

    ssqp_params = ssqpy.solvers.ssqpParams()

    if INVERSE_MODE:
        ssqp_params.tolerance = 1e-3
        ssqp_params.admmParams.init_rho = 5.0
        ssqp_params.admmParams.adaptive_rho_tolerance = 5.0
        ssqp_params.admmParams.rho_update_iter_freq = 25
        ssqp_params.admmParams.abs_tolerance = 1e-3
        ssqp_params.admmParams.rel_tolerance = 1e-3
        ssqp_params.admmParams.warm_start = False

        mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)
    else:
        ssqp_params.tolerance = 1e-2
        ssqp_params.admmParams.abs_tolerance = 1e-3
        ssqp_params.admmParams.rel_tolerance = 1e-3
        ssqp_params.admmParams.warm_start = False

        mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

    simulator = SSQPYSimulator(model, dt, np.zeros(4))
    delayed_control = []

    normalize_angle = lambda a: np.arctan2(np.sin(a), np.cos(a))

    state_data = []
    frequency_data = []

    for i in range(1, args.steps + 1):
        frame_start = time()

        pre = time()
        try:
            control = mpc.step(simulator.state)
        except RuntimeError:
            control = np.zeros(nv)

        sim_ctrl = np.zeros(2)
        if INVERSE_MODE:
            torques = model.inverseDynamics(
                simulator.state[:2], simulator.state[2:], control
            )
            delayed_control.append(torques[0])

            if len(delayed_control) == args.control_delay:
                sim_ctrl[0] = np.clip(delayed_control.pop(0), -0.15, 0.15)
        else:
            delayed_control.append(control[0])
            if len(delayed_control) == args.control_delay:
                sim_ctrl[0] = np.clip(delayed_control.pop(0), -0.15, 0.15)

        elapsed = time() - pre

        if i == args.disturbance_step:
            sim_ctrl += np.random.uniform(
                -args.disturbance_within, args.disturbance_within, 2
            )

        simulator.step(sim_ctrl)

        state_data.append(
            (
                np.array(list(map(normalize_angle, simulator.state[:nq]))),
                simulator.state[nq:],
                sim_ctrl,
            )
        )

        frequency_data.append(1 / elapsed)

    return state_data, frequency_data


all_state_data = []
all_frequency_data = []

for _ in range(args.repeats):
    data = run()
    all_state_data.append(data[0])
    all_frequency_data.append(data[1])


positions = np.array(
    [[item[0] for item in run] for run in all_state_data]
)  # (R, N, 2)
velocities = np.array(
    [[item[1] for item in run] for run in all_state_data]
)  # (R, N, 2)
torques = np.array(
    [[item[2] for item in run] for run in all_state_data]
)  # (R, N, 2)
frequencies = np.array(all_frequency_data)

# Time axis (sample index)
t = np.arange(args.steps) * dt


def stats(x):
    median = np.percentile(x, 50, axis=0)
    p25 = np.percentile(x, 25, axis=0)
    p75 = np.percentile(x, 75, axis=0)
    return median, p25, p75


pos_med, pos_25, pos_75 = stats(positions)
vel_med, vel_25, vel_75 = stats(velocities)
tau_med, tau_25, tau_75 = stats(torques[:, :, 0])
freq_med, freq_25, freq_75 = stats(frequencies)

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])

# --- Position ---
ax_pos = fig.add_subplot(gs[0, 0])

for i, label in enumerate(["q₀", "q₁"]):
    ax_pos.plot(t, pos_med[:, i], label=label)
    ax_pos.fill_between(t, pos_25[:, i], pos_75[:, i], alpha=0.3)

ax_pos.set_title("Position")
ax_pos.set_ylabel("Position")
ax_pos.legend()
ax_pos.grid(True)

# --- Velocity ---
ax_vel = fig.add_subplot(gs[1, 0], sharex=ax_pos)

for i, label in enumerate(["v₀", "v₁"]):
    ax_vel.plot(t, vel_med[:, i], label=label)
    ax_vel.fill_between(t, vel_25[:, i], vel_75[:, i], alpha=0.3)

ax_vel.set_title("Velocity")
ax_vel.set_ylabel("Velocity")
ax_vel.legend()
ax_vel.grid(True)

# --- Torque (τ₀) ---
ax_tau = fig.add_subplot(gs[2, 0], sharex=ax_pos)

ax_tau.plot(t, tau_med, label="τ₀")
ax_tau.fill_between(t, tau_25, tau_75, alpha=0.3)

ax_tau.set_title("Torque")
ax_tau.set_ylabel("τ₀")
ax_tau.grid(True)

# --- Frequency ---
ax_freq = fig.add_subplot(gs[3, 0], sharex=ax_pos)

ax_freq.plot(t, freq_med, label="f")
ax_freq.fill_between(t, freq_25, freq_75, alpha=0.3)

ax_freq.set_title("Frequency")
ax_freq.set_xlabel("Time (s)")
ax_freq.set_ylabel("Frequency (Hz)")
ax_freq.grid(True)

plt.tight_layout()

plot_name = f"data/plots/pendubot_{args.dynamics_mode}"

if args.control_delay > 1:
    plot_name += f"_d{args.control_delay}"

if args.disturbance_step is not None:
    plot_name += f"_disturbance-{args.disturbance_within}"

if args.repeats > 1:
    plot_name += f"_repeats-{args.repeats}"

plot_name += ".pdf"

plt.savefig(plot_name)
plt.show()
