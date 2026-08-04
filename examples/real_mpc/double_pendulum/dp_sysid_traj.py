import socket
import os
import numpy as np
import argparse
from time import time

from real_mpc.socket_utils import send, recv

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Double pendulum sys-id data collection server"
    )

    parser.add_argument(
        "-p",
        "--plot",
        action=argparse.BooleanOptionalAction,
        help="Whether to plot the collected trajectory after the run",
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=30.0,
        help="Duration of the excitation trajectory (s)",
    )

    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=10,
        help="Number of harmonics in the finite Fourier series excitation",
    )

    parser.add_argument(
        "--base-freq",
        type=float,
        default=0.1,
        help="Base frequency of the excitation Fourier series (Hz)",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.08,
        help="Torque amplitude scaling (Nm) applied to the excitation",
    )

    parser.add_argument(
        "--ramp",
        type=float,
        default=2.0,
        help="Raised-cosine ramp up/down time at the start/end of the run (s)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the excitation trajectory coefficients",
    )

    parser.add_argument(
        "--torque-clip",
        type=float,
        default=0.1,
        help="Per-joint torque saturation limit (Nm) for the commanded excitation",
    )

    parser.add_argument(
        "--damping",
        type=float,
        default=4e-3,
        help=(
            "Virtual viscous damping gain (Nm per rad/s) added on top of the "
            "excitation. Bounds energy build-up from resonance; does not "
            "meaningfully hurt persistence of excitation at small values."
        ),
    )

    parser.add_argument(
        "--max-vel",
        type=float,
        default=45.0,
        help=(
            "Safety trip threshold (rad/s). If either joint exceeds this, "
            "torque is ramped to zero and the run is stopped early."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/double_pendulum_sysid.npz",
        help="Where to save the collected (q, v, a, tau) dataset",
    )

    parser.add_argument(
        "--accel-filter-cutoff",
        type=float,
        default=None,
        help=(
            "Low-pass cutoff (Hz) applied to velocity before differentiating "
            "to get acceleration. Defaults to 5x the highest excited "
            "frequency (n_harmonics * base_freq), capped below the Nyquist "
            "frequency of the actual sample rate."
        ),
    )

    return parser.parse_args()


args = parse_args()

if args.plot:
    import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt

FILTER_ORDER = 4

SOCK_PATH = "/tmp/mpc_socket"

if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)

# Fully actuated double pendulum: state = [q(2), v(2), tau_actual(2)]
nq = 2
nv = 2
nu = 2

torque_bounds = np.full(nu, args.torque_clip)


def generate_exciting_trajectory(n_joints, n_harmonics, base_freq, seed):
    """
    Build a band-limited, persistently exciting finite Fourier series
    (Swevers-style) for each joint. Random sine/cosine coefficients are
    drawn per joint and normalized so the raw series has roughly unit
    amplitude before overall scaling is applied at command time.
    """
    rng = np.random.default_rng(seed)
    omega_f = 2 * np.pi * base_freq

    a_coef = rng.uniform(-1.0, 1.0, size=(n_joints, n_harmonics))
    b_coef = rng.uniform(-1.0, 1.0, size=(n_joints, n_harmonics))

    for j in range(n_joints):
        norm = np.sqrt(np.sum(a_coef[j] ** 2 + b_coef[j] ** 2))
        if norm > 0:
            a_coef[j] /= norm
            b_coef[j] /= norm

    harmonics = np.arange(1, n_harmonics + 1)
    return a_coef, b_coef, harmonics, omega_f


def excitation_torque(t, a_coef, b_coef, harmonics, omega_f, amplitude):
    n_joints = a_coef.shape[0]
    tau = np.empty(n_joints)
    for j in range(n_joints):
        tau[j] = np.sum(
            a_coef[j] * np.sin(harmonics * omega_f * t)
            + b_coef[j] * np.cos(harmonics * omega_f * t)
        )
    return amplitude * tau


def envelope(t, duration, ramp):
    """Raised-cosine ramp so the excitation starts/ends at zero torque."""
    if ramp <= 0:
        return 1.0
    if t < ramp:
        return 0.5 * (1 - np.cos(np.pi * t / ramp))
    elif t > duration - ramp:
        return 0.5 * (1 - np.cos(np.pi * (duration - t) / ramp))
    return 1.0


a_coef, b_coef, harmonics, omega_f = generate_exciting_trajectory(
    nu, args.n_harmonics, args.base_freq, args.seed
)

# Socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

print("Sys-id server ready, waiting for connection…")

conn, _ = server.accept()
print("Hardware connected")

data_t = []
data_q = []
data_v = []
data_tau = []

t0 = time()
state = None
tripped = False
trip_start = None
TRIP_RAMP_DOWN = 1.0  # seconds to bring commanded torque to zero after a trip

while True:
    state = recv(conn)
    if state is None:
        break

    t = time() - t0
    if t > args.duration:
        break

    pos = state[:nq]
    vel = state[nq : nq + nv]
    tau_actual = state[nq + nv : nq + nv + nu]

    if not tripped and np.any(np.abs(vel) > args.max_vel):
        tripped = True
        trip_start = t
        print(
            f"[SAFETY] |vel|={vel} exceeded max-vel={args.max_vel} at t={t:.2f}s "
            "— ramping torque to zero"
        )

    # Feed-forward excitation, windowed in and out at the start/end of the run.
    tau_cmd = excitation_torque(
        t, a_coef, b_coef, harmonics, omega_f, args.amplitude
    )
    tau_cmd *= envelope(t, args.duration, args.ramp)

    # Virtual viscous damping, always active: bounds energy build-up from
    # resonance without meaningfully hurting persistence of excitation.
    tau_cmd += -args.damping * vel

    if tripped:
        # Kill the feed-forward excitation and ramp everything to zero.
        decay = max(0.0, 1.0 - (t - trip_start) / TRIP_RAMP_DOWN)
        tau_cmd = decay * (-args.damping * vel)

    tau_cmd = np.clip(tau_cmd, -torque_bounds, torque_bounds)

    data_t.append(t)
    data_q.append(pos.copy())
    data_v.append(vel.copy())
    data_tau.append(tau_actual.copy())

    send(conn, tau_cmd)

    if tripped and (t - trip_start) > TRIP_RAMP_DOWN:
        break

# Safety: command zero torque before closing the connection
if state is not None:
    try:
        send(conn, np.zeros(nu))
    except (BrokenPipeError, OSError):
        pass

conn.close()

if tripped:
    print(f"Run stopped early by the safety cutoff at t={trip_start:.2f}s")

# --- Post-processing ---
t_arr = np.array(data_t)
q_arr = np.array(data_q)
v_arr = np.array(data_v)
tau_arr = np.array(data_tau)

n_samples = len(t_arr)
dt_mean = np.mean(np.diff(t_arr)) if n_samples > 1 else args.base_freq

# Acceleration is not measured directly, so it's estimated from the (noisy)
# measured velocity in two separate steps:
#   1) zero-phase low-pass filter the velocity to remove sensor/quantization
#      noise, with a cutoff set relative to the actual sample rate and to the
#      highest excited frequency (rather than a fixed sample-count window,
#      which silently stops doing any real smoothing at high sample rates);
#   2) differentiate the now-clean signal with a central difference.
# Filtering and differentiating in one fixed-size Savitzky-Golay pass is what
# caused the blown-up, oscillating accelerations: a short window relative to
# a fast sample rate barely smooths anything, and dividing that leftover
# noise by a small dt amplifies it enormously.
fs = 1.0 / dt_mean if dt_mean > 0 else 1.0
nyquist = fs / 2.0

highest_excited_freq = args.n_harmonics * args.base_freq
default_cutoff = min(5.0 * highest_excited_freq, 0.4 * nyquist)
requested_cutoff = (
    args.accel_filter_cutoff
    if args.accel_filter_cutoff is not None
    else default_cutoff
)
cutoff = min(requested_cutoff, 0.9 * nyquist)

print(
    f"Sample rate ≈ {fs:.1f} Hz, low-pass filtering velocity at "
    f"{cutoff:.2f} Hz before differentiating for acceleration"
)

min_filtfilt_samples = 3 * (
    2 * FILTER_ORDER + 1
)  # scipy's filtfilt padlen rule
if n_samples > min_filtfilt_samples and 0 < cutoff < nyquist:
    sos = butter(FILTER_ORDER, cutoff, btype="low", fs=fs, output="sos")
    v_smooth = sosfiltfilt(sos, v_arr, axis=0)
else:
    print(
        "Not enough samples (or invalid cutoff) for a reliable filtfilt — "
        "falling back to unfiltered velocity for differentiation."
    )
    v_smooth = v_arr

a_arr = np.gradient(v_smooth, t_arr, axis=0)


np.savez(args.output, t=t_arr, q=q_arr, v=v_arr, a=a_arr, tau=tau_arr)
print(f"Saved {n_samples} samples to {args.output}")

if args.plot:
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])

    ax_pos = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[1, 0], sharex=ax_pos)
    ax_acc = fig.add_subplot(gs[2, 0], sharex=ax_pos)
    ax_tau = fig.add_subplot(gs[3, 0], sharex=ax_pos)

    for i, label in enumerate(["q₀", "q₁"]):
        ax_pos.plot(t_arr, q_arr[:, i], label=label)
    ax_pos.set_title("Position")
    ax_pos.set_ylabel("rad")
    ax_pos.legend()
    ax_pos.grid(True)

    for i, label in enumerate(["v₀", "v₁"]):
        ax_vel.plot(t_arr, v_arr[:, i], label=label)
    ax_vel.set_title("Velocity")
    ax_vel.set_ylabel("rad/s")
    ax_vel.legend()
    ax_vel.grid(True)

    for i, label in enumerate(["a₀", "a₁"]):
        ax_acc.plot(t_arr, a_arr[:, i], label=label)
    ax_acc.set_title("Acceleration (Savitzky-Golay estimate)")
    ax_acc.set_ylabel("rad/s²")
    ax_acc.legend()
    ax_acc.grid(True)

    for i, label in enumerate(["τ₀", "τ₁"]):
        ax_tau.plot(t_arr, tau_arr[:, i], label=label)
    ax_tau.set_title("Measured Torque")
    ax_tau.set_xlabel("Time (s)")
    ax_tau.set_ylabel("Nm")
    ax_tau.legend()
    ax_tau.grid(True)

    plt.tight_layout()
    # plt.savefig(f"../plots/double_pendulum_sysid_{args.duration}.pdf")
    plt.show()
