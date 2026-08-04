import numpy as np
import ssqpy
import argparse

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline double pendulum dynamic parameter identification"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/double_pendulum_sysid.npz",
        help="Path to the .npz dataset saved by collect_double_pendulum_sysid.py",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action=argparse.BooleanOptionalAction,
        help="Whether to plot measured vs. predicted torque after identification",
    )

    parser.add_argument(
        "--unknown-cols",
        type=str,
        default=None,
        help=(
            "Comma-separated dynamic parameter indices to identify (e.g. "
            "'10,12,13,14'). If omitted, ALL parameters are identified in a "
            "single least-squares solve; if given, only these columns are "
            "solved for via residual regression, using the model's current "
            "values (from the URDF) for every other column — mirrors how the "
            "pendubot script identifies just link 2's parameters."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/identified_params.npy",
        help="Where to save the identified dynamic parameter vector",
    )

    return parser.parse_args()


args = parse_args()

if args.plot:
    import matplotlib.pyplot as plt

# Build the model purely for its kinematics/dynamics (regressor, current
# parameters) — no MPC solver needed for offline identification.
ssqpy.setSilentMode()

dt = 0.01
MPC_H = 1

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/double_pendulum.urdf",
    actuated_joints=[0, 1],
    solver_mode=ssqpy.model.SolverMode.InverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# --- Load the collected trajectory ---
dataset = np.load(args.input)
q = dataset["q"]
v = dataset["v"]
a = dataset["a"]
tau = dataset["tau"]

n_samples = q.shape[0]
print(f"Loaded {n_samples} samples from {args.input}")

# --- Build the stacked regressor, exactly like the online adaptation loop ---
Ys = []
taus = []

for k in range(n_samples):
    Y = model.getJointTorqueRegressor(q[k], v[k], a[k])
    Ys.append(Y)
    taus.append(tau[k])

Y_stack = np.vstack(Ys)
tau_stack = np.hstack(taus)

params = model.getAllDynamicParameters()
n_params = params.shape[0]

if args.unknown_cols is not None:
    cols = np.array([int(c) for c in args.unknown_cols.split(",")])
    known_cols = np.setdiff1d(np.arange(n_params), cols)

    Y_known = Y_stack[:, known_cols]
    Y_unknown = Y_stack[:, cols]

    # Subtract off the contribution of the columns we're trusting from the
    # URDF, then solve only for the unknown ones — same residual-regression
    # trick as the pendubot script.
    tau_residual = tau_stack - Y_known @ params[known_cols]

    pi_hat, *_ = np.linalg.lstsq(Y_unknown, tau_residual, rcond=None)

    print("Previous parameters (unknown cols):", params[cols])
    print("Identified parameters (unknown cols):", pi_hat)

    params[cols] = pi_hat
else:
    pi_hat, *_ = np.linalg.lstsq(Y_stack, tau_stack, rcond=None)

    print("Previous parameters:", params)
    print("Identified parameters:", pi_hat)

    params = pi_hat

model.setAllDynamicParameters(params)
tau_pred = Y_stack @ params

print("Final parameters:", params)
np.save(args.output, params)
print(f"Saved identified parameters to {args.output}")

# --- Fit quality ---
residual = tau_stack - tau_pred
rmse = np.sqrt(np.mean(residual**2))
ss_res = np.sum(residual**2)
ss_tot = np.sum((tau_stack - np.mean(tau_stack)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print(f"RMSE: {rmse:.5f} Nm")
print(f"R^2:  {r2:.5f}")

if args.plot:
    tau_measured = tau_stack.reshape(n_samples, nu)
    tau_predicted = tau_pred.reshape(n_samples, nu)

    fig, axes = plt.subplots(nu, 1, figsize=(10, 3 * nu), sharex=True)
    if nu == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(tau_measured[:, i], label="measured")
        ax.plot(tau_predicted[:, i], label="predicted", linestyle="--")
        ax.set_title(f"Joint {i} torque: measured vs. predicted")
        ax.set_ylabel("Nm")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Sample")

    plt.tight_layout()
    plt.savefig("data/double_pendulum_identification_fit.pdf")
    plt.show()
