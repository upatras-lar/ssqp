import ssqpy
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import pi, sin, cos, sqrt
from time import time

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

    return parser.parse_args()


args = parse_args()


nq = 9
nv = 9
nu = 9

H = 400
dt = 0.01
mpc_H = 8
mpc_dt = 0.1

T_WGT = 10.0
R_WGT = 1.0
V_WGT = 1e-2
U_WGT = 1e-3
C_WGT = 0.1

INVERSE_MODE = True if args.dynamics_mode == "inverse" else False

cx = 0.3
cy = 0
cz = 0.26
cr = 0.22
distance = 2 * pi
ω = distance / (H * dt)

traj = []

for k in range(H + 1):
    φk = ω * k * dt
    traj.append(np.array([cx, cy + cr * sin(φk), cz + cr * cos(φk)]))

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

square_side_hlf = cr / sqrt(2)
constraint_hlines = [cz - square_side_hlf, cz + square_side_hlf]
constraint_vlines = [cy - square_side_hlf, cy + square_side_hlf]


def run(dynamics_mode, constrain):
    inverse_mode = dynamics_mode == ssqpy.model.DynamicsMode.Inverse

    model = ssqpy.model.Model(
        mpc_H,
        mpc_dt,
        urdf_path="../../assets/urdf/panda/panda.urdf",
        dynamics_mode=dynamics_mode,
    )

    init_state = np.zeros(2 * nq)
    init_state[:nq] = [
        0.0013,
        -0.9115,
        -0.0010,
        -2.6316,
        -0.0008,
        1.7431,
        0.7862,
        0.0,
        0.0,
    ]

    pose_costs = []

    lb = np.array((-np.inf, constraint_vlines[0], constraint_hlines[0]))
    ub = np.array((np.inf, constraint_vlines[1], constraint_hlines[1]))
    box_constraint = ssqpy.model.constraints.FramePositionBound(
        model, "panda_ee", lb, ub, nq, nv
    )

    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        init_state[:nq], nq, nv, C_WGT, False
    )
    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)

    for k in range(mpc_H):
        pose_costs.append(
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                model, "panda_ee", traj[k], Rd, nq, nv, T_WGT, R_WGT
            )
        )

        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, config_cost)
        model.addCost(k, pose_costs[-1])

        if constrain:
            model.addConstraints(k, box_constraint)

    pose_costs.append(
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, "panda_ee", traj[mpc_H], Rd, nq, nv, T_WGT, R_WGT
        )
    )

    model.addCost(mpc_H, v_cost)
    model.addCost(mpc_H, pose_costs[-1])

    if constrain:
        model.addConstraints(mpc_H, box_constraint)

    def update_pose_costs(current_stage):
        for k in range(current_stage, current_stage + mpc_H):
            pose_costs[k - current_stage].set_desired_translation(traj[k % H])

    ssqp_params = ssqpy.solvers.ssqpParams()
    ssqp_params.tolerance = 1e-2

    admm_params = ssqpy.solvers.admmParams()
    admm_params.abs_tolerance = 1e-3
    admm_params.rel_tolerance = 1e-3
    admm_params.warm_start = False

    if inverse_mode:
        admm_params.init_rho = 10.0
        admm_params.adaptive_rho_tolerance = 25.0
        admm_params.rho_update_iter_freq = 25
    else:
        admm_params.init_rho = 5.0
        admm_params.adaptive_rho_tolerance = 10.0
        admm_params.rho_update_iter_freq = 25

    ssqp_params.admmParams = admm_params

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=125)

    simulator = SSQPYSimulator(model, dt, init_state)

    state_data = [model.forwardKinematics(simulator.state[:nq], "panda_ee")[0]]
    frequency_data = []

    for stage in range(1, H + 1):
        pre = time()

        if inverse_mode:
            sim_ctrl = model.inverseDynamics(
                simulator.state[:nq],
                simulator.state[nq:],
                mpc.step(simulator.state),
            )
        else:
            sim_ctrl = mpc.step(simulator.state)

        elapsed = time() - pre

        state_data.append(
            model.forwardKinematics(simulator.state[:nq], "panda_ee")[0]
        )
        frequency_data.append(1 / elapsed)

        simulator.step(sim_ctrl)

        update_pose_costs(stage)

    return state_data, frequency_data


unconstrained_state_data, unconstrained_frequency_data = run(
    (
        ssqpy.model.DynamicsMode.Inverse
        if INVERSE_MODE
        else ssqpy.model.DynamicsMode.Forward
    ),
    False,
)

constrained_state_data, constrained_frequency_data = run(
    (
        ssqpy.model.DynamicsMode.Inverse
        if INVERSE_MODE
        else ssqpy.model.DynamicsMode.Forward
    ),
    True,
)

traj = np.array(traj)
traj = np.delete(traj, 0, axis=1)

constraint_traj = traj.copy()
constraint_traj[:, 0] = np.clip(
    constraint_traj[:, 0], constraint_vlines[0], constraint_vlines[1]
)
constraint_traj[:, 1] = np.clip(
    constraint_traj[:, 1], constraint_hlines[0], constraint_hlines[1]
)

fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, figure=fig)

ax_big = fig.add_subplot(gs[:, 0])
ax_big.set_title("Trajectory")
ax_big.plot(traj[:, 0], traj[:, 1], label="Cost Trajectory", linewidth=5)
ax_big.hlines(
    y=constraint_hlines,
    xmin=constraint_vlines[0],
    xmax=constraint_vlines[1],
    colors="red",
)
ax_big.vlines(
    x=constraint_vlines,
    ymin=constraint_hlines[0],
    ymax=constraint_hlines[1],
    colors="darkred",
)

ax_top = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, 1])


def add_plots(state_data, frequency_data, ref_traj, legend_pref=""):
    state_data = np.array(state_data)
    state_data = np.delete(state_data, 0, axis=1)

    ax_big.plot(
        state_data[:, 0],
        state_data[:, 1],
        label=f"{legend_pref} Trajectory",
        linewidth=3,
    )

    ax_big.legend()

    ax_top.plot(
        np.arange(H + 1),
        np.linalg.norm(state_data - ref_traj, axis=1),
        label=f"{legend_pref} - e",
    )
    ax_top.set_title("Error from Reference Trajectory")
    ax_top.set_ylabel("Error Norm")
    ax_top.legend()
    ax_top.grid(True)

    ax_bottom.plot(
        np.arange(H), np.array(frequency_data), label=f"{legend_pref} - f"
    )
    ax_bottom.set_title("Frequency")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Frequency (Hz)")
    # ax_bottom.set_yscale("log")
    ax_bottom.legend()
    ax_bottom.grid(True)


add_plots(
    unconstrained_state_data,
    unconstrained_frequency_data,
    traj,
    "Unconstrained",
)
add_plots(
    constrained_state_data,
    constrained_frequency_data,
    constraint_traj,
    "Constrained",
)

plt.tight_layout()
plt.savefig(f"data/plots/panda_{args.dynamics_mode}.pdf")
plt.show()
