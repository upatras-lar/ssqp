import ssqpy
import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import pi
from time import time

from simulators.simulator import SSQPYSimulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmarking MPC script for the Go2 standing problem"
    )

    parser.add_argument(
        "-m",
        "--dynamics-mode",
        choices=["reduced-inverse", "inverse", "forward"],
        default="forward",
        help="The dynamics mode to use",
    )

    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100,
        help="Number of steps to simulate",
    )

    return parser.parse_args()


args = parse_args()


dt = 0.01
mpc_H = 15
mpc_dt = 0.05

V_WGT = 1e-3
U_WGT = 1e-4
C_WGT = 10.0

CONTACT_MU = 0.6

mode_map = {
    "reduced-inverse": ssqpy.model.DynamicsMode.ReducedInverse,
    "inverse": ssqpy.model.DynamicsMode.Inverse,
    "forward": ssqpy.model.DynamicsMode.Forward,
}


def run(dynamics_mode):
    inverse_mode = (
        dynamics_mode == ssqpy.model.DynamicsMode.Inverse
        or dynamics_mode == ssqpy.model.DynamicsMode.ReducedInverse
    )

    model = ssqpy.model.Model(
        mpc_H,
        mpc_dt,
        urdf_path="../../assets/urdf/go2/go2.urdf",
        free_flyer=True,
        dynamics_mode=dynamics_mode,
    )

    nq = model.getnq()
    nv = model.getnv()
    nu = model.getnu()

    base_pos = np.array([0.0, 0.0, 0.3689268220122])
    base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

    # Initial configuration
    init_state = np.zeros(37)
    init_state[:3] = base_pos
    init_state[6] = 1.0
    init_state[7:19] = base_config

    config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
        model, init_state[7:nq], nq, nv, C_WGT, True
    )
    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT, True)
    u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)

    fl_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
        model, "FL_foot", np.zeros(3), np.zeros(3), nq, nv
    )

    fr_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
        model, "FR_foot", np.zeros(3), np.zeros(3), nq, nv
    )

    rl_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
        model, "RL_foot", np.zeros(3), np.zeros(3), nq, nv
    )

    rr_foot_vel_bound = ssqpy.model.constraints.FrameLinearVelocityBound(
        model, "RR_foot", np.zeros(3), np.zeros(3), nq, nv
    )

    for k in range(mpc_H):
        model.addCost(k, config_cost)
        model.addCost(k, v_cost)
        model.addCost(k, u_cost)

        model.addContactConstraint(k, "FL_foot", CONTACT_MU)
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(k, "FR_foot", CONTACT_MU)
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(k, "RL_foot", CONTACT_MU)
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(k, "RR_foot", CONTACT_MU)
        model.addConstraints(k, rr_foot_vel_bound)

    model.addCost(mpc_H, v_cost)

    model.finalize(init_states=[init_state])

    ssqp_params = ssqpy.solvers.ssqpParams()

    if inverse_mode:
        ssqp_params.tolerance = 1e-3
        ssqp_params.admmParams.init_rho = 5.0
        ssqp_params.admmParams.adaptive_rho_tolerance = 5.0
        ssqp_params.admmParams.rho_update_iter_freq = 25
        ssqp_params.admmParams.abs_tolerance = 1e-3
        ssqp_params.admmParams.rel_tolerance = 1e-3
        ssqp_params.admmParams.warm_start = False
    else:
        ssqp_params.tolerance = 1e-2
        ssqp_params.admmParams.abs_tolerance = 1e-3
        ssqp_params.admmParams.rel_tolerance = 1e-3
        ssqp_params.admmParams.warm_start = False

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)
    simulator = SSQPYSimulator(model, dt, init_state)

    state_data = [init_state]
    frequency_data = []

    for _ in range(1, args.steps + 1):
        pre = time()

        try:
            control = mpc.step(simulator.state)[1].stage(0)[:nu]
        except RuntimeError:
            control = np.zeros(nu)

        # TODO: clip
        sim_ctrl = np.zeros(nv)
        if inverse_mode:
            torques = model.inverseDynamics(
                simulator.state[:nq],
                simulator.state[nq:],
                model.getFullAccelerationVector(simulator.state, control),
            )
            sim_ctrl[6:] = torques[6:]
        else:
            sim_ctrl[6:] = control[6:]

        print(sim_ctrl)

        elapsed = time() - pre

        simulator.step(sim_ctrl)

        state_data.append(simulator.state)
        frequency_data.append(1 / elapsed)

    return state_data, frequency_data


state_data, _ = run(dynamics_mode=mode_map[args.dynamics_mode])

model = ssqpy.model.Model(
    args.steps,
    0.01,
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    dynamics_mode=ssqpy.model.DynamicsMode.Forward,
)
cost = ssqpy.model.costs.SquaredJointVelocityCost(model.getnv(), V_WGT, True)

for k in range(args.steps + 1):
    model.addCost(k, cost)

model.finalize(init_states=state_data)

solver = ssqpy.solvers.ssqp(
    model, line_search_method=ssqpy.solvers.LineSearchMethod.Merit
)
solver.write("test.json")
