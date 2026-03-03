import ssqpy
import numpy as np
from math import pi
from time import time, sleep

from simulators.simulator import SSQPYSimulator

nq = 2
nv = 2
MPC_H = 45
dt = 0.01

V_WGT = 1e-3
U_WGT = 1e-3
TIP_WGT = 5.0

REAL_TIME = True
NOISE = False
INVERSE_MODE = False


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
    np.array((1.0, 0.0)),
    False,
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, tip_cost)

model.addCost(MPC_H, tip_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.adaptive_rho_tolerance = 20
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=3, qp_iters=100)

simulator = SSQPYSimulator(model, dt, np.zeros(4))

elapsed = 0
steps = 0
total_steps = int(1 / dt)

reg_fails = 0

np.set_printoptions(suppress=True)

while True:
    frame_start = time()

    pre = time()
    try:
        control = mpc.step(simulator.state)
    except RuntimeError:
        reg_fails += 1
        control = np.zeros(nv)

    sim_ctrl = np.zeros(2)
    if INVERSE_MODE:
        torques = model.inverseDynamics(
            simulator.state[:2], simulator.state[2:], control
        )
        sim_ctrl[0] = torques[0]
    else:
        sim_ctrl[0] = control[0]

    if NOISE:
        sim_ctrl[0] += (
            np.random.uniform(0, 0.1) if np.random.uniform(0, 10) > 8 else 0.0
        )

    elapsed += time() - pre
    steps += 1

    if steps == total_steps:
        print(
            f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz, regularization fails: {reg_fails}\t",
            end="",
        )
        elapsed = 0
        steps = 0

    simulator.step(sim_ctrl)

    print(np.arctan2(np.sin(simulator.state), np.cos(simulator.state)))

    if REAL_TIME:
        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)
