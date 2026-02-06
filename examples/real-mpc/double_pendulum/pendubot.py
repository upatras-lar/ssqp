import socket
import os
import numpy as np
from math import pi
import ssqpy
import sys

sys.path.insert(0, "..")
from socket_utils import send, recv

SOCK_PATH = "/tmp/mpc_socket"

if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)

# Build MPC
nq = 2
nv = 2
dt = 0.01
MPC_H = 45

tip_cost_wgt = 10.0

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, 1e-6)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, 1e-6)
tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, np.array((0.0, 0.0, 0.1)), "tip", nq, nv, tip_cost_wgt
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, tip_cost)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, tip_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.adaptive_rho_tolerance = 10
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

mpc = ssqpy.solvers.MPC(
    model, ssqp_params, admm_params, sqp_iters=4, admm_iters=200
)

# Socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

print("MPC ready, waiting for connection…")

conn, _ = server.accept()
print("Hardware connected")

while True:
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

    send(conn, tau)
