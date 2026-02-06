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

# -------- Build MPC ONCE --------
nq = 2
nv = 2
dt = 0.01
MPC_H = 40

tip_cost_wgt = 25.0
link_cost_wgt = 0.0

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(
    nq, np.array([1e-8, 1e-6])
)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, 1e-1)
link2_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, np.array((0.0, 0.0, 0.05)), "link2", nq, nv, link_cost_wgt
)
tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, np.array((0.0, 0.0, 0.1)), "tip", nq, nv, tip_cost_wgt
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    # model.addCost(k, link2_cost)
    model.addCost(k, tip_cost)

config_cost = ssqpy.model.costs.SquaredConfigurationDifferenceCost(
    np.array((pi, 0.0)),
    nq,
    nv,
    np.array((1e-5, 0.0)),
    False,
)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, link2_cost)
model.addCost(MPC_H, tip_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.adaptive_rho_tolerance = 10
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = True

mpc = ssqpy.solvers.MPC(
    model, ssqp_params, admm_params, sqp_iters=4, admm_iters=75
)

# -------- Socket server --------
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

    print(u)

    send(conn, u)
