import socket
import os
import numpy as np
import ssqpy
import sys

sys.path.insert(0, "..")
from socket_utils import send, recv

INVERSE_MODE = False

SOCK_PATH = "/tmp/mpc_socket"

if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)

# Build MPC
dt = 0.01
MPC_H = 50

V_WGT = 1e-3
U_WGT = 3e-5 if INVERSE_MODE else 1.0
TIP_WGT = 3.0

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/double_pendulum.urdf",
    dynamics_mode=(
        ssqpy.model.DynamicsMode.Inverse
        if INVERSE_MODE
        else ssqpy.model.DynamicsMode.Forward
    ),
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)
tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, "tip", np.array((0.0, 0.0, 0.1)), nq, nv, TIP_WGT
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

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

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

    if INVERSE_MODE:
        tau = model.inverseDynamics(pos, vel, u)
    else:
        tau = u

    send(conn, tau)
