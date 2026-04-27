import socket
import os
import sys
import numpy as np
import ssqpy
from time import time

sys.path.insert(0, "..")
from socket_utils import send, recv

SOCK_PATH = "/tmp/mpc_socket"

if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)

# Build MPC
ssqpy.setSilentMode()

dt = 0.01
MPC_H = 45

V_WGT = 1.0
U_WGT = 1e-1
C_WGT = 16.0

model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/pendubot_cont.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Forward,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

DESIRED_CONFIG = np.array((-1.0, 0.0, 1.0, 0.0))

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, DESIRED_CONFIG, nq, nv, C_WGT, False
)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, config_cost)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.adaptive_rho_tolerance = 10
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=8, qp_iters=200)

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

i = 0
while True:
    state = recv(conn)
    if state is None:
        break

    # pos = state[:nq]
    # vel = state[nq:]
    pos = state[:2]
    pos = np.array(
        (np.cos(pos[0]), np.sin(pos[0]), np.cos(pos[1]), np.sin(pos[1]))
    )
    vel = state[2:]

    state = np.hstack((pos, vel))

    start = time()
    try:
        u = mpc.step(state)[1].stage(0)
    except RuntimeError:
        u = np.zeros(nv)

    # tau = model.inverseDynamics(pos, vel, u)
    # tau[1] = 0.0
    tau = u

    elapsed = time() - start

    send(conn, tau)

conn.close()
