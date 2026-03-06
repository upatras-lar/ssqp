import ssqpy
import numpy as np
import sys
import argparse
import socket
from time import time

sys.path.insert(0, "..")
from socket_utils import send, recv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control script moving the end-effector of the real panda "
        "robot to a certain location"
    )

    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        required=True,
        help="The IP to bind the control server to",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        required=True,
        help="The port to bind the control server to",
    )

    parser.add_argument(
        "-t",
        "--torque",
        type=argparse.BooleanOptionalAction,
        help="Whether to enable torque control",
    )

    return parser.parse_args()


args = parse_args()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((args.ip, args.port))
server.listen(1)

while True:
    print("Waiting for controller device...")
    conn, _ = server.accept()
    print("Controller device connected")

    nq = 9
    nv = 9
    MPC_H = 20
    dt = 0.01

    T_WGT = 4.0
    R_WGT = 3.0
    V_WGT = 1e-1
    U_WGT = 1e-3

    Rd = np.eye(3)
    Rd[1, 1] = -1
    Rd[2, 2] = -1

    ####
    # (0): Determine target position
    target_pos = np.array((0.4, 0.2, 0.4))
    ####

    desired_pose = np.zeros((4, 4))
    desired_pose[:3, :3] = Rd
    desired_pose[:3, 3] = target_pos
    desired_pose[3, 3] = 1

    model = ssqpy.model.Model(
        MPC_H,
        dt,
        urdf_path="../assets/urdf/panda/panda.urdf",
        dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
    )

    #####
    # (1): Get initial state from real
    data = recv(conn)

    if not data:
        continue

    init_state = np.zeros(2 * nq)
    init_state[:nq] = data
    #####

    start_pos = model.forwardKinematics(init_state[:nq], "panda_ee")[0]

    v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
    u_cost = ssqpy.model.costs.SquaredControlCost(nq, U_WGT)
    pose_error = ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, desired_pose, "panda_ee", nq, nq, T_WGT, R_WGT
    )

    for k in range(MPC_H):
        model.addCost(k, v_cost)
        model.addCost(k, u_cost)
        model.addCost(k, pose_error)

    model.addCost(MPC_H, pose_error)
    model.addCost(MPC_H, v_cost)

    ssqp_params = ssqpy.solvers.ssqpParams()
    ssqp_params.tolerance = 1e-2

    admm_params = ssqpy.solvers.admmParams()
    admm_params.abs_tolerance = 1e-3
    admm_params.rel_tolerance = 1e-3
    admm_params.warm_start = False

    ssqp_params.admmParams = admm_params

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=125)

    while True:
        # frame_start = time()

        ####
        # (2): Get state from real
        state = recv(conn)
        if not state:
            break
        ####

        # pre = time()
        if args.t:
            controls = model.inverseDynamics(
                state[:nq], state[nq:], mpc.step(state)
            )
        else:
            controls = mpc.step(state, True)
        # elapsed += time() - pre

        ####
        # (3): Send torques to real
        send(conn, controls)
        ###
