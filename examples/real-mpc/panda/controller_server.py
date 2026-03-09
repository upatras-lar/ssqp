import ssqpy
import numpy as np
import sys
import argparse
import socket

import threading
import time

sys.path.insert(0, "..")
from socket_utils import send, recv

np.set_printoptions(suppress=True)


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
        action="store_true",
        default=False,
        help="Whether to enable torque control",
    )

    return parser.parse_args()


args = parse_args()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Increase receive buffer (SO_RCVBUF) and send buffer (SO_SNDBUF)
recv_buf_size = 256 * 1024 * 1024
send_buf_size = 256 * 1024 * 1024

server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf_size)
server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buf_size)

server.bind((args.ip, args.port))
server.listen(1)

# Shared variable for latest data
latest_data = None
data_lock = threading.Lock()
running = True


# TODO: Improve
def recv_thread(conn):
    global latest_data, running
    while running:
        data = recv(conn)
        if data is None:
            running = False
            break
        with data_lock:
            latest_data = data


try:
    while True:
        print("Waiting for controller device...")
        conn, _ = server.accept()
        print("Controller device connected")

        # Start receiver thread
        threading.Thread(target=recv_thread, args=(conn,), daemon=True).start()

        nq = 9
        nv = 9
        MPC_H = 10
        dt = 0.1

        T_WGT = 4.0
        R_WGT = 2.0
        V_WGT = 0.05
        U_WGT = 1e-3

        Rd = np.eye(3)
        Rd[1, 1] = -1
        Rd[2, 2] = -1

        # Rd = np.array([[0.73061795, 0.68194389, -0.0339109], [0.68206378, -0.73122844, -0.00969386], [-0.03140728, -0.0160469, -0.99937785]])

        ####
        # (0): Determine target position
        target_pos = np.array((0.46, 0.35, -0.05))
        # 0.4634960889816284, -0.008664117194712162, 0.6517822742462158
        ####

        # Tf = np.array([[0.9991192817687988, -0.007720937021076679, -0.0412437878549099, 0.0], [-0.007166385184973478, -0.999882161617279, 0.013576673343777657, 0.0], [-0.041343752294778824, -0.013269146904349327, -0.999056875705719, 0.0], [0.4624573290348053, 0.0001711864460958168, 0.6479776501655579, 1.0]]).T

        # print(Tf)
        # print(Rd)
        # exit(1)

        desired_pose = np.zeros((4, 4))
        desired_pose[:3, :3] = Rd
        desired_pose[:3, 3] = target_pos
        desired_pose[3, 3] = 1

        model = ssqpy.model.Model(
            MPC_H,
            dt,
            urdf_path="../../assets/urdf/panda/panda.urdf",
            dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
        )

        #####
        # (1): Get initial state from real
        data = recv(conn)

        if data is None:
            continue

        init_state = np.zeros(2 * nq)
        init_state[:7] = data[:7]
        init_state[9:16] = data[7:]
        print("Init state:", init_state)
        #####

        start_pos, R = model.forwardKinematics(init_state[:nq], "panda_ee")

        # print(R)
        # print(Rd)

        # # print(start_pos)
        # exit(1)

        constraint_vlines = [-np.inf, 0.1]

        lb = np.array((-np.inf, constraint_vlines[0], -np.inf))
        ub = np.array((np.inf, constraint_vlines[1], np.inf))
        box_constraint = ssqpy.model.constraints.FramePositionBound(
            model, "panda_ee", lb, ub, nq, nv
        )

        v_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
        u_cost = ssqpy.model.costs.SquaredControlCost(nq, U_WGT)
        pose_error = ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, desired_pose, "panda_ee", nq, nq, T_WGT, R_WGT
        )
        qq = init_state[:nq]
        c_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(qq, nq, nv, 1.0, False)

        for k in range(MPC_H):
            model.addCost(k, v_cost)
            model.addCost(k, u_cost)
            model.addCost(k, pose_error)
            # model.addCost(k, c_cost)

            model.addConstraints(k, box_constraint)

        model.addCost(MPC_H, pose_error)
        model.addCost(MPC_H, v_cost)
        # model.addCost(MPC_H, c_cost)
        model.addConstraints(MPC_H, box_constraint)

        ssqp_params = ssqpy.solvers.ssqpParams()
        ssqp_params.tolerance = 1e-2

        admm_params = ssqpy.solvers.admmParams()
        admm_params.abs_tolerance = 1e-3
        admm_params.rel_tolerance = 1e-3
        admm_params.init_rho = 5.0
        admm_params.warm_start = False

        ssqp_params.admmParams = admm_params

        mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=125)

        while running:
            t0 = time.time()
            # frame_start = time()

            ####
            # # (2): Get state from real
            # data = recv(conn)
            # if data is None:
            #     break
            # ####
            # Get the latest data
            with data_lock:
                data = latest_data
                latest_data = None  # optional: clear after reading

            if data is None:
                time.sleep(0.001)  # avoid busy wait
                continue
            state = np.zeros(nq + nv)
            state[:7] = data[:7]
            state[9:16] = data[7:]

            # pre = time()
            if args.torque:
                controls = model.inverseDynamics(
                    state[:nq], state[nq:], mpc.step(state)
                )
            else:
                controls = mpc.step(state, True)
            # elapsed += time() - pre
            # controls = data[:7]
            print("Control:", controls)

            ####
            # (3): Send torques to real
            send(conn, controls)
            ###
            Dt = time.time() - t0
            print("freq:", 1 / Dt)
except KeyboardInterrupt:
    running = False
    server.close()
