import ssqpy
import numpy as np
import sys
import argparse
import socket

import threading
import time

import models

sys.path.insert(0, "..")
from socket_utils import send, recv

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control script moving the end-effector of the real panda "
        "robot to a certain location"
    )

    subparsers = parser.add_subparsers(
        dest="model", help="The model to execute"
    )

    parser_reset = subparsers.add_parser(
        "reset", help="Resets the Panda's configuration to the initial"
    )

    parser_target = subparsers.add_parser(
        "target", help="Moves the Panda's end-effector to a target position"
    )
    parser_target.add_argument(
        "--pos",
        nargs=3,
        type=float,
        default=(0.46, 0.35, 0.0),
        help="The target end-effector position",
    )

    parser_circle = subparsers.add_parser(
        "circle",
        help="Performs a circle in the yz-plane with the Panda's end-effector",
    )
    parser_circle.add_argument(
        "--constrain",
        type=bool,
        default=False,
        help="Whether to add the inscribed square constraints",
    )

    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        default="localhost",
        help="The IP to bind the control server to",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default="11880",
        help="The port to bind the control server to",
    )

    parser.add_argument(
        "-t",
        "--torque",
        action="store_true",
        default=False,
        help="Whether to enable torque control",
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

if args.model is None:
    args.model = "reset"

print(f"Setting model as: {args.model}")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Increase receive buffer (SO_RCVBUF) and send buffer (SO_SNDBUF)
recv_buf_size = 256 * 1024 * 1024
send_buf_size = 256 * 1024 * 1024

server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf_size)
server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buf_size)

server.bind((args.ip, args.port))
print(f"Started server on {args.ip}:{args.port}")
server.listen(1)

# Shared variable for latest data
latest_data = None
data_lock = threading.Lock()
running = True

mode_map = {
    "inverse": ssqpy.model.DynamicsMode.Inverse,
    "forward": ssqpy.model.DynamicsMode.Forward,
}

get_model_func = None
model_func_params = []
if args.model == "target":
    get_model_func = models.get_target_model
    model_func_params.append(args.pos)
elif args.model == "reset":
    get_model_func = models.get_reset_model
elif args.model == "circle":
    get_model_func = models.get_circle_model
    model_func_params.append(args.constrain)


# TODO: Improve
def recv_thread(conn):
    global latest_data, running
    while running:
        try:
            data = recv(conn)
        except:
            continue
        if data is None:
            running = False
            break
        with data_lock:
            latest_data = data


ssqpy.setSilentMode()

try:
    while True:
        print("Waiting for hardware controller...")
        conn, _ = server.accept()
        print("Hardware controller connected")

        # Start receiver thread
        running = True
        threading.Thread(target=recv_thread, args=(conn,), daemon=True).start()

        model, callback = get_model_func(
            mode_map[args.dynamics_mode], *model_func_params
        )

        nq = model.getnq()
        nv = model.getnv()

        ssqp_params = ssqpy.solvers.ssqpParams()
        ssqp_params.tolerance = 1e-2

        admm_params = ssqpy.solvers.admmParams()
        admm_params.abs_tolerance = 1e-3
        admm_params.rel_tolerance = 1e-3
        admm_params.init_rho = 0.1
        admm_params.warm_start = False

        ssqp_params.admmParams = admm_params

        mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=125)

        stage = 0

        while running:
            # Get the latest data
            with data_lock:
                data = latest_data
                latest_data = None

            if data is None:
                time.sleep(0.001)
                continue

            state = np.zeros(nq + nv)
            state[:7] = data[:7]
            state[9:16] = data[7:]

            if args.torque:
                if args.dynamics_mode == "inverse":
                    controls = model.inverseDynamics(
                        state[:nq],
                        state[nq:],
                        model.getFullAccelerationVector(
                            state, mpc.step(state)[1].stage(0)
                        ),
                    )
                elif args.dynamics_mode == "forward":
                    controls = mpc.step(state)[1].stage(0)
            else:
                controls = mpc.step(state)[0].stage(0)

            try:
                send(conn, controls)
                stage += 1
                callback(stage)

            except BlockingIOError:
                pass
except KeyboardInterrupt:
    running = False
    server.close()
