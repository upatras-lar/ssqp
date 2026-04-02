import socket
import sys
import numpy as np
import argparse
from pylibfranka import ControllerMode, RealtimeConfig, Robot, JointPositions, Torques
import time

import fcntl, termios, array

buf_queued = array.array("I", [0])

sys.path.insert(0, "..")
from socket_utils import send, recv

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hardware control script using pylibfranka"
    )

    parser.add_argument(
        "-r",
        "--robot-ip",
        type=str,
        required=True,
        help="The robot IP",
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

    parser.add_argument(
        "-a",
        "--interpolation-alpha",
        type=float,
        default=0.99,
        help="The interpolation alpha parameter",
    )

    return parser.parse_args()


args = parse_args()

# Connect to MPC server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Increase receive buffer (SO_RCVBUF) and send buffer (SO_SNDBUF)
recv_buf_size = 1024 * 1024 * 1024
send_buf_size = 1024 * 1024 * 1024

sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf_size)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buf_size)

sock.connect((args.ip, args.port))

sock.setblocking(False)

# Initialize hardware
robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)

# Set default behavior
robot.set_collision_behavior(
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
)

# Create a model instance from robot
model = robot.load_model()

prev_controls = None
alpha = 0.99

if args.torque:
    active_control = robot.start_torque_control()
else:
    active_control = robot.start_joint_position_control(
        ControllerMode.CartesianImpedance
    )

target_control = None
current_control = None

# Compliance parameters
joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]

frame_start = time.time()

while True:
    t0 = time.time()
    robot_state, _ = active_control.readOnce()

    solver_state = np.hstack((robot_state.q, robot_state.dq))

    coriolis = np.array(model.coriolis(robot_state))
    q = np.array(robot_state.q)
    dq = np.array(robot_state.dq)

    # fcntl.ioctl(sock, termios.TIOCOUTQ, buf_queued)
    # print(f"Bytes in send buffer: {buf_queued[0]}")
    try:
        send(sock, solver_state)
    except BlockingIOError:
        pass

    if target_control is None:
        target_control = q.copy()
        current_control = target_control.copy()
    try:
        target_control = recv(sock)[:7]
    except:
        pass

    current_control = alpha * current_control + (1.0 - alpha) * target_control
    if args.torque:
        # Compute error to desired equilibrium joint configuration
        position_error = q - current_control

        # Compute joint-space impedance control
        tau_task = np.zeros(7)
        for i in range(7):
            tau_task[i] = (
                -joint_stiffness[i] * position_error[i] - joint_damping[i] * dq[i]
            )

        # Add coriolis compensation
        tau_d = tau_task + coriolis
        command = Torques(tau_d.tolist())
        command.motion_finished = False
    else:
        command = JointPositions(current_control.tolist())

    active_control.writeOnce(command)
    Dt = time.time() - t0
    time.sleep(max(0.0008 - Dt, 0.0))

    if time.time() - frame_start > 1.0:
        print(f"\rFrequency: {1/Dt:.2f}Hz\t", end="")
        frame_start = time.time()
