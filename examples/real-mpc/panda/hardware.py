import socket
import sys
import numpy as np
import argparse
from pylibfranka import RealtimeConfig, Robot, JointPositions, Torques

sys.path.insert(0, "..")
from socket_utils import send, recv


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
        type=argparse.BooleanOptionalAction,
        help="Whether to enable torque control",
    )

    return parser.parse_args()


args = parse_args()

# Connect to MPC server
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect((args.ip, args.port))

# Initialize hardware
robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)

lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

robot.set_collision_behavior(
    lower_torque_thresholds,
    upper_torque_thresholds,
    lower_force_thresholds,
    upper_force_thresholds,
)

init_robot_state = robot.read_once()
init_solver_state = np.hstack((init_robot_state.q, init_robot_state.dq))

send(sock, init_solver_state)

if args.torque:
    active_control = robot.start_torque_control()
else:
    active_control = robot.start_joint_position_control()

while True:
    robot_state, duration = active_control.readOnce()

    solver_state = np.hstack((robot_state.q, robot_state.dq))

    send(sock, solver_state)

    controls = recv(sock)

    if args.torque:
        command = Torques(controls.tolist())
    else:
        command = JointPositions(controls.tolist())

    active_control.writeOnce(command)
