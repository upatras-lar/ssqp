import pyCandle
import socket
import numpy as np
import time
import argparse

from real_mpc.socket_utils import send, recv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hardware control script using pyCandle"
    )

    parser.add_argument(
        "-b",
        "--pendubot",
        action=argparse.BooleanOptionalAction,
        help="Whether to actuate only shoulder joint (pendubot)",
    )

    parser.add_argument(
        "-a",
        "--acrobot",
        action=argparse.BooleanOptionalAction,
        help="Whether to actuate only elbow joint (acrobot)",
    )

    parser.add_argument(
        "-t",
        "--torque-limit",
        type=float,
        default=0.15,
        help="Maximum allowed torque",
    )

    return parser.parse_args()


args = parse_args()


SOCK_PATH = "/tmp/mpc_socket"

# Connect to MPC server
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(SOCK_PATH)

# Initialize hardware
# candle2 = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
ids = candle.ping()

for id in ids:
    candle.addMd80(id)

for md in candle.md80s:
    candle.controlMd80SetEncoderZero(md)
    candle.controlMd80Mode(md, pyCandle.IMPEDANCE)
    md.setImpedanceControllerParams(0.0, 0.0)
    candle.controlMd80Enable(md, True)

candle.begin()

dt = 0.01

steps = 0
total_steps = int(1 / dt)

start = time.time()
while True:
    pos = np.array(
        (
            candle.md80s[1].getPosition(),
            candle.md80s[0].getPosition(),
        )
    )

    vel = np.array(
        (
            candle.md80s[1].getVelocity(),
            candle.md80s[0].getVelocity(),
        )
    )

    torque = np.array(
        (
            candle.md80s[1].getTorque(),
            candle.md80s[0].getTorque(),
        )
    )

    state = np.hstack((pos, vel, torque))

    send(sock, state)
    tau = np.clip(recv(sock), -args.torque_limit, args.torque_limit)

    if not args.pendubot:
        candle.md80s[0].setTargetTorque(tau[1])

    if not args.acrobot:
        candle.md80s[1].setTargetTorque(tau[0])

    steps += 1

    if steps == total_steps:
        print(f"\rFrequency: {1 / dt / (time.time() - start):.2f}Hz", end="")
        start = time.time()
        steps = 0
