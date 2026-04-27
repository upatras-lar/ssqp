import pyCandle
import socket
import numpy as np
import time
import argparse
import sys

sys.path.insert(0, "..")
from socket_utils import send, recv


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

if not args.pendubot:
    candle.controlMd80SetEncoderZero(ids[0])
    candle.controlMd80Mode(ids[0], pyCandle.RAW_TORQUE)
    candle.controlMd80Enable(ids[0], True)

candle.controlMd80SetEncoderZero(ids[1])
candle.controlMd80Mode(ids[1], pyCandle.RAW_TORQUE)
candle.controlMd80Enable(ids[1], True)
candle.begin()

dt = 0.01

MAX_TORQUE = args.torque_limit

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

    state = np.hstack((pos, vel))

    send(sock, state)
    tau = np.clip(recv(sock), -MAX_TORQUE, MAX_TORQUE)

    if not args.pendubot:
        candle.md80s[0].setTargetTorque(tau[1])
    candle.md80s[1].setTargetTorque(tau[0])

    steps += 1

    if steps == total_steps:
        print(f"\rFrequency: {1 / dt / (time.time() - start):.2f}Hz", end="")
        start = time.time()
        steps = 0
