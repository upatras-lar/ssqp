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
        description="Trajectory visualizer via Viser"
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

# -------- Connect to MPC --------
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(SOCK_PATH)

# -------- Hardware init --------
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
    tau = recv(sock)

    if not args.pendubot:
        candle.md80s[0].setTargetTorque(
            np.sign(tau[1]) * min(abs(tau[1]), MAX_TORQUE)
        )
    candle.md80s[1].setTargetTorque(
        np.sign(tau[0]) * min(abs(tau[0]), MAX_TORQUE)
    )

    steps += 1

    if steps == total_steps:
        print(f"\rFrequency: {1 / dt / (time.time() - start):.2f}Hz", end="")
        start = time.time()
        steps = 0
