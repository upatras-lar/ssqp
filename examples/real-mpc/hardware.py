import pyCandle
import socket
import numpy as np
import time
from socket_utils import send, recv

SOCK_PATH = "/tmp/mpc_socket"

# -------- Connect to MPC --------
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(SOCK_PATH)

# -------- Hardware init --------
candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
ids = candle.ping()

for id in ids:
    candle.addMd80(id)

candle.controlMd80SetEncoderZero(ids[0])
candle.controlMd80Mode(ids[0], pyCandle.RAW_TORQUE)
candle.controlMd80Enable(ids[0], True)

candle.controlMd80SetEncoderZero(ids[1])
candle.controlMd80Mode(ids[1], pyCandle.RAW_TORQUE)
candle.controlMd80Enable(ids[1], True)
candle.begin()

dt = 0.01

MAX_TORQUE = 0.15

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

    candle.md80s[0].setTargetTorque(
        np.sign(tau[1]) * min(abs(tau[1]), MAX_TORQUE)
    )
    candle.md80s[1].setTargetTorque(
        np.sign(tau[0]) * min(abs(tau[0]), MAX_TORQUE)
    )
    print(tau)

    time.sleep(dt)
