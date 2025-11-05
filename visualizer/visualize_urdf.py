
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from sys import argv
from os import getcwd

if len(argv) == 2:
    # Assume data file name is same as robot name
    data_file = argv[1]
    robot_name = argv[1][argv[1].rfind("/") + 1:argv[1].rfind(".")]
elif len(argv) == 3:
    robot_name = argv[1]
    data_file = argv[2]

robot = rtb.Robot.URDF(getcwd() + f"/../examples/urdf/{robot_name}.urdf")

with open(data_file, "r") as f:
    data = f.readlines()

file_state_size = int(data[0].split()[0]) // 2

state_data = [[float(pos) for pos in line.split()] for line in data[1 : 1 + file_state_size]]

trajectory = np.array(state_data).transpose()

robot.plot(trajectory, backend='pyplot', block=True, dt=0.01)
