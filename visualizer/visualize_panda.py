
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from swift import Swift
from sys import argv

file_state_size = 9
state_size = 7
robot = rtb.models.Panda()

with open(argv[1], "r") as f:
    data = f.readlines()

state_data = [[float(pos) for pos in line.split()] for line in data[1 : 1 + file_state_size]]
#velocity_data = [[float(pos) for pos in line.split()] for line in data[1 + file_state_size : 1 + 2 * file_state_size]]

trajectory = np.array(state_data).transpose()
#velocity = np.array(velocity_data).transpose()

# Set initial position
robot.q = trajectory[0][:state_size]
robot.grippers[0].q = trajectory[0][state_size:file_state_size]

# Make and instance of the Swift simulator and open it
env = Swift()
env.launch(realtime=True)

# Add the robot to the simulator
env.add(robot)

# Simulate the robot while it has not arrived at the goal
for q in trajectory:
    # Set the Panda's joint velocities
    robot.q = q[:state_size]
    robot.grippers[0].q = q[state_size:file_state_size]
    
    # Step the simulator by 10 milliseconds
    env.step(0.01)