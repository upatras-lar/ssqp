import viser
import viser.extras
from pathlib import Path
import numpy as np
import argparse
import json
from time import sleep


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Trajectory visualizer via Viser"
    )
    parser.add_argument(
        "-u",
        "--urdf-path",
        type=str,
        required=True,
        help="Path to URDF robot description",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input trajectory file (.json)",
    )

    parser.add_argument(
        "-d",
        "--delay-factor",
        type=float,
        default=1.0,
        help="Factor to delay each time step by during visualization",
    )

    parser.add_argument(
        "-l",
        "--loop",
        action=argparse.BooleanOptionalAction,
        help="Whether to loop solution after ending",
    )

    parser.add_argument(
        "-s",
        "--state_size",
        type=int,
        default=-1,
        help="Number of (first) elements to use for state",
    )

    return parser.parse_args()


args = parse_args()

# Read robot trajectory
with open(args.input, "r") as f:
    data = json.loads(f.read())

# Create the server object
server = viser.ViserServer(port=8080)

# Add grid to represent the ground
grid = server.scene.add_grid(
    "/ground",
    position=(0, 0, 0),
    cell_color=(0.7, 0.7, 0.7),
    cell_size=0.5,
)

# Create a parent frame for the robot
robot_base = server.scene.add_frame("/robot", show_axes=False)

# Load given URDF
urdf = viser.extras.ViserUrdf(
    server,
    urdf_or_path=Path(args.urdf_path),
    root_node_name="/robot",
)


# Move client camera to initial position
@server.on_client_connect
def _(client):
    client.camera.position = (0.8, 1, 0.8)
    client.camera.look_at = (0, 0, 0)


# Main loop
while True:
    # Animate each stage
    for k in range(data["properties"]["horizon"]):
        joint_positions = np.array(data["stage_data"][k]["state"])

        # Slice stage if custom size was given
        if args.state_size > 0:
            joint_positions = joint_positions[: args.state_size]

        urdf.update_cfg(joint_positions)

        # Move floating base if free flyer
        if data["properties"]["stages"][k]["floating_base"]:
            robot_base.position = np.array(data["stage_data"][k]["position"])
            q = data["stage_data"][k]["orientation"].copy()
            q.insert(0, q.pop())
            robot_base.wxyz = np.array(q)

        sleep(data["properties"]["dt"] * args.delay_factor)

    if not args.loop:
        break
