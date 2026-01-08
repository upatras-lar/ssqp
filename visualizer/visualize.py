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
        "-g",
        "--no-ground",
        action=argparse.BooleanOptionalAction,
        help="Whether to skip drawing the ground",
    )

    return parser.parse_args()


args = parse_args()

# Read robot trajectory
with open(args.input, "r") as f:
    data = json.loads(f.read())

# Create the server object
server = viser.ViserServer(port=8080)

# Add box to represent the ground
if not args.no_ground:
    server.scene.add_box(
        "/ground",
        color=(225, 225, 205),
        dimensions=(10, 10, 0.001),
    )

# Create a parent frame for the robot
robot_base = server.scene.add_frame("/robot", show_axes=False)

# Load given URDF
urdf = viser.extras.ViserUrdf(
    server,
    urdf_or_path=Path(args.urdf_path),
    root_node_name="/robot",
)

urdf.show_frames = True


# Move client camera to initial position
@server.on_client_connect
def _(client):
    client.camera.position = (0.8, 1, 0.8)
    client.camera.look_at = (0, 0, 0)


# Main loop
while True:
    # Animate each stage
    for k in range(data["properties"]["horizon"]):
        state = np.array(data["stage_data"][k]["state"])
        data_joints = data["properties"]["stages"][k]["state_joints"]

        joints = urdf.get_actuated_joint_names()
        if len(state) < len(joints):
            print(
                f"State data size ({len(state)}) smaller than required ({len(joints)}) (stage {k})"
            )
            exit(10)
        if len(data_joints) != len(state):
            print(f"Invalid state for listed joints (stage {k})")
            exit(11)

        indices = []
        for joint in joints:
            if joint not in data_joints:
                print(
                    f"Joint {joint} expected, but not found in stage {k} data"
                )
                exit(12)

            indices.append(data_joints.index(joint))

        joint_positions = [state[i] for i in indices]
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
