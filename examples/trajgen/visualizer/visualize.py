import viser
import viser.extras
from pathlib import Path
import numpy as np
import argparse
import json
from time import sleep


def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array(
        [
            [
                2 * (q0**2 + q1**2) - 1,
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                2 * (q0**2 + q2**2) - 1,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                2 * (q0**2 + q3**2) - 1,
            ],
        ]
    )


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

    parser.add_argument(
        "-f",
        "--forces",
        action=argparse.BooleanOptionalAction,
        help="Whether to visualize forces",
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
        dimensions=(20, 20, 0.0001),
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

if args.forces:
    forces = server.scene.add_line_segments(
        "/arrow_shaft",
        points=np.array([[np.zeros(3), np.zeros(3)]]),
        colors=np.array([225, 165, 60]),
        line_width=4.0,
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
        state = np.array(data["stage_data"][k]["state"])
        data_joints = data["properties"]["stages"][k]["state_joints"]

        joints = urdf.get_actuated_joint_names()
        if len(state) < len(joints):
            print(
                f"State data size ({len(state)}) smaller than required ({len(joints)}) (stage {k})"
            )
            exit(10)

        indices = []
        for joint in joints:
            if joint not in data_joints:
                print(
                    f"Joint {joint} expected, but not found in stage {k} data"
                )
                exit(11)

            indices.append(data_joints.index(joint))

        joint_positions = [state[i] for i in indices]
        urdf.update_cfg(joint_positions)

        # Move floating base if free flyer
        if data["properties"]["stages"][k]["floating_base"]:
            robot_base.position = np.array(data["stage_data"][k]["position"])
            q = data["stage_data"][k]["orientation"].copy()
            q.insert(0, q.pop())
            robot_base.wxyz = np.array(q)

        if args.forces:
            if data["properties"]["stages"][k]["nforces"] == 0:
                forces.visible = False
            else:
                R_base = quaternion_to_rotation_matrix(robot_base.wxyz)

                all_forces = []
                for force_idx, frame in enumerate(
                    data["properties"]["stages"][k]["ordered_force_frames"]
                ):
                    frame_T = urdf._urdf.get_transform(frame)
                    frame_R = frame_T[:3, :3]
                    frame_t = frame_T[:3, 3]

                    f_local = np.array(
                        data["stage_data"][k]["forces"][
                            3 * force_idx : 3 * force_idx + 3
                        ]
                    )

                    p0 = R_base @ frame_t + robot_base.position
                    p1 = p0 + 0.005 * R_base @ frame_R @ f_local

                    all_forces.append([p0, p1])

                forces.points = np.array(all_forces)

                forces.visible = True

        sleep(data["properties"]["dt"] * args.delay_factor)

    if not args.loop:
        break
