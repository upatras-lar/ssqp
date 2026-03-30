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

    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        help="A goal 3D point to visualize",
    )

    parser.add_argument(
        "--box-region",
        nargs=6,
        type=float,
        help="A feasible region box (xyzxyz, 3 lower followed by 3 upper)",
    )

    parser.add_argument(
        "--shaded-model-base",
        nargs=7,
        type=float,
        help="Position (xyz) and quaternion (wxyz) for a shaded model of the "
        "provided robot; if not passed, the position (0, 0, 0) and quaternion"
        "(1 0 0 0) are used",
    )

    parser.add_argument(
        "--shaded-model-configuration",
        nargs="+",
        type=float,
        help="Configuration for the shaded model of the provided robot; if no "
        "configuration is passed, the robot's first actual configuration is used",
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

if args.goal:
    server.scene.add_point_cloud(
        name="/goal",
        points=np.array((args.goal,)),
        colors=(1.0, 0.0, 0.0),
        point_size=0.02,
    )

if args.box_region:
    lower = args.box_region[:3]
    upper = args.box_region[3:]

    # Create the 8 corners of the box
    corners = np.array(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ]
    )

    # Define the 12 edges of the cuboid
    edges = np.array(
        [
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],  # bottom
            [corners[4], corners[5]],
            [corners[5], corners[6]],
            [corners[6], corners[7]],
            [corners[7], corners[4]],  # top
            [corners[0], corners[4]],
            [corners[1], corners[5]],
            [corners[2], corners[6]],
            [corners[3], corners[7]],  # verticals
        ]
    )

    server.scene.add_line_segments(
        name="/feasible_region",
        points=edges,
        colors=(0.0, 1.0, 0.0),  # green
        line_width=2.0,
    )

shaded_urdf = None
updated_cfg = False
if args.shaded_model_base or args.shaded_model_configuration:
    if args.shaded_model_base is None:
        args.shaded_model_base = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    shaded_base = server.scene.add_frame("/shaded-model", show_axes=False)

    shaded_urdf = viser.extras.ViserUrdf(
        server,
        urdf_or_path=Path(args.urdf_path),
        root_node_name="/shaded-model",
        mesh_color_override=(0.0, 0.0, 0.5, 0.3),
    )

    shaded_base.position = np.array(args.shaded_model_base[:3])
    shaded_base.wxyz = np.array(args.shaded_model_base[3:])

    if args.shaded_model_configuration:
        actuated_joints = shaded_urdf.get_actuated_joint_names()
        if len(args.shaded_model_configuration) != len(actuated_joints):
            print(
                f"Shaded model configuration size ({len(args.shaded_model_configuration)}) does not match the required ({len(actuated_joints)})"
            )
            exit(12)

        shaded_urdf.update_cfg(args.shaded_model_configuration)
        updated_cfg = True


# Move client camera to initial position
@server.on_client_connect
def _(client):
    client.camera.position = (0.8, 1, 0.8)
    client.camera.look_at = (0, 0, 0)


JOINT_TYPE_NVALS = {
    "revolute": 1,
    "continuous": 2,
    "prismatic": 1,
    "spherical": 4,
    "spherical-euler": 3,
}

data_joints = data["properties"]["state_joints"]

vis_to_data_indices = []
joints = urdf.get_actuated_joint_names()
for joint in joints:
    if joint not in data_joints:
        print(f"Joint {joint} expected, but not found in data")
        exit(11)

    vis_to_data_indices.append(data_joints.index(joint))

joint_types = data["properties"]["joint_types"]

if len(joint_types) != len(data_joints):
    print(
        f"Length of joint types array ({len(joint_types)}) does not match that of state joints ({len(data_joints)})"
    )
    exit(13)

joint_to_state_indices = {}

state_offset = 0
for joint, joint_type in zip(data_joints, joint_types):
    joint_to_state_indices[joint] = state_offset
    state_offset += JOINT_TYPE_NVALS[joint_type]

expected_config_size = state_offset


def process_state(state):
    if len(state) <= expected_config_size:
        print(
            f"Stage {k} state data size ({len(state)}) smaller than expected configuration size ({expected_config_size})"
        )
        exit(10)

    angles = []

    for index in vis_to_data_indices:
        joint = data_joints[index]
        joint_type = joint_types[index]

        state_index = joint_to_state_indices[joint]

        if joint_type == "revolute" or joint_type == "prismatic":
            angles.append(state[state_index])
        elif joint_type == "continuous":
            angles.append(
                np.arctan2(state[state_index + 1], state[state_index])
            )
        # TODO: figure out spherical (cannot via URDF)
        elif joint_type == "spherical":
            angles.append(0.0)
        elif joint_type == "spherical-euler":
            angles.append(0.0)

    return np.array(angles)


# Main loop
while True:
    # Animate each stage
    for k in range(data["properties"]["horizon"]):
        configuration = process_state(data["stage_data"][k]["state"])

        urdf.update_cfg(configuration)

        if shaded_urdf and not updated_cfg:
            shaded_urdf.update_cfg(configuration)
            updated_cfg = True

        # Move floating base if free flyer
        if data["properties"]["floating_base"]:
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
