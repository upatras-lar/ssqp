import viser
import viser.extras
from pathlib import Path
import numpy as np
import argparse
import json
from time import sleep, time


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
        "--circle",
        nargs=4,
        type=float,
        help="A circle to visualize, on the yz-plane (cx cy cz radius)",
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

TERRAIN_MARGIN = 1.0


def heightmap_heights(terrain, x, y):
    # Evaluates the bicubic Catmull-Rom surface of a heightmap terrain
    # Coordinates outside the grid are clamped to its edge
    heights = np.array(terrain["heights"])
    res = terrain["resolution"]
    nx, ny = heights.shape

    gx = np.clip((x - terrain["x0"]) / res, 0, nx - 1)
    gy = np.clip((y - terrain["y0"]) / res, 0, ny - 1)
    ci = np.minimum(np.floor(gx).astype(int), nx - 2)
    cj = np.minimum(np.floor(gy).astype(int), ny - 2)
    tx = gx - ci
    ty = gy - cj

    def weights(t):
        return 0.5 * np.stack(
            [
                -(t**3) + 2 * t**2 - t,
                3 * t**3 - 5 * t**2 + 2,
                -3 * t**3 + 4 * t**2 + t,
                t**3 - t**2,
            ]
        )

    wx = weights(tx)
    wy = weights(ty)

    z = np.zeros_like(x)
    for a in range(4):
        for b in range(4):
            z += (
                wx[a]
                * wy[b]
                * heights[
                    np.clip(ci - 1 + a, 0, nx - 1),
                    np.clip(cj - 1 + b, 0, ny - 1),
                ]
            )
    return z


def grid_mesh(x, y, z):
    # Triangulates a grid of vertices, with z of shape (len(x), len(y))
    gx, gy = np.meshgrid(x, y, indexing="ij")
    vertices = np.stack([gx, gy, z], axis=-1).reshape(-1, 3)

    idx = np.arange(len(x) * len(y)).reshape(len(x), len(y))
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])

    return vertices, faces


def terrain_mesh(terrain, x_range, y_range):
    # Builds a mesh of the terrain over the given range
    kind = terrain["type"]
    x0, x1 = x_range
    y0, y1 = y_range

    if kind == "heightmap":
        step = terrain["resolution"] / 4
        x = np.linspace(x0, x1, max(2, int(np.ceil((x1 - x0) / step)) + 1))
        y = np.linspace(y0, y1, max(2, int(np.ceil((y1 - y0) / step)) + 1))
        gx, gy = np.meshgrid(x, y, indexing="ij")
        return grid_mesh(x, y, heightmap_heights(terrain, gx, gy))

    base = terrain.get("base_height", 0.0)
    if kind == "flat":
        return grid_mesh([x0, x1], [y0, y1], np.full((2, 2), base))
    if kind == "slope":
        gx, gy = np.meshgrid([x0, x1], [y0, y1], indexing="ij")
        return grid_mesh(
            [x0, x1],
            [y0, y1],
            base + terrain["slope_x"] * gx + terrain["slope_y"] * gy,
        )
    if kind == "step":
        # A flat tread per level, with a vertical wall at every step edge
        num_steps = terrain["num_steps"]
        edges = terrain["x_start"] + terrain["step_length"] * np.arange(
            num_steps
        )
        levels = base + terrain["step_height"] * np.arange(num_steps + 1)
        bounds = np.clip(np.concatenate([[x0], edges, [x1]]), x0, x1)

        parts = []
        for i, level in enumerate(levels):
            if bounds[i + 1] > bounds[i]:
                parts.append(
                    grid_mesh(
                        [bounds[i], bounds[i + 1]],
                        [y0, y1],
                        np.full((2, 2), level),
                    )
                )
        for i, edge in enumerate(edges):
            if x0 < edge < x1:
                parts.append(
                    grid_mesh(
                        [edge, edge],
                        [y0, y1],
                        np.array(
                            [[levels[i]] * 2, [levels[i + 1]] * 2]
                        ),
                    )
                )

        vertices, faces, offset = [], [], 0
        for v, f in parts:
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        return np.concatenate(vertices), np.concatenate(faces)

    return None


# Draw the terrain around the robot's trajectory
# Files without a terrain are drawn as flat ground
terrain = data["properties"].get("terrain", {"type": "flat", "base_height": 0.0})

if not args.no_ground:
    if data["properties"]["floating_base"]:
        base_positions = np.array(
            [stage["position"] for stage in data["stage_data"]]
        )
        x_range = (
            base_positions[:, 0].min() - TERRAIN_MARGIN,
            base_positions[:, 0].max() + TERRAIN_MARGIN,
        )
        y_range = (
            base_positions[:, 1].min() - TERRAIN_MARGIN,
            base_positions[:, 1].max() + TERRAIN_MARGIN,
        )
    else:
        x_range = y_range = (-TERRAIN_MARGIN, TERRAIN_MARGIN)

    mesh = terrain_mesh(terrain, x_range, y_range)
    if mesh is None:
        print(f"Cannot draw terrain of type '{terrain['type']}'")
    else:
        server.scene.add_mesh_simple(
            "/ground",
            vertices=mesh[0],
            faces=mesh[1],
            color=(225, 225, 205),
            side="double",
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

if args.circle:
    cx, cy, cz, cr = args.circle
    theta = np.linspace(0, 2 * np.pi, 64)
    server.scene.add_spline_catmull_rom(
        "/circle",
        points=np.stack(
            [np.full_like(theta, cx), cy + cr * np.cos(theta), cz + cr * np.sin(theta)],
            axis=1,
        ),
        closed=True,
        color=(0.0, 1.0, 0.0),
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
        colors=(1.0, 0.0, 0.0),  # red
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
    if len(state) < expected_config_size:
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


configuration_key = (
    "joint_configuration" if data["properties"]["floating_base"] else "state"
)

# Main loop
while True:
    # Animate each stage
    for k in range(data["properties"]["horizon"]):
        start = time()
        configuration = process_state(data["stage_data"][k][configuration_key])

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
                    frame_visuals = urdf._urdf.link_map[frame].visuals
                    if frame_visuals:
                        frame_T_visual = (
                            urdf._urdf.link_map[frame].visuals[0].origin
                        )
                    else:
                        frame_T_visual = np.eye(4)
                    frame_T = urdf._urdf.get_transform(frame) @ frame_T_visual
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

        sleep(
            max(
                0,
                data["properties"]["dt"] * args.delay_factor
                - (time() - start),
            )
        )

    if not args.loop:
        break
