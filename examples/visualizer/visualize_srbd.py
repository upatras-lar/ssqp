import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args():
    parser = argparse.ArgumentParser(description="SRBD trajectory visualizer")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input trajectory file (.json)",
    )
    parser.add_argument(
        "--body-length",
        type=float,
        default=0.4,
        help="Body rectangle length along body-X (default: 0.4)",
    )
    parser.add_argument(
        "--body-width",
        type=float,
        default=0.2,
        help="Body rectangle width along body-Y (default: 0.2)",
    )
    parser.add_argument(
        "--view-radius",
        type=float,
        default=0.6,
        help="Half-extent of the camera window around the CoM (default: 0.8)",
    )
    parser.add_argument(
        "--fixed-limits",
        action=argparse.BooleanOptionalAction,
        help="Whether to fix axis limits",
    )

    return parser.parse_args()


def quat_to_rotmat(q):
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def body_rect_world(com, R, L, W):
    """Four world-frame corners centred at `com`: FL, FR, RR, RL."""
    corners_body = np.array(
        [
            [L / 2, W / 2, 0],
            [L / 2, -W / 2, 0],
            [-L / 2, -W / 2, 0],
            [-L / 2, W / 2, 0],
        ]
    )
    return (R @ corners_body.T).T + com  # (4, 3)


args = parse_args()

with open(args.input, "r") as f:
    data = json.loads(f.read())

states = np.array([stage["state"] for stage in data["stage_data"]])
com_pos = states[:, 0:3]
quat = states[:, 3:7]
com_vel = states[:, 7:10]
com_ang_vel = states[:, 10:13]
fp1 = states[:, 13:16]
fp2 = states[:, 16:19]
fp3 = states[:, 19:22]
fp4 = states[:, 22:25]
feet = [fp1, fp2, fp3, fp4]
foot_labels = ["FL", "FR", "RL", "RR"]
foot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
foot_corner_idx = [0, 1, 3, 2]  # FL→0, FR→1, RL→3, RR→2

BODY_L = args.body_length
BODY_W = args.body_width
R_VIEW = args.view_radius  # fixed half-extent for every axis

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

if args.fixed_limits:
    all_points = np.vstack([com_pos, fp1, fp2, fp3, fp4])
    xmin, ymin, zmin = np.min(all_points, axis=0)
    xmax, ymax, zmax = np.max(all_points, axis=0)
    pad = 0.4
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_zlim(0.0, zmax + pad)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# ── Artists ───────────────────────────────────────────────────────────────────
(com_artist,) = ax.plot(
    [], [], [], "o", markersize=6, color="black", label="CoM", zorder=5
)
(com_trail,) = ax.plot([], [], [], linewidth=1.5, color="gray", alpha=0.5)

foot_artists, leg_lines, leg_trails = [], [], []
for label, color in zip(foot_labels, foot_colors):
    (fa,) = ax.plot([], [], [], "s", markersize=8, color=color, label=label)
    (ll,) = ax.plot([], [], [], linewidth=1.5, color=color, alpha=0.6)
    (lt,) = ax.plot([], [], [], linewidth=1.5, color=color, alpha=0.3)
    foot_artists.append(fa)
    leg_lines.append(ll)
    leg_trails.append(lt)

(body_outline,) = ax.plot(
    [], [], [], linewidth=2, color="steelblue", label="Body"
)
(front_line,) = ax.plot(
    [], [], [], linewidth=3, color="crimson", label="Front"
)

body_fill = [None]


def _refresh_fill(verts):
    if body_fill[0] is not None:
        body_fill[0].remove()
    poly = Poly3DCollection(
        [verts], alpha=0.25, facecolor="steelblue", edgecolor="none"
    )
    ax.add_collection3d(poly)
    body_fill[0] = poly


_refresh_fill(
    body_rect_world(com_pos[0], quat_to_rotmat(quat[0]), BODY_L, BODY_W)
)
ax.legend(loc="upper left", fontsize=8)

# ── Animation update ──────────────────────────────────────────────────────────


def update(frame):
    cx, cy, cz = com_pos[frame]

    # ── Follow camera: re-centre every axis on the current CoM ───────────────
    if not args.fixed_limits:
        ax.set_xlim(cx - R_VIEW, cx + R_VIEW)
        ax.set_ylim(cy - R_VIEW, cy + R_VIEW)
        ax.set_zlim(0.0, cz + R_VIEW)

    # CoM
    com_artist.set_data([cx], [cy])
    com_artist.set_3d_properties([cz])

    com_trail.set_data(com_pos[: frame + 1, 0], com_pos[: frame + 1, 1])
    com_trail.set_3d_properties(com_pos[: frame + 1, 2])

    # Body rectangle
    R = quat_to_rotmat(quat[frame])
    verts = body_rect_world(com_pos[frame], R, BODY_L, BODY_W)

    # Feet + legs
    for i, foot in enumerate(feet):
        fx, fy, fz = foot[frame]
        foot_artists[i].set_data([fx], [fy])
        foot_artists[i].set_3d_properties([fz])
        kx, ky, kz = verts[foot_corner_idx[i]]
        leg_lines[i].set_data([kx, fx], [ky, fy])
        leg_lines[i].set_3d_properties([kz, fz])
        leg_trails[i].set_data(foot[:frame, 0], foot[:frame, 1])
        leg_trails[i].set_3d_properties(foot[:frame, 2])

    # Outline
    loop = np.vstack([verts, verts[0]])
    body_outline.set_data(loop[:, 0], loop[:, 1])
    body_outline.set_3d_properties(loop[:, 2])

    # Front edge
    fe = verts[[0, 1]]
    front_line.set_data(fe[:, 0], fe[:, 1])
    front_line.set_3d_properties(fe[:, 2])

    _refresh_fill(verts)

    ax.set_title(f"SRBD Playback | Frame {frame}/{len(states)-1}")
    return (
        [com_artist, com_trail, body_outline, front_line]
        + foot_artists
        + leg_lines
    )


anim = FuncAnimation(
    fig, update, frames=len(states), interval=50, blit=False, repeat=True
)
plt.show()
