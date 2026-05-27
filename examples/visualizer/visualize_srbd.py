import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SRBD trajectory visualizer")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input trajectory file (.json)",
    )

    return parser.parse_args()


args = parse_args()

# Read trajectory
with open(args.input, "r") as f:
    data = json.loads(f.read())


# Extract states
# State format:
# [
#   CoM pos (3),
#   quaternion (4),
#   CoM vel (3),
#   CoM ang vel (3),
#   fp1 (3), fp2 (3), fp3 (3), fp4 (3)
# ]

states = np.array([stage["state"] for stage in data["stage_data"]])

# CoM position
com_pos = states[:, 0:3]

# Quaternion (unused here)
quat = states[:, 3:7]

# Velocities
com_vel = states[:, 7:10]
com_ang_vel = states[:, 10:13]

# Feet positions
fp1 = states[:, 13:16]
fp2 = states[:, 16:19]
fp3 = states[:, 19:22]
fp4 = states[:, 22:25]

feet = [fp1, fp2, fp3, fp4]
foot_labels = ["FL", "FR", "RL", "RR"]


# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


# Compute plotting bounds
all_points = np.vstack([com_pos, fp1, fp2, fp3, fp4])

xmin, ymin, zmin = np.min(all_points, axis=0)
xmax, ymax, zmax = np.max(all_points, axis=0)

padding = 0.1

ax.set_xlim(xmin - padding, xmax + padding)
ax.set_ylim(ymin - padding, ymax + padding)
ax.set_zlim(min(0.0, zmin - padding), zmax + padding)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("SRBD CoM / Foot Playback")


# Artists

# CoM point
(com_artist,) = ax.plot([], [], [], "o", markersize=10, label="CoM")

# CoM trajectory trail
(com_trail,) = ax.plot([], [], [], linewidth=2)

# Foot points
foot_artists = []
for label in foot_labels:
    (artist,) = ax.plot([], [], [], "s", markersize=8, label=label)
    foot_artists.append(artist)

# Lines from CoM to each foot
leg_lines = []
for _ in range(4):
    (line,) = ax.plot([], [], [], linewidth=1.5)
    leg_lines.append(line)

ax.legend()


# Animation update
def update(frame):
    # Current CoM
    cx, cy, cz = com_pos[frame]

    # Update CoM point
    com_artist.set_data([cx], [cy])
    com_artist.set_3d_properties([cz])

    # Update trail
    com_trail.set_data(com_pos[: frame + 1, 0], com_pos[: frame + 1, 1])
    com_trail.set_3d_properties(com_pos[: frame + 1, 2])

    # Update feet
    for i, foot in enumerate(feet):
        fx, fy, fz = foot[frame]

        foot_artists[i].set_data([fx], [fy])
        foot_artists[i].set_3d_properties([fz])

        # Leg line
        leg_lines[i].set_data([cx, fx], [cy, fy])
        leg_lines[i].set_3d_properties([cz, fz])

    ax.set_title(f"SRBD Playback | Frame {frame}/{len(states)-1}")

    return [com_artist, com_trail] + foot_artists + leg_lines


# Run animation
anim = FuncAnimation(
    fig,
    update,
    frames=len(states),
    interval=50,
    blit=False,
    repeat=True,
)

plt.show()
