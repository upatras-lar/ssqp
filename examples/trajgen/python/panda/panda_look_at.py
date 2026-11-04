import ssqpy
import numpy as np
import argparse
import json
from math import pi, sin, cos

"""
Panda Look At example
Keep the Panda's end effector pointed at a target moving on a circle on the
table in front of it, like a camera following an object, while the end effector
stays near where it started

The pointing cost and the distance bound are expressed in the end effector's own
(Local) axes, which turn with it, so the solver meets them by turning the end
effector towards the target rather than by moving it. With --reference-frame
world, they are expressed in the world axes instead, and the end effector moves
over the target, staying above it, without turning to point at it
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Panda end effector look at trajectory generation"
    )

    parser.add_argument(
        "-r",
        "--reference-frame",
        choices=["local", "world"],
        default="local",
        help="Axes the pointing cost and the distance bound are expressed in: \
            the end effector's own axes, which turn with it, or the world axes",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/panda_look_at.json",
        help="Path of the output JSON file",
    )

    parser.add_argument(
        "-t",
        "--target-output",
        default="../../data/panda_look_at_target.json",
        help="Path of the output JSON file with the target's position at each \
            stage, which visualize.py draws with --target",
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm", "piqp"],
        default="hpipm",
        help="QP solver used at each SSQP iteration",
    )

    return parser.parse_args()


args = parse_args()

H = 400
dt = 0.01

# Weights
LOOK_WGT = 10.0
HOLD_WGT = 1.0
V_WGT = 3e-3
U_WGT = 3e-3

# Stages over which the pointing cost's weight ramps up from zero, so that the
# end effector turns towards the target smoothly from where it starts
RAMP_STAGES = 50

# Distance range of the target ahead of the end effector, along its z axis
MIN_DISTANCE = 0.4
MAX_DISTANCE = 0.8

# Target circle on the table, in front of the robot
cx = 0.55
cy = 0
cz = 0.0
cr = 0.2

DISTANCE = 2 * pi
ω = DISTANCE / (H * dt)
φ0 = 0

# Target trajectory
target = []
for k in range(H + 1):
    φk = φ0 + ω * k * dt
    target.append(np.array([cx + cr * cos(φk), cy + cr * sin(φk), cz]))

# Create an SSQP model with H and dt, from the URDF at the below path
model = ssqpy.model.Model(
    H,
    dt,
    from_preset=ssqpy.model.Preset.Panda,
    solver_mode=ssqpy.model.SolverMode.InverseDynamics,
)

nq = model.getnq()
nv = model.getnv()

# Initial state, with the end effector pointing down
init_state = np.zeros(nq + nv)
init_state[:nq] = [
    0.0013,
    -0.9115,
    -0.0010,
    -2.6316,
    -0.0008,
    1.7431,
    0.7862,
    0.0,
    0.0,
]

# The end effector looks along its z axis, and stays near where it starts
start, _ = model.forwardKinematics(init_state[:nq], "panda_ee")

ReferenceFrame = ssqpy.model.ReferenceFrame
look_frame = (
    ReferenceFrame.Local
    if args.reference_frame == "local"
    else ReferenceFrame.World
)

# The end effector's position relative to the target along the z axis is
# negative in its own axes, whose z axis points at the target, and positive in
# the world axes, whose z axis points up away from it
distance_bounds = (
    (-MAX_DISTANCE, -MIN_DISTANCE)
    if args.reference_frame == "local"
    else (MIN_DISTANCE, MAX_DISTANCE)
)

hold_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, "panda_ee", start, HOLD_WGT
)

for k in range(H + 1):
    # Penalize the target's offset across the x and y axes, leaving the z axis
    # free. In the end effector's own axes, the target then lies on its z axis,
    # the direction it looks along, when the cost is zero, so the solver turns
    # the end effector to point at it. In the world axes, the end effector moves
    # over the target instead
    model.addCost(
        k,
        ssqpy.model.costs.FrameSquaredTranslationErrorCost(
            model,
            "panda_ee",
            target[k],
            min(1.0, k / RAMP_STAGES) * np.array([LOOK_WGT, LOOK_WGT, 0.0]),
            reference_frame=look_frame,
        ),
    )
    model.addCost(k, hold_cost)
    model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT))
    if k < H:
        model.addCost(k, ssqpy.model.costs.SquaredControlCost(model, U_WGT))

    # Keep the end effector between the two distances from the target along
    # the z axis: ahead of it along the end effector's own axes, or above it
    # along the world axes. The bound is taken around the target, so it
    # measures the end effector's position relative to it. The first stage's
    # state is fixed, so it is left out
    if k > 0:
        model.addConstraints(
            k,
            ssqpy.model.constraints.FrameTranslationBound(
                model,
                "panda_ee",
                np.array([-np.inf, -np.inf, distance_bounds[0]]),
                np.array([np.inf, np.inf, distance_bounds[1]]),
                reference_frame=look_frame,
                center=target[k],
            ),
        )

model.finalize(init_states=init_state)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-3,
    admm_adaptive_rho_tolerance=10,
    qp_solver_type=getattr(ssqpy.solvers.QPSolverType, args.qp_solver.upper()),
    hpipm_max_iter=250,
)

state, x, u = solver.solve()

# Write the solution to an output file, and the target's position at each
# stage to its own file
# The generated trajectory can be visualized using the `visualize.py` script,
# with `--target` pointing to the target file
solver.write(args.output)
with open(args.target_output, "w") as f:
    json.dump({"positions": [p.tolist() for p in target]}, f)

print(f"Solver state: {state}")
