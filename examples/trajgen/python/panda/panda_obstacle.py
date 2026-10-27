import ssqpy
import numpy as np
import argparse
from math import pi, sin, cos

"""
Panda Obstacle example
From the starting end-effector position, move to the target end-effector
position while avoiding the obstacle defined by the constraints
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Panda obstacle avoidance trajectory generation"
    )

    parser.add_argument(
        "-t",
        "--target",
        type=float,
        nargs=3,
        default=(0.3, 0.3, 0.1),
        metavar=("X", "Y", "Z"),
        help="The target end effector position",
    )

    parser.add_argument(
        "--min-height",
        type=float,
        default=0.4,
        help="Height under which the end effector may not go for the middle \
            of the trajectory (the obstacle)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/panda_constrained.json",
        help="Path of the output JSON file",
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm", "piqp"],
        default="admm",
        help="QP solver used at each SSQP iteration",
    )

    return parser.parse_args()


args = parse_args()
H = 400
dt = 0.01

# Weights
T_WGT = 1e4
R_WGT = 10.0
V_WGT = 1e-1
U_WGT = 1e-2

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

# Create an SSQP model with H and dt, from the URDF at the below path
model = ssqpy.model.Model(
    H,
    dt,
    from_preset=ssqpy.model.Preset.Panda,
    solver_mode=ssqpy.model.SolverMode.InverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

init_state = np.zeros(2 * nq)
init_state[:nq] = [
    -0.3958,
    0.1247,
    -0.3661,
    -2.6497,
    0.1197,
    2.7639,
    -0.0852,
    0.0,
    0.0,
]

joint_vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
rot_cost = ssqpy.model.costs.FrameSquaredRotationErrorCost(
    model, "panda_ee", Rd, R_WGT
)
pose_cost = ssqpy.model.costs.FrameSquaredPoseErrorCost(
    model, "panda_ee", np.array(args.target), Rd, T_WGT, R_WGT
)

eef_bound = ssqpy.model.constraints.FrameTranslationBound(
    model,
    "panda_ee",
    np.array((-np.inf, -np.inf, args.min_height)),
    np.array((np.inf, np.inf, np.inf)),
)

for k in range(H):
    model.addCost(k, rot_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)

    if k > 50 and k < H - 50:
        model.addConstraints(k, eef_bound)

model.addCost(H, pose_cost)
model.addCost(H, joint_vel_cost)

model.finalize(init_states=init_state)

params = ssqpy.solvers.ssqpParams()
params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)
params.tolerance = 1e-2
params.admmParams.abs_tolerance = 1e-4
params.admmParams.rel_tolerance = 1e-4
params.admmParams.rho_update = False
params.admmParams.reset_rho = False
params.admmParams.reset_y = False

solver = ssqpy.solvers.ssqp(model, params)

state, x, u = solver.solve()

print("Solver state:", state)

solver.write(args.output)
