import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse
from math import pi, sin, cos

"""
Go2 circular trunk motion example
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 circular trunk motion trajectory generation"
    )

    parser.add_argument(
        "-n",
        "--circles",
        type=float,
        default=1,
        help="Number of circles to trace",
    )

    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=0.15,
        help="Radius of the circle",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_circle.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

# ----------------------------
# Dimensions and parameters
# ----------------------------

HORIZON = 400
dt = 0.01

# Weights
T_WGT = 10.0
R_WGT = 3.0
V_WGT = 3e-2
U_WGT = 3e-2
F_WGT = 1e-4

N_circles = args.circles


# ----------------------------
# Trajectory generation
# ----------------------------


def make_trajectory(H):
    z = 0.3879793820122  # Go2 nominal base height
    r = args.radius

    traj = []

    # Circular segment
    for k in range(HORIZON):
        theta = N_circles * 2.0 * pi * (float(k) / float(HORIZON))
        x = r * sin(theta)
        y = -r * cos(theta)
        traj.append(np.array([x, y, z]))

    return traj


traj = make_trajectory(HORIZON)


# ----------------------------
# Initial configuration
# ----------------------------

base_pos = np.array(
    (0.00977120909615503, 0.0010149333444776712, 0.3879793820122)
)
base_config = np.array(
    (
        -0.00301867560886747,
        0.5453115231859124,
        -1.0453655393403025,
        -0.003017174731528249,
        0.5428733284855946,
        -1.0418771136626226,
        -0.003914740418553554,
        0.5379606244219666,
        -1.0308890625183473,
        -0.003911815987037944,
        0.5346780323166637,
        -1.025727017171215,
    )
)

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0  # quaternion w
init[7:19] = base_config


# ----------------------------
# Create model (Go2)
# ----------------------------

model = ssqpy.model.Model(
    HORIZON,
    dt,
    from_preset=ssqpy.model.Preset.Go2,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()


# ----------------------------
# Build stage costs & constraints
# ----------------------------

for k in range(HORIZON):

    # Trunk pose tracking
    trunk_cost = costs.FrameSquaredPoseErrorCost(
        model,
        "base",  # Go2 trunk frame name
        traj[k],
        np.eye(3),
        T_WGT,
        R_WGT,
    )
    model.addCost(k, trunk_cost)

    # Regularization
    joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
    control_cost = costs.SquaredControlCost(model, U_WGT)
    force_cost = costs.SquaredForceCost(model, F_WGT)

    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # 4-foot contact phase (standing)
    for foot in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:

        model.addContactConstraint(k, foot, eq_tol=0.01)

        vel_bound = constraints.FrameLinearVelocityBound(
            model, foot, np.zeros(3), np.zeros(3)
        )

        model.addConstraints(k, vel_bound)


# Terminal trunk tracking cost
terminal_cost = costs.FrameSquaredPoseErrorCost(
    model,
    "base",
    traj[-1],
    np.eye(3),
    T_WGT,
    R_WGT,
)

model.addCost(HORIZON, terminal_cost)


# ----------------------------
# Finalize model
# ----------------------------

model.finalize(init_states=init)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = False
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.max_iter = 500
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
# hpipm_params.tol_stat = 1e-4
# hpipm_params.tol_comp = 1e-4
# hpipm_params.tol_eq = 1e-4
# hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

# ----------------------------
# Create solver
# ----------------------------

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

print("Solver state:", state)

solver.write(args.output)
