import ssqpy
import ssqpy.model.costs as costs
import numpy as np
import argparse
from math import pi

"""
A1 Jump example
Perform a jump with the A1
To do this, we add contacts for each foot for a certain number of steps at the
start and end of the time horizon, while leaving a span of time with no contacts
in the middle
"""

# Model parameters
HORIZON = 150
dt = 0.01

# Weights
A_WGT = 1e-1
V_WGT = 1e-2
U_WGT = 1e-3
F_WGT_START = 1e-4
F_WGT_END = 3e-3
TF_WGT = 10
RF_WGT = 10

# Number of steps with contacts at the start & end of time horizon


def parse_args():
    parser = argparse.ArgumentParser(
        description="A1 jump trajectory generation"
    )

    parser.add_argument(
        "--steps-before",
        type=int,
        default=60,
        help="Number of steps with contacts at the start of the time horizon",
    )

    parser.add_argument(
        "--steps-after",
        type=int,
        default=50,
        help="Number of steps with contacts at the end of the time horizon",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../data/a1_jump.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

STEPS_BEFORE = args.steps_before
STEPS_AFTER = args.steps_after

# Initial position
base_pos = np.array([0, 0, 0.3464])

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

# Create an SSQP model
model = ssqpy.model.Model(
    HORIZON,
    dt,
    from_preset=ssqpy.model.Preset.A1,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Create regularization costs for joint angles, joint velocity and control
joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)

for k in range(HORIZON):
    # Add regularization costs at each time step
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)

    # If before jump
    if k < STEPS_BEFORE:
        # Add an increasing weight to the forces as we move closer to the jump
        force_cost_k = costs.SquaredForceCost(
            model, F_WGT_START + k * (F_WGT_END - F_WGT_START) / STEPS_BEFORE
        )
        model.addCost(k, force_cost_k)
    # If after jump
    elif k > HORIZON - STEPS_AFTER:
        # Add the regular force weight
        force_cost_k = costs.SquaredForceCost(model, F_WGT_START)
        model.addCost(k, force_cost_k)

    # If not on the air
    if k < STEPS_BEFORE or k > HORIZON - STEPS_AFTER:
        # Add a contact constraint for each foot
        model.addContactConstraint(k, "FL_foot")
        model.addContactConstraint(k, "FR_foot")
        model.addContactConstraint(k, "RL_foot")
        model.addContactConstraint(k, "RR_foot")

# Add joint velocity regularization cost to terminal stage
model.addCost(k, joint_angle_cost)
model.addCost(HORIZON, joint_vel_cost)

# Add cost for `"trunk"` frame (base) to stay in the initial position and
# the base rotation
model.addCost(
    HORIZON,
    costs.FrameSquaredPoseErrorCost(
        model, "trunk", base_pos, np.eye(3), TF_WGT, RF_WGT
    ),
)

# Finalize the model using the same initial state for all stages, as well as the
# controls we put together
model.finalize(init_states=init)

# Creare a solver based on the model
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    filter_size=100,
    stop_after_constraint_satisfaction=True,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-4,
    admm_max_iter=750,
)

# Solve
state, x, u = solver.solve()

print("Solver state:", state)

# Write output trajectory to a file
solver.write(args.output)
