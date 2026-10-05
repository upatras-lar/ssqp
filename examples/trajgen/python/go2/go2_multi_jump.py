import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse
from math import pi

"""
Go2 Multi-Jump example
Performs multiple jumps with the Go2 robot, using a predefined sequence of
steps, as well as which (combinations of) feet to land on in between the jumps
"""

# Model Parameters
HORIZON = 340
dt = 0.01

# Weights
A_WGT = 1e-5
V_WGT = 1e-3
U_WGT = 1e-4
F_WGT = 1e-4
TF_WGT = 1e-1
RF_WGT = 1e-1

MAX_FORCE_NORMAL = 147



def step_range(text):
    start, end = text.split(":")
    return int(start), int(end)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 multiple jumps trajectory generation"
    )

    parser.add_argument(
        "-j",
        "--jump-steps",
        type=step_range,
        nargs="+",
        default=[(50, 70), (100, 120), (150, 180), (210, 240), (270, 300)],
        metavar="START:END",
        help="The step ranges for which the robot should be in the air",
    )

    parser.add_argument(
        "-l",
        "--land-seq",
        nargs="+",
        choices=["B", "F", "L", "R", "FL", "FR", "RL", "RR"],
        default=["B", "R", "L", "F"],
        help="Which feet the robot should land on in between the jumps, one \
            for each gap between consecutive jumps: back (B), front (F), left \
            (L) or right (R) feet, or a single foot (FL, FR, RL or RR)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_multi_jump.json",
        help="Path of the output JSON file",
    )

    args = parser.parse_args()

    if len(args.land_seq) != len(args.jump_steps) - 1:
        parser.error(
            "--land-seq needs one entry for each gap between consecutive jumps"
        )

    return args


args = parse_args()

JUMP_STEPS = args.jump_steps

multiplier = 1

HORIZON = int(multiplier * HORIZON)
JUMP_STEPS = [(multiplier * x, multiplier * y) for (x, y) in JUMP_STEPS]

LAND_SEQ = args.land_seq

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3879793820122])
final_pos = np.array([0.0, 0.0, 0.3879793820122])

# Initial configuration
init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

# Initial controls for 4 forces
init_control = np.zeros(30)
init_control[20] = 3.75 * 9.81
init_control[23] = 3.75 * 9.81
init_control[26] = 3.75 * 9.81
init_control[29] = 3.75 * 9.81

# Initial controls for 2 forces
init_control_2 = np.zeros(24)
init_control_2[20] = 7.5 * 9.81
init_control_2[23] = 7.5 * 9.81

# Initial controls for 1 force
init_control_1 = np.zeros(21)
init_control_1[20] = 15 * 9.81

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    from_preset=ssqpy.model.Preset.Go2,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Define regularization costs
joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

# Define velocity bounds for feet
# Used in conjunction with contact constraints
fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3)
)

fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FR_foot", np.zeros(3), np.zeros(3)
)

rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RL_foot", np.zeros(3), np.zeros(3)
)

rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RR_foot", np.zeros(3), np.zeros(3)
)

# Define position bounds for feet, such that their z-coordinate is positive and
# the y-coordinate of the front two and back two feet does not get too close,
# respectively
# This is done so that:
#  1) Feet that are on the air stay above the ground
#  2) Front and back feet don't get too close to each other
fl_foot_pos_bound = constraints.FrameTranslationBound(
    model,
    "FL_foot",
    np.array((-np.inf, 0.1, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

fr_foot_pos_bound = constraints.FrameTranslationBound(
    model,
    "FR_foot",
    np.array((-np.inf, -np.inf, -0.01)),
    np.array((np.inf, -0.1, np.inf)),
)

rl_foot_pos_bound = constraints.FrameTranslationBound(
    model,
    "RL_foot",
    np.array((-np.inf, 0.1, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

rr_foot_pos_bound = constraints.FrameTranslationBound(
    model,
    "RR_foot",
    np.array((-np.inf, -np.inf, -0.01)),
    np.array((np.inf, -0.1, np.inf)),
)

# Define a bound to keep the head position above the ground
head_pos_bound = constraints.FrameTranslationBound(
    model,
    "Head_lower",
    np.array((-np.inf, -np.inf, 0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

# Define a bound to keep base above a certain height
base_pos_bound = constraints.FrameTranslationBound(
    model,
    "base",
    np.array((-np.inf, -np.inf, 0.2)),
    np.array((np.inf, np.inf, np.inf)),
)

init_controls = []

for k in range(HORIZON):
    # Add regularization costs
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add bounds for head and feet to keep valid positions
    model.addConstraints(k, head_pos_bound)

    model.addConstraints(k, fl_foot_pos_bound)
    model.addConstraints(k, fr_foot_pos_bound)
    model.addConstraints(k, rl_foot_pos_bound)
    model.addConstraints(k, rr_foot_pos_bound)

    model.addConstraints(k, base_pos_bound)

    # If at start or end of horizon
    if k < JUMP_STEPS[0][0] or k >= JUMP_STEPS[-1][-1]:
        # Add a standing phase (4 feet on the ground)
        model.addContactConstraint(
            k, "FL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
        )
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(
            k, "FR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
        )
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(
            k, "RL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
        )
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(
            k, "RR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
        )
        model.addConstraints(k, rr_foot_vel_bound)

        init_controls.append(init_control)
        continue

    on_air = True
    # For each on-air range
    for i, jmp_range in enumerate(JUMP_STEPS):
        # If not on air
        if k < jmp_range[0]:
            on_air = False
            break

        # If on air
        if k >= jmp_range[0] and k < jmp_range[1]:
            init_controls.append(np.zeros(nv))
            break

    # If we are on the ground, determine which feet should actually make contact
    # and add constraints for them
    if not on_air:
        if LAND_SEQ[i - 1] == "B":
            model.addContactConstraint(
                k, "RL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rl_foot_vel_bound)

            model.addContactConstraint(
                k, "RR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rr_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "F":
            model.addContactConstraint(
                k, "FL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fl_foot_vel_bound)

            model.addContactConstraint(
                k, "FR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fr_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "L":
            model.addContactConstraint(
                k, "FL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fl_foot_vel_bound)

            model.addContactConstraint(
                k, "RL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rl_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "R":
            model.addContactConstraint(
                k, "FR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fr_foot_vel_bound)

            model.addContactConstraint(
                k, "RR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rr_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "FL":
            model.addContactConstraint(
                k, "FL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fl_foot_vel_bound)

            init_controls.append(init_control_1)
        elif LAND_SEQ[i - 1] == "FR":
            model.addContactConstraint(
                k, "FR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, fr_foot_vel_bound)

            init_controls.append(init_control_1)
        elif LAND_SEQ[i - 1] == "RL":
            model.addContactConstraint(
                k, "RL_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rl_foot_vel_bound)

            init_controls.append(init_control_1)
        elif LAND_SEQ[i - 1] == "RR":
            model.addContactConstraint(
                k, "RR_foot", max_normal_force=MAX_FORCE_NORMAL, eq_tol=0.01
            )
            model.addConstraints(k, rr_foot_vel_bound)

            init_controls.append(init_control_1)

# Add joint regularization cost
model.addCost(HORIZON, joint_vel_cost)

# Add cost to end up in the final position, with the base orientation
model.addCost(
    HORIZON,
    costs.FrameSquaredPoseErrorCost(
        model, "base", final_pos, np.eye(3), TF_WGT, RF_WGT
    ),
)
# Add constraint for the final velocity of the base to be 0
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = False
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.max_iter = 50
admm_params.reset_rho = False
admm_params.reset_y = False
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-3
hpipm_params.tol_comp = 1e-3
hpipm_params.tol_eq = 1e-3
hpipm_params.tol_ineq = 1e-3
ssqp_params.hpipmParams = hpipm_params

# Create a solver with the model
solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

solver.write(args.output)

print("x:")
x.print()
print("u:")
u.print()
