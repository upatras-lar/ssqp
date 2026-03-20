import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

"""
Go2 Multi-Jump example
Performs multiple jumps with the Go2 robot, using a predefined sequence of
steps, as well as which (combinations of) feet to land on in between the jumps
"""

# Model Parameters
nq = 19
nv = 18
nu = 18
HORIZON = 300
dt = 0.01

# Weights
A_WGT = 1e-1
V_WGT = 1e-3
U_WGT = 1e-2
F_WGT = 1e-4
TF_WGT = 7.0
RF_WGT = 3.0

# @script-parameter
# List of tuples defining the step ranges for which the robot should be on air
JUMP_STEPS = [(50, 70), (100, 120), (150, 180), (200, 220), (240, 260)]
# @script-parameter
# List of strings defining which feet the robot should land on in between
# the above sequences
# Accepted values:
#  - `"B"`: Back feet
#  - `"F"`: Front feet
#  - `"L"`L Left feet
#  - `"R"`: Right feet
#  - `"FL"`: Front left foot
#  - `"FR"`: Front right foot
#  - `"RL"`: Rear left foot
#  - `"RR"`: Rear right foot
LAND_SEQ = ["F", "RR", "FL", "B"]

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3689])
final_pos = np.array([0.0, 0.0, 0.3689])

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
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    # dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

# Define regularization costs
joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, A_WGT, True)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, nu, U_WGT)
force_cost = costs.SquaredForceCost(nv, nu, F_WGT)

# Define velocity bounds for feet
# Used in conjunction with contact constraints
fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3), nq, nv
)

fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FR_foot", np.zeros(3), np.zeros(3), nq, nv
)

rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RL_foot", np.zeros(3), np.zeros(3), nq, nv
)

rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RR_foot", np.zeros(3), np.zeros(3), nq, nv
)

# Define position bounds for feet, such that their z-coordinate is positive and
# the y-coordinate of the front two and back two feet does not get too close,
# respectively
# This is done so that:
#  1) Feet that are on the air stay above the ground
#  2) Front and back feet don't get too close to each other
fl_foot_pos_bound = constraints.FramePositionBound(
    model,
    "FL_foot",
    np.array((-np.inf, 0.1, 0.0)),
    np.array((np.inf, np.inf, np.inf)),
    nq,
    nv,
)

fr_foot_pos_bound = constraints.FramePositionBound(
    model,
    "FR_foot",
    np.array((-np.inf, -np.inf, 0.0)),
    np.array((np.inf, -0.1, np.inf)),
    nq,
    nv,
)

rl_foot_pos_bound = constraints.FramePositionBound(
    model,
    "RL_foot",
    np.array((-np.inf, 0.1, 0.0)),
    np.array((np.inf, np.inf, np.inf)),
    nq,
    nv,
)

rr_foot_pos_bound = constraints.FramePositionBound(
    model,
    "RR_foot",
    np.array((-np.inf, -np.inf, 0.0)),
    np.array((np.inf, -0.1, np.inf)),
    nq,
    nv,
)

# Define a bound to keep the head position above the ground
head_pos_bound = constraints.FramePositionBound(
    model,
    "Head_lower",
    np.array((-np.inf, -np.inf, 0.01)),
    np.array((np.inf, np.inf, np.inf)),
    nq,
    nv,
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

    # If at start or end of horizon
    if k < JUMP_STEPS[0][0] or k >= JUMP_STEPS[-1][-1]:
        # Add a standing phase (4 feet on the ground)
        model.addContactConstraint(k, "FL_foot")
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(k, "FR_foot")
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(k, "RL_foot")
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(k, "RR_foot")
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
            model.addContactConstraint(k, "RL_foot")
            model.addConstraints(k, rl_foot_vel_bound)

            model.addContactConstraint(k, "RR_foot")
            model.addConstraints(k, rr_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "F":
            model.addContactConstraint(k, "FL_foot")
            model.addConstraints(k, fl_foot_vel_bound)

            model.addContactConstraint(k, "FR_foot")
            model.addConstraints(k, fr_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "L":
            model.addContactConstraint(k, "FL_foot")
            model.addConstraints(k, fl_foot_vel_bound)

            model.addContactConstraint(k, "RL_foot")
            model.addConstraints(k, rl_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "R":
            model.addContactConstraint(k, "FR_foot")
            model.addConstraints(k, fr_foot_vel_bound)

            model.addContactConstraint(k, "RR_foot")
            model.addConstraints(k, rr_foot_vel_bound)

            init_controls.append(init_control_2)
        elif LAND_SEQ[i - 1] == "FL":
            model.addContactConstraint(k, "FL_foot")
            model.addConstraints(k, fl_foot_vel_bound)

            init_controls.append(init_control_1)
        elif LAND_SEQ[i - 1] == "FR":
            model.addContactConstraint(k, "FR_foot")
            model.addConstraints(k, fr_foot_vel_bound)

            init_controls.append(init_control_1)
        elif LAND_SEQ[i - 1] == "RL":
            model.addContactConstraint(k, "RL_foot")
            model.addConstraints(k, rl_foot_vel_bound)

            init_controls.append(init_control_1)
        elif LAND_SEQ[i - 1] == "RR":
            model.addContactConstraint(k, "RR_foot")
            model.addConstraints(k, rr_foot_vel_bound)

            init_controls.append(init_control_1)

# Add joint regularization cost
model.addCost(HORIZON, joint_vel_cost)

# Add cost to end up in the final position, with the base orientation
model.addCost(
    HORIZON,
    costs.FrameSquaredPoseErrorCost(
        model, "base", final_pos, np.eye(3), nq, nv, TF_WGT, RF_WGT
    ),
)
# Add constraint for the final velocity of the base to be 0
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init, init_controls=init_controls)

# Create a solver with the model
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-1,
    tolerance_final=5e-2,
    tolerance_update_steps=25,
    line_search_method=ssqpy.solvers.LineSearchMethod.Merit,
    merit_mu_c=2.0,
    merit_mu_gamma=1.0,
    # qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    # hpipm_max_iter=250,
    admm_abs_tolerance=1e-2,
    admm_abs_tolerance_final=1e-3,
    admm_rel_tolerance=1e-3,
    admm_tolerance_update_steps=250,
    admm_max_iter=500,
)

state, x, u = solver.solve()

solver.write("../data/go2_multi_jump.json")

print("x:")
x.print()
print("u:")
u.print()
