import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

"""
A1 Jump example
Perform a jump with the A1
To do this, we add contacts for each foot for a certain number of steps at the
start and end of the time horizon, while leaving a span of time with no contacts
in the middle
"""

# Model parameters
nq = 19
nv = 18
HORIZON = 150
dt = 0.01

# Weights
A_WGT = 1e-2
V_WGT = 1e-4
U_WGT = 1e-6
F_WGT_START = 1e-8
F_WGT_END = 1e-5
TP_WGT = 100
TO_WGT = 100

# Number of steps with contacts at the start & end of time horizon
STEPS_BEFORE = 60
STEPS_AFTER = 50

# Initial position
base_pos = np.array([0, 0, 0.3464])

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

# Initial values for forces
init_control = np.zeros(30)
init_control[20] = 3.43525 * 9.81
init_control[23] = 3.43525 * 9.81
init_control[26] = 3.43525 * 9.81
init_control[29] = 3.43525 * 9.81

# Create an SSQP model
# Since the A1 has a floating base, we must set free_flyer to True
model = ssqpy.model.Model(
    HORIZON, dt, urdf_path="../../assets/urdf/a1/a1.urdf", free_flyer=True
)

# Create regularization costs for joint angles, joint velocity and control
joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, A_WGT, True)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, U_WGT)

# Velocity bounds for feet frames
# These add constraints, such that the feet have no velocity
fl_foot_vel_bound = constraints.FrameVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3), nq, nv
)

fr_foot_vel_bound = constraints.FrameVelocityBound(
    model, "FR_foot", np.zeros(3), np.zeros(3), nq, nv
)

rl_foot_vel_bound = constraints.FrameVelocityBound(
    model, "RL_foot", np.zeros(3), np.zeros(3), nq, nv
)

rr_foot_vel_bound = constraints.FrameVelocityBound(
    model, "RR_foot", np.zeros(3), np.zeros(3), nq, nv
)

init_controls = []

for k in range(HORIZON):
    # Add regularization costs at each time step
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)

    # If before jump
    if k < STEPS_BEFORE:
        # Add an increasing weight to the forces as we move closer to the jump
        force_cost_k = costs.SquaredForceCost(
            nv, F_WGT_START + k * (F_WGT_END - F_WGT_START) / STEPS_BEFORE
        )
        model.addCost(k, force_cost_k)
    # If after jump
    elif k > HORIZON - STEPS_AFTER:
        # Add the regular force weight
        force_cost_k = costs.SquaredForceCost(nv, F_WGT_START)
        model.addCost(k, force_cost_k)

    # If not on the air
    if k < STEPS_BEFORE or k > HORIZON - STEPS_AFTER:
        # Add a contact constraint & velocity bound for each foot
        model.addContactConstraint(k, "FL_foot")
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(k, "FR_foot")
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(k, "RL_foot")
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(k, "RR_foot")
        model.addConstraints(k, rr_foot_vel_bound)

        # Our controls for this stage must include initial values for the 4
        # forces we add through the contact constraints
        # As such, we will use `init_control`
        init_controls.append(init_control)
    # If on the air
    else:
        # If there are no forces, we only need initial values for the
        # the joint controls; we will simply initialize to 0
        init_controls.append(np.zeros(nv))

# Add joint velocity regularization cost to terminal stage
model.addCost(k, joint_angle_cost)
model.addCost(HORIZON, joint_vel_cost)

# Add cost for `"trunk"` frame (base) to stay in the initial position and
# the base rotation
model.addCost(
    HORIZON,
    costs.FrameSquaredPoseErrorCost(
        model, base_pos, np.eye(3), "trunk", nq, nv, TP_WGT, TO_WGT
    ),
)

# Finalize the model using the same initial state for all stages, as well as the
# controls we put together
model.finalize(init_states=init, init_controls=init_controls)

# Creare a solver based on the model
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    filter_size=100,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-4,
    admm_max_iter=750,
)

# Solve
state, x, u = solver.solve()

# Write output trajectory to a file
solver.write("../data/a1_jump.json")

print("x:")
x.print()
print("u:")
u.print()
