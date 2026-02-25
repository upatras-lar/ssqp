import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

"""
Go2 Standing example
Stands in place with the Go2
"""

# Model Parameters
nq = 19
nv = 18
HORIZON = 100
dt = 0.01

# Weights
A_WGT = 1.0
V_WGT = 1e-3
U_WGT = 1e-4
F_WGT = 1e-6

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3689])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

# Initial configuration
init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = base_config

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

# Define regularization costs
config_cost = costs.SquaredConfigurationErrorCost(
    base_config, nq, nv, A_WGT, True
)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, U_WGT)
force_cost = costs.SquaredForceCost(nv, F_WGT)

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

for k in range(HORIZON):
    # Add regularization costs
    model.addCost(k, config_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add a standing phase (4 feet on the ground)
    model.addContactConstraint(k, "FL_foot")
    model.addConstraints(k, fl_foot_vel_bound)

    model.addContactConstraint(k, "FR_foot")
    model.addConstraints(k, fr_foot_vel_bound)

    model.addContactConstraint(k, "RL_foot")
    model.addConstraints(k, rl_foot_vel_bound)

    model.addContactConstraint(k, "RR_foot")
    model.addConstraints(k, rr_foot_vel_bound)

model.addCost(HORIZON, config_cost)
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)
# TODO: fix crash with this constraint
# model.addConstraints(
# HORIZON,
# constraints.JointControlConstraint(np.zeros(nv - 6), np.zeros(nv - 6), nv, True),
# )

# Finalize the model with our state and controls
model.finalize(init_states=init)

# Create a solver with the model
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    line_search_method=ssqpy.solvers.LineSearchMethod.Merit,
    merit_mu_c=1.0,
    merit_mu_gamma=1.0,
    qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    hpipm_max_iter=500,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-3,
    admm_max_iter=500,
)

state, x, u = solver.solve()

solver.write("../data/go2_standing.json")

print("x:")
x.print()
print("u:")
u.print()
