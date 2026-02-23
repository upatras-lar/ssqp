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
A_WGT = 1e-1
V_WGT = 1e-3
U_WGT = 1e-2
F_WGT = 1e-4
TF_WGT = 7.0
RF_WGT = 3.0

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3689])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

# Initial configuration
init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = base_config

# Initial controls for 4 forces
init_control = np.zeros(30)
init_control[20] = 3.75 * 9.81
init_control[23] = 3.75 * 9.81
init_control[26] = 3.75 * 9.81
init_control[29] = 3.75 * 9.81

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

# Define regularization costs
# joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, A_WGT, True)
config_cost = costs.SquaredConfigurationErrorCost(base_config, nq, nv, A_WGT, True)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, U_WGT)
force_cost = costs.SquaredForceCost(nv, F_WGT)
pose_cost = costs.FrameSquaredPoseErrorCost(
    model, base_pos, np.eye(3), "base", nq, nv, TF_WGT, RF_WGT
)

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

init_controls = []

for k in range(HORIZON):
    # Add regularization costs
    model.addCost(k, config_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)
    model.addCost(k, pose_cost)

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

model.addCost(k, config_cost)
# model.addCost(HORIZON, joint_vel_cost)
model.addCost(HORIZON, costs.SquaredJointVelocityCost(nv, 1e3, True))
model.addCost(HORIZON, pose_cost)
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
model.finalize(init_states=init, init_controls=init_controls)

# Create a solver with the model
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    use_filter=False,
    merit_mu_c=1.0,
    merit_mu_gamma=1.0,
    qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    hpipm_max_iter=500,
    hpipm_tol_stat=1e-4,
    hpipm_tol_comp=1e-4,
    hpipm_tol_eq=1e-4,
    hpipm_tol_ineq=1e-4,
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
