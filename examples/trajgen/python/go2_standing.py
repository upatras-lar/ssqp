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
HORIZON = 100
dt = 0.01

# Weights
A_WGT = 1.0
V_WGT = 1e-3
U_WGT = 1e-4
F_WGT = 1e-6

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3879793820122])
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
    # mjcf_path="../../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    # solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Define regularization costs
config_cost = costs.SquaredConfigurationErrorCost(
    model, base_config, A_WGT, "Configuration cost"
)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT, "Velocity cost")
control_cost = costs.SquaredControlCost(model, U_WGT, "Control cost")
force_cost = costs.SquaredForceCost(model, F_WGT, "Force cost")

# Define velocity bounds for feet
# Used in conjunction with contact constraints
fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model,
    "FL_foot",
    np.zeros(3),
    np.zeros(3),
    name="FL foot velocity bound",
)

fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model,
    "FR_foot",
    np.zeros(3),
    np.zeros(3),
    name="FR foot velocity bound",
)

rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model,
    "RL_foot",
    np.zeros(3),
    np.zeros(3),
    name="RL foot velocity bound",
)

rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model,
    "RR_foot",
    np.zeros(3),
    np.zeros(3),
    name="RR foot velocity bound",
)

b1 = constraints.FramePositionBound(
    model,
    "FL_foot",
    np.array((-np.inf, -np.inf, 0.0)),
    np.array((np.inf, np.inf, np.inf)),
)

b2 = constraints.FramePositionBound(
    model,
    "FL_foot",
    np.array((-np.inf, -np.inf, 0.0)),
    np.array((np.inf, np.inf, np.inf)),
)

for k in range(HORIZON):
    # Add regularization costs
    # model.addCost(k, config_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add a standing phase (4 feet on the ground)
    model.addContactConstraint(k, "FL_foot", name="FL foot contact constraint")
    model.addConstraints(k, fl_foot_vel_bound)

    model.addContactConstraint(k, "FR_foot", name="FR foot contact constraint")
    model.addConstraints(k, fr_foot_vel_bound)

    model.addContactConstraint(k, "RL_foot", name="RL foot contact constraint")
    model.addConstraints(k, rl_foot_vel_bound)

    model.addContactConstraint(k, "RR_foot", name="RR foot contact constraint")
    model.addConstraints(k, rr_foot_vel_bound)

    model.addConstraints(k, b1)
    model.addConstraints(k, b2)

model.addCost(HORIZON, config_cost)
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), True
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = False
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
ssqp_params.stop_after_constraint_satisfaction = True

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.max_iter = 200
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.max_iter = 500
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

# Create a solver with the model
solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

solver.write("../data/go2_standing.json")
