import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

"""
Go2 Jump example
Performs a simple jump with the Go2 robot
"""

# Model Parameters
HORIZON = 150
dt = 0.01

START_STEP = 50
END_STEP = 100

# Weights
A_WGT = 0.0
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-4
TF_WGT = 1.0
RF_WGT = 1.0
C_WGT = 1e-3

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    dynamics_mode=ssqpy.model.DynamicsMode.ReducedInverse,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3689])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)
final_pos = np.array([0.0, 0.0, 0.3689])

# Initial configuration
init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = base_config

init_control = np.zeros(nu + 12)
init_control[-10] = 3.75 * 9.81
init_control[-7] = 3.75 * 9.81
init_control[-4] = 3.75 * 9.81
init_control[-1] = 3.75 * 9.81

joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, A_WGT, True)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, nu, U_WGT)
force_cost = costs.SquaredForceCost(nv, nu, F_WGT)
config_cost = costs.SquaredConfigurationErrorCost(
    base_config, nq, nv, C_WGT, True
)

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

init_controls = []

for k in range(HORIZON):
    # Add regularization costs
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)
    model.addCost(k, config_cost)

    model.addConstraints(k, fl_foot_pos_bound)
    model.addConstraints(k, fr_foot_pos_bound)
    model.addConstraints(k, rl_foot_pos_bound)
    model.addConstraints(k, rr_foot_pos_bound)

    if k < START_STEP or k >= END_STEP:
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

    init_controls.append(np.zeros(nu))

model.addCost(HORIZON, joint_vel_cost)

model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(
        model,
        "FL_foot",
        np.array((0.1934, 0.142, -0.00002682)),
        np.array((0.1934, 0.142, -0.00002682)),
        nq,
        nv,
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(
        model,
        "FR_foot",
        np.array((0.1934, -0.142, -0.00002682)),
        np.array((0.1934, -0.142, -0.00002682)),
        nq,
        nv,
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(
        model,
        "RL_foot",
        np.array((-0.1934, 0.142, -0.00002682)),
        np.array((-0.1934, 0.142, -0.00002682)),
        nq,
        nv,
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(
        model,
        "RR_foot",
        np.array((-0.1934, -0.142, -0.00002682)),
        np.array((-0.1934, -0.142, -0.00002682)),
        nq,
        nv,
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init, init_controls=init_controls)

# Create a solver with the model
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    stop_after_constraint_satisfaction=True,
    tolerance=1e-2,
    line_search_method=ssqpy.solvers.LineSearchMethod.Merit,
    merit_mu_c=4.0,
    merit_mu_gamma=0.1,
    # qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    hpipm_tol_stat=1e-3,
    hpipm_tol_eq=1e-3,
    hpipm_tol_ineq=1e-3,
    hpipm_tol_comp=1e-3,
    # hpipm_max_iter=250,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-3,
    admm_max_iter=500,
)

state, x, u = solver.solve()

solver.write("../data/go2_jump.json")
