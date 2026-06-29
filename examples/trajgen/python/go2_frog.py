import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

"""
Go2 Jump example
Performs a simple jump with the Go2 robot
"""

# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)

# Model Parameters
HORIZON = 300
dt = 0.01

# Weights
A_WGT = 1e-7
V_WGT = 1e-4
U_WGT = 5e-5
F_WGT = 5e-4
FEET_GROUND_WGT = 1e-1

MAX_NORMAL_FORCE = 100
FOOT_EQ_TOL = 0.01

# ssqpy.unsetLogLevel(ssqpy.LogLevel.Warning)

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3879793820122])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

final_pos = np.array([2.0, 0.0, 0.3879793820122])

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

joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

fl_foot_ag_bound = constraints.FrameTranslationBound(
    model,
    "FL_foot",
    np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
    np.array((np.inf, np.inf, np.inf)),
)

fr_foot_ag_bound = constraints.FrameTranslationBound(
    model,
    "FR_foot",
    np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
    np.array((np.inf, np.inf, np.inf)),
)

rl_foot_ag_bound = constraints.FrameTranslationBound(
    model,
    "RL_foot",
    np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
    np.array((np.inf, np.inf, np.inf)),
)

rr_foot_ag_bound = constraints.FrameTranslationBound(
    model,
    "RR_foot",
    np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
    np.array((np.inf, np.inf, np.inf)),
)

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

fl_foot_pos_bound = constraints.FrameTranslationBound(
    model,
    "FL_foot",
    np.array((-np.inf, 0.1, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

fl_foot_z_cost = costs.FrameSquaredTranslationErrorCost(
    model,
    "FL_foot",
    np.array((0.0, 0.0, 0.0)),
    np.array((0.0, 0.0, FEET_GROUND_WGT)),
)

fr_foot_z_cost = costs.FrameSquaredTranslationErrorCost(
    model,
    "FR_foot",
    np.array((0.0, 0.0, 0.0)),
    np.array((0.0, 0.0, FEET_GROUND_WGT)),
)

rl_foot_z_cost = costs.FrameSquaredTranslationErrorCost(
    model,
    "RL_foot",
    np.array((0.0, 0.0, 0.0)),
    np.array((0.0, 0.0, FEET_GROUND_WGT)),
)

rr_foot_z_cost = costs.FrameSquaredTranslationErrorCost(
    model,
    "RR_foot",
    np.array((0.0, 0.0, 0.0)),
    np.array((0.0, 0.0, FEET_GROUND_WGT)),
)

# Disallow pitch and yaw at differing limits
rot_constraint = constraints.FrameRotationBound(
    model,
    "base",
    np.eye(3),
    np.array((-np.inf, -1.0, -0.01)),
    np.array((np.inf, 1.0, 0.01)),
)

init_controls = []

for k in range(HORIZON):
    # Add regularization costs
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    model.addCost(k, fl_foot_z_cost)
    model.addCost(k, fr_foot_z_cost)
    model.addCost(k, rl_foot_z_cost)
    model.addCost(k, rr_foot_z_cost)

    model.addConstraints(k, rot_constraint)
    model.addConstraints(k, fl_foot_ag_bound)
    model.addConstraints(k, fr_foot_ag_bound)
    model.addConstraints(k, rl_foot_ag_bound)
    model.addConstraints(k, rr_foot_ag_bound)


def add_cc_to_foot(k, foot_frame):
    model.addContactConstraint(
        k, foot_frame, max_normal_force=MAX_NORMAL_FORCE, eq_tol=FOOT_EQ_TOL
    )
    model.addConstraints(
        k,
        constraints.FrameLinearVelocityBound(
            model, foot_frame, np.zeros(3), np.zeros(3)
        ),
    )


for k in range(0, round(0.08 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.08 * HORIZON), round(0.1 * HORIZON)):
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.1 * HORIZON), round(0.15 * HORIZON)):
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.18 * HORIZON), round(0.23 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")

for k in range(round(0.23 * HORIZON), round(0.27 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.27 * HORIZON), round(0.29 * HORIZON)):
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.29 * HORIZON), round(0.32 * HORIZON)):
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.37 * HORIZON), round(0.39 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")

for k in range(round(0.39 * HORIZON), round(0.43 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")

for k in range(round(0.43 * HORIZON), round(0.47 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.47 * HORIZON), round(0.58 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.58 * HORIZON), round(0.59 * HORIZON)):
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.59 * HORIZON), round(0.61 * HORIZON)):
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.68 * HORIZON), round(0.7 * HORIZON)):
    add_cc_to_foot(k, "FR_foot")

for k in range(round(0.7 * HORIZON), round(0.71 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")

for k in range(round(0.71 * HORIZON), round(0.76 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.76 * HORIZON), round(0.78 * HORIZON)):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(round(0.8 * HORIZON), round(0.81 * HORIZON)):
    add_cc_to_foot(k, "FR_foot")

for k in range(round(0.81 * HORIZON), round(0.83 * HORIZON)):
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")

for k in range(round(0.84 * HORIZON), HORIZON):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")


model.addCost(HORIZON, joint_angle_cost)
model.addCost(HORIZON, joint_vel_cost)

model.addConstraints(
    HORIZON,
    constraints.FrameTranslationBound(model, "base", final_pos, final_pos),
)

model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), True
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FrameRotationBound(
        model, "base", np.eye(3), np.zeros(3), np.zeros(3)
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.max_iter = 100
admm_params.reset_rho = False
admm_params.reset_y = False
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.max_iter = 25
hpipm_params.split_step = True
hpipm_params.prediction_correction = True
hpipm_params.tol_stat = 1e-3
hpipm_params.tol_comp = 1e-3
hpipm_params.tol_eq = 1e-3
hpipm_params.tol_ineq = 1e-3
ssqp_params.hpipmParams = hpipm_params

# Create a solver with the model
solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

solver.write("../data/go2_frog.json")
