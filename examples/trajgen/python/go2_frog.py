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
HORIZON = 100
dt = 0.03

# Weights
A_WGT = 1e-3
V_WGT = 1e-3
U_WGT = 1e-4
F_WGT = 1e-4

MAX_NORMAL_FORCE = 145
FOOT_EQ_TOL = 0.02

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

fl_foot_ag_bound = constraints.FramePositionBound(
    model,
    "FL_foot",
    np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
    np.array((np.inf, np.inf, np.inf)),
)

fr_foot_ag_bound = constraints.FramePositionBound(
    model,
    "FR_foot",
    np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
    np.array((np.inf, np.inf, np.inf)),
)

rl_foot_ag_bound = constraints.FramePositionBound(
    model,
    "RL_foot",
    np.array((-np.inf, -np.inf, -FOOT_EQ_TOL)),
    np.array((np.inf, np.inf, np.inf)),
)

rr_foot_ag_bound = constraints.FramePositionBound(
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

fl_foot_pos_bound = constraints.FramePositionBound(
    model,
    "FL_foot",
    np.array((-np.inf, 0.1, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
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


for k in range(0, 10):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(10, 12):
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(12, 18):
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(18, 22):
    pass

for k in range(22, 28):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")

for k in range(28, 33):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(33, 35):
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(35, 39):
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(39, 45):
    pass

for k in range(45, 47):
    add_cc_to_foot(k, "FL_foot")

for k in range(47, 52):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")

for k in range(52, 57):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(57, 70):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(70, 71):
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(71, 74):
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")

for k in range(74, 82):
    pass

for k in range(82, 84):
    add_cc_to_foot(k, "FR_foot")

for k in range(84, 85):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")

for k in range(85, 100):
    add_cc_to_foot(k, "FL_foot")
    add_cc_to_foot(k, "FR_foot")
    add_cc_to_foot(k, "RL_foot")
    add_cc_to_foot(k, "RR_foot")


model.addCost(HORIZON, joint_vel_cost)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(model, "base", final_pos, final_pos),
)

model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), True
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
# ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
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
