import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse
from math import pi

"""
Go2 Jump example
Performs a simple jump with the Go2 robot
"""

# Model Parameters
HORIZON = 150
dt = 0.01



def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 jump trajectory generation"
    )

    parser.add_argument(
        "--start-step",
        type=int,
        default=60,
        help="Step at which the robot leaves the ground",
    )

    parser.add_argument(
        "--end-step",
        type=int,
        default=90,
        help="Step at which the robot lands",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_jump.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

START_STEP = args.start_step
END_STEP = args.end_step

# Weights
A_WGT = 1e-3
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-3
C_WGT = 1e-3

MAX_NORMAL_FORCE = 100

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

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3879793820122])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

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
config_cost = costs.SquaredConfigurationErrorCost(model, base_config, C_WGT)

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

head_constraint = constraints.FrameTranslationBound(
    model,
    "Head_lower",
    np.array((-np.inf, -np.inf, 0.1)),
    np.array((np.inf, np.inf, np.inf)),
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

    model.addConstraints(k, head_constraint)

    if k < START_STEP or k >= END_STEP:
        t1 = model.addContactConstraint(
            k, "FL_foot", max_normal_force=MAX_NORMAL_FORCE, eq_tol=0.01
        )
        t1.set_enabled(False)
        model.addConstraints(k, fl_foot_vel_bound)
        fl_foot_vel_bound.set_enabled(False)

        t2 = model.addContactConstraint(
            k, "FR_foot", max_normal_force=MAX_NORMAL_FORCE, eq_tol=0.01
        )
        # t2.set_enabled(False)
        model.addConstraints(k, fr_foot_vel_bound)

        t3 = model.addContactConstraint(
            k, "RL_foot", max_normal_force=MAX_NORMAL_FORCE, eq_tol=0.01
        )
        # t3.set_enabled(False)
        model.addConstraints(k, rl_foot_vel_bound)

        t4 = model.addContactConstraint(
            k, "RR_foot", max_normal_force=MAX_NORMAL_FORCE, eq_tol=0.01
        )
        # t4.set_enabled(False)
        model.addConstraints(k, rr_foot_vel_bound)

        init_controls.append(init_control)
        continue

    init_controls.append(np.zeros(nu))

model.addCost(HORIZON, joint_vel_cost)

model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)

model.addConstraints(HORIZON, head_constraint)

model.addConstraints(
    HORIZON,
    constraints.FrameTranslationBound(
        model,
        "FL_foot",
        np.array((0.1934, 0.142, -0.00002682)),
        np.array((0.1934, 0.142, -0.00002682)),
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FrameTranslationBound(
        model,
        "FR_foot",
        np.array((0.1934, -0.142, -0.00002682)),
        np.array((0.1934, -0.142, -0.00002682)),
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FrameTranslationBound(
        model,
        "RL_foot",
        np.array((-0.1934, 0.142, -0.00002682)),
        np.array((-0.1934, 0.142, -0.00002682)),
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FrameTranslationBound(
        model,
        "RR_foot",
        np.array((-0.1934, -0.142, -0.00002682)),
        np.array((-0.1934, -0.142, -0.00002682)),
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = True
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.max_iter = 25
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

solver.write(args.output)
