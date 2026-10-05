import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse

"""
Go2 Handstand example
Balances the Go2 upside down on its front left leg
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 handstand trajectory generation"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_handstand.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

# Model Parameters
HORIZON = 1000
dt = 0.01

# Weights
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-4
C_WGT = 1.0
T_BASE_WGT = 10.0
R_BASE_WGT = 10.0

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

# Handstand pose, with the center of mass directly above the front left foot
# The base is vertical and leans slightly to the side, the front right leg is
# tucked in and the rear legs are raised in the air
base_pos = np.array([0.0, 0.0, 0.46367])
base_quat = np.array([-0.10991, 0.69851, -0.10991, 0.69851])
base_config = np.array(
    [-0.13151, -0.49889, -1.96967] + [0.0, 1.8, -2.6] + [0.0, 0.8, -1.6] * 2
)

# Initial configuration
init = np.zeros(nq + nv)
init[:3] = base_pos
init[3:7] = base_quat
init[7:19] = base_config

base_rot = model.forwardKinematics(init[:nq], "base")[1]

# Initial control, with the front left foot carrying the robot's weight
# The contact force is expressed in the foot frame
weight = np.linalg.norm(
    model.inverseDynamics(init[:nq], init[nq:], np.zeros(nv))[:3]
)
foot_rot = model.forwardKinematics(init[:nq], "FL_foot")[1]

init_control = np.zeros(nu + 3)
init_control[-3:] = foot_rot.T @ np.array((0.0, 0.0, weight))

# Create regularization costs
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

# Costs to hold the handstand pose
config_cost = costs.SquaredConfigurationErrorCost(model, base_config, C_WGT)
base_pose_cost = costs.FrameSquaredPoseErrorCost(
    model, "base", base_pos, base_rot, T_BASE_WGT, R_BASE_WGT
)

# Velocity bound for the front left foot
# Used in conjunction with the contact constraint
fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3)
)

for k in range(HORIZON):
    # Add regularization costs
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)
    model.addCost(k, config_cost)
    model.addCost(k, base_pose_cost)

    # Stand on the front left foot only
    model.addContactConstraint(k, "FL_foot", eq_tol=0.01)
    model.addConstraints(k, fl_foot_vel_bound)

# Add costs at the terminal stage
model.addCost(HORIZON, joint_vel_cost)
model.addCost(HORIZON, config_cost)
model.addCost(HORIZON, base_pose_cost)

# End the trajectory at rest
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)
model.addConstraints(
    HORIZON,
    constraints.FrameAngularVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init, init_controls=[init_control] * HORIZON)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = False
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

# Create a solver with the model
solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

solver.write(args.output)
