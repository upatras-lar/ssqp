import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse
from math import pi, atan, sin, cos

"""
Go2 Standing example
Stands in place with the Go2, optionally on a slope that rises along the x axis
"""

# Model Parameters
HORIZON = 100
dt = 0.01

# Weights
A_WGT = 1.0
V_WGT = 1e-3
U_WGT = 1e-4
F_WGT = 1e-6



def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 standing trajectory generation"
    )

    parser.add_argument(
        "-s",
        "--slope",
        type=float,
        default=0.0,
        help="Terrain slope along the x axis (rise over run), 0 is flat \
            ground",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_standing.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

SLOPE = args.slope

# Standing height of the base above flat ground
STAND_HEIGHT = 0.3879793820122

# Initial configuration
# The base is pitched to be parallel to the terrain and offset from it along the
# terrain normal, with the same leg configuration as when standing on flat ground
slope_angle = atan(SLOPE)
base_pos = STAND_HEIGHT * np.array([-sin(slope_angle), 0.0, cos(slope_angle)])
base_quat = np.array([0.0, -sin(slope_angle / 2), 0.0, cos(slope_angle / 2)])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

init = np.zeros(37)
init[:3] = base_pos
init[3:7] = base_quat
init[7:19] = base_config

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    from_preset=ssqpy.model.Preset.Go2,
    terrain=ssqpy.model.terrain.SlopeTerrain(SLOPE),
    # mjcf_path="../../../assets/mjcf/go2/go2.xml",
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

for k in range(HORIZON):
    # Add regularization costs
    # model.addCost(k, config_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add a standing phase (4 feet on the ground)
    model.addContactConstraint(k, "FL_foot", name="FL foot contact constraint")

    model.addContactConstraint(k, "FR_foot", name="FR foot contact constraint")

    model.addContactConstraint(k, "RL_foot", name="RL foot contact constraint")

    model.addContactConstraint(k, "RR_foot", name="RR foot contact constraint")

model.addCost(HORIZON, config_cost)
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
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

print("Solver state:", state)

solver.write(args.output)
