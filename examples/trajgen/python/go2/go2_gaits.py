import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse
from math import pi, atan, sin, cos
from utils.GaitManager import GaitManager

"""
Go2 Gaits example
Performs different types of gaits (walking/running styles) with the Go2 robot,
optionally on a slope that rises along the x axis
The `GaitManager` class is used to handle gait creation
"""

# Model Parameters
HORIZON = 300
dt = 0.01

# Weights
C_WGT = 1e-1
V_WGT = 1e-2
U_WGT = 1e-3
F_WGT = 1e-5
ROT_WGT = 1e-1
TF_WGT = 5.0

GAITS = {
    "walk": GaitManager.setWalk,
    "amble": GaitManager.setAmble,
    "trot": GaitManager.setTrot,
    "pace": GaitManager.setPace,
    "canter": GaitManager.setCanter,
    "gallop": GaitManager.setGallop,
    "rotatory-gallop": GaitManager.setRotatoryGallop,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 gaits trajectory generation"
    )

    parser.add_argument(
        "-g",
        "--gait",
        choices=list(GAITS),
        default="walk",
        help="Which gait to use",
    )

    parser.add_argument(
        "-c",
        "--cycle-steps",
        type=int,
        default=None,
        help="How many steps to take per gait cycle, by default the cycle \
            step value of the gait being used",
    )

    parser.add_argument(
        "-f",
        "--final-standing-steps",
        type=int,
        default=30,
        help="The number of steps to stand on all 4 feet for at the end of \
            the trajectory",
    )

    parser.add_argument(
        "-s",
        "--slope",
        type=float,
        default=0.2,
        help="Terrain slope along the x axis (rise over run), 0 is flat \
            ground",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_gaits.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

GAIT = GAITS[args.gait]
CYCLE_STEPS = args.cycle_steps
FINAL_STANDING_STEPS = args.final_standing_steps
SLOPE = args.slope

# Standing height of the base above the terrain
STAND_HEIGHT = 0.3879793820122

# The base stands parallel to the terrain, offset from it along its normal
slope_angle = atan(SLOPE)
terrain_normal = np.array([-sin(slope_angle), 0.0, cos(slope_angle)])
terrain_pitch = np.array(
    (
        (cos(slope_angle), 0.0, -sin(slope_angle)),
        (0.0, 1.0, 0.0),
        (sin(slope_angle), 0.0, cos(slope_angle)),
    )
)


def standing_position(x):
    return np.array([x, 0.0, SLOPE * x]) + STAND_HEIGHT * terrain_normal


# Initial and final positions
base_pos = standing_position(-1.0)
final_pos = standing_position(1.0)
final_yaw = 0.0
final_R = terrain_pitch @ np.array(
    (
        (np.cos(final_yaw), -np.sin(final_yaw), 0.0),
        (np.sin(final_yaw), np.cos(final_yaw), 0.0),
        (0.0, 0.0, 1.0),
    )
)

# Initial configuration
init = np.zeros(37)
init[:3] = base_pos
init[3:7] = [0.0, -sin(slope_angle / 2), 0.0, cos(slope_angle / 2)]
init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    from_preset=ssqpy.model.Preset.Go2,
    terrain=ssqpy.model.terrain.SlopeTerrain(SLOPE),
    # solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Create regularization costs
config_cost = costs.SquaredConfigurationErrorCost(model, init[7:19], C_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

for k in range(HORIZON):
    # Add the regularization costs for each stage
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add a cost to keep the orientation of the robot roughly parallel to the
    # terrain
    model.addCost(
        k,
        costs.FrameSquaredRotationErrorCost(
            model, "base", terrain_pitch, ROT_WGT
        ),
    )


for k in range(HORIZON + 1):
    model.addCost(k, config_cost)

# Add a velocity regularization cost for the final stage
model.addCost(HORIZON, costs.SquaredJointVelocityCost(model, 1e3))

# Add a cost for the final translation of the base to be that of the final
# position
model.addCost(
    HORIZON,
    costs.FrameSquaredPoseErrorCost(
        model, "base", final_pos, final_R, TF_WGT, TF_WGT
    ),
)

# Add a bound such that the base has no velocity at the end of the trajectory
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)

# List of initial controls
init_controls = []

# Create a `GaitManager`, passing the model, the configuration and velocity
# sizes and the ordered frames for each of the 4 feet
gait_manager = GaitManager(
    model,
    nq,
    nv,
    ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    max_normal_force=80,
    contact_equality_tolerance=0.01,
    swing_cost_height=0.1,
)

# Add a gait of the specified type for the range [0, H - FSS - 1]
# Gait methods will return the initial controls as a list, so we extend
# `init_controls` by their return value
if CYCLE_STEPS is None:
    init_controls.extend(
        GAIT(gait_manager, 0, HORIZON - FINAL_STANDING_STEPS - 1)
    )
else:
    init_controls.extend(
        GAIT(
            gait_manager,
            0,
            HORIZON - FINAL_STANDING_STEPS - 1,
            cycle_steps=CYCLE_STEPS,
        )
    )

# Add standing stages at the end, depending on the FINAL_STANDING_STEPS value
for k in range(HORIZON - FINAL_STANDING_STEPS, HORIZON):
    init_controls.extend(gait_manager.addStandingStage(k))

# Finalize the model
model.finalize(init_states=init)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = True
ssqp_params.tolerance = 1e-2
# ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.max_iter = 200
admm_params.reset_rho = False
admm_params.reset_y = False
admm_params.warm_start = True
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

# Create an SSQP solver for the model
solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

print("Solver state:", state)

solver.write(args.output)
