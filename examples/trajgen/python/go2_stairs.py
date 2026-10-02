import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from utils.GaitManager import GaitManager

"""
Go2 Stairs example
Walks the Go2 up a few steps along the x axis
"""

# Model Parameters
HORIZON = 500
dt = 0.01

# Weights
C_WGT = 1e-1
V_WGT = 1e-2
U_WGT = 1e-3
F_WGT = 1e-5
ROT_WGT = 1e-1
TF_WGT = 5.0

# @script-parameter
# Which gait to use
# Available options:
#  - `setWalk`
#  - `setAmble`
#  - `setTrot`
GAIT = GaitManager.setWalk

# @script-parameter
# The number of steps to stand on all 4 feet for at the end of the trajectory
FINAL_STANDING_STEPS = 40

# @script-parameter
# Number of steps, their height and length (along x) and the position of the
# first one along the x axis
NUM_STEPS = 6
STEP_HEIGHT = 0.05
STEP_LENGTH = 0.2
STEP_X = 0.2

# @script-parameter
# Standing height of the base above the terrain
# A lower base leaves the legs more bent, giving the feet more room to lift
# (the Go2 stands at about 0.388 with its default leg configuration)
STAND_HEIGHT = 0.3

terrain = ssqpy.model.terrain.StepTerrain(
    STEP_X, STEP_HEIGHT, STEP_LENGTH, NUM_STEPS
)

# Create an SSQP model for the Go2 on the stairs
model = ssqpy.model.Model(
    HORIZON,
    dt,
    from_preset=ssqpy.model.Preset.Go2,
    terrain=terrain,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()


def leg_configuration(thigh_angle):
    # Joint angles for the 4 legs, with each foot kept below its hip
    return np.array([0.0, thigh_angle, -2 * thigh_angle] * 4)


def height_above_feet(thigh_angle):
    # Height of the base above the feet for a given thigh angle
    q = np.zeros(nq)
    q[6] = 1.0
    q[7:19] = leg_configuration(thigh_angle)
    return -model.forwardKinematics(q, "FL_foot")[0][2]


# Find the thigh angle that puts the base at the standing height (the height
# decreases as the thigh angle increases)
lo, hi = 0.0, 1.2
for _ in range(50):
    mid = (lo + hi) / 2
    if height_above_feet(mid) > STAND_HEIGHT:
        lo = mid
    else:
        hi = mid
base_config = leg_configuration((lo + hi) / 2)

# Initial and final positions
# The base ends on top of the stairs
base_pos = np.array([-0.6, 0.0, STAND_HEIGHT])
final_pos = np.array(
    [
        STEP_X + NUM_STEPS * STEP_LENGTH + 0.4,
        0.0,
        NUM_STEPS * STEP_HEIGHT + STAND_HEIGHT,
    ]
)

# Initial configuration
init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = base_config

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

    # Add a cost to keep the base level
    model.addCost(
        k,
        costs.FrameSquaredRotationErrorCost(model, "base", np.eye(3), ROT_WGT),
    )

for k in range(HORIZON + 1):
    model.addCost(k, config_cost)

# Add a velocity regularization cost for the final stage
model.addCost(HORIZON, costs.SquaredJointVelocityCost(model, 1e3))

# Add a cost for the final pose of the base
model.addCost(
    HORIZON,
    costs.FrameSquaredPoseErrorCost(
        model, "base", final_pos, np.eye(3), TF_WGT, TF_WGT
    ),
)

# Add a bound such that the base has no velocity at the end of the trajectory
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)

# Create a `GaitManager`, passing the model, the configuration and velocity
# sizes and the ordered frames for each of the 4 feet
gait_manager = GaitManager(
    model,
    nq,
    nv,
    ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    max_normal_force=80,
    contact_equality_tolerance=0.01,
    swing_cost_height=0.15,
)

# Add the gait, followed by standing stages at the end
GAIT(gait_manager, 0, HORIZON - FINAL_STANDING_STEPS - 1)

for k in range(HORIZON - FINAL_STANDING_STEPS, HORIZON):
    gait_manager.addStandingStage(k)

# Finalize the model
model.finalize(init_states=init)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = True
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

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

solver.write("../data/go2_stairs.json")
