import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

from trajgen.python.utils.GaitManager import GaitManager

"""
Go2 Gaits example
Performs different types of gaits (walking/running styles) with the Go2 robot
The `GaitManager` class is used to handle gait creation
"""

# Model Parameters
HORIZON = 600
dt = 0.01

# Weights
C_WGT = 1e-1
V_WGT = 1e-2
U_WGT = 1e-3
F_WGT = 1e-5
ROT_WGT = 1.0
TF_WGT = 5.0

# @script-parameter
# Which gait to use
# Available options:
#  - `setWalk`
#  - `setAmble`
#  - `setTrot`
#  - `setPace`
#  - `setCanter`
#  - `setGallop`
#  - `setRotatoryGallop`
GAIT = GaitManager.setWalk

# @script-parameter
# How many steps to take per gait cycle
# With `None`, the default cycle step value for the gait being used is used
CYCLE_STEPS = 120

# @script-parameter
# The number of steps to stand on all 4 feet for at the end of the trajectory
FINAL_STANDING_STEPS = 10

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.389])
final_pos = np.array([1.0, 0.0, 0.389])

# Initial configuration
init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    mjcf_path="../../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.ForwardDynamics,
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

    # Add a cost to keep the orientation of the robot roughly at the base
    # orientation
    model.addCost(
        k,
        costs.FrameSquaredRotationErrorCost(
            model, "base_link", np.eye(3), ROT_WGT
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
    costs.FrameSquaredTranslationErrorCost(
        model, "base_link", final_pos, TF_WGT
    ),
)

# Add a bound such that the base has no velocity at the end of the trajectory
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base_link", np.zeros(3), np.zeros(3), True
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
    swing_cost_height=0.04,
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

# Create an SSQP solver for the model
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-3,
    admm_max_iter=500,
    admm_warm_start=False,
)

state, x, u = solver.solve()

solver.write("tmp.json")

print("x:")
x.print()
print("u:")
u.print()
