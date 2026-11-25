import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse
from math import pi

"""
Go2 Frontflip example
Performs a forward flip with the Go2 robot, tipping forward over its front feet
before pushing off, rotating once about the pitch axis and landing back on its
feet
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 frontflip trajectory generation"
    )

    parser.add_argument(
        "--rear-lead",
        type=int,
        default=10,
        help="Steps the rear feet leave the ground before the front feet, \
            negative for the front feet to leave first",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_frontflip.json",
        help="Path of the output JSON file",
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm", "piqp"],
        default="hpipm",
        help="QP solver used at each SSQP iteration",
    )

    return parser.parse_args()


args = parse_args()

# The first pair of feet leaves the ground at a fixed step and the other pair
# follows after the rear lead
FIRST_TAKEOFF_STEP = 55
REAR_TAKEOFF_STEP = FIRST_TAKEOFF_STEP + max(-args.rear_lead, 0)
FRONT_TAKEOFF_STEP = FIRST_TAKEOFF_STEP + max(args.rear_lead, 0)
LAST_TAKEOFF_STEP = max(REAR_TAKEOFF_STEP, FRONT_TAKEOFF_STEP)

# The robot is in the air for a fixed time, then the rear feet land first and
# the front feet a few steps later
REAR_LANDING_STEP = LAST_TAKEOFF_STEP + 50
FRONT_LANDING_STEP = REAR_LANDING_STEP + 5

# Model Parameters
HORIZON = FRONT_LANDING_STEP + 60
dt = 0.01

# Weights
A_WGT = 1e-3
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-4
C_WGT = 1e-2

MAX_NORMAL_FORCE = 250

# Allowed pitch deviation from the reference rotation
PITCH_MARGIN = 1.0

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

# Standing pose
base_pos = np.array([0.0, 0.0, 0.3879793820122])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)


def pitch_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def reference_pitch(k):
    # Turns once forward at a constant rate, from when the first feet leave the
    # ground until the front feet land
    t = (k - FIRST_TAKEOFF_STEP) / (FRONT_LANDING_STEP - FIRST_TAKEOFF_STEP)
    return 2 * pi * min(max(t, 0.0), 1.0)


# Initial guess, with the base following the reference pitch and a ballistic
# arc in the air
flight_time = (REAR_LANDING_STEP - LAST_TAKEOFF_STEP) * dt
vz_takeoff = 9.81 * flight_time / 2

init_states = []
for k in range(HORIZON + 1):
    pitch = reference_pitch(k)
    pitch_rate = (reference_pitch(k + 1) - pitch) / dt

    t = (k - LAST_TAKEOFF_STEP) * dt
    height = base_pos[2]
    vz = 0.0
    if LAST_TAKEOFF_STEP <= k < REAR_LANDING_STEP:
        height += vz_takeoff * t - 9.81 * t**2 / 2
        vz = vz_takeoff - 9.81 * t

    state = np.zeros(nq + nv)
    state[2] = height
    state[3:7] = np.array((0.0, np.sin(pitch / 2), 0.0, np.cos(pitch / 2)))
    state[7:19] = base_config

    # Base velocities are expressed in the base frame
    state[nq : nq + 3] = pitch_rotation(pitch).T @ np.array((0.0, 0.0, vz))
    state[nq + 4] = pitch_rate
    init_states.append(state)


def contact_feet(k):
    feet = []
    if k < FRONT_TAKEOFF_STEP or k >= FRONT_LANDING_STEP:
        feet += ["FL_foot", "FR_foot"]
    if k < REAR_TAKEOFF_STEP or k >= REAR_LANDING_STEP:
        feet += ["RL_foot", "RR_foot"]
    return feet


# Regularization costs
joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)
config_cost = costs.SquaredConfigurationErrorCost(model, base_config, C_WGT)

model.addRunningCost(joint_angle_cost)
model.addRunningCost(joint_vel_cost)
model.addRunningCost(control_cost)
model.addRunningCost(force_cost)
model.addRunningCost(config_cost)

# Keep the feet above the ground and apart from each other
foot_bounds = [
    constraints.FrameTranslationBound(
        model,
        foot,
        np.array((-np.inf, y_min, -0.01)),
        np.array((np.inf, y_max, np.inf)),
    )
    for foot, y_min, y_max in [
        ("FL_foot", 0.1, np.inf),
        ("FR_foot", -np.inf, -0.1),
        ("RL_foot", 0.1, np.inf),
        ("RR_foot", -np.inf, -0.1),
    ]
]

# Keep the head and the knees above the ground
body_bounds = [
    constraints.FrameTranslationBound(
        model,
        frame,
        np.array((-np.inf, -np.inf, 0.05)),
        np.array((np.inf, np.inf, np.inf)),
    )
    for frame in ["Head_lower", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
]

# Keep the base from crouching too low before takeoff and after landing
base_height_bound = constraints.FrameTranslationBound(
    model,
    "base",
    np.array((-np.inf, -np.inf, 0.15)),
    np.array((np.inf, np.inf, np.inf)),
)

init_controls = []
for k in range(HORIZON):
    for bound in foot_bounds:
        model.addConstraints(k, bound)
    for bound in body_bounds:
        model.addConstraints(k, bound)
    model.addConstraints(k, base_height_bound)

    # Keep the base near the reference pitch, so that it turns forward, and
    # stop it from rolling or yawing
    model.addConstraints(
        k,
        constraints.FloatingBaseRotationBound(
            model,
            pitch_rotation(reference_pitch(k)),
            np.array((-0.2, -PITCH_MARGIN, -0.2)),
            np.array((0.2, PITCH_MARGIN, 0.2)),
        ),
    )

    feet = contact_feet(k)
    for foot in feet:
        model.addContactConstraint(
            k, foot, max_normal_force=MAX_NORMAL_FORCE, eq_tol=0.01
        )

    # Split the robot's weight evenly between the feet in contact
    init_control = np.zeros(nu + 3 * len(feet))
    init_control[nu + 2 :: 3] = 15.0 * 9.81 / max(len(feet), 1)
    init_controls.append(init_control)

# End the trajectory standing still, facing forward again
model.addCost(HORIZON, joint_vel_cost)
model.addCost(HORIZON, config_cost)

for bound in foot_bounds:
    model.addConstraints(HORIZON, bound)

model.addConstraints(
    HORIZON,
    constraints.FloatingBaseTranslationBound(
        model,
        np.array((-np.inf, -np.inf, base_pos[2])),
        np.array((np.inf, np.inf, base_pos[2])),
    ),
)

model.addConstraints(
    HORIZON,
    constraints.FloatingBaseRotationBound(
        model, np.eye(3), np.zeros(3), np.zeros(3)
    ),
)
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)

# Finalize the model with our state and controls
model.finalize(init_states=init_states, init_controls=init_controls)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = True
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)

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

print("Solver state:", state)

solver.write(args.output)
