import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse
from math import pi

"""
G1 push-ups example
Performs push-ups with the 29 DoF G1, starting and ending in a plank on its
toes and flat palms
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="G1 push-ups trajectory generation"
    )

    parser.add_argument(
        "-n",
        "--num-pushups",
        type=int,
        default=2,
        help="Number of push-ups",
    )

    parser.add_argument(
        "-d",
        "--depth",
        type=float,
        default=0.14,
        help="How far the torso is lowered in each push-up",
    )

    parser.add_argument(
        "-p",
        "--period",
        type=float,
        default=1.6,
        help="Duration of each push-up, in seconds",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/g1_pushups.json",
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

# Model Parameters
dt = 0.02
PUSHUP_STEPS = round(args.period / dt)

# Steps in the plank before the first push-up and after the last, the latter
# longer so the robot comes to rest
START_HOLD_STEPS = 20
END_HOLD_STEPS = 40

H = START_HOLD_STEPS + args.num_pushups * PUSHUP_STEPS + END_HOLD_STEPS

# Weights
V_WGT = 1e-2
U_WGT = 1e-3
F_WGT = 1e-4
BODY_WGT = 1.0
ARM_WGT = 0.1
HEIGHT_WGT = 3.0

# The 29 DoF G1 URDF with contact frames on the palms and stronger, more
# flexible wrists
URDF_PATH = "../../../assets/urdf/g1/g1_29dof_hands.urdf"

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path=URDF_PATH,
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Mirrors a left arm configuration onto the right arm
ARM_MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])

# Plank pose, on the front corners of the feet and the palms, with the body
# straight, the arms almost straight and the palms flat, fingers forward,
# slightly behind the shoulders
base_pos = np.array([0.0, 0.0, 0.2772])
base_quat = np.array([0.0, 0.61758, 0.0, 0.7865])
leg_config = np.array([0.0, 0.0, 0.0, 0.0, 0.5236, 0.0])
waist_config = np.zeros(3)
left_arm_config = np.array(
    [-0.76519, 0.16366, 0.35562, 1.4, 1.71647, -0.11291, 1.97018]
)
plank_config = np.concatenate(
    [
        leg_config,
        leg_config,
        waist_config,
        left_arm_config,
        left_arm_config * ARM_MIRROR,
    ]
)

# Pose at the bottom of each push-up, with the forearms upright and the
# elbows close to the body
bottom_leg_config = np.array([0.0, 0.0, 0.0, 0.0, 0.24203, 0.0])
bottom_left_arm_config = np.array(
    [0.30051, 0.1932, -0.05335, -0.30618, 1.38954, 0.02651, 1.51925]
)
bottom_config = np.concatenate(
    [
        bottom_leg_config,
        bottom_leg_config,
        waist_config,
        bottom_left_arm_config,
        bottom_left_arm_config * ARM_MIRROR,
    ]
)

init = np.zeros(nq + nv)
init[:3] = base_pos
init[3:7] = base_quat
init[7:nq] = plank_config

contact_frames = [
    "left_pp",
    "left_pm",
    "right_pp",
    "right_pm",
    "left_heel_thumb",
    "left_heel_pinky",
    "left_palm_front",
    "right_heel_thumb",
    "right_heel_pinky",
    "right_palm_front",
]

# Initial control, with the robot's weight split evenly between the contacts
# Each contact force is expressed in its own frame
weight = np.linalg.norm(
    model.inverseDynamics(init[:nq], init[nq:], np.zeros(nv))[:3]
)
init_control = np.zeros(nu + 3 * len(contact_frames))
for i, frame in enumerate(contact_frames):
    frame_rot = model.forwardKinematics(init[:nq], frame)[1]
    init_control[nu + 3 * i : nu + 3 * i + 3] = frame_rot.T @ np.array(
        (0.0, 0.0, weight / len(contact_frames))
    )

# Torso height at the top of each push-up
top_height = model.forwardKinematics(init[:nq], "torso_link")[0][2]


def dip(k):
    # Goes from 0 in the plank to 1 at the bottom and back once in each
    # push-up, holding the plank before the first and after the last
    t = k - START_HOLD_STEPS
    if t < 0 or t >= args.num_pushups * PUSHUP_STEPS:
        return 0.0
    phase = 2 * pi * (t % PUSHUP_STEPS) / PUSHUP_STEPS
    return (1 - np.cos(phase)) / 2


# Regularization costs
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

# Keep the legs and waist close to the reference pose, so the body stays
# straight, while guiding the arms more loosely
config_weights = np.full(nu, BODY_WGT)
config_weights[15:] = ARM_WGT
config_cost = costs.SquaredConfigurationErrorCost(
    model, plank_config, config_weights
)

model.addRunningCost(joint_vel_cost)
model.addRunningCost(control_cost)
model.addRunningCost(force_cost)

# Keep the elbows from bending past straight, so the arms bend the right way
elbow_upper = np.full(nu, np.inf)
elbow_upper[[18, 25]] = 1.5
elbow_bound = constraints.JointAngleConstraint(
    model, np.full(nu, -np.inf), elbow_upper
)

for k in range(H):
    model.addConstraints(k, elbow_bound)

    # Follow the torso height reference, with the pose moving between the
    # plank and the bottom pose along with it
    model.addCost(
        k,
        costs.FrameSquaredTranslationErrorCost(
            model,
            "torso_link",
            np.array((0.0, 0.0, top_height - args.depth * dip(k))),
            np.array((0.0, 0.0, HEIGHT_WGT)),
        ),
    )
    model.addCost(
        k,
        costs.SquaredConfigurationErrorCost(
            model,
            plank_config + dip(k) * (bottom_config - plank_config),
            config_weights,
        ),
    )

    for frame in contact_frames:
        model.addContactConstraint(k, frame, eq_tol=0.01)

# End the trajectory at rest in the plank
model.addCost(H, joint_vel_cost)
model.addCost(H, config_cost)

model.addConstraints(
    H,
    constraints.FrameLinearVelocityBound(
        model, "torso_link", np.zeros(3), np.zeros(3)
    ),
)
model.addConstraints(
    H,
    constraints.FloatingBaseAngularVelocityBound(
        model, np.zeros(3), np.zeros(3)
    ),
)
model.addConstraints(
    H, constraints.JointVelocityConstraint(model, np.zeros(nu), np.zeros(nu))
)

model.finalize(init_states=init, init_controls=[init_control] * H)

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

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()
print("Solver state:", state)
solver.write(args.output)
