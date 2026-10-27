import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse

"""
Talos Walking example
Performs a walk with the Talos humanoid robot
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Talos walking trajectory generation"
    )

    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        default=2.0,
        help="Distance to walk along the x axis",
    )

    parser.add_argument(
        "--foot-swing-steps",
        type=int,
        default=40,
        help="Number of steps for each foot swing",
    )

    parser.add_argument(
        "--final-standing-steps",
        type=int,
        default=20,
        help="Number of steps to stand upright for at the end of the \
            trajectory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/talos_walk.json",
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
H = 200
dt = 0.02

# Weights
V_WGT = 3e-2
U_WGT = 1e-3
F_WGT = 1e-4
TF_WGT = 3.0
RF_WGT = 3.0

# Number of steps to stand upright for at the end of the trajectory
FINAL_STANDING_STEPS = args.final_standing_steps

# Number of steps for each swing
FOOT_SWING_STEPS = args.foot_swing_steps

# Create an SSQP model for Talos
model = ssqpy.model.Model(
    H,
    dt,
    from_preset=ssqpy.model.Preset.Talos,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Initial position
base_pos = np.array((0, 0, 1.0611))

# Final position
final_pos = np.array((args.distance, 0.0, 1.0611))

# Initial configuration
init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0
init[7:51] = [
    # Left leg
    0.0,  # leg_left_1_joint  (hip yaw)
    0.0,  # leg_left_2_joint  (hip roll)
    -0.25,  # leg_left_3_joint  (hip pitch)
    0.50,  # leg_left_4_joint  (knee)
    -0.25,  # leg_left_5_joint  (ankle pitch)
    0.0,  # leg_left_6_joint  (ankle roll)
    # Right leg
    0.0,  # leg_right_1_joint
    0.0,  # leg_right_2_joint
    -0.25,  # leg_right_3_joint
    0.50,  # leg_right_4_joint
    -0.25,  # leg_right_5_joint
    0.0,  # leg_right_6_joint
    # Torso
    0.0,  # torso_1_joint (yaw)
    0.05,  # torso_2_joint (pitch)
    # Left arm
    0.0,  # arm_left_1_joint
    0.15,  # arm_left_2_joint
    0.30,  # arm_left_3_joint
    -0.60,  # arm_left_4_joint
    0.0,  # arm_left_5_joint
    0.0,  # arm_left_6_joint
    0.0,  # arm_left_7_joint
    # Left gripper
    0.0,  # gripper_left_inner_double_joint
    0.0,  # gripper_left_fingertip_1_joint
    0.0,  # gripper_left_fingertip_2_joint
    0.0,  # gripper_left_inner_single_joint
    0.0,  # gripper_left_fingertip_3_joint
    0.0,  # gripper_left_joint
    0.0,  # gripper_left_motor_single_joint
    # Right arm (mirrored)
    0.0,  # arm_right_1_joint
    -0.15,  # arm_right_2_joint
    0.30,  # arm_right_3_joint
    -0.60,  # arm_right_4_joint
    0.0,  # arm_right_5_joint
    0.0,  # arm_right_6_joint
    0.0,  # arm_right_7_joint
    # Right gripper
    0.0,  # gripper_right_inner_double_joint
    0.0,  # gripper_right_fingertip_1_joint
    0.0,  # gripper_right_fingertip_2_joint
    0.0,  # gripper_right_inner_single_joint
    0.0,  # gripper_right_fingertip_3_joint
    0.0,  # gripper_right_joint
    0.0,  # gripper_right_motor_single_joint
    # Head
    0.0,  # head_1_joint
    0.0,  # head_2_joint
]


# Initial forces for 2 feet on the ground
init_control = np.zeros(nu + 24)
for f in range(1, 9):
    init_control[nu + 3 * f - 1] = 116

# Initial forces for 1 foot on the ground
init_control_left = np.zeros(nu + 12)
init_control_right = np.zeros(nu + 12)
for f in range(1, 5):
    init_control_left[nu + 3 * f - 1] = 232
    init_control_right[nu + 3 * f - 1] = 232

# Create cost to move robot to final position
pose_cost = costs.FrameSquaredPoseErrorCost(
    model,
    "torso_1_link",
    final_pos,
    np.eye(3),
    TF_WGT,
    RF_WGT,
)

# Create regularization costs
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

init_controls = []

for k in range(H - FINAL_STANDING_STEPS):
    # Add regularization costs
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add contact for the appropriate foot
    if (k // FOOT_SWING_STEPS) % 2 == 0:
        model.addContactConstraint(k, "left_mm", eq_tol=0.01)
        model.addContactConstraint(k, "left_mp", eq_tol=0.01)
        model.addContactConstraint(k, "left_pp", eq_tol=0.01)
        model.addContactConstraint(k, "left_pm", eq_tol=0.01)

        init_controls.append(init_control_left)
    else:
        model.addContactConstraint(k, "right_mm", eq_tol=0.01)
        model.addContactConstraint(k, "right_mp", eq_tol=0.01)
        model.addContactConstraint(k, "right_pp", eq_tol=0.01)
        model.addContactConstraint(k, "right_pm", eq_tol=0.01)

        init_controls.append(init_control_right)

# Add final standing phase
for k in range(H - FINAL_STANDING_STEPS, H):
    # Add regularization costs
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add contacts for both feet
    model.addContactConstraint(k, "left_mm", eq_tol=0.01)
    model.addContactConstraint(k, "left_mp", eq_tol=0.01)
    model.addContactConstraint(k, "left_pp", eq_tol=0.01)
    model.addContactConstraint(k, "left_pm", eq_tol=0.01)

    model.addContactConstraint(k, "right_mm", eq_tol=0.01)
    model.addContactConstraint(k, "right_mp", eq_tol=0.01)
    model.addContactConstraint(k, "right_pp", eq_tol=0.01)
    model.addContactConstraint(k, "right_pm", eq_tol=0.01)

    init_controls.append(init_control)

    # Add constraints to keep the base and head still at the end
    model.addConstraints(
        k,
        constraints.FrameLinearVelocityBound(
            model, "base_link", np.zeros(3), np.zeros(3)
        ),
    )
    model.addConstraints(
        k,
        constraints.FrameLinearVelocityBound(
            model, "head_1_link", np.zeros(3), np.zeros(3)
        ),
    )

# Add terminal stage costs & constraints
model.addCost(H, joint_vel_cost)
model.addCost(H, pose_cost)
model.addConstraints(
    H,
    constraints.FrameLinearVelocityBound(
        model, "base_link", np.zeros(3), np.zeros(3)
    ),
)
model.addConstraints(
    k,
    constraints.FrameLinearVelocityBound(
        model, "head_1_link", np.zeros(3), np.zeros(3)
    ),
)

model.finalize(init_states=init, init_controls=init_controls)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = False
ssqp_params.filter_size = 20
ssqp_params.tolerance = 5e-2
ssqp_params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 5e-3
admm_params.rel_tolerance = 5e-3
admm_params.max_iter = 100
admm_params.reset_rho = False
admm_params.reset_y = False
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-2
hpipm_params.tol_comp = 1e-2
hpipm_params.tol_eq = 1e-2
hpipm_params.tol_ineq = 1e-2
ssqp_params.hpipmParams = hpipm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()
print("Solver state:", state)
solver.write(args.output)
