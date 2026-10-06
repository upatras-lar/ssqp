import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse

"""
Talos Jump example
Performs an in-place jump with the Talos robot
"""

# Model Parameters
H = 120
dt = 0.01



def parse_args():
    parser = argparse.ArgumentParser(
        description="Talos jump trajectory generation"
    )

    parser.add_argument(
        "--steps-before",
        type=int,
        default=45,
        help="Number of steps before the jump",
    )

    parser.add_argument(
        "--steps-after",
        type=int,
        default=45,
        help="Number of steps after the jump",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/talos_jump.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

STEPS_BEFORE = args.steps_before
STEPS_AFTER = args.steps_after

# Weights
A_WGT = 1e-1
V_WGT = 1e-3
U_WGT = 1e-3
F_WGT = 1e-4
T_WGT = 4.4
R_WGT = 3.0

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
base_pos = np.array((0, 0, 0.96))

# Initial configuration
init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0
init[7:51] = [
    # Left leg
    0.0,  # leg_left_1_joint
    0.0,  # leg_left_2_joint
    -0.6,  # leg_left_3_joint
    1.2,  # leg_left_4_joint
    -0.6,  # leg_left_5_joint
    0.0,  # leg_left_6_joint
    # Right leg
    0.0,  # leg_right_1_joint
    0.0,  # leg_right_2_joint
    -0.6,  # leg_right_3_joint
    1.2,  # leg_right_4_joint
    -0.6,  # leg_right_5_joint
    0.0,  # leg_right_6_joint
    # Torso
    0.0,  # torso_1_joint
    0.1,  # torso_2_joint
    # Left arm
    0.0,  # arm_left_1_joint
    0.2,  # arm_left_2_joint
    0.4,  # arm_left_3_joint
    -0.8,  # arm_left_4_joint
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
    # Right arm
    0.0,  # arm_right_1_joint
    -0.2,  # arm_right_2_joint
    0.4,  # arm_right_3_joint
    -0.8,  # arm_right_4_joint
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

# Initial forces
init_control = np.zeros(nu + 24)
for f in range(1, 9):
    init_control[nu + 3 * f - 1] = 116

# To make convergence easier, we would like to provide more initialization for
# the controls than just initial force values
# To do this, we will use the initial configuration and force values to perform
# inverse dynamics, calculating initial joint torques to use as initial controls

# First, we create a dictionary
# For each force frame as a key, we provide a list of vectors (3D forces) for
# its value
# forces = {}
# forces["left_mm"] = [init_control[nv : nv + 3]]
# forces["left_mp"] = [init_control[nv + 3 : nv + 6]]
# forces["left_pp"] = [init_control[nv + 6 : nv + 9]]
# forces["left_pm"] = [init_control[nv + 9 : nv + 12]]
# forces["right_mm"] = [init_control[nv + 12 : nv + 15]]
# forces["right_mp"] = [init_control[nv + 15 : nv + 18]]
# forces["right_pp"] = [init_control[nv + 18 : nv + 21]]
# forces["right_pm"] = [init_control[nv + 21 : nv + 24]]

# We then use the `Model` object we have created to perform inverse dynamics
# This is valid because the model was initialized with a URDF
# We pass the initial configuration, velocity and 0 acceleration, as well as
# the above force dictionary
# The output will be joint torques, which we store in our initial controls
# vector, at the torque part
# init_control[6:nv] = model.inverseDynamics(
#     init[:nq], init[nq:], np.zeros(nv), forces
# )[-nv + 6 :]

# Create a cost for the torso to remain at the base position
pose_cost = costs.FrameSquaredPoseErrorCost(
    model,
    "torso_1_link",
    base_pos,
    np.eye(3),
    T_WGT,
    R_WGT,
)

# Create regularization costs
joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

# Create velocity bounds for feet frames that are in contact
left_mm_vel_bound = constraints.FrameLinearVelocityBound(
    model, "left_mm", np.zeros(3), np.zeros(3)
)
left_mp_vel_bound = constraints.FrameLinearVelocityBound(
    model, "left_mp", np.zeros(3), np.zeros(3)
)
left_pp_vel_bound = constraints.FrameLinearVelocityBound(
    model, "left_pp", np.zeros(3), np.zeros(3)
)
left_pm_vel_bound = constraints.FrameLinearVelocityBound(
    model, "left_pm", np.zeros(3), np.zeros(3)
)

right_mm_vel_bound = constraints.FrameLinearVelocityBound(
    model, "right_mm", np.zeros(3), np.zeros(3)
)
right_mp_vel_bound = constraints.FrameLinearVelocityBound(
    model, "right_mp", np.zeros(3), np.zeros(3)
)
right_pp_vel_bound = constraints.FrameLinearVelocityBound(
    model, "right_pp", np.zeros(3), np.zeros(3)
)
right_pm_vel_bound = constraints.FrameLinearVelocityBound(
    model, "right_pm", np.zeros(3), np.zeros(3)
)

init_controls = []

for k in range(H):
    # Add regularization costs
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # If on the ground
    if k < STEPS_BEFORE or k > H - STEPS_AFTER:
        # Add contact and velocity constraints for feet frames
        model.addContactConstraint(
            k, "left_mm", max_normal_force=900, eq_tol=0.01
        )
        model.addContactConstraint(
            k, "left_mp", max_normal_force=900, eq_tol=0.01
        )
        model.addContactConstraint(
            k, "left_pp", max_normal_force=900, eq_tol=0.01
        )
        model.addContactConstraint(
            k, "left_pm", max_normal_force=900, eq_tol=0.01
        )
        model.addConstraints(k, left_mm_vel_bound)
        model.addConstraints(k, left_mp_vel_bound)
        model.addConstraints(k, left_pp_vel_bound)
        model.addConstraints(k, left_pm_vel_bound)

        model.addContactConstraint(
            k, "right_mm", max_normal_force=900, eq_tol=0.01
        )
        model.addContactConstraint(
            k, "right_mp", max_normal_force=900, eq_tol=0.01
        )
        model.addContactConstraint(
            k, "right_pp", max_normal_force=900, eq_tol=0.01
        )
        model.addContactConstraint(
            k, "right_pm", max_normal_force=900, eq_tol=0.01
        )
        model.addConstraints(k, right_mm_vel_bound)
        model.addConstraints(k, right_mp_vel_bound)
        model.addConstraints(k, right_pp_vel_bound)
        model.addConstraints(k, right_pm_vel_bound)

        init_controls.append(init_control)
    else:
        init_controls.append(np.zeros(nu))

# Add regularization cost
model.addCost(H, joint_vel_cost)

# Add cost for torso to remain at initial position
model.addCost(H, pose_cost)
# Add velocity bound for torso to stay still
model.addConstraints(
    H,
    constraints.FrameLinearVelocityBound(
        model, "torso_1_link", np.zeros(3), np.zeros(3)
    ),
)

model.finalize(init_states=init, init_controls=init_controls)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = False
ssqp_params.filter_size = 20
ssqp_params.tolerance = 5e-2
# ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 5e-3
admm_params.rel_tolerance = 5e-3
admm_params.max_iter = 100
admm_params.reset_rho = False
admm_params.reset_y = False
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()
print("Solver state:", state)
solver.write(args.output)
