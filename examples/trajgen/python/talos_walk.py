import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

"""
Talos Walking example
Performs a walk with the Talos humanoid robot
"""

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
FINAL_STANDING_STEPS = 20

# Number of steps for each swing
FOOT_SWING_STEPS = 20

# Create an SSQP model for Talos
model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/talos/talos.urdf",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.InverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Initial position
base_pos = np.array((0, 0, 1.0611))

# Final position
final_pos = np.array((1.0, 0.0, 1.0611))

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
init_control = np.zeros(nv + 24)
for f in range(1, 9):
    init_control[nv + 3 * f - 1] = 116

# Initial forces for 1 foot on the ground
init_control_left = np.zeros(nv + 12)
init_control_right = np.zeros(nv + 12)
for f in range(1, 5):
    init_control_left[nv + 3 * f - 1] = 232
    init_control_right[nv + 3 * f - 1] = 232

# forces_left = {}
# forces_left["left_mm"] = [init_control_left[nv : nv + 3]]
# forces_left["left_mp"] = [init_control_left[nv + 3 : nv + 6]]
# forces_left["left_pp"] = [init_control_left[nv + 6 : nv + 9]]
# forces_left["left_pm"] = [init_control_left[nv + 9 : nv + 12]]

# forces_right = {}
# forces_right["right_mm"] = [init_control_right[nv : nv + 3]]
# forces_right["right_mp"] = [init_control_right[nv + 3 : nv + 6]]
# forces_right["right_pp"] = [init_control_right[nv + 6 : nv + 9]]
# forces_right["right_pm"] = [init_control_right[nv + 9 : nv + 12]]

# # Use inverse dynamics to calculate initial joint torques
# init_control_left[6:nv] = model.inverseDynamics(
#     init[:nq], init[nq:], np.zeros(nv), forces_left
# )[-nv + 6 :]

# init_control_right[6:nv] = model.inverseDynamics(
#     init[:nq], init[nq:], np.zeros(nv), forces_right
# )[-nv + 6 :]

# init_control[6:nv] = model.inverseDynamics(
#     init[:nq], init[nq:], np.zeros(nv), forces_left | forces_right
# )[-nv + 6 :]

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

# For each contact frame, create a zero velocity bound
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

for k in range(H - FINAL_STANDING_STEPS):
    # Add regularization costs
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add contact for the appropriate foot
    if (k // FOOT_SWING_STEPS) % 2 == 0:
        model.addContactConstraint(k, "left_mm")
        model.addContactConstraint(k, "left_mp")
        model.addContactConstraint(k, "left_pp")
        model.addContactConstraint(k, "left_pm")
        model.addConstraints(k, left_mm_vel_bound)
        model.addConstraints(k, left_mp_vel_bound)
        model.addConstraints(k, left_pp_vel_bound)
        model.addConstraints(k, left_pm_vel_bound)

        init_controls.append(init_control_left)
    else:
        model.addContactConstraint(k, "right_mm")
        model.addContactConstraint(k, "right_mp")
        model.addContactConstraint(k, "right_pp")
        model.addContactConstraint(k, "right_pm")
        model.addConstraints(k, right_mm_vel_bound)
        model.addConstraints(k, right_mp_vel_bound)
        model.addConstraints(k, right_pp_vel_bound)
        model.addConstraints(k, right_pm_vel_bound)

        init_controls.append(init_control_right)

# Add final standing phase
for k in range(H - FINAL_STANDING_STEPS, H):
    # Add regularization costs
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add contacts for both feet
    model.addContactConstraint(k, "left_mm")
    model.addContactConstraint(k, "left_mp")
    model.addContactConstraint(k, "left_pp")
    model.addContactConstraint(k, "left_pm")
    model.addConstraints(k, left_mm_vel_bound)
    model.addConstraints(k, left_mp_vel_bound)
    model.addConstraints(k, left_pp_vel_bound)
    model.addConstraints(k, left_pm_vel_bound)

    model.addContactConstraint(k, "right_mm")
    model.addContactConstraint(k, "right_mp")
    model.addContactConstraint(k, "right_pp")
    model.addContactConstraint(k, "right_pm")
    model.addConstraints(k, right_mm_vel_bound)
    model.addConstraints(k, right_mp_vel_bound)
    model.addConstraints(k, right_pp_vel_bound)
    model.addConstraints(k, right_pm_vel_bound)

    init_controls.append(init_control)

    # Add constraints to keep the base and head still at the end
    model.addConstraints(
        k,
        constraints.FrameLinearVelocityBound(
            model, "base_link", np.zeros(3), np.zeros(3), True
        ),
    )
    model.addConstraints(
        k,
        constraints.FrameLinearVelocityBound(
            model, "head_1_link", np.zeros(3), np.zeros(3), True
        ),
    )

# Add terminal stage costs & constraints
model.addCost(H, joint_vel_cost)
model.addCost(H, pose_cost)
model.addConstraints(
    H,
    constraints.FrameLinearVelocityBound(
        model, "base_link", np.zeros(3), np.zeros(3), True
    ),
)
model.addConstraints(
    k,
    constraints.FrameLinearVelocityBound(
        model, "head_1_link", np.zeros(3), np.zeros(3), True
    ),
)

model.finalize(init_states=init, init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    line_search_method=ssqpy.solvers.LineSearchMethod.Merit,
    tolerance=5e-2,
    admm_abs_tolerance=5e-3,
    admm_rel_tolerance=5e-3,
    admm_max_iter=500,
    admm_reset_rho=False,
    admm_reset_y=False,
    admm_warm_start=False,
)

state, x, u = solver.solve()
solver.write("../data/talos_walk.json")

print("x:")
x.print()
print("u:")
u.print()
