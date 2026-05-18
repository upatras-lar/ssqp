import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

"""
Talos Standing on 1 Foot example
Balances the Talos humanoid robot on 1 foot
"""

# Model Parameters
H = 160
dt = 0.01

# Weights
A_WGT = 3e-2
V_WGT = 1e-2
U_WGT = 1e-3
F_WGT = 1e-4
T_BASE_WGT = 3.0
R_BASE_WGT = 1.0
T_KNEE_RAISE_WGT = 3.0
T_ARM_RAISE_WGT = 1.4

# For Talos, we must apply 4 forces per foot
# As such, we create a rectangle (in the URDF) around the foot and apply forces
# to each of the rectangle's corners

# Create an SSQP model for Talos
model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/talos/talos.urdf",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Initial position for the torso
base_pos = np.array((0, 0, 1.072))

# Initial configuration
init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0

# Initial control (for forces)
init_control = np.zeros(nu + 12)
for f in range(1, 5):
    init_control[nu + 3 * f - 1] = 2 * 116


# Create regularization costs
joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

# Create velocity bounds for each frame that will have a force applied
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

init_controls = []

for k in range(H):
    # Add regularization costs
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add a frame squared pose error cost for the torso to remain in the
    # position (0.0, 0.15, 1.072)
    # This moves it slightly closer to the left foot that's on the ground,
    # thus improving balance
    model.addCost(
        k,
        costs.FrameSquaredPoseErrorCost(
            model,
            "torso_1_link",
            np.array((0.0, 0.15, 1.072)),
            np.eye(3),
            T_BASE_WGT,
            R_BASE_WGT,
        ),
    )

    # After the middle of the trajectory
    if k > H // 2:
        # Add a cost to raise the right knee
        model.addCost(
            k,
            costs.FrameSquaredTranslationErrorCost(
                model,
                "leg_right_4_link",
                np.array((0.4, -0.25, 0.65)),
                T_KNEE_RAISE_WGT,
            ),
        )

        # Add costs to raise arms
        model.addCost(
            k,
            costs.FrameSquaredTranslationErrorCost(
                model,
                "arm_left_4_link",
                np.array((0.0, 0.6, 1.0)),
                T_ARM_RAISE_WGT,
            ),
        )

        model.addCost(
            k,
            costs.FrameSquaredTranslationErrorCost(
                model,
                "arm_right_4_link",
                np.array((0.0, -0.6, 1.0)),
                T_ARM_RAISE_WGT,
            ),
        )

    # Add contact constraints and velocity bounds for points on left foot
    model.addContactConstraint(k, "left_mm")
    model.addContactConstraint(k, "left_mp")
    model.addContactConstraint(k, "left_pp")
    model.addContactConstraint(k, "left_pm")
    model.addConstraints(k, left_mm_vel_bound)
    model.addConstraints(k, left_mp_vel_bound)
    model.addConstraints(k, left_pp_vel_bound)
    model.addConstraints(k, left_pm_vel_bound)

    init_controls.append(init_control)

# Add regularization costs at the terminal stage
model.addCost(H, joint_angle_cost)
model.addCost(H, joint_vel_cost)

# Add velocity bound for the torso at the terminal stage
model.addConstraints(
    H,
    constraints.FrameLinearVelocityBound(
        model, "torso_1_link", np.zeros(3), np.zeros(3), True
    ),
)

model.finalize(init_states=init, init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    filter_size=10,
    tolerance=1e-1,
    admm_abs_tolerance=1e-2,
    admm_rel_tolerance=1e-2,
    admm_max_iter=500,
)

state, x, u = solver.solve()
solver.write("../data/talos_stand_1foot.json")

print("x:")
x.print()
print("u:")
u.print()
