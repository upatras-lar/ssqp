import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import argparse

"""
G1 standing example
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="G1 standing trajectory generation"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/g1_standing.json",
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
dt = 0.01

# Weights
A_WGT = 3e-2
V_WGT = 1e-2
U_WGT = 1e-3
F_WGT = 1e-4
T_BASE_WGT = 3.0
R_BASE_WGT = 1.0

MAX_NORMAL_FORCE = 340

# Create an SSQP model for Talos
model = ssqpy.model.Model(
    H,
    dt,
    from_preset=ssqpy.model.Preset.G1,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Initial position for the torso
base_pos = np.array((0, 0, 0.78686375))

# Initial configuration
init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0

# Create regularization costs
joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

right_hand_translation_cost = costs.FrameSquaredTranslationErrorCost(
    model, "right_wrist_roll_rubber_hand", np.array((0.1, -0.3, 1.0)), 0.5
)
left_hand_translation_cost = costs.FrameSquaredTranslationErrorCost(
    model, "left_wrist_roll_rubber_hand", np.array((0, 0.2, 0.7)), 0.5
)

for k in range(H):
    # Add regularization costs
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    model.addCost(
        k,
        costs.FrameSquaredPoseErrorCost(
            model,
            "torso_link",
            base_pos,
            np.eye(3),
            T_BASE_WGT,
            R_BASE_WGT,
        ),
    )

    model.addCost(k, right_hand_translation_cost)
    model.addCost(k, left_hand_translation_cost)

    # Add contact constraints for points on left foot
    model.addContactConstraint(
        k, "left_mm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )
    model.addContactConstraint(
        k, "left_mp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )
    model.addContactConstraint(
        k, "left_pp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )
    model.addContactConstraint(
        k, "left_pm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )

    model.addContactConstraint(
        k, "right_mm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )
    model.addContactConstraint(
        k, "right_mp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )
    model.addContactConstraint(
        k, "right_pp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )
    model.addContactConstraint(
        k, "right_pm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
    )

# Add regularization costs at the terminal stage
model.addCost(H, joint_angle_cost)
model.addCost(H, joint_vel_cost)

model.addCost(H, right_hand_translation_cost)
model.addCost(H, left_hand_translation_cost)

# Add velocity bound for the torso at the terminal stage
model.addConstraints(
    H,
    constraints.FrameLinearVelocityBound(
        model, "torso_link", np.zeros(3), np.zeros(3)
    ),
)

model.finalize(init_states=init)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)

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
