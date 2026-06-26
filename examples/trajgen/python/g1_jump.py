import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

"""
G1 standing example
"""

# Model Parameters
H = 200
dt = 0.01

# Weights
A_WGT = 1e-2
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-4
T_BASE_WGT = 1.0
R_BASE_WGT = 1.0

MAX_NORMAL_FORCE = 160

JUMP_START = 80
JUMP_END = 120

# Create an SSQP model for Talos
model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/g1/g1_23dof.urdf",
    free_flyer=True,
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

    if k < JUMP_START or k > JUMP_END:
        # Add contact constraints and velocity bounds for points on left foot
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
        model.addConstraints(k, left_mm_vel_bound)
        model.addConstraints(k, left_mp_vel_bound)
        model.addConstraints(k, left_pp_vel_bound)
        model.addConstraints(k, left_pm_vel_bound)

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
        model.addConstraints(k, right_mm_vel_bound)
        model.addConstraints(k, right_mp_vel_bound)
        model.addConstraints(k, right_pp_vel_bound)
        model.addConstraints(k, right_pm_vel_bound)

# Add regularization costs at the terminal stage
model.addCost(H, joint_angle_cost)
model.addCost(H, joint_vel_cost)

# Add velocity bound for the torso at the terminal stage
model.addConstraints(
    H,
    constraints.FrameLinearVelocityBound(
        model, "torso_link", np.zeros(3), np.zeros(3), True
    ),
)

model.finalize(init_states=init)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.max_iter = 20
hpipm_params.tol_stat = 1e-3
hpipm_params.tol_comp = 1e-3
hpipm_params.tol_eq = 1e-3
hpipm_params.tol_ineq = 1e-3
ssqp_params.hpipmParams = hpipm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()
solver.write("../data/g1_jump.json")
