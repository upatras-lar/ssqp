import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

"""
G1 walking example
"""

# Model Parameters
H = 200
dt = 0.02

# fmt: off
INIT_CONFIG = np.array((
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0,
    np.pi/6, 0.0, 0.0, np.pi/6, 0.0,
    np.pi/6, 0.0, 0.0, np.pi/6, 0.0,
))
# fmt: on

# Weights
A_WGT = np.hstack(((1e-6,) * 12, 1e-2, (1e-6,) * 10))
# Do not move hip yaw joints too much
# Prevents stepping sideways
A_WGT[2] = 5e-3
A_WGT[3] = 0.0
A_WGT[8] = 5e-3
A_WGT[9] = 0.0

V_WGT = 5e-4
U_WGT = 1e-4
F_WGT = 1e-4

MAX_NORMAL_FORCE = 100

FOOT_SWING_STEPS = 25
GRADUAL_STEP = 1
STANDING_STEPS = 15

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
base_pos = np.array((0.0, 0.0, 0.78686375))
final_pos = np.array((3.0, 0.0, 0.78686375))

# Initial configuration
init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0
init[7:nq] = INIT_CONFIG

# Create regularization costs
# joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
config_cost = costs.SquaredConfigurationErrorCost(model, INIT_CONFIG, A_WGT)
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

left_foot_ag_bound_1 = constraints.FrameTranslationBound(
    model,
    "left_pp",
    np.array((-np.inf, -np.inf, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

left_foot_ag_bound_2 = constraints.FrameTranslationBound(
    model,
    "left_pm",
    np.array((-np.inf, -np.inf, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

right_foot_ag_bound_1 = constraints.FrameTranslationBound(
    model,
    "right_pp",
    np.array((-np.inf, -np.inf, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

right_foot_ag_bound_2 = constraints.FrameTranslationBound(
    model,
    "right_pm",
    np.array((-np.inf, -np.inf, -0.01)),
    np.array((np.inf, np.inf, np.inf)),
)

fb_height_bound = constraints.FloatingBaseTranslationBound(
    model,
    np.array((-np.inf, -np.inf, 0.65)),
    np.array((np.inf, np.inf, np.inf)),
)

pose_cost = costs.FloatingBaseSquaredPoseErrorCost(
    model, final_pos, np.eye(3), 1e-5
)

rot_cost = costs.FloatingBaseSquaredRotationErrorCost(model, np.eye(3), 5e-2)

final_config_cost = costs.SquaredConfigurationErrorCost(
    model, INIT_CONFIG, 1e-1
)

final_pose_bound = constraints.FloatingBasePoseBound(
    model, final_pos, np.eye(3)
)
final_vel_bound = constraints.FloatingBaseLinearVelocityBound(
    model, np.zeros(3), np.zeros(3)
)


def add_left_cc(stance_start_steps):

    if stance_start_steps <= FOOT_SWING_STEPS - GRADUAL_STEP:
        model.addContactConstraint(
            k, "left_mm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )
        model.addContactConstraint(
            k, "left_mp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )

        model.addConstraints(k, left_mm_vel_bound)
        model.addConstraints(k, left_mp_vel_bound)

    if stance_start_steps >= GRADUAL_STEP:
        model.addContactConstraint(
            k, "left_pp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )
        model.addContactConstraint(
            k, "left_pm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )

        model.addConstraints(k, left_pp_vel_bound)
        model.addConstraints(k, left_pm_vel_bound)


def add_right_cc(stance_start_steps):
    if stance_start_steps <= FOOT_SWING_STEPS - GRADUAL_STEP:
        model.addContactConstraint(
            k, "right_mm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )
        model.addContactConstraint(
            k, "right_mp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )

        model.addConstraints(k, right_mm_vel_bound)
        model.addConstraints(k, right_mp_vel_bound)

    if stance_start_steps >= GRADUAL_STEP:
        model.addContactConstraint(
            k, "right_pp", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )
        model.addContactConstraint(
            k, "right_pm", eq_tol=0.01, max_normal_force=MAX_NORMAL_FORCE
        )

        model.addConstraints(k, right_pp_vel_bound)
        model.addConstraints(k, right_pm_vel_bound)


stance_start_steps = 0

for k in range(H):
    # Add regularization costs
    model.addCost(k, config_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    model.addCost(k, pose_cost)
    model.addCost(k, rot_cost)

    model.addConstraints(k, fb_height_bound)

    model.addConstraints(k, left_foot_ag_bound_1)
    model.addConstraints(k, left_foot_ag_bound_2)
    model.addConstraints(k, right_foot_ag_bound_1)
    model.addConstraints(k, right_foot_ag_bound_1)

    if k < STANDING_STEPS or k >= H - STANDING_STEPS:
        add_left_cc(10)
        add_right_cc(10)
        continue

    # Add contact constraints and velocity bounds for points on left foot
    if (k // FOOT_SWING_STEPS) % 2 == 0:
        if k % FOOT_SWING_STEPS == 0:
            stance_start_steps = 0

        add_left_cc(stance_start_steps)

        stance_start_steps += 1

    else:
        if k % FOOT_SWING_STEPS == 0:
            stance_start_steps = 0

        add_right_cc(stance_start_steps)

        stance_start_steps += 1

model.addCost(H, final_config_cost)
model.addCost(H, joint_vel_cost)

model.addConstraints(H, final_pose_bound)
model.addConstraints(H, final_vel_bound)

model.finalize(init_states=init)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.min_alpha = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-3
hpipm_params.tol_comp = 1e-3
hpipm_params.tol_eq = 1e-3
hpipm_params.tol_ineq = 1e-3
hpipm_params.max_iter = 25
ssqp_params.hpipmParams = hpipm_params

admm_params = ssqpy.solvers.admmParams()
admm_params.max_iter = 50
# admm_params.reset_rho = False
# admm_params.reset_y = False
# admm_params.warm_start = False
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
ssqp_params.admmParams = admm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()
solver.write("../data/g1_walk.json")
