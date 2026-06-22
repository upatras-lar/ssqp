import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi, sin, cos

HORIZON = 180
dt = 0.01

A_WGT = 1e-2
V_WGT = 2e-2
U_WGT = 1e-4
F_WGT = 1e-4
T_WGT = 4.4
R_WGT = 1.0

MAX_NORMAL_FORCE = 140

STEPS_BEFORE = 60
STEPS_AFTER = 60

model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.ForwardDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

base_pos = np.array([0, 0, 0.3689])

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0

init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

init_control = np.zeros(nu + 12)
init_control[-10] = 3.75 * 9.81
init_control[-7] = 3.75 * 9.81
init_control[-4] = 3.75 * 9.81
init_control[-1] = 3.75 * 9.81

joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)
pose_cost = costs.FrameSquaredPoseErrorCost(
    model, "base", base_pos, np.eye(3), T_WGT, R_WGT
)

fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3)
)

fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FR_foot", np.zeros(3), np.zeros(3)
)

rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RL_foot", np.zeros(3), np.zeros(3)
)

rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RR_foot", np.zeros(3), np.zeros(3)
)

above_ground_lb = np.array((-np.inf, -np.inf, 0.1))
above_ground_ub = np.array((np.inf, np.inf, np.inf))

base_above_ground_bound = constraints.FramePositionBound(
    model, "base", above_ground_lb, above_ground_ub
)

head_above_ground_bound = constraints.FramePositionBound(
    model, "Head_lower", above_ground_lb, above_ground_ub
)

init_states = []
init_controls = []

rotation_traj = []
rotation_range = HORIZON - STEPS_BEFORE - STEPS_AFTER
for k in range(rotation_range):
    θk = pi / 2 * (1 - cos(2 * pi * k / (rotation_range - 1)))
    R = np.eye(3)
    R[1, 1] = cos(θk)
    R[1, 2] = sin(θk) if k <= rotation_range / 2 else -sin(θk)
    R[2, 1] = -sin(θk) if k <= rotation_range / 2 else sin(θk)
    R[2, 2] = cos(θk)
    rotation_traj.append(R)

for k in range(HORIZON):
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    model.addConstraints(k, base_above_ground_bound)
    model.addConstraints(k, head_above_ground_bound)

    if k < STEPS_BEFORE or k >= HORIZON - STEPS_AFTER:
        model.addContactConstraint(
            k, "FL_foot", max_normal_force=MAX_NORMAL_FORCE
        )
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(
            k, "FR_foot", max_normal_force=MAX_NORMAL_FORCE
        )
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(
            k, "RL_foot", max_normal_force=MAX_NORMAL_FORCE
        )
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(
            k, "RR_foot", max_normal_force=MAX_NORMAL_FORCE
        )
        model.addConstraints(k, rr_foot_vel_bound)

        init_states.append(init)
        init_controls.append(init_control)
        continue

    # model.addCost(
    #     k,
    #     costs.FrameSquaredRotationErrorCost(
    #         model, "base", rotation_traj[k - STEPS_BEFORE], R_WGT
    #     ),
    # )

    init_states.append(np.zeros(nq + nv))
    init_states[-1][3] = 1.0
    init_states[-1][7:19] = np.array([0.0, 0.0, -pi / 3] * 4)
    init_controls.append(np.zeros(nu))

model.addCost(HORIZON, pose_cost)
model.addCost(HORIZON, joint_vel_cost)
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), True
    ),
)

init_states.append(init)

model.finalize(init_states=init_states, init_controls=init_controls)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = True
ssqp_params.tolerance = 5e-2
ssqp_params.line_search_method = ssqpy.solvers.LineSearchMethod.Merit
ssqp_params.merit_mu_c = 3.0
# ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.max_iter = 500
admm_params.adaptive_rho_tolerance = 15
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.split_step = True
hpipm_params.prediction_correction = True
hpipm_params.tol_stat = 1e-3
hpipm_params.tol_comp = 1e-3
hpipm_params.tol_eq = 1e-3
hpipm_params.tol_ineq = 1e-3
ssqp_params.hpipmParams = hpipm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

solver.write("../data/go2_sideflip.json")

print("x:")
x.print()
print("u:")
u.print()
