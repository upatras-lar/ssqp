import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi, sin, cos

nq = 19
nv = 18
nu = 12
HORIZON = 180
dt = 0.01

A_WGT = 1e-1
V_WGT = 1e-3
U_WGT = 1e-4
F_WGT = 1e-4
T_WGT = 4.0
R_WGT = 4.0
C_WGT = 4.0

STEPS_BEFORE = 60
STEPS_AFTER = 60

base_pos = np.array([0, 0, 0.3689])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0

init[7:19] = base_config

init_control = np.zeros(30)
init_control[20] = 3.75 * 9.81
init_control[23] = 3.75 * 9.81
init_control[26] = 3.75 * 9.81
init_control[29] = 3.75 * 9.81

model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../assets/urdf/go2/go2.urdf",
    free_flyer=True,
    dynamics_mode=ssqpy.model.DynamicsMode.ReducedInverse,
)

joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, A_WGT, True)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, nu, U_WGT)
force_cost = costs.SquaredForceCost(nv, nu, F_WGT)
pose_cost = costs.FrameSquaredPoseErrorCost(
    model, "base", base_pos, np.eye(3), nq, nv, T_WGT, R_WGT
)
config_cost = costs.SquaredConfigurationErrorCost(
    base_config, nq, nv, C_WGT, True
)

fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3), nq, nv
)

fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "FR_foot", np.zeros(3), np.zeros(3), nq, nv
)

rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RL_foot", np.zeros(3), np.zeros(3), nq, nv
)

rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
    model, "RR_foot", np.zeros(3), np.zeros(3), nq, nv
)

above_ground_lb = np.array((-np.inf, -np.inf, 0.1))
above_ground_ub = np.array((np.inf, np.inf, np.inf))

base_above_ground_bound = constraints.FramePositionBound(
    model, "base", above_ground_lb, above_ground_ub, nq, nv
)

head_above_ground_bound = constraints.FramePositionBound(
    model, "Head_lower", above_ground_lb, above_ground_ub, nq, nv
)

init_states = []
init_controls = []

rotation_range = HORIZON - STEPS_BEFORE - STEPS_AFTER
omega_traj = [np.zeros(6)]


def phi(k):
    return np.pi / 2 * (1 - np.cos(2 * np.pi * k / (rotation_range - 1)))


for k in range(rotation_range - 1):
    omega_x = -abs((phi(k + 1) - phi(k)) / dt)
    omega_traj.append(np.array([0.0, 0.0, 0.0, omega_x, 0.0, 0.0]))

for k in range(HORIZON):
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)
    model.addCost(k, joint_angle_cost)

    model.addConstraints(k, base_above_ground_bound)
    model.addConstraints(k, head_above_ground_bound)

    if k < STEPS_BEFORE or k >= HORIZON - STEPS_AFTER:
        model.addContactConstraint(k, "FL_foot")
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(k, "FR_foot")
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(k, "RL_foot")
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(k, "RR_foot")
        model.addConstraints(k, rr_foot_vel_bound)

        init_states.append(init)
        init_controls.append(init_control)
        continue

    model.addCost(
        k,
        costs.FrameSquaredVelocityErrorCost(
            model,
            "base",
            omega_traj[k - STEPS_BEFORE],
            nq,
            nv,
            np.array((0.2, 0.2, 0.0, 1.0, 0.2, 0.2)),
        ),
    )

    init_states.append(np.zeros(nq + nv))
    init_states[-1][3] = 1.0
    init_states[-1][7:19] = np.array([0.0, 0.0, -pi / 3] * 4)
    init_controls.append(np.zeros(nv))

model.addCost(HORIZON, joint_vel_cost)
model.addCost(HORIZON, pose_cost)
model.addCost(HORIZON, config_cost)

model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)

fl_pos = np.array((0.1934, 0.142, 0.0))
fr_pos = np.array((0.1934, -0.142, 0.0))
rl_pos = np.array((-0.1934, 0.142, 0.0))
rr_pos = np.array((-0.1934, -0.142, 0.0))

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(model, "FL_foot", fl_pos, fl_pos, nq, nv),
)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(model, "FR_foot", fr_pos, fr_pos, nq, nv),
)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(model, "RL_foot", rl_pos, rl_pos, nq, nv),
)

model.addConstraints(
    HORIZON,
    constraints.FramePositionBound(model, "RR_foot", rr_pos, rr_pos, nq, nv),
)

init_states.append(init)

model.finalize(init_states=init_states)  # , init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    line_search_method=ssqpy.solvers.LineSearchMethod.Merit,
    merit_mu_c=4.0,
    tolerance=1e-1,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-3,
    admm_max_iter=500,
    admm_adaptive_rho_tolerance=15,
    admm_warm_start=False,
    # admm_reset_rho=False,
)

state, x, u = solver.solve()

solver.write("../data/go2_sideflip.json")

print("x:")
x.print()
print("u:")
u.print()
