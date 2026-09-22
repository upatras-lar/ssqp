import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi, sin, cos

"""
Go2 Heart example
Traces a heart with the Go2's front feet
"""

# Model Parameters
HORIZON = 200
dt = 0.01

# Weights
A_WGT = 1e-2
V_WGT = 1e-3
U_WGT = 1e-4
F_WGT = 1e-4
TRACE_ARM_WGT = 3.0
TF_WGT = 10.0


# Steps to stand for at beginning/end of trajectory
STANDING_STEPS = 20
# Steps to transition from/to standing to/from end of trace
TRANSITIONAL_STEPS = 20
# x-coordinate for trace
TRACE_X_COORD = 0.2

# Total steps for the trace
trace_steps = HORIZON - 2 * STANDING_STEPS - 2 * TRANSITIONAL_STEPS

# Create 2D half-heart trajectories (per arm)
traj_l = []
traj_r = []

x = lambda t: 0.02 * (16 * sin(t) ** 3)
y = (
    lambda t: 0.02
    * (13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t))
    + 0.5
)

for t in np.arange(0, pi, pi / trace_steps):
    traj_l.append((x(t), y(t)))
    traj_r.append((x(2 * pi - t), y(2 * pi - t)))

# Create an SSQP model for the Go2
model = ssqpy.model.Model(
    HORIZON,
    dt,
    from_preset=ssqpy.model.Preset.Go2,
    solver_mode=ssqpy.model.SolverMode.ReducedInverseDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

# Initial and final positions
base_pos = np.array([0.0, 0.0, 0.3689])

# Initial configuration
init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

# Initial controls for 4 forces
init_control = np.zeros(nu + 12)
init_control[-10] = 3.75 * 9.81
init_control[-7] = 3.75 * 9.81
init_control[-4] = 3.75 * 9.81
init_control[-1] = 3.75 * 9.81

# Initial controls for 2 forces
init_control_2 = np.zeros(nu + 6)
init_control_2[-4] = 7.5 * 9.81
init_control_2[-1] = 7.5 * 9.81

# forces = {}
# forces["FR_foot"] = [init_control[nv : nv + 3]]
# forces["FL_foot"] = [init_control[nv + 3 : nv + 6]]
# forces["RR_foot"] = [init_control[nv + 6 : nv + 9]]
# forces["RL_foot"] = [init_control[nv + 9 : nv + 12]]

# forces_2 = {}
# forces["RR_foot"] = [init_control[nv : nv + 3]]
# forces["RL_foot"] = [init_control[nv + 3 : nv + 6]]

# init_control[6:nv] = model.inverseDynamics(
#     init[:nq], init[nq:], np.zeros(nv), forces
# )[-nv + 6 :]

# init_control_2[6:nv] = model.inverseDynamics(
#     init[:nq], init[nq:], np.zeros(nv), forces_2
# )[-nv + 6 :]

# Create regularization costs
joint_angle_cost = costs.SquaredJointAngleCost(model, A_WGT)
joint_vel_cost = costs.SquaredJointVelocityCost(model, V_WGT)
control_cost = costs.SquaredControlCost(model, U_WGT)
force_cost = costs.SquaredForceCost(model, F_WGT)

# Define velocity bounds for feet
# Used in conjunction with contact constraints
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


init_controls = []

for k in range(HORIZON):
    # Add the regularization costs for each stage
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    # Add rear feet contacts
    model.addContactConstraint(k, "RL_foot", eq_tol=0.01)
    model.addConstraints(k, rl_foot_vel_bound)

    model.addContactConstraint(k, "RR_foot", eq_tol=0.01)
    model.addConstraints(k, rr_foot_vel_bound)

    # If at start or end of trajectory, add front feet contacts
    if k < STANDING_STEPS or k > HORIZON - STANDING_STEPS:
        model.addContactConstraint(k, "FL_foot", eq_tol=0.01)
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(k, "FR_foot", eq_tol=0.01)
        model.addConstraints(k, fr_foot_vel_bound)

        init_controls.append(init_control)
        continue

    # If within trace
    if k < HORIZON - STANDING_STEPS - TRANSITIONAL_STEPS:
        trace_k = k - STANDING_STEPS - TRANSITIONAL_STEPS
        model.addCost(
            k,
            costs.FrameSquaredTranslationErrorCost(
                model,
                "FL_foot",
                np.array(
                    (TRACE_X_COORD, traj_l[trace_k][0], traj_l[trace_k][1])
                ),
                TRACE_ARM_WGT,
            ),
        )

        model.addCost(
            k,
            costs.FrameSquaredTranslationErrorCost(
                model,
                "FR_foot",
                np.array(
                    (TRACE_X_COORD, traj_r[trace_k][0], traj_r[trace_k][1])
                ),
                TRACE_ARM_WGT,
            ),
        )

    init_controls.append(init_control_2)


# Add a velocity regularization cost for the final stage
model.addCost(HORIZON, joint_vel_cost)

# Add a cost for the final translation of the base to be that of the final
# position
model.addCost(
    HORIZON,
    costs.FrameSquaredTranslationErrorCost(model, "base", base_pos, TF_WGT),
)

# Add a bound such that the base has no velocity at the end of the trajectory
model.addConstraints(
    HORIZON,
    constraints.FrameLinearVelocityBound(
        model, "base", np.zeros(3), np.zeros(3)
    ),
)

# Finalize the model
model.finalize(init_states=init, init_controls=init_controls)

# SSQP model parameters
ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.loose_success = False
ssqp_params.tolerance = 1e-2
# ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.max_iter = 500
admm_params.warm_start = False
ssqp_params.admmParams = admm_params

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_stat = 1e-4
hpipm_params.tol_comp = 1e-4
hpipm_params.tol_eq = 1e-4
hpipm_params.tol_ineq = 1e-4
ssqp_params.hpipmParams = hpipm_params

# Create an SSQP solver for the model
solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

solver.write("../data/go2_heart.json")

print("x:")
x.print()
print("u:")
u.print()
