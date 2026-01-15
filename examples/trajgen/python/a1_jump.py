import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

nq = 19
nv = 18
HORIZON = 140
dt = 0.01

V_WGT = 1e-3
U_WGT = 1e-6
F_WGT = 1e-8
TP_WGT = 100
TO_WGT = 10

STEPS_BEFORE = 50
STEPS_AFTER = 60

base_pos = np.array([0, 0, 0.3464])

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0

init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

init_control = np.zeros(30)
init_control[20] = 3.43525 * 9.81
init_control[23] = 3.43525 * 9.81
init_control[26] = 3.43525 * 9.81
init_control[29] = 3.43525 * 9.81

model = ssqpy.model.Model(
    HORIZON, dt, urdf_path="../../urdf/a1/a1.urdf", free_flyer=True
)

joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, U_WGT)
force_cost = costs.SquaredForceCost(nv, F_WGT)

fl_foot_vel_bound = constraints.FrameVelocityBound(
    model, "FL_foot", np.zeros(3), np.zeros(3), nq, nv
)

fr_foot_vel_bound = constraints.FrameVelocityBound(
    model, "FR_foot", np.zeros(3), np.zeros(3), nq, nv
)

rl_foot_vel_bound = constraints.FrameVelocityBound(
    model, "RL_foot", np.zeros(3), np.zeros(3), nq, nv
)

rr_foot_vel_bound = constraints.FrameVelocityBound(
    model, "RR_foot", np.zeros(3), np.zeros(3), nq, nv
)

init_controls = []

for k in range(HORIZON):
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    if k < STEPS_BEFORE or k > HORIZON - STEPS_AFTER:
        model.addContactConstraint(k, "FL_foot")
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(k, "FR_foot")
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(k, "RL_foot")
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(k, "RR_foot")
        model.addConstraints(k, rr_foot_vel_bound)

        init_controls.append(init_control)
    else:
        init_controls.append(np.zeros(nv))

model.addCost(HORIZON, joint_vel_cost)
model.addCost(
    HORIZON,
    costs.FrameSquaredPoseErrorCost(
        model, base_pos, np.eye(3), "trunk", nq, nv, TP_WGT, TO_WGT
    ),
)

model.finalize(init_states=[init], init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    filter_size=100,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-4,
    admm_max_iter=400,
)

state, x, u = solver.solve()

solver.write("../data/a1_jump")

print("x:")
x.print()
print("u:")
u.print()
