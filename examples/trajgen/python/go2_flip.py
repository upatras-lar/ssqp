import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi, sin, cos

nq = 19
nv = 18
HORIZON = 160
dt = 0.01

A_WGT = 1e-4
V_WGT = 1e-4
U_WGT = 1e-5
F_WGT = 1e-9

STEPS_BEFORE = 30
STEPS_BEFORE_TWO = 30
STEPS_AFTER = 50

base_pos = np.array([0, 0, 0.3689])
Rf = np.eye(3)
Rf[1, 1] = -1
Rf[2, 2] = -1

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0
init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

init_flipped = np.zeros(37)
init_flipped[:3] = base_pos
init_flipped[3] = 1.0
init_flipped[7:19] = init[7:19]


init_control = np.zeros(30)
init_control[20] = 3.75 * 9.81
init_control[23] = 3.75 * 9.81
init_control[26] = 3.75 * 9.81
init_control[29] = 3.75 * 9.81

init_control_2 = np.zeros(24)
init_control_2[20] = 7.5 * 9.81
init_control_2[23] = 7.5 * 9.81

model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../urdf/go2/go2.urdf",
    free_flyer=True,
)

joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, A_WGT, True)
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

init_states = []
init_controls = []

for k in range(HORIZON):
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

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
    elif k < STEPS_BEFORE_TWO:
        model.addContactConstraint(k, "FR_foot")
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(k, "RR_foot")
        model.addConstraints(k, rr_foot_vel_bound)

        init_states.append(init)
        init_controls.append(init_control_2)
        continue
    else:
        pass
        # model.addCost(
        #     k,
        #     costs.FrameSquaredRotationErrorCost(
        #         model, Rf, "base", nq, nv, 1.0
        #     ),
        # )

    init_states.append(init_flipped)
    init_controls.append(np.zeros(nv))

model.addCost(HORIZON, joint_vel_cost)
model.addConstraints(
    HORIZON,
    constraints.FrameVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)

init_states.append(init)

model.finalize(init_states=init_states, init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-1,
    filter_size=1,
    admm_abs_tolerance=1e-2,
    admm_rel_tolerance=1e-3,
    admm_max_iter=500,
)

state, x, u = solver.solve()

solver.write("../data/go2_flip.json")

print("x:")
x.print()
print("u:")
u.print()
