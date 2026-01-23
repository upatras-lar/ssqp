import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi

nq = 19
nv = 18
HORIZON = 280
dt = 0.01

A_WGT = 1e-3
V_WGT = 1e-5
U_WGT = 1e-4
F_WGT = 1e-8

JUMP_STEPS = [(50, 70), (100, 120), (150, 180), (200, 220)]
LAND_SEQ = [0, 1, 1]

base_pos = np.array([0, 0, 0.3689])

final_pos = np.array([1.0, 0.0, 0.3689])

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0

init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)

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

fl_foot_pos_bound = constraints.FramePositionBound(
    model,
    "FL_foot",
    np.array((-np.inf, 0.1, 0.0)),
    np.array((np.inf, np.inf, np.inf)),
    nq,
    nv,
)

fr_foot_pos_bound = constraints.FramePositionBound(
    model,
    "FR_foot",
    np.array((-np.inf, -np.inf, 0.0)),
    np.array((np.inf, -0.1, np.inf)),
    nq,
    nv,
)

rl_foot_pos_bound = constraints.FramePositionBound(
    model,
    "RL_foot",
    np.array((-np.inf, 0.1, 0.0)),
    np.array((np.inf, np.inf, np.inf)),
    nq,
    nv,
)

rr_foot_pos_bound = constraints.FramePositionBound(
    model,
    "RR_foot",
    np.array((-np.inf, -np.inf, 0.0)),
    np.array((np.inf, -0.1, np.inf)),
    nq,
    nv,
)

head_pos_bound = constraints.FramePositionBound(
    model,
    "Head_lower",
    np.array((-np.inf, -np.inf, 0.01)),
    np.array((np.inf, np.inf, np.inf)),
    nq,
    nv,
)

init_controls = []

for k in range(HORIZON):
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    model.addConstraints(k, head_pos_bound)

    model.addConstraints(k, fl_foot_pos_bound)
    model.addConstraints(k, fr_foot_pos_bound)
    model.addConstraints(k, rl_foot_pos_bound)
    model.addConstraints(k, rr_foot_pos_bound)

    if k < JUMP_STEPS[0][0] or k >= JUMP_STEPS[-1][-1]:
        model.addContactConstraint(k, "FL_foot")
        model.addConstraints(k, fl_foot_vel_bound)

        model.addContactConstraint(k, "FR_foot")
        model.addConstraints(k, fr_foot_vel_bound)

        model.addContactConstraint(k, "RL_foot")
        model.addConstraints(k, rl_foot_vel_bound)

        model.addContactConstraint(k, "RR_foot")
        model.addConstraints(k, rr_foot_vel_bound)

        init_controls.append(init_control)
        continue

    on_air = True
    for i, jmp_range in enumerate(JUMP_STEPS):
        if k < jmp_range[0]:
            on_air = False
            break

        if k >= jmp_range[0] and k < jmp_range[1]:
            init_controls.append(np.zeros(nv))
            break

    if not on_air:
        if LAND_SEQ[i - 1]:
            model.addContactConstraint(k, "FL_foot")
            model.addConstraints(k, fl_foot_vel_bound)

            model.addContactConstraint(k, "FR_foot")
            model.addConstraints(k, fr_foot_vel_bound)
        else:
            model.addContactConstraint(k, "RL_foot")
            model.addConstraints(k, rl_foot_vel_bound)

            model.addContactConstraint(k, "RR_foot")
            model.addConstraints(k, rr_foot_vel_bound)

        init_controls.append(init_control_2)

model.addCost(HORIZON, joint_vel_cost)
# model.addCost(
#     HORIZON,
#     costs.SquaredConfigurationDifferenceCost(init[7:19], nq, nv, 5.0, True),
# )
# model.addCost(
#     HORIZON,
#     costs.FrameSquaredPoseErrorCost(
#         model, final_pos, np.eye(3), "base", nq, nv, 50.0, 10.0
#     ),
# )
model.addConstraints(
    HORIZON,
    constraints.FrameVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)

model.finalize(init_states=[init], init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-1,
    tolerance_final=5e-2,
    tolerance_update_steps=40,
    filter_size=1,
    admm_abs_tolerance=1e-2,
    admm_abs_tolerance_final=1e-3,
    admm_rel_tolerance=1e-3,
    admm_tolerance_update_steps=250,
    admm_max_iter=500,
)

state, x, u = solver.solve()

solver.write("../data/go2_djump")

print("x:")
x.print()
print("u:")
u.print()
