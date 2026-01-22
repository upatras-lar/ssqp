import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi
from utils.GaitManager import GaitManager

nq = 19
nv = 18
HORIZON = 300
dt = 0.01

A_WGT = 1e-2
V_WGT = 1e-5
U_WGT = 1e-6
F_WGT = 1e-8

FINAL_STANDING_STEPS = 10

base_pos = np.array([-3.0, 0.0, 0.3689])
final_pos = np.array([3.0, 0.0, 0.3689])

init = np.zeros(37)
init[:3] = base_pos
init[6] = 1.0

init[7:19] = np.array([0.0, pi / 6, -pi / 3] * 4)


model = ssqpy.model.Model(
    HORIZON,
    dt,
    urdf_path="../../urdf/go2/go2.urdf",
    free_flyer=True,
    # dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, A_WGT, True)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, V_WGT, True)
control_cost = costs.SquaredControlCost(nv, U_WGT)
force_cost = costs.SquaredForceCost(nv, F_WGT)

for k in range(HORIZON):
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    model.addCost(
        k,
        costs.FrameSquaredRotationErrorCost(
            model, np.eye(3), "base", nq, nv, 1.0
        ),
    )

    # model.addConstraints(
    #     k,
    #     constraints.FramePositionBound(
    #         model,
    #         "base",
    #         np.array((-np.inf, -np.inf, 0.32)),
    #         np.array((np.inf, np.inf, -0.32)),
    #         nq,
    #         nv,
    #     ),
    # )

model.addCost(HORIZON, joint_vel_cost)
model.addCost(
    HORIZON,
    costs.FrameSquaredTranslationErrorCost(
        model, final_pos, "base", nq, nv, 100.0
    ),
)
# model.addCost(
#     HORIZON,
#     costs.SquaredConfigurationDifferenceCost(init[7:19], nq, nv, 1.0, True),
# )
model.addConstraints(
    HORIZON,
    constraints.FrameVelocityBound(
        model, "base", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)

init_controls = []

gait_manager = GaitManager(
    model, nq, nv, ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
)
init_controls.extend(
    gait_manager.setRotatoryGallop(0, HORIZON - FINAL_STANDING_STEPS - 1)
)
# for k in range(HORIZON // 2 + 1, HORIZON // 2 + 1 + FINAL_STANDING_STEPS):
#     init_controls.extend(gait_manager.addStandingStage(k))
# init_controls.extend(
#     gait_manager.setWalk(
#         HORIZON // 2 + FINAL_STANDING_STEPS + 1, HORIZON - FINAL_STANDING_STEPS - 1
#     )
# )
for k in range(HORIZON - FINAL_STANDING_STEPS, HORIZON):
    init_controls.extend(gait_manager.addStandingStage(k))

model.finalize(init_states=[init], init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-1,
    tolerance_final=5e-2,
    tolerance_update_steps=20,
    filter_size=100,
    admm_abs_tolerance=1e-2,
    admm_abs_tolerance_final=5e-3,
    admm_rel_tolerance=1e-3,
    admm_rel_tolerance_final=5e-4,
    admm_tolerance_update_steps=400,
    admm_max_iter=500,
)

state, x, u = solver.solve()

solver.write("../data/go2_walk")

print("x:")
x.print()
print("u:")
u.print()
