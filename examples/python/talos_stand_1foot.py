import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

nq = 51
nv = 50
H = 40
dt = 0.01

a_wgt = 1e-3
v_wgt = 1e-4
u_wgt = 1e-6
f_wgt = 1e-8

base_pos = np.array((0, 0, 1.072))

init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0

init_control = np.zeros(nv + 12)
for f in range(1, 5):
    init_control[nv + 3 * f - 1] = 2 * 116

forces = {}
forces["left_mm"] = [init_control[nv : nv + 3]]
forces["left_mp"] = [init_control[nv + 3 : nv + 6]]
forces["left_pp"] = [init_control[nv + 6 : nv + 9]]
forces["left_pm"] = [init_control[nv + 9 : nv + 12]]

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../examples/urdf/talos/talos.urdf",
    free_flyer=True,
    # ignore_innate_constraints=True,
    # dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)
init_control[6:nv] = model.inverseDynamics(
    init[:nq], init[nq:], np.zeros(nv), forces
)[-nv + 6 :]

joint_angle_cost = costs.SquaredJointAngleCost(nq, nv, a_wgt, True)
joint_vel_cost = costs.SquaredJointVelocityCost(nv, v_wgt, True)
control_cost = costs.SquaredControlCost(nv, u_wgt)
force_cost = costs.SquaredForceCost(nv, f_wgt)

left_mm_vel_bound = constraints.FrameVelocityBound(
    model, "left_mm", np.zeros(3), np.zeros(3), nq, nv
)
left_mp_vel_bound = constraints.FrameVelocityBound(
    model, "left_mp", np.zeros(3), np.zeros(3), nq, nv
)
left_pp_vel_bound = constraints.FrameVelocityBound(
    model, "left_pp", np.zeros(3), np.zeros(3), nq, nv
)
left_pm_vel_bound = constraints.FrameVelocityBound(
    model, "left_pm", np.zeros(3), np.zeros(3), nq, nv
)

init_controls = []

for k in range(H):
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    model.addCost(
        k,
        costs.FrameSquaredPoseErrorCost(
            model,
            np.array((0.0, 0.15, 1.072)),
            np.eye(3),
            "torso_1_link",
            nq,
            nv,
            10.0,
            1.0,
        ),
    )

    if k > H // 2:
        model.addCost(
            k,
            costs.FrameSquaredPositionErrorCost(
                model,
                np.array((0.4, -0.25, 0.65)),
                "leg_right_4_link",
                nq,
                nv,
                10.0,
            ),
        )

        model.addCost(
            k,
            costs.FrameSquaredPositionErrorCost(
                model,
                np.array((0.0, 0.6, 1.0)),
                "arm_left_4_link",
                nq,
                nv,
                2.0,
            ),
        )

        model.addCost(
            k,
            costs.FrameSquaredPositionErrorCost(
                model,
                np.array((0.0, -0.6, 1.0)),
                "arm_right_4_link",
                nq,
                nv,
                2.0,
            ),
        )

    model.addContactConstraint(k, "left_mm")
    model.addContactConstraint(k, "left_mp")
    model.addContactConstraint(k, "left_pp")
    model.addContactConstraint(k, "left_pm")
    model.addConstraints(k, left_mm_vel_bound)
    model.addConstraints(k, left_mp_vel_bound)
    model.addConstraints(k, left_pp_vel_bound)
    model.addConstraints(k, left_pm_vel_bound)

    init_controls.append(init_control)

model.addCost(H, joint_angle_cost)
model.addCost(H, joint_vel_cost)
model.addConstraints(
    H,
    constraints.FrameVelocityBound(
        model, "torso_1_link", np.zeros(3), np.zeros(3), nq, nv, True
    ),
)

model.finalize(init_states=[init], init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    filter_size=10,
    tolerance=1e-1,
    admm_abs_tolerance=1e-2,
    admm_rel_tolerance=1e-3,
    admm_max_iter=500,
)

state, x, u = solver.solve()
solver.write("../data/talos_jump")

print("x:")
x.print()
print("u:")
u.print()
