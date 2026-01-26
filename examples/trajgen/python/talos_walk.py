import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

nq = 51
nv = 50
H = 60
dt = 0.01

pos_wgt = 100
rot_wgt = 10

v_wgt = 1e-3
u_wgt = 1e-6
f_wgt = 1e-8

base_pos = np.array((0, 0, 1.072))

init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0

init_control = np.zeros(nv + 12)
for f in range(1, 5):
    init_control[nv + 3 * f - 1] = 116

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../urdf/talos/talos.urdf",
    free_flyer=True,
    # dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

pose_cost = costs.FrameSquaredPoseErrorCost(
    model,
    base_pos + np.array((4.0, 0, 0)),
    np.eye(3),
    "torso_1_link",
    nq,
    nv,
    pos_wgt,
    rot_wgt,
)

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

right_mm_vel_bound = constraints.FrameVelocityBound(
    model, "right_mm", np.zeros(3), np.zeros(3), nq, nv
)
right_mp_vel_bound = constraints.FrameVelocityBound(
    model, "right_mp", np.zeros(3), np.zeros(3), nq, nv
)
right_pp_vel_bound = constraints.FrameVelocityBound(
    model, "right_pp", np.zeros(3), np.zeros(3), nq, nv
)
right_pm_vel_bound = constraints.FrameVelocityBound(
    model, "right_pm", np.zeros(3), np.zeros(3), nq, nv
)

init_controls = []

for k in range(H):
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    if (k // 15) % 2 == 0:
        model.addContactConstraint(k, "left_mm")
        model.addContactConstraint(k, "left_mp")
        model.addContactConstraint(k, "left_pp")
        model.addContactConstraint(k, "left_pm")
        model.addConstraints(k, left_mm_vel_bound)
        model.addConstraints(k, left_mp_vel_bound)
        model.addConstraints(k, left_pp_vel_bound)
        model.addConstraints(k, left_pm_vel_bound)
    else:
        model.addContactConstraint(k, "right_mm")
        model.addContactConstraint(k, "right_mp")
        model.addContactConstraint(k, "right_pp")
        model.addContactConstraint(k, "right_pm")
        model.addConstraints(k, right_mm_vel_bound)
        model.addConstraints(k, right_mp_vel_bound)
        model.addConstraints(k, right_pp_vel_bound)
        model.addConstraints(k, right_pm_vel_bound)

    init_controls.append(init_control)

model.addCost(H, joint_vel_cost)
model.addCost(H, pose_cost)
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
    filter_size=5,
    tolerance=5e-1,
    admm_abs_tolerance=5e-2,
    admm_rel_tolerance=5e-2,
    admm_max_iter=250,
)

state, x, u = solver.solve()
solver.write("../data/talos_walk.json")

print("x:")
x.print()
print("u:")
u.print()
