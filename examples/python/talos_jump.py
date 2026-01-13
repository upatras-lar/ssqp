import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

nq = 51
nv = 50
H = 30
dt = 0.01

STEPS_BEFORE = 10
STEPS_AFTER = 10

pos_wgt = 20
rot_wgt = 10

a_wgt = 1e-1
v_wgt = 1e-3
u_wgt = 1e-6
f_wgt = 1e-8

base_pos = np.array((0, 0, 1.072))

init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0

init_control = np.zeros(nv + 24)
for f in range(1, 9):
    init_control[nv + 3 * f - 1] = 116

forces = {}
forces["left_mm"] = [init_control[nv : nv + 3]]
forces["left_mp"] = [init_control[nv + 3 : nv + 6]]
forces["left_pp"] = [init_control[nv + 6 : nv + 9]]
forces["left_pm"] = [init_control[nv + 9 : nv + 12]]
forces["right_mm"] = [init_control[nv + 12 : nv + 15]]
forces["right_mp"] = [init_control[nv + 15 : nv + 18]]
forces["right_pp"] = [init_control[nv + 18 : nv + 21]]
forces["right_pm"] = [init_control[nv + 21 : nv + 24]]

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

pose_cost = costs.FrameSquaredPoseErrorCost(
    model,
    base_pos + np.array((0, 0, 0)),
    np.eye(3),
    "torso_1_link",
    nq,
    nv,
    pos_wgt,
    rot_wgt,
)

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
    model.addCost(k, joint_angle_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    if k < STEPS_BEFORE or k > STEPS_AFTER:
        model.addContactConstraint(k, "left_mm")
        model.addContactConstraint(k, "left_mp")
        model.addContactConstraint(k, "left_pp")
        model.addContactConstraint(k, "left_pm")
        model.addConstraints(k, left_mm_vel_bound)
        model.addConstraints(k, left_mp_vel_bound)
        model.addConstraints(k, left_pp_vel_bound)
        model.addConstraints(k, left_pm_vel_bound)

        model.addContactConstraint(k, "right_mm")
        model.addContactConstraint(k, "right_mp")
        model.addContactConstraint(k, "right_pp")
        model.addContactConstraint(k, "right_pm")
        model.addConstraints(k, right_mm_vel_bound)
        model.addConstraints(k, right_mp_vel_bound)
        model.addConstraints(k, right_pp_vel_bound)
        model.addConstraints(k, right_pm_vel_bound)

        init_controls.append(init_control)
    else:
        init_controls.append(np.zeros(nv))

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
    tolerance=1e-2,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-4,
    admm_max_iter=500,
)

state, x, u = solver.solve()
solver.write("../data/talos_jump")

print("x:")
x.print()
print("u:")
u.print()
