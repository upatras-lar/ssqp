import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

nq = 51
nv = 50
H = 100
dt = 0.01

pos_wgt = 1
rot_wgt = 1

v_wgt = 1e-4
u_wgt = 1e-6
f_wgt = 1e-8

base_pos = np.array((0, 0, 1.072))

init = np.zeros(nq + nv)
init[:3] = base_pos
init[6] = 1.0

init_control = np.zeros(nv + 24)
for f in range(9):
    init_control[nv + 3 * f - 1] = 116

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../examples/urdf/talos/talos.urdf",
    free_flyer=True,
    # dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

pose_cost = costs.FrameSquaredPoseErrorCost(
    model,
    base_pos,
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

left_vel_bound = constraints.FrameVelocityBound(
    model, "left_sole_link", np.zeros(3), np.zeros(3), nq, nv
)
right_vel_bound = constraints.FrameVelocityBound(
    model, "right_sole_link", np.zeros(3), np.zeros(3), nq, nv
)

init_controls = []

for k in range(H):
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)
    model.addCost(k, force_cost)

    if k < H // 3 or k > 2 * H // 3:
        model.addCost(k, pose_cost)

        model.addContactConstraint(k, "left_mm")
        model.addContactConstraint(k, "left_mp")
        model.addContactConstraint(k, "left_pp")
        model.addContactConstraint(k, "left_pm")
        model.addConstraints(k, left_vel_bound)

        model.addContactConstraint(k, "right_mm")
        model.addContactConstraint(k, "right_mp")
        model.addContactConstraint(k, "right_pp")
        model.addContactConstraint(k, "right_pm")
        model.addConstraints(k, right_vel_bound)

        init_controls.append(init_control)
    else:
        init_controls.append(np.zeros(nv))

model.addCost(H, joint_vel_cost)
model.addCost(H, pose_cost)

model.finalize(init_states=[init], init_controls=init_controls)

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-2,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-3,
    admm_max_iter=1000,
)

state, x, u = solver.solve()
solver.write("../data/talos_jump")

print("x:")
x.print()
print("u:")
u.print()
