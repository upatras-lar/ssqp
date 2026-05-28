import ssqpy
import numpy as np

import models.quadruped_hlplanner.planner as go2_planner
import models.quadruped_srbd.model as go2_srbd

v_cmd = np.array((1.0, 0.0, 0.0))

H = 200
dt = 0.01

model = ssqpy.model.Model(H, dt)

com_ref, xb_ref, p_ref = go2_planner.generate_references(
    v_cmd, go2_planner.config.INIT_CONFIG, np.zeros(3), 0.0, H, dt
)

for i, p in enumerate(p_ref["RR"]):
    print(f"Position {i}: {p[0]}, swing: {p[1]}")

for k in range(H):
    stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

    com_cost = go2_srbd.CoMTrackingCost(com_ref[k])
    orientation_cost = go2_srbd.OrientationTrackingCost(xb_ref[k])
    angvel_cost = go2_srbd.AngularVelocityCost()
    foot_cost = go2_srbd.FootTrackingCost(stage_p_ref)
    control_cost = go2_srbd.ControlRegularizationCost(stage_p_ref)

    model.addCost(k, com_cost)
    model.addCost(k, orientation_cost)
    model.addCost(k, angvel_cost)
    model.addCost(k, foot_cost)
    model.addCost(k, control_cost)

    height_constraint = go2_srbd.CoMHeightConstraint()
    model.addConstraints(k, height_constraint)

    for i, foot in enumerate(stage_p_ref):
        foot_constraint = go2_srbd.FrictionAndForcesConstraint(stage_p_ref, i)
        foot_vel_constraint = go2_srbd.FootVelocityConstraint(stage_p_ref, i)

        model.addConstraints(k, foot_constraint)
        model.addConstraints(k, foot_vel_constraint)


stage_p_ref = {foot: p_ref[foot][H] for foot in p_ref}

com_cost = go2_srbd.CoMTrackingCost(com_ref[H])
orientation_cost = go2_srbd.OrientationTrackingCost(xb_ref[H])
angvel_cost = go2_srbd.AngularVelocityCost()
foot_cost = go2_srbd.FootTrackingCost(stage_p_ref)

model.addCost(H, com_cost)
model.addCost(H, orientation_cost)
model.addCost(H, angvel_cost)
model.addCost(H, foot_cost)

height_constraint = go2_srbd.CoMHeightConstraint()
model.addConstraints(H, height_constraint)

model.finalize(
    dynamics=[
        go2_srbd.SRBDynamics(
            dt,
            go2_srbd.config.SRBDParameters.mass,
            go2_srbd.config.SRBDParameters.inertia,
        )
        for _ in range(H + 1)
    ],
    init_states=[
        np.hstack(
            (
                go2_planner.config.SRBDParameters.INIT_STATE,
                *[
                    p_ref[foot][0][0]
                    for foot in go2_planner.config.FRAME_NAMES.keys()
                ],
            )
        )
    ],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.max_iter = 25
ssqp_params.loose_success = True

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

step = 0
solver = ssqpy.solvers.ssqp(model, ssqp_params)
state, x, u = solver.solve()
print("Solver state:", state)
solver.write("../data/go2_srbd_walk.json")
