import ssqpy
import numpy as np

import models.quadruped_hlplanner.planner as go2_planner
import models.quadruped_srbd.model as go2_srbd

v_cmd = np.array((1.0, 0.0, 0.0))

H = 100
dt = 0.01

model = ssqpy.model.Model(H, dt)

com_ref, xb_ref, p_ref = go2_planner.generate_references(
    v_cmd, go2_planner.config.INIT_CONFIG, np.zeros(3), 0.0, H, dt
)

stage_costs = []
stage_constraints = []

for k in range(H):
    stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

    cost = go2_srbd.SRBCost(com_ref[k], xb_ref[k], stage_p_ref)
    constr = go2_srbd.SRBConstraints(stage_p_ref)

    stage_costs.append(cost)
    stage_constraints.append(constr)

    model.addCost(k, cost)
    model.addConstraints(k, constr)

stage_p_ref = {foot: p_ref[foot][H] for foot in p_ref}

cost = go2_srbd.SRBCost(com_ref[H], xb_ref[H], stage_p_ref, terminal=True)
stage_costs.append(cost)
model.addCost(H, cost)

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
