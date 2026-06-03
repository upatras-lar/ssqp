import ssqpy
import numpy as np

import models.quadruped_hlplanner.planner as go2_planner
import models.quadruped_kinematic.model as go2_kinematic

v_cmd = np.array((1.0, 0.0, 0.0))

H = 500
dt = 0.01

model = ssqpy.model.Model(H, dt)

com_ref, xb_ref, p_ref = go2_planner.generate_references(
    v_cmd,
    go2_planner.config.INIT_CONFIG,
    np.zeros(3),
    0.0,
    H,
    dt,
)

stage_costs = []
constr = go2_kinematic.KinematicConstraints()

for k in range(H):
    stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

    cost = go2_kinematic.KinematicCost(
        com_ref[k],
        xb_ref[k],
        stage_p_ref,
        go2_planner.config.INIT_JOINT_CONFIG,
        np.zeros(go2_planner.config.KinematicParameters.nj),
    )
    stage_costs.append(cost)

    model.addCost(k, cost)
    model.addConstraints(k, constr)

stage_p_ref = {foot: p_ref[foot][H] for foot in p_ref}
cost = go2_kinematic.KinematicCost(
    com_ref[H],
    xb_ref[H],
    stage_p_ref,
    go2_planner.config.INIT_JOINT_CONFIG,
    np.zeros(go2_planner.config.KinematicParameters.nj),
    terminal=True,
)
stage_costs.append(cost)

model.addCost(H, cost)

model.finalize(
    dynamics=[go2_kinematic.Kinematics(dt) for _ in range(H + 1)],
    init_states=[go2_planner.config.INIT_CONFIG],
)

custom_properties = ssqpy.model.CustomProperties()
custom_properties.nq = go2_planner.config.KinematicParameters.nq
custom_properties.nv = go2_planner.config.KinematicParameters.ndq
custom_properties.nu = go2_planner.config.KinematicParameters.ndq
custom_properties.njoints = go2_planner.config.KinematicParameters.nj
custom_properties.floating_base = True
custom_properties.ordered_joints = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
custom_properties.joint_types = ["revolute"] * 12

model.setCustomProperties(custom_properties)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

step = 0
solver = ssqpy.solvers.ssqp(model, ssqp_params)
state, x, u = solver.solve()
print("Solver state:", state)
solver.write("../data/go2_kinematic_walk.json")
