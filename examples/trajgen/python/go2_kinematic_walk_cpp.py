import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import sys

sys.path.insert(0, "../../models")

import go2_kinematic

v_cmd = np.array((0.5, 0.0, -0.4))

H = 500
dt = 0.01


model = ssqpy.model.Model(
    H,
    dt,
    mjcf_path="../../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.Kinematics,
)

nq = model.getnq()
nv = model.getnv()

com_ref, xb_ref, p_ref = go2_kinematic.generate_references(
    v_cmd, go2_kinematic.init_config, np.zeros(3), 0.0, H, dt, ssqp_model=model
)

q_q = 1e-2
q_v = 1e-3
q_c = 0.1
q_b = 0.1
q_o = 0.1
q_sw = 0.5

q_sp = 7.0
l_sigma = 0.5

config_cost = costs.SquaredConfigurationErrorCost(
    model,
    go2_kinematic.init_config[7:],
    q_q,
    "Configuration Regularization Cost",
)

vel_cost = costs.SquaredControlCost(model, q_v, "Velocity Regularization Cost")

base_height_constraint = constraints.FramePositionBound(
    model,
    "base_link",
    np.array((-np.inf, -np.inf, go2_kinematic.zb_min)),
    np.array((np.inf, np.inf, go2_kinematic.zb_max)),
    "Base Height Constraint",
)

for k in range(H + 1):
    com_cost = costs.CenterOfMassTrackingCost(
        model, com_ref[k], q_c, f"CoM Cost {k}"
    )

    model.addCost(k, com_cost)

    pose_cost = costs.FrameSquaredPoseErrorCost(
        model,
        "base_link",
        xb_ref[k][:3],
        go2_kinematic.model_utils.quaternion_to_rotation_matrix(
            xb_ref[k][3:7]
        ),
        q_b,
        q_o,
        f"Pose Cost {k}",
    )

    model.addCost(k, pose_cost)

    for foot in p_ref:
        if p_ref[foot][k][1]:  # swing
            foot_translation_cost = costs.FrameSquaredTranslationErrorCost(
                model,
                go2_kinematic.frame_names[foot],
                p_ref[foot][k][0],
                q_sw,
                f"Foot {foot} Swing Cost {k}",
            )

            model.addCost(k, foot_translation_cost)
        else:
            foot_translation_bound = constraints.FramePositionBound(
                model,
                go2_kinematic.frame_names[foot],
                p_ref[foot][k][0],
                p_ref[foot][k][0],
                f"Foot {foot} Stance Constraint {k}",
            )

            model.addConstraints(k, foot_translation_bound)

    line_segment_cost = go2_kinematic.LineSegmentCost(
        model,
        {foot: p_ref[foot][k] for foot in p_ref},
        q_sp,
        l_sigma,
        terminal=k == H,
    )

    model.addCost(k, line_segment_cost)

    model.addCost(k, config_cost)

    if k != H:
        model.addCost(k, vel_cost)

    model.addConstraints(k, base_height_constraint)

model.finalize(
    init_states=[go2_kinematic.init_config],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)

step = 0
solver = ssqpy.solvers.ssqp(model, ssqp_params)
state, x, u = solver.solve()
print("Solver state:", state)
solver.write("../data/go2_kinematic_walk_cpp.json")
