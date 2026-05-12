import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import sys

sys.path.insert(0, "../../models")

import go2_kinematic

v_cmd = np.array((0.2, 0.0, 0.0))

H = 500
dt = 0.01

model = ssqpy.model.Model(
    H,
    dt,
    mjcf_path="../../assets/mjcf/go2/go2.xml",
    solver_mode=ssqpy.model.SolverMode.Kinematics,
)

nq = model.getnq()
nv = model.getnv()

com_ref, xb_ref, p_ref = go2_kinematic.generate_references(
    v_cmd, go2_kinematic.init_config, np.zeros(3), 0.0, H, dt, ssqp_model=model
)

config_cost = costs.SquaredConfigurationErrorCost(
    model,
    go2_kinematic.init_config[7:],
    go2_kinematic.q_q,
    "Configuration Regularization Cost",
)

vel_cost = costs.SquaredControlCost(
    model, go2_kinematic.q_v, "Velocity Regularization Cost"
)


base_height_constraint = constraints.FramePositionBound(
    model,
    "base_link",
    np.array((-np.inf, -np.inf, go2_kinematic.zb_min)),
    np.array((np.inf, np.inf, go2_kinematic.zb_max))
    "Base Height Constraint",
)

for k in range(H):
    com_cost = costs.CenterOfMassTrackingCost(
        model, com_ref[k], go2_kinematic.q_c, f"CoM Cost {k}"
    )

    model.addCost(k, com_cost)

    desired_pose = np.zeros((4, 4))
    desired_pose[:3, 3] = xb_ref[k][:3]
    desired_pose[:3, :3] = go2_kinematic.model_utils.quaternion_to_rotation_matrix(
        xb_ref[k][3:7]
    )
    desired_pose[3, 3] = 1.0

    pose_cost = costs.FrameSquaredPoseErrorCost(
        model,
        "base_link",
        desired_pose,
        go2_kinematic.q_b,
        go2_kinematic.q_o,
        f"Pose Cost {k}",
    )

    model.addCost(k, pose_cost)

    for foot in p_ref:
        if p_ref[foot][k][1]:  # swing
            foot_translation_cost = costs.FrameSquaredTranslationErrorCost(
                model,
                go2_kinematic.frame_names[foot],
                p_ref[foot][k][0],
                go2_kinematic.q_sw,
                f"Foot {foot} Swing Cost {k}",
            )
        else:
            foot_translation_cost = costs.FrameSquaredTranslationErrorCost(
                model,
                go2_kinematic.frame_names[foot],
                p_ref[foot][k][0],
                go2_kinematic.q_st,
                f"Foot {foot} Stance Cost {k}",
            )

        model.addCost(k, foot_translation_cost)

    model.addCost(k, config_cost)
    model.addCost(k, vel_cost)

    model.addConstraints(k, base_height_constraint)

model.addCost(H, config_cost)

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

ssqpy.setLogLevel(ssqpy.LogLevel.Debug)

step = 0
solver = ssqpy.solvers.ssqp(model, ssqp_params)
state, x, u = solver.solve()
print("Solver state:", state)
solver.write("../data/go2_kinematic_walk_cpp.json")
