import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np

import models.go2_kinematic.model as go2_kinematic

go2_kinematic.config.ContactScheduling.hip_to_foot_offsets = {
    "FL": np.array((0.3, 0.14, -0.36892682)),
    "FR": np.array((0.3, -0.14, -0.36892682)),
    "RL": np.array((-0.18, 0.14, -0.36892682)),
    "RR": np.array((-0.18, -0.14, -0.36892682)),
}

go2_kinematic.config.ReferenceGeneration.h_sw = 0.1

ssqpy.setSilentMode()


def get_controller(init_config):
    MPC_H = 10
    dt = 0.02

    model = ssqpy.model.Model(
        MPC_H,
        dt,
        mjcf_path="../../assets/mjcf/go2/go2.xml",
        free_flyer=True,
        solver_mode=ssqpy.model.SolverMode.Kinematics,
    )

    com_ref, xb_ref, p_ref = go2_kinematic.generate_references_MPC(
        np.array((0.0, 0.0, 0.0)),
        init_config,
        np.zeros(3),
        0.0,
        MPC_H,
        dt,
        ssqp_model=model,
    )

    q_q = 1e-1
    q_v = 1e-4
    q_c = 1.5
    q_b = 3.0
    q_o = 6.0
    q_sw = 2.0
    q_sp = 10.0
    l_sigma = 0.44

    config_cost = costs.SquaredConfigurationErrorCost(
        model,
        init_config[7:],
        q_q,
        "Configuration Regularization Cost",
    )

    vel_cost = costs.SquaredControlCost(
        model, q_v, "Velocity Regularization Cost"
    )

    base_height_constraint = constraints.FramePositionBound(
        model,
        "base_link",
        np.array(
            (
                -np.inf,
                -np.inf,
                go2_kinematic.config.ReferenceGeneration.zb_min + 0.03,
            )
        ),
        np.array(
            (
                np.inf,
                np.inf,
                go2_kinematic.config.ReferenceGeneration.zb_max + 0.03,
            )
        ),
        "Base Height Constraint",
    )

    com_costs = []
    pose_costs = []
    feet_costs = {foot: [] for foot in go2_kinematic.config.FRAME_NAMES}
    feet_constraints = {foot: [] for foot in go2_kinematic.config.FRAME_NAMES}
    line_segment_costs = []

    for k in range(MPC_H + 1):
        com_cost = costs.CenterOfMassTrackingCost(
            model, com_ref[k], q_c, f"CoM Cost {k}"
        )

        com_costs.append(com_cost)
        model.addCost(k, com_cost)

        pose_cost = costs.FrameSquaredPoseErrorCost(
            model,
            "base_link",
            xb_ref[k][:3],
            go2_kinematic.utils.quaternion_to_rotation_matrix(xb_ref[k][3:7]),
            q_b,
            q_o,
            f"Pose Cost {k}",
        )

        pose_costs.append(pose_cost)
        model.addCost(k, pose_cost)

        for foot in p_ref:
            foot_translation_cost = costs.FrameSquaredTranslationErrorCost(
                model,
                go2_kinematic.config.FRAME_NAMES[foot],
                p_ref[foot][k][0],
                q_sw,
                f"Foot {foot} Swing Cost {k}",
            )

            feet_costs[foot].append(foot_translation_cost)
            model.addCost(k, foot_translation_cost)

            foot_translation_bound = constraints.FramePositionBound(
                model,
                go2_kinematic.config.FRAME_NAMES[foot],
                p_ref[foot][k][0],
                p_ref[foot][k][0],
                f"Foot {foot} Stance Constraint {k}",
            )

            feet_constraints[foot].append(foot_translation_bound)
            model.addConstraints(k, foot_translation_bound)

            if p_ref[foot][k][1]:  # swing
                foot_translation_bound.set_enabled(False)
            else:  # stance
                foot_translation_cost.set_enabled(False)

        line_segment_cost = go2_kinematic.LineSegmentCost(
            model,
            {foot: p_ref[foot][k] for foot in p_ref},
            q_sp,
            l_sigma,
            terminal=k == MPC_H,
        )

        line_segment_costs.append(line_segment_cost)
        model.addCost(k, line_segment_cost)

        model.addCost(k, config_cost)

        if k != MPC_H:
            model.addCost(k, vel_cost)

        model.addConstraints(k, base_height_constraint)

    model.finalize(
        init_states=[init_config],
    )

    ssqp_params = ssqpy.solvers.ssqpParams()
    ssqp_params.tolerance = 1e-1

    admm_params = ssqpy.solvers.admmParams()
    admm_params.abs_tolerance = 1e-2
    admm_params.rel_tolerance = 1e-2
    admm_params.warm_start = True
    admm_params.init_rho = 10.0
    admm_params.adaptive_rho_tolerance = 0.1

    ssqp_params.admmParams = admm_params

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=3, qp_iters=200)

    np.set_printoptions(suppress=True)

    # ssqpy.setLogLevel(ssqpy.LogLevel.Debug)

    def update_func(v_cmd, config, vel, step):
        vel_3 = np.hstack((vel[:2], vel[5]))
        com_ref, xb_ref, p_ref = go2_kinematic.generate_references_MPC(
            v_cmd,
            config,
            vel_3,
            step,
            MPC_H,
            dt,
            ssqp_model=model,
        )

        for k in range(MPC_H + 1):
            com_costs[k].set_cref(com_ref[k])
            pose_costs[k].set_desired_translation(xb_ref[k][:3])
            pose_costs[k].set_desired_rotation(
                go2_kinematic.utils.quaternion_to_rotation_matrix(
                    xb_ref[k][3:7]
                )
            )

            stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

            for foot in go2_kinematic.config.FRAME_NAMES:
                if stage_p_ref[foot][1]:  # swing
                    feet_constraints[foot][k].set_enabled(False)
                    feet_costs[foot][k].set_enabled(True)
                    feet_costs[foot][k].set_desired_translation(
                        stage_p_ref[foot][0]
                    )
                else:  # stance
                    feet_costs[foot][k].set_enabled(False)
                    feet_constraints[foot][k].set_enabled(True)
                    feet_constraints[foot][k].set_lower_bound(
                        stage_p_ref[foot][0]
                    )
                    feet_constraints[foot][k].set_upper_bound(
                        stage_p_ref[foot][0]
                    )

            line_segment_costs[k].update_reference(stage_p_ref)

    return mpc, update_func
