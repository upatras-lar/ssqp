import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
import time
from go2_controllers.JointSpacePD import KinematicJointSpacePD
from utils_go2_mujoco import (
    ssqp_to_mujoco_state_kin,
    ssqp_to_mujoco_vel_kin,
    ssqp_to_mujoco_ctrl_kin,
    mujoco_to_ssqp_state_kin,
)

import models.quadruped_hlplanner.planner as go2_planner

go2_planner.config.ContactScheduling.T = 25
go2_planner.config.ContactScheduling.hip_to_foot_offsets = {
    "FL": np.array((0.3, 0.14, -0.38797938)),
    "FR": np.array((0.3, -0.14, -0.38797938)),
    "RL": np.array((-0.15, 0.14, -0.38797938)),
    "RR": np.array((-0.15, -0.14, -0.38797938)),
}
go2_planner.config.ContactScheduling.calculate()

go2_planner.config.ReferenceGeneration.h_sw = 0.1

ssqpy.setSilentMode()

v_cmd = np.array((0.0, 0.0, 0.0))

MPC_H = 10
dt = 0.02

model = ssqpy.model.Model(
    MPC_H,
    dt,
    mjcf_path="../../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.Kinematics,
)

com_ref, xb_ref, p_ref = go2_planner.generate_references_MPC(
    v_cmd,
    go2_planner.config.INIT_CONFIG,
    np.zeros(3),
    0.0,
    MPC_H,
    dt,
    ssqp_model=model,
)

q_q = 1e-3
q_v = 1e-3
q_c = 1.5
q_b = 3.0
q_o = 1.0
q_sw = 1.5
q_sp = 10.0
l_sigma = 0.5

config_cost = costs.SquaredConfigurationErrorCost(
    model,
    go2_planner.config.INIT_JOINT_CONFIG,
    q_q,
    "Configuration Regularization Cost",
)

vel_cost = costs.SquaredControlCost(model, q_v, "Velocity Regularization Cost")

base_height_constraint = constraints.FramePositionBound(
    model,
    "base_link",
    np.array(
        (-np.inf, -np.inf, go2_planner.config.ReferenceGeneration.zb_min)
    ),
    np.array((np.inf, np.inf, go2_planner.config.ReferenceGeneration.zb_max)),
    "Base Height Constraint",
)

com_costs = []
pose_costs = []
feet_costs = {foot: [] for foot in go2_planner.config.FRAME_NAMES}
feet_constraints = {foot: [] for foot in go2_planner.config.FRAME_NAMES}
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
        go2_planner.utils.quat_to_rotation_matrix(xb_ref[k][3:7]),
        q_b,
        q_o,
        f"Pose Cost {k}",
    )

    pose_costs.append(pose_cost)
    model.addCost(k, pose_cost)

    for foot in p_ref:
        foot_translation_cost = costs.FrameSquaredTranslationErrorCost(
            model,
            go2_planner.config.FRAME_NAMES[foot],
            p_ref[foot][k][0],
            q_sw,
            f"Foot {foot} Swing Cost {k}",
        )

        feet_costs[foot].append(foot_translation_cost)
        model.addCost(k, foot_translation_cost)

        foot_translation_bound = constraints.FramePositionBound(
            model,
            go2_planner.config.FRAME_NAMES[foot],
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

    line_segment_cost = go2_planner.LineSegmentCost(
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
    init_states=[go2_planner.config.INIT_CONFIG],
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
    com_ref, xb_ref, p_ref = go2_planner.generate_references_MPC(
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
            go2_planner.utils.quat_to_rotation_matrix(xb_ref[k][3:7])
        )

        stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

        for foot in go2_planner.config.FRAME_NAMES:
            if stage_p_ref[foot][1]:  # swing
                feet_constraints[foot][k].set_enabled(False)
                feet_costs[foot][k].set_enabled(True)
                feet_costs[foot][k].set_desired_translation(
                    stage_p_ref[foot][0]
                )
            else:  # stance
                feet_costs[foot][k].set_enabled(False)
                feet_constraints[foot][k].set_enabled(True)
                feet_constraints[foot][k].set_lower_bound(stage_p_ref[foot][0])
                feet_constraints[foot][k].set_upper_bound(stage_p_ref[foot][0])

        line_segment_costs[k].update_reference(stage_p_ref)


Kp = 40.0
Kv = 2.0 * np.sqrt(Kp)

joint_space_pd_controller = KinematicJointSpacePD(
    mpc,
    go2_planner.config.INIT_CONFIG,
    Kp=Kp,
    Kd=Kv,
    update_func=update_func,
    max_vel_front=go2_planner.config.ReferenceGeneration.max_vel_front,
    max_vel_back=go2_planner.config.ReferenceGeneration.max_vel_back,
    max_vel_side=go2_planner.config.ReferenceGeneration.max_vel_side,
    max_vel_yaw=go2_planner.config.ReferenceGeneration.max_vel_yaw,
)

joint_space_pd_controller.start_mpc()

import mujoco
import mujoco.viewer

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state_kin(go2_planner.config.INIT_CONFIG)


# mjmodel.opt.gravity[:] = [0.0, 0.0, 0.0]
# step = 0

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        # print(f"\rSimulation time: {mjdata.time:.2f}s")
        # frame_start = time.time()

        pos = mujoco_to_ssqp_state_kin(mjdata.qpos)
        vel = ssqp_to_mujoco_vel_kin(mjdata.qvel)

        # pre = time.time()

        # mjdata.qpos = ssqp_to_mujoco_state_kin(mpc.step(pos)[0].stage(1))

        mjdata.ctrl = ssqp_to_mujoco_ctrl_kin(
            joint_space_pd_controller.step(pos, vel)
        )

        # mpc_pos, mpc_vel = mpc.step(pos)

        # mjdata.ctrl = ssqp_to_mujoco_ctrl_kin(
        #     Kp * (mpc_pos.stage(1)[7:] - pos[7:])
        #     + Kv * (mpc_vel.stage(0)[6:] - vel[6:])
        # )

        # elapsed = time.time() - pre

        # step += 1
        # update_func(v_cmd, pos, vel, step)

        # print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t", end="")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        # time.sleep(0.01)

print()
