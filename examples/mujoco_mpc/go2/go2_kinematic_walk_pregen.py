import ssqpy
import numpy as np
import time
from go2_controllers.JointSpacePD import KinematicJointSpacePD
from utils_go2_mujoco import (
    ssqp_to_mujoco_state_kin,
    ssqp_to_mujoco_vel_kin,
    mujoco_to_ssqp_state_kin,
    ssqp_to_mujoco_ctrl_kin,
    clip_joints_to_limits,
)

import models.quadruped_hlplanner.planner as go2_planner
import models.quadruped_kinematic.model as go2_kinematic

go2_planner.config.ContactScheduling.hip_to_foot_offsets = {
    "FL": np.array((0.4, 0.14, -0.38797938)),
    "FR": np.array((0.4, -0.14, -0.38797938)),
    "RL": np.array((-0.1, 0.14, -0.38797938)),
    "RR": np.array((-0.1, -0.14, -0.38797938)),
}

ssqpy.setSilentMode()


v_cmd = np.array((0.2, 0.0, 0.0))

MPC_H = 15
dt = 0.01

model = ssqpy.model.Model(MPC_H, dt)

com_ref, xb_ref, p_ref = go2_planner.generate_references(
    v_cmd,
    go2_planner.config.INIT_CONFIG,
    np.zeros(3),
    0,
    10000,
    dt,
)

stage_costs = []
constr = go2_kinematic.KinematicConstraints()

for k in range(MPC_H):
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

stage_p_ref = {foot: p_ref[foot][MPC_H] for foot in p_ref}
cost = go2_kinematic.KinematicCost(
    com_ref[MPC_H],
    xb_ref[MPC_H],
    stage_p_ref,
    go2_planner.config.INIT_JOINT_CONFIG,
    np.zeros(go2_planner.config.KinematicParameters.nj),
    terminal=True,
)
stage_costs.append(cost)

model.addCost(MPC_H, cost)

model.finalize(
    dynamics=[go2_kinematic.Kinematics(dt) for _ in range(MPC_H + 1)],
    init_states=[go2_planner.config.INIT_CONFIG],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=6, qp_iters=100)

import mujoco
import mujoco.viewer
import glfw

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state_kin(go2_planner.config.INIT_CONFIG)

np.set_printoptions(suppress=True)


def update_func(_1, _2, _3, step):
    for k, cost in enumerate(stage_costs):
        stage_p_ref = {foot: p_ref[foot][step + k] for foot in p_ref}
        cost.update_references(
            com_ref[step + k],
            xb_ref[step + k],
            stage_p_ref,
            go2_planner.config.INIT_JOINT_CONFIG,
            np.zeros(go2_planner.config.KinematicParameters.nj),
        )


Kp = 40.0
Kv = 2.0 * np.sqrt(Kp)

joint_space_pd_controller = KinematicJointSpacePD(
    mpc,
    go2_planner.config.INIT_CONFIG,
    Kp=Kp,
    Kd=Kv,
    update_func=update_func,
)

# Start the viewer maximized
glfw.init()
glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        pos = mujoco_to_ssqp_state_kin(mjdata.qpos)
        pos = clip_joints_to_limits(pos, mjmodel)
        vel = ssqp_to_mujoco_vel_kin(mjdata.qvel)

        pre = time.time()

        mjdata.ctrl = ssqp_to_mujoco_ctrl_kin(
            joint_space_pd_controller.step(pos, vel)
        )

        elapsed = time.time() - pre

        # print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t", end="")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        # time.sleep(0.1)

print()
