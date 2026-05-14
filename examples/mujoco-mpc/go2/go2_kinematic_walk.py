import ssqpy
import numpy as np
import time
from go2_controllers.JointSpacePD import KinematicJointSpacePD
from utils_go2_mujoco import (
    ssqp_to_mujoco_state_kin,
    ssqp_to_mujoco_vel_kin,
    ssqp_to_mujoco_ctrl_kin,
    mujoco_to_ssqp_state_kin,
)

import sys

sys.path.insert(0, "../../models")

import go2_kinematic

# go2_kinematic.init_config[3:7] = np.array((0.0, 0.0, 1.0, 0.0))
go2_kinematic.hip_to_foot_offsets = {
    "FL": np.array((0.35, 0.16, -0.36892682)),
    "FR": np.array((0.35, -0.16, -0.36892682)),
    "RL": np.array((-0.15, 0.16, -0.36892682)),
    "RR": np.array((-0.15, -0.16, -0.36892682)),
}

go2_kinematic.load_model("../../assets/mjcf/go2/go2.xml")

ssqpy.setSilentMode()

v_cmd = np.array((0.0, 0.0, 0.0))

MPC_H = 15
dt = 0.01

model = ssqpy.model.Model(MPC_H, dt)

com_ref, xb_ref, p_ref = go2_kinematic.generate_references(
    v_cmd, go2_kinematic.init_config, np.zeros(3), 0, MPC_H, dt
)

stage_costs = []
constr = go2_kinematic.KinematicConstraints()

for k in range(MPC_H):
    stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

    cost = go2_kinematic.KinematicCost(
        com_ref[k],
        xb_ref[k],
        stage_p_ref,
        go2_kinematic.init_config[7:],
        np.zeros(go2_kinematic.nj),
    )
    stage_costs.append(cost)

    model.addCost(k, cost)
    model.addConstraints(k, constr)

stage_p_ref = {foot: p_ref[foot][MPC_H] for foot in p_ref}
cost = go2_kinematic.KinematicCost(
    com_ref[MPC_H],
    xb_ref[MPC_H],
    stage_p_ref,
    go2_kinematic.init_config[7:],
    np.zeros(go2_kinematic.nj),
    terminal=True,
)
stage_costs.append(cost)

model.addCost(MPC_H, cost)

model.finalize(
    dynamics=[go2_kinematic.Kinematics(dt) for _ in range(MPC_H + 1)],
    init_states=[go2_kinematic.init_config],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=100)

import mujoco
import mujoco.viewer

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state_kin(go2_kinematic.init_config)

np.set_printoptions(suppress=True)


def update_func(config, vel, step):
    vel_3 = np.hstack((vel[:2], vel[5]))
    com_ref, xb_ref, p_ref = go2_kinematic.generate_references_MPC(
        v_cmd, config, vel_3, step, MPC_H, dt
    )

    for k, cost in enumerate(stage_costs):
        stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}
        cost.update_references(
            com_ref[k],
            xb_ref[k],
            stage_p_ref,
            go2_kinematic.init_config[7:],
            np.zeros(go2_kinematic.nj),
        )

        model.setInitState(k, config)


Kp = 60.0
Kv = 2.0 * np.sqrt(Kp)

joint_space_pd_controller = KinematicJointSpacePD(
    mpc, go2_kinematic.init_config, Kp=Kp, Kd=Kv, update_func=update_func
)

# mjmodel.opt.gravity[:] = [0.0, 0.0, 0.0]
step = 0

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        # print(f"\rSimulation time: {mjdata.time:.2f}s")
        # frame_start = time.time()

        pos = mujoco_to_ssqp_state_kin(mjdata.qpos)
        vel = ssqp_to_mujoco_vel_kin(mjdata.qvel)

        pre = time.time()

        # mjdata.qpos = ssqp_to_mujoco_state_kin(mpc.step(pos)[0].stage(1))

        mjdata.ctrl = ssqp_to_mujoco_ctrl_kin(
            joint_space_pd_controller.step(pos, vel)
        )

        # mpc_pos, mpc_vel = mpc.step(pos)

        # mjdata.ctrl = ssqp_to_mujoco_ctrl_kin(
        #     Kp * (mpc_pos.stage(1)[7:] - pos[7:])
        #     + Kv * (mpc_vel.stage(1)[6:] - vel[6:])
        # )

        elapsed = time.time() - pre

        step += 1
        update_func(pos, vel, step)

        # print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t", end="")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        # time.sleep(0.01)

print()
