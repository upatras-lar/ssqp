import ssqpy
import numpy as np
from time import time
import sys

sys.path.insert(0, "../../models")

import go2_kinematic

go2_kinematic.load_model("../../assets/mjcf/go2/go2.xml")


def ssqp_to_mujoco_config(ssqp_config):
    mujoco_config = ssqp_config.copy()
    mujoco_config[3:7] = go2_kinematic.model_utils.quat_xyzw_to_wxyz(
        ssqp_config[3:7]
    )
    mujoco_config[7:10] = ssqp_config[10:13]
    mujoco_config[10:13] = ssqp_config[7:10]
    mujoco_config[13:16] = ssqp_config[16:19]
    mujoco_config[16:19] = ssqp_config[13:16]
    return mujoco_config


def ssqp_to_mujoco_vel(ssqp_vel):
    mujoco_vel = ssqp_vel.copy()
    mujoco_vel[6:9] = ssqp_vel[9:12]
    mujoco_vel[9:12] = ssqp_vel[6:9]
    mujoco_vel[12:15] = ssqp_vel[15:18]
    mujoco_vel[15:18] = ssqp_vel[12:15]
    return mujoco_vel


def mujoco_to_ssqp_config(mujoco_config):
    ssqp_config = mujoco_config.copy()
    ssqp_config[3:7] = go2_kinematic.model_utils.quat_wxyz_to_xyzw(
        mujoco_config[3:7]
    )
    ssqp_config[7:10] = mujoco_config[10:13]
    ssqp_config[10:13] = mujoco_config[7:10]
    ssqp_config[13:16] = mujoco_config[16:19]
    ssqp_config[16:19] = mujoco_config[13:16]
    return ssqp_config


v_cmd = np.array((-0.02, 0.0, 0.0))

MPC_H = 15
dt = 0.05

model = ssqpy.model.Model(MPC_H, dt)

com_ref, xb_ref, p_ref = go2_kinematic.generate_references(
    v_cmd, go2_kinematic.init_config, np.zeros(3), 0, 2000, dt
)

constr = go2_kinematic.KinematicConstraints()

stage_costs = []
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
    dynamics=[go2_kinematic.KinematicDynamics(dt) for _ in range(MPC_H + 1)],
    init_states=[go2_kinematic.init_config],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=400)

import mujoco
import mujoco.viewer

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_config(go2_kinematic.init_config)

np.set_printoptions(suppress=True)

step = 0

ssqpy.setSilentMode()

Kp = 80.0
Kv = 2.0 * np.sqrt(Kp)

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        # print(f"\rSimulation time: {mjdata.time:.2f}s")
        # frame_start = time()

        pos = mujoco_to_ssqp_config(mjdata.qpos)
        vel = ssqp_to_mujoco_vel(mjdata.qvel)

        # pre = time()

        p, v = mpc.step(pos)

        next_pos = p.stage(1)
        next_vel = v.stage(0)

        # mjdata.qpos = next_pos
        # mjdata.qpos[3:7] = go2_kinematic.model_utils.quat_xyzw_to_wxyz(
        #     mjdata.qpos[3:7]
        # )

        mjdata.ctrl = -Kp * (pos[7:] - next_pos[7:]) - Kv * (
            vel[6:] - next_vel[6:]
        )

        # elapsed = time() - pre

        step += 1
        # com_ref, xb_ref, p_ref = go2_kinematic.generate_references(
        #     v_cmd, mjpos, np.hstack((mjvel[:2], mjvel[5])), step, MPC_H, dt
        # )

        for k, cost in enumerate(stage_costs):
            stage_p_ref = {foot: p_ref[foot][step + k] for foot in p_ref}
            cost.update_references(
                com_ref[step + k],
                xb_ref[step + k],
                stage_p_ref,
                go2_kinematic.init_config[7:],
                np.zeros(go2_kinematic.nj),
            )

        # print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

print()
