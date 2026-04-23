import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep

ssqpy.setSilentMode()


def quat_wxyz_to_xyzw(wxyz):
    xyzw = np.zeros(4)
    xyzw[:3] = wxyz[1:]
    xyzw[3] = wxyz[0]
    return xyzw


def quat_xyzw_to_wxyz(xyzw):
    wxyz = np.zeros(4)
    wxyz[0] = xyzw[3]
    wxyz[1:] = xyzw[:3]
    return wxyz


def ssqp_to_mujoco_state(ssqp_state):
    mujoco_state = ssqp_state.copy()
    mujoco_state[3:7] = quat_xyzw_to_wxyz(ssqp_state[3:7])
    mujoco_state[7:10] = ssqp_state[10:13]
    mujoco_state[10:13] = ssqp_state[7:10]
    mujoco_state[13:16] = ssqp_state[16:19]
    mujoco_state[16:19] = ssqp_state[13:16]
    mujoco_state[25:28] = ssqp_state[28:31]
    mujoco_state[28:31] = ssqp_state[25:28]
    mujoco_state[31:34] = ssqp_state[34:37]
    mujoco_state[34:37] = ssqp_state[31:34]
    return mujoco_state


def ssqp_to_mujoco_control(ssqp_control):
    mujoco_control = ssqp_control.copy()
    mujoco_control[6:9] = ssqp_control[9:12]
    mujoco_control[9:12] = ssqp_control[6:9]
    mujoco_control[12:15] = ssqp_control[15:18]
    mujoco_control[15:18] = ssqp_control[12:15]
    return mujoco_control


traj_dt = 0.01
sim_dt = 0.002

steps_per_stage = int(traj_dt / sim_dt)

nq = 19
nv = 18
nu = 18

Kp = 1.8
Kd = 0.01

x, u = ssqpy.solvers.loadTrajectory("tmp.json")

mjmodel = mujoco.MjModel.from_xml_path("../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state(x.stage(0))[:nq]

np.set_printoptions(suppress=True)

k = 0
step = 0
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        print(f"\rSimulation time: {mjdata.time:.2f}s")
        frame_start = time()

        state = np.hstack((mjdata.qpos, mjdata.qvel))

        pre = time()

        xk = ssqp_to_mujoco_state(x.stage(k + 1))
        uk = ssqp_to_mujoco_control(u.stage(k))

        mjdata.qpos[:3] = np.array((0.0, 0.0, 0.5))
        # mjdata.qpos[0:19] = xk[0:19]

        mjdata.ctrl = (
            Kp * (xk[7:19] - state[7:19])
            + Kd * (xk[25:] - state[25:])
            + uk[6:18]
        )
        print(mjdata.ctrl)
        step += 1

        if step % steps_per_stage == 0:
            k += 1

        elapsed = time() - pre

        print(f"\rControl frequency: {(1 / elapsed):.2f}Hz\t")

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        if k == u.stages():
            break

        sleep(0.01)

print()
