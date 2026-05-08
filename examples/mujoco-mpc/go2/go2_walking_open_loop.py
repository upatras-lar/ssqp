import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep
from utils_go2_mujoco import (
    ssqp_to_mujoco_state,
    ssqp_to_mujoco_control,
)

ssqpy.setSilentMode()


traj_dt = 0.01
sim_dt = 0.002

steps_per_stage = int(traj_dt / sim_dt)

nq = 19
nv = 18
nu = 18

Kp = 1.8
Kd = 0.01

x, u = ssqpy.solvers.loadTrajectory("tmp.json")

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
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
