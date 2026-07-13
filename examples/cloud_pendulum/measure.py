import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from time import time, sleep
import pandas as pd

traj = pd.read_csv("data/trajectory-pos-50.csv")

t_data = traj["time"].to_numpy()
pos1_data = traj["pos1"].to_numpy()
vel1_data = traj["vel1"].to_numpy()
acc1_data = traj["acc1"].to_numpy()

pos2_data = traj["pos2"].to_numpy()
vel2_data = traj["vel2"].to_numpy()
acc2_data = traj["acc2"].to_numpy()

dt = 0.0033333

# Initial Parameter Guess
m1 = 0.100
m2 = 0.070
I1 = 0.0005
I2 = 0.0001
r1 = 0.012
r2 = 0.02
b1 = 0.0004
b2 = 0.0004
cf1 = 0.00305
cf2 = 0.000777

initial_params = np.array((r1 * m1, I1, cf1, b1, r2 * m2, m2, I2, cf2, b2))

bounds = np.array(
    [
        [0.0001, 0.5],  # r1*m1
        [0.00001, 0.2],  # I1
        [0.0, 0.5],  # cf1
        [0.0, 0.5],  # b1
        [0.0001, 0.5],  # r2*m2
        [0.06, 1.0],  # m2
        [0.00001, 0.2],  # I2
        [0.0, 0.5],  # cf2
        [0.0, 0.5],  # b2
    ]
).T

print("Initial parameters:", initial_params)

dynamics = ssqpy.model.custom.double_pendulum.DoublePendulumAmbientDynamics(
    initial_params,
    dt=dt,
    # actuation_mode=ssqpy.model.custom.double_pendulum.ActuationMode.Pendubot,
)

KP = 5.0
KD = 0.4
KI = 0.0


def get_ref(t):
    # clamp time inside dataset
    t = np.clip(t, t_data[0], t_data[-1])

    q_ref = [np.interp(t, t_data, pos1_data), np.interp(t, t_data, pos2_data)]
    dq_ref = [np.interp(t, t_data, vel1_data), np.interp(t, t_data, vel2_data)]
    ddq_ref = [
        np.interp(t, t_data, acc1_data),
        np.interp(t, t_data, acc2_data),
    ]

    return q_ref, dq_ref, ddq_ref


mjmodel = mujoco.MjModel.from_xml_path(
    "../assets/mjcf/double_pendulum/double_pendulum.xml"
)
mjdata = mujoco.MjData(mjmodel)

elapsed = 0
steps = 0
total_steps = int(1 / dt)

np.set_printoptions(suppress=True)

data_states = []
data_controls = []
data_accelerations = []

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running() and steps < len(t_data):
        frame_start = time()

        t = t_data[steps]

        q_ref, dq_ref, ddq_ref = get_ref(t)

        q = mjdata.qpos
        dq = mjdata.qvel

        # PD tracking
        u = [
            KI * ddq_ref[0]
            + KP * (q_ref[0] - q[0])
            + KD * (dq_ref[0] - dq[0]),
            KI * ddq_ref[1]
            + KP * (q_ref[1] - q[1])
            + KD * (dq_ref[1] - dq[1]),
        ]

        mjdata.ctrl = u

        steps += 1

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        state = np.hstack(
            (
                np.sin(mjdata.qpos[0]),
                np.cos(mjdata.qpos[0]),
                np.sin(mjdata.qpos[1]),
                np.cos(mjdata.qpos[1]),
                mjdata.qvel,
            )
        )

        data_states.append(state.copy())
        data_controls.append(mjdata.ctrl.copy())
        data_accelerations.append(mjdata.qacc.copy())

        to_sleep = mjmodel.opt.timestep - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)

N = len(data_states)
Y_stacked = np.zeros((2 * N, 9))
tau_stacked = np.zeros((2 * N))

for i, z in enumerate(zip(data_states, data_controls, data_accelerations)):
    state, control, accel = z

    Y, tau = dynamics.computeRegressorAndTorque(state, accel, control)

    Y_stacked[2 * i : 2 * i + 2, :] = Y
    tau_stacked[2 * i : 2 * i + 2] = tau

print(Y_stacked)

print(np.linalg.matrix_rank(Y_stacked))
print(np.linalg.cond(Y_stacked))

phi = np.linalg.lstsq(Y_stacked, tau_stacked)[0]
phi = np.clip(phi, bounds[0, :], bounds[1, :])

print("Optimal parameters:", phi)
