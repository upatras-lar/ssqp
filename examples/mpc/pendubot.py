import mujoco
import mujoco.viewer
import ssqpy
import numpy as np
from math import pi, sin, cos
from time import time, sleep

nq = 2
nv = 2
MPC_H = 400
dt = 0.01

cost_wgt = 10.0
vel_wgt = 1e-3
u_wgt = 1e-9


def state_mod(state):
    return [x % (2 * pi) for x in state]


class UprightCost(ssqpy.model.costs.StageCost):
    def evaluate(self, x, u):
        # q = np.array((sin(x[0]), sin(x[1]), cos(x[0]) + 1, cos(x[1]) - 1))
        # return 0.5 * cost_wgt * q.T * q
        return cost_wgt * (cos(x[0]) - cos(x[1]) + 2)

    def gradient(self, x, u):
        s0 = sin(x[0])
        s1 = sin(x[1])

        dx = np.array(
            (
                -cost_wgt * s0,
                cost_wgt * s1,
                0.0,
                0.0,
            )
        )

        du = np.zeros(u.size)

        return dx, du

    def hessian_xx(self, x, u):
        c0 = cos(x[0])
        c1 = cos(x[1])

        Hxx = np.zeros((4, 4))
        Hxx[0, 0] = -cost_wgt * c0
        Hxx[1, 1] = cost_wgt * c1

        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((4, 2))

    def hessian_uu(self, x, u):
        return np.zeros((2, 2))


model = ssqpy.model.Model(
    MPC_H,
    dt,
    urdf_path="../urdf/double_pendulum/double_pendulum_2.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, vel_wgt)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, u_wgt)

for k in range(MPC_H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    config_cost = ssqpy.model.costs.SquaredConfigurationDifferenceCost(
        np.array((pi, 0.0)), nq, nv, cost_wgt / (MPC_H - k), False
    )
    model.addCost(k, config_cost)

config_cost = ssqpy.model.costs.SquaredConfigurationDifferenceCost(
    np.array((pi, 0.0)), nq, nv, cost_wgt, False
)

model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, config_cost)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.accept_worse_update = True

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = False
admm_params.reset_rho = False
admm_params.reset_y = False

mpc = ssqpy.solvers.MPC(
    model, ssqp_params, admm_params, sqp_iters=3, admm_iters=160
)

mjmodel = mujoco.MjModel.from_xml_path(
    "../mjcf/double_pendulum/double_pendulum.xml"
)
mjdata = mujoco.MjData(mjmodel)

elapsed = 0
steps = 0
total_steps = int(1 / dt)

reg_fails = 0

np.set_printoptions(suppress=True)

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        frame_start = time()

        state = np.hstack((state_mod(mjdata.qpos), mjdata.qvel))

        pre = time()
        try:
            control = mpc.step(state)
        except RuntimeError:
            reg_fails += 1
            control = np.zeros(nv)

        torques = model.inverseDynamics(mjdata.qpos, mjdata.qvel, control)
        mjdata.ctrl[0] = torques[0]

        # mjdata.ctrl = control

        elapsed += time() - pre
        steps += 1

        if steps == total_steps:
            print(
                f"\rControl frequency: {(1 / (elapsed*dt)):.2f}Hz, regularization fails: {reg_fails}\t",
                end="",
            )
            elapsed = 0
            steps = 0

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

        to_sleep = dt - (time() - frame_start)
        if to_sleep > 0:
            sleep(to_sleep)
