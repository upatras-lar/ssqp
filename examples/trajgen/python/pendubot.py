import ssqpy
import numpy as np
from math import pi

"""
Pendubot example
Using the pendubot (double pendulum with only the shoulder joint actuated),
move and remain at a vertical position in the opposite direction of gravity
"""

nq = 2
nv = 2

H = 200
dt = 0.01

V_WGT = 1e-3
U_WGT = 1e-4
C_WGT = 4.0

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    # dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
    # integration=ssqpy.model.Integration.ExplicitEuler,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, U_WGT)
tau_cost = ssqpy.model.costs.SquaredIDTorqueCost(
    model, nq, nv, np.array((1e-1, 0.0))
)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    np.array((pi, 0.0)), nq, nv, C_WGT, False
)

for k in range(H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    # model.addCost(k, tau_cost)

    if k > 100:
        model.addCost(k, config_cost)

model.finalize(terminal_cost=config_cost)

solver = ssqpy.solvers.ssqp(
    model,
    tolerance=1e-2,
    # qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    filter_size=10,
    hpipm_warm_start=False,
    hpipm_prediction_correction=True,
    hpipm_square_root_riccati=False,
    hpipm_split_step=True,
    admm_adaptive_rho_tolerance=25,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-4,
    admm_max_iter=500,
    admm_warm_start=False,
    # admm_reset_rho=False,
    # admm_reset_y=False,
)

state, x, u = solver.solve()

solver.write("../data/pendubot.json")
