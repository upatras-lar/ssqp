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

H = 500
dt = 0.01

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
    # integration=ssqpy.model.Integration.ExplicitEuler,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, 1e-6)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, 1e-8)
config_cost = ssqpy.model.costs.SquaredConfigurationDifferenceCost(
    np.array((pi, 0.0)), nq, nv, 10.0, False
)

for k in range(H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    if k > 100:
        model.addCost(k, config_cost)

model.finalize(terminal_cost=config_cost)

solver = ssqpy.solvers.ssqp(
    model,
    tolerance=1e-2,
    use_filter=True,
    admm_adaptive_rho_tolerance=20,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-3,
    admm_max_iter=500,
    admm_warm_start=False,
    # admm_reset_rho=False,
    # admm_reset_y=False,
)

state, x, u = solver.solve()

solver.write("../data/pendubot.json")
