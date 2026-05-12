import ssqpy
import numpy as np

"""
Ball joint example
"""

H = 200
dt = 0.01

V_WGT = 1.0
U_WGT = 1.0
C_WGT = 5.0

DESIRED_CONFIG = np.array((0.0, 0.0, 0.0, 1.0))

model = ssqpy.model.Model(
    H,
    dt,
    mjcf_path="../../assets/mjcf/ball_joint.xml",
    solver_mode=ssqpy.model.SolverMode.ForwardDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, DESIRED_CONFIG, C_WGT
)

for k in range(H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, config_cost)

model.finalize(terminal_cost=config_cost)

solver = ssqpy.solvers.ssqp(model, tolerance=1e-2)

state, x, u = solver.solve()

solver.write("../data/ball_joint.json")
