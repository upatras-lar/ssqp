import ssqpy
import numpy as np
import sys

sys.path.insert(0, "..")

import generate_paramset

V_WGT = 1e-3
U_WGT = 1e-4
C_WGT = 1.0
TIP_WGT = 4.0

nq = 2
nv = 2
dt = 0.01

H = 100

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, U_WGT)
tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
    model, np.array((0.0, 0.0, 0.1)), "tip", nq, nv, TIP_WGT
)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    np.array((np.pi, 0.0)), nq, nv, C_WGT, False
)

for k in range(H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)
    model.addCost(k, config_cost)
    model.addCost(k, tip_cost)

model.addCost(H, vel_cost)
model.addCost(H, config_cost)
model.addCost(H, tip_cost)

model.finalize()

hyperparams_admm = generate_paramset.generate_admm_paramset(
    adaptive_rho_tols=[5, 10, 25], rho_updates=[True, False]
)
param_search = ssqpy.solvers.ParamSearch(hyperparams_admm, [model])

result = param_search.search()

result[0].print()
