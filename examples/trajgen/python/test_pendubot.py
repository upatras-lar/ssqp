import ssqpy
import numpy as np

nq = 2
nv = 2
nu = 2

horizon = 200
dt = 0.01

V_WGT = 1e-3
U_WGT = 1e-3
C_WGT = 10.0

dynamics_mode = ssqpy.model.DynamicsMode.Inverse

model = ssqpy.model.Model(
    horizon,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
    dynamics_mode=dynamics_mode,
)

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    np.array((np.pi, 0.0)), nq, nv, C_WGT, False
)

for k in range(horizon):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    if k > horizon // 2:
        model.addCost(k, config_cost)

model.addCost(horizon, config_cost)

model.finalize(init_states=[np.array((0.0, 0.0, 0.0, 0.0))])

params = ssqpy.solvers.ssqpParams(
    "../../benchmarks/data/parameters/pendubot_admm_params_H200_inverse.json"
)
params.max_iter = 1000
params.tolerance = 1e-3

params.admmParams.abs_tolerance = 1e-4
params.admmParams.rel_tolerance = 1e-4

solver = ssqpy.solvers.ssqp(
    model,
    params,
    # max_iter=500,
    # tolerance=1e-3,
    # admm_adaptive_rho_tolerance=1,
    # admm_rho_update_freq=1,
    # admm_abs_tolerance=1e-4,
    # admm_rel_tolerance=1e-4,
    # admm_max_iter=500,
    # admm_warm_start=False,
    # qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
)

state, _, _ = solver.solve()

print("Solver state:", state)

solver.write("../data/test_pendubot.json")
