import ssqpy
import numpy as np

"""
Pendubot example
Using the pendubot (double pendulum with only the shoulder joint actuated),
move and remain at a vertical position in the opposite direction of gravity
"""

H = 200
dt = 0.01

V_WGT = 1e-2
U_WGT = 1e-3
C_WGT = 5.0

INIT_STATE = np.array((1.0, 0.0, 1.0, 0.0, 0.0, 0.0))
DESIRED_CONFIG = np.array((-1.0, 0.0, 1.0, 0.0))

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/double_pendulum/pendubot_cont.urdf",
    actuated_joints=[0],
    dynamics_mode=ssqpy.model.DynamicsMode.Forward,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, DESIRED_CONFIG, nq, nv, C_WGT, False
)

for k in range(H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    if k > 150:
        model.addCost(k, config_cost)

model.finalize(
    terminal_cost=config_cost,
    init_states=[INIT_STATE],
)

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
