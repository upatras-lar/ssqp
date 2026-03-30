import ssqpy
import numpy as np
from math import pi, sin, cos

"""
Panda Obstacle example
From the starting end-effector position, move to the target end-effector
position while avoiding the obstacle defined by the constraints
"""
H = 400
dt = 0.01

# Weights
T_WGT = 10.0
R_WGT = 10.0
V_WGT = 1e-1
U_WGT = 1e-2

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

# Create an SSQP model with H and dt, from the URDF at the below path
model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/panda/panda.urdf",
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

init_state = np.zeros(2 * nq)
init_state[:nq] = [
    -0.3958,
    0.1247,
    -0.3661,
    -2.6497,
    0.1197,
    2.7639,
    -0.0852,
    0.0,
    0.0,
]

joint_vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nv, V_WGT)
control_cost = ssqpy.model.costs.SquaredControlCost(nv, nu, U_WGT)
rot_cost = ssqpy.model.costs.FrameSquaredRotationErrorCost(
    model, "panda_ee", Rd, nq, nq, R_WGT
)
pose_cost = ssqpy.model.costs.FrameSquaredPoseErrorCost(
    model, "panda_ee", np.array((0.3, 0.3, 0.1)), Rd, nq, nq, T_WGT, R_WGT
)

eef_bound = ssqpy.model.constraints.FramePositionBound(
    model,
    "panda_ee",
    np.array((-np.inf, -np.inf, 0.4)),
    np.array((np.inf, np.inf, np.inf)),
    nq,
    nv,
)

for k in range(H):
    model.addCost(k, rot_cost)
    model.addCost(k, joint_vel_cost)
    model.addCost(k, control_cost)

    if k > 50 and k < H - 50:
        model.addConstraints(k, eef_bound)

model.addCost(H, pose_cost)
model.addCost(H, joint_vel_cost)

model.finalize(init_states=init_state)

params = ssqpy.solvers.ssqpParams(
    "../../benchmarks/data/parameters/panda_admm_params_inverse.json"
)
params.admmParams.abs_tolerance = 1e-4
params.admmParams.rel_tolerance = 1e-4

solver = ssqpy.solvers.ssqp(
    model,
    params,
    tolerance=1e-3,
    # loose_success=False,
    # tolerance=1e-3,
    # line_search_method=ssqpy.solvers.LineSearchMethod.Merit,
    # merit_mu_c=2.0,
    # merit_mu_gamma=0.5,
    # admm_init_rho=0.1,
    # admm_abs_tolerance=1e-4,
    # admm_rel_tolerance=1e-4,
    # admm_adaptive_rho_tolerance=1,
    # admm_rho_update_iter_freq=10,
    # admm_max_iter=1000,
    # qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    # hpipm_max_iter=250,
)

state, x, u = solver.solve()

solver.write("../data/panda_constrained.json")
