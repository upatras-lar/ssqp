import ssqpy
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Pendubot controller server")

    parser.add_argument(
        "--hpipm",
        action=argparse.BooleanOptionalAction,
        help="Whether to use HPIPM as the QP backend",
    )

    parser.add_argument(
        "--merit",
        action=argparse.BooleanOptionalAction,
        help="Whether to use merit function based line search",
    )

    return parser.parse_args()


args = parse_args()

H = 2500
dt = 0.001

V_WGT = 1e-4
U_WGT = 1e-4
C_WGT = 1.0

DESIRED_CONFIG = np.array((np.pi, 0.0))

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../../assets/urdf/double_pendulum/pendubot.urdf",
    actuated_joints=[0],
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

    if k > H - 500:
        model.addCost(k, config_cost)

model.finalize(terminal_cost=config_cost)

# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
if args.merit:
    ssqp_params.line_search_method = ssqpy.solvers.LineSearchMethod.Merit

if args.hpipm:
    ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
    hpipm_params = ssqpy.solvers.hpipmParams()
    hpipm_params.tol_comp = 1e-3
    hpipm_params.tol_stat = 1e-3
    hpipm_params.tol_eq = 1e-3
    hpipm_params.tol_ineq = 1e-3
    ssqp_params.hpipmParams = hpipm_params
else:
    admm_params = ssqpy.solvers.admmParams()
    admm_params.max_iter = 100
    admm_params.abs_tolerance = 1e-3
    admm_params.rel_tolerance = 1e-3
    admm_params.adaptive_rho_tolerance = 1
    admm_params.rho_update_iter_freq = 25
    admm_params.warm_start = True
    ssqp_params.admmParams = admm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

print("xs:")
x.print()
print("us:")
u.print()

solver.write("pendubot.json")
