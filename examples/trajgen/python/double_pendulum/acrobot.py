import ssqpy
import numpy as np
import argparse

"""
Acrobot example
Using the acrobot (double pendulum with only the elbow joint actuated),
move and remain at a vertical position in the opposite direction of gravity
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Acrobot trajectory generation"
    )

    parser.add_argument(
        "-a",
        "--target-angle",
        type=float,
        default=np.pi,
        help="The angle of the shoulder joint to move to and remain at, in \
            radians (pi is vertical, opposite to gravity)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/acrobot.json",
        help="Path of the output JSON file",
    )

    return parser.parse_args()


args = parse_args()

H = 200
dt = 0.01

V_WGT = 1e-4
U_WGT = 1e-5
C_WGT = 0.5

# DESIRED_CONFIG = np.array((-1.0, 0.0, 1.0, 0.0))
DESIRED_CONFIG = np.array((args.target_angle, 0.0))

model = ssqpy.model.Model(
    H,
    dt,
    from_preset=ssqpy.model.Preset.DoublePendulum,
    actuated_joints=[1],
    solver_mode=ssqpy.model.SolverMode.InverseDynamics,
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

    if k > 150:
        model.addCost(k, config_cost)

model.finalize(terminal_cost=config_cost)

solver = ssqpy.solvers.ssqp(
    model,
    tolerance=1e-2,
    qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    # filter_size=10,
    hpipm_warm_start=False,
    hpipm_prediction_correction=True,
    hpipm_square_root_riccati=False,
    hpipm_split_step=True,
    hpipm_tol_comp=1e-3,
    hpipm_tol_rel=1e-3,
    hpipm_tol_eq=1e-3,
    hpipm_tol_ineq=1e-3,
    admm_adaptive_rho_tolerance=25,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-4,
    admm_max_iter=500,
    admm_warm_start=False,
    # admm_reset_rho=False,
    # admm_reset_y=False,
)

state, x, u = solver.solve()

solver.write(args.output)
