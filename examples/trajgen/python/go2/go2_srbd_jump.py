import ssqpy
import numpy as np
import argparse

import models.quadruped_hlplanner.planner as go2_planner
import models.quadruped_srbd.model as go2_srbd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 single rigid body dynamics jump trajectory \
            generation"
    )

    parser.add_argument(
        "-t",
        "--com-target",
        type=float,
        nargs=3,
        default=(0.4, 0.0, 0.36),
        metavar=("X", "Y", "Z"),
        help="The position of the center of mass to jump to",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_srbd_jump.json",
        help="Path of the output JSON file",
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm", "piqp"],
        default="hpipm",
        help="QP solver used at each SSQP iteration",
    )

    return parser.parse_args()


args = parse_args()

H = 160
dt = 0.01

# Cost Weights
com_weight = 100.0
weight_vel = 2.0
weight_angularvel = 1.0
weight_forces = 1e-5
weight_foot_vel = 1e-2

# Constraint Parameters
L_max = 0.7

com_target = np.array(args.com_target)

model = ssqpy.model.Model(H, dt)

vel_cost = go2_srbd.VelocityCost(weight_vel)
angvel_cost = go2_srbd.AngularVelocityCost(weight_angularvel)
control_cost = go2_srbd.ControlRegularizationCost(
    weight_forces, weight_foot_vel
)

for k in range(H):
    model.addCost(k, vel_cost)
    model.addCost(k, angvel_cost)
    model.addCost(k, control_cost)

    for i, foot in enumerate(go2_srbd.FEET):
        foot_reachability_constraint = go2_srbd.FootReachabilityConstraint(
            i, L_max
        )
        model.addConstraints(k, foot_reachability_constraint)

        if k < H // 3 or k > H // 3 + 40:
            foot_constraint = go2_srbd.FrictionConstraint(i)
            model.addConstraints(k, foot_constraint)

            foot_vel_constraint = go2_srbd.FootVelocityConstraint(
                i, [0.0] * 3, [0.0] * 3
            )
            model.addConstraints(k, foot_vel_constraint)
        else:
            for i, foot in enumerate(go2_srbd.FEET):
                force_constraint = go2_srbd.ForceConstraint(
                    i, [0.0] * 3, [0.0] * 3
                )
                model.addConstraints(k, force_constraint)

com_pos_constraint = go2_srbd.CoMPositionConstraint(com_target, com_target)
com_vel_constraint = go2_srbd.CoMVelocityConstraint([-0.01] * 3, [0.01] * 3)

model.addCost(H, vel_cost)

model.addConstraints(H, com_pos_constraint)
model.addConstraints(H, com_vel_constraint)

model.finalize(
    dynamics=[
        go2_srbd.SRBDynamics(
            dt,
            go2_srbd.config.SRBDParameters.mass,
            go2_srbd.config.SRBDParameters.inertia,
        )
        for _ in range(H + 1)
    ],
    init_states=[
        np.hstack(
            (
                go2_planner.config.SRBDParameters.INIT_STATE,
                [0.20514, 0.142, 0.0],
                [0.20514, -0.142, 0.0],
                [-0.20514, 0.142, 0.0],
                [-0.20514, -0.142, 0.0],
            )
        )
    ],
)

# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)

admm_params = ssqpy.solvers.admmParams()
admm_params.rho_update = False
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
# admm_params.warm_start = True
ssqp_params.admmParams = admm_params

step = 0
solver = ssqpy.solvers.ssqp(model, ssqp_params)
solver.setReducedOutput(True)
state, x, u = solver.solve()
print("Solver state:", state)
solver.write(args.output)
