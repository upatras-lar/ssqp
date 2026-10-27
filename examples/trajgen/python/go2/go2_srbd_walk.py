import ssqpy
import numpy as np
import argparse

import models.quadruped_hlplanner.planner as go2_planner
import models.quadruped_srbd.model as go2_srbd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 single rigid body dynamics walk trajectory generation"
    )

    parser.add_argument(
        "-v",
        "--velocity",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        metavar=("FORWARD", "SIDEWAYS", "YAW"),
        help="The commanded velocity to walk with",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../data/go2_srbd_walk.json",
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

v_cmd = np.array(args.velocity)

H = 500
dt = 0.01

# Cost Weights
weight_com = 100.0
weight_orientation = 10.0
weight_vel = 1.0
weight_angularvel = 1.0
weight_feet = 100.0
weight_forces = 1e-4
weight_foot_vel = 1e-2

# Constraint Parameters
z_min = 0.3
L_max = 0.8

model = ssqpy.model.Model(H, dt)

# Full Go2 model, used only to get the initial feet and CoM positions
go2_model = ssqpy.model.Model(
    H, dt, mjcf_path="../../../assets/mjcf/go2/go2.xml"
)

com_ref, xb_ref, p_ref = go2_planner.generate_references(
    v_cmd, go2_planner.config.INIT_CONFIG, np.zeros(3), 0.0, H, dt, go2_model
)

for k in range(H):
    stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

    com_cost = go2_srbd.CoMTrackingCost(com_ref[k], weight_com)
    orientation_cost = go2_srbd.OrientationTrackingCost(
        xb_ref[k], weight_orientation
    )
    vel_cost = go2_srbd.VelocityCost(weight_vel)
    angvel_cost = go2_srbd.AngularVelocityCost(weight_angularvel)
    foot_cost = go2_srbd.FootTrackingCost(stage_p_ref, weight_feet)
    control_cost = go2_srbd.ControlRegularizationCost(
        weight_forces, weight_foot_vel
    )

    model.addCost(k, com_cost)
    model.addCost(k, orientation_cost)
    model.addCost(k, vel_cost)
    model.addCost(k, angvel_cost)
    model.addCost(k, foot_cost)
    model.addCost(k, control_cost)

    height_constraint = go2_srbd.CoMHeightConstraint(z_min)
    model.addConstraints(k, height_constraint)

    for i, foot in enumerate(stage_p_ref):
        foot_reachability_constraint = go2_srbd.FootReachabilityConstraint(
            i, L_max
        )
        model.addConstraints(k, foot_reachability_constraint)

        if not stage_p_ref[foot][1]:
            friction_constraint = go2_srbd.FrictionConstraint(i)
            model.addConstraints(k, friction_constraint)

            foot_vel_constraint = go2_srbd.FootVelocityConstraint(
                i, [0.0] * 3, [0.0] * 3
            )
        else:
            force_constraint = go2_srbd.ForceConstraint(
                i, [0.0] * 3, [0.0] * 3
            )
            model.addConstraints(k, force_constraint)

            foot_vel_constraint = go2_srbd.FootVelocityConstraint(
                i, [-2.0] * 3, [2.0] * 3
            )

        model.addConstraints(k, foot_vel_constraint)


stage_p_ref = {foot: p_ref[foot][H] for foot in p_ref}

com_cost = go2_srbd.CoMTrackingCost(com_ref[H], weight_com)
orientation_cost = go2_srbd.OrientationTrackingCost(
    xb_ref[H], weight_orientation
)
vel_cost = go2_srbd.VelocityCost(weight_vel)
angvel_cost = go2_srbd.AngularVelocityCost(weight_angularvel)
foot_cost = go2_srbd.FootTrackingCost(stage_p_ref, weight_feet)

model.addCost(H, com_cost)
model.addCost(H, orientation_cost)
model.addCost(H, vel_cost)
model.addCost(H, angvel_cost)
model.addCost(H, foot_cost)

height_constraint = go2_srbd.CoMHeightConstraint(z_min)
model.addConstraints(H, height_constraint)

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
                *[
                    p_ref[foot][0][0]
                    for foot in go2_planner.config.FRAME_NAMES.keys()
                ],
            )
        )
    ],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)

admm_params = ssqpy.solvers.admmParams()
# admm_params.rho_update = False
admm_params.abs_tolerance = 1e-3
admm_params.rel_tolerance = 1e-3
admm_params.warm_start = True
ssqp_params.admmParams = admm_params

step = 0
solver = ssqpy.solvers.ssqp(model, ssqp_params)
solver.setReducedOutput(True)
state, x, u = solver.solve()
print("Solver state:", state)
solver.write(args.output)
