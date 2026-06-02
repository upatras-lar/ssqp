import ssqpy
import numpy as np

import models.quadruped_hlplanner.planner as go2_planner
import models.quadruped_srbd.model as go2_srbd

from go2_controllers.WholeBody import WholeBodyQP
from utils_go2_mujoco import (
    ssqp_to_mujoco_state_kin,
    ssqp_to_mujoco_ctrl_kin,
    mujoco_to_ssqp_state_kin,
)

v_cmd = np.array((0.0, 0.0, 0.0))

MPC_H = 10
dt = 0.02

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

model = ssqpy.model.Model(MPC_H, dt)

com_ref, xb_ref, p_ref = go2_planner.generate_references_mpc(
    v_cmd, go2_planner.config.INIT_CONFIG, np.zeros(3), 0.0, MPC_H, dt
)

for k in range(MPC_H):
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


stage_p_ref = {foot: p_ref[foot][MPC_H] for foot in p_ref}

com_cost = go2_srbd.CoMTrackingCost(com_ref[MPC_H], weight_com)
orientation_cost = go2_srbd.OrientationTrackingCost(
    xb_ref[MPC_H], weight_orientation
)
vel_cost = go2_srbd.VelocityCost(weight_vel)
angvel_cost = go2_srbd.AngularVelocityCost(weight_angularvel)
foot_cost = go2_srbd.FootTrackingCost(stage_p_ref, weight_feet)

model.addCost(MPC_H, com_cost)
model.addCost(MPC_H, orientation_cost)
model.addCost(MPC_H, vel_cost)
model.addCost(MPC_H, angvel_cost)
model.addCost(MPC_H, foot_cost)

height_constraint = go2_srbd.CoMHeightConstraint(z_min)
model.addConstraints(MPC_H, height_constraint)

model.finalize(
    dynamics=[
        go2_srbd.SRBDynamics(
            dt,
            go2_srbd.config.SRBDParameters.mass,
            go2_srbd.config.SRBDParameters.inertia,
        )
        for _ in range(MPC_H + 1)
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
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=3, qp_iters=200)


# TODO
def update_func(v_cmd, config, vel, step):
    pass


wbc = WholeBodyQP()

import mujoco
import mujoco.viewer

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state_kin(go2_planner.config.INIT_CONFIG)

with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        pos = mujoco_to_ssqp_state_kin(mjdata.qpos)
        vel = mjdata.qvel

        # TODO
        # mjdata.ctrl = ...

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

print()
