"""
Go2 quadruped standing trajectory optimization, translated from ssqpy to
aligator.
"""

import numpy as np
import pinocchio as pin
import aligator
from aligator import manifolds, dynamics, constraints
from math import pi
from time import time

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
HORIZON = 100
dt = 0.01

A_WGT = 10.0
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-5

MAX_NORMAL_FORCE = 50.0
FOOT_EQ_TOL = 0.01  # ssqpy's per-contact equality tolerance -> proximal
# solver tolerance for the constrained-dynamics solve, see prox_settings.

MU = 0.6

FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

base_pos = np.array([0.0, 0.0, 0.3879793820122])
base_config = np.array([0.0, pi / 6, -pi / 3] * 4)  # 12 leg-joint angles

# ---------------------------------------------------------------------------
# Build the Pinocchio model with a free-flyer base
# ---------------------------------------------------------------------------
URDF_PATH = "urdf/go2/go2.urdf"

model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()

nq, nv = model.nq, model.nv  # 19, 18 for Go2
nu = 12  # 3 actuated joints per leg x 4 legs

# Base (free-flyer) config: [pos(3), quat_xyzw(4), leg_joints(12)]
q0 = np.zeros(nq)
q0[:3] = base_pos
q0[6] = 1.0  # quaternion w-component -> identity orientation
q0[7:19] = base_config
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])

# Contact-constraint target height h_z per foot
foot_h_z = {foot: 0.0 for foot in FOOT_FRAMES}

# Reference state used only to build the joint-angle part
q_ref = q0.copy()
x_ref = np.concatenate([q_ref, np.zeros(nv)])

# Actuation matrix: base is unactuated (6 dof), legs are fully actuated (12)
B = np.zeros((nv, nu))
B[6:, :] = np.eye(nu)

# ---------------------------------------------------------------------------
# State space: floating-base tangent bundle, ndx = 2*nv = 36
# ---------------------------------------------------------------------------
space = manifolds.MultibodyPhaseSpace(model)
ndx = space.ndx  # 36

# ---------------------------------------------------------------------------
# Per-foot rigid contact constraints (feet pinned to the ground)
# ---------------------------------------------------------------------------
constraint_models = []
for foot in FOOT_FRAMES:
    frame_id = model.getFrameId(foot)
    cm = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_3D,
        model,
        model.frames[frame_id].parentJoint,
        model.frames[frame_id].placement,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    cm.name = foot
    cm.corrector.Kp = np.zeros(6)
    cm.corrector.Kd = np.zeros(6)
    constraint_models.append(cm)

prox_settings = pin.ProximalSettings(FOOT_EQ_TOL, 1e-10, 10)

ode = dynamics.MultibodyConstraintFwdDynamics(
    space, B, constraint_models, prox_settings
)
disc_dynamics = dynamics.IntegratorEuler(ode, dt)

# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
# Configuration cost
state_err_to_ref = aligator.StateErrorResidual(space, nu, x_ref)
config_err = state_err_to_ref[6 : 6 + nu]
config_cost = aligator.QuadraticResidualCost(
    space, config_err, A_WGT * np.eye(nu)
)

# Velocity cost
state_err_zero = aligator.StateErrorResidual(space, nu, x_ref)
vel_err = state_err_zero[nv + 6 : nv + 6 + nu]
vel_cost = aligator.QuadraticResidualCost(space, vel_err, V_WGT * np.eye(nu))

control_err = aligator.ControlErrorResidual(ndx, nu)
control_cost = aligator.QuadraticResidualCost(
    space, control_err, U_WGT * np.eye(nu)
)

# Force cost
force_costs = []
for i, foot in enumerate(FOOT_FRAMES):
    force_res = aligator.ContactForceResidual(
        ndx, model, B, constraint_models, prox_settings, np.zeros(3), foot
    )
    fc = aligator.QuadraticResidualCost(space, force_res, F_WGT * np.eye(3))
    force_costs.append(fc)
# ---------------------------------------------------------------------------
# Innate constraints from URDF
# ---------------------------------------------------------------------------
zero_state = np.zeros(nq + nv)
zero_state[6] = 1.0

zero_residual = aligator.StateErrorResidual(space, nu, zero_state)

config_bound = constraints.BoxConstraint(
    model.lowerPositionLimit[7:], model.upperPositionLimit[7:]
)
config_err = zero_residual[6:nv]

vel_bound = constraints.BoxConstraint(
    -model.velocityLimit, model.velocityLimit
)
vel_err = zero_residual[nv + 6 : 2 * nv]

torque_bound = constraints.BoxConstraint(-model.effortLimit, model.effortLimit)
torque_err = control_err

# ---------------------------------------------------------------------------
# Per-foot velocity bound
# ---------------------------------------------------------------------------
foot_vel_bound = constraints.BoxConstraint(np.zeros(3), np.zeros(3))
foot_vel_residuals = []
for foot in FOOT_FRAMES:
    frame_id = model.getFrameId(foot)
    fv = aligator.FrameVelocityResidual(
        ndx,
        nu,
        model,
        pin.Motion.Zero(),
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    foot_vel_residuals.append(fv)

# Terminal base-velocity bound
base_frame_id = model.getFrameId("base")
base_vel_residual = aligator.FrameVelocityResidual(
    ndx,
    nu,
    model,
    pin.Motion.Zero(),
    base_frame_id,
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
)
base_vel_bound = constraints.BoxConstraint(
    np.concatenate([-np.inf * np.ones(3), np.zeros(3)]),
    np.concatenate([np.inf * np.ones(3), np.zeros(3)]),
)

# ---------------------------------------------------------------------------
# Linearized (4-facet pyramidal) friction cone per foot, replacing the
# earlier plain z-only box:
#     f_n in [0, MAX_NORMAL_FORCE]
#     |f1| <= mu * f_n
#     |f2| <= mu * f_n
# Force order is assumed [f1, f2, f_n] = [f_x, f_y, f_z] in the
# LOCAL_WORLD_ALIGNED frame. Expressed as A @ f <= b via
# LinearFunctionComposition, then boxed to (-inf, b].
# ---------------------------------------------------------------------------
A_cone = np.array(
    [
        [0.0, 0.0, -1.0],  # -f_n <= 0            -> f_n >= 0
        [0.0, 0.0, 1.0],  # f_n <= MAX_NORMAL_FORCE
        [1.0, 0.0, -MU],  # f1 - mu*f_n <= 0
        [-1.0, 0.0, -MU],  # -f1 - mu*f_n <= 0
        [0.0, 1.0, -MU],  # f2 - mu*f_n <= 0
        [0.0, -1.0, -MU],  # -f2 - mu*f_n <= 0
    ]
)
b_cone = np.array([0.0, MAX_NORMAL_FORCE, 0.0, 0.0, 0.0, 0.0])
friction_cone_bound = constraints.BoxConstraint(-np.inf * np.ones(6), b_cone)

# ---------------------------------------------------------------------------
# Per-foot p_z = h_z position constraint.
# ---------------------------------------------------------------------------
foot_pz_bound = constraints.BoxConstraint(np.zeros(1), np.zeros(1))
foot_pz_residuals = []
for foot in FOOT_FRAMES:
    frame_id = model.getFrameId(foot)
    p_ref = np.array([0.0, 0.0, foot_h_z[foot]])
    fp = aligator.FrameTranslationResidual(ndx, nu, model, p_ref, frame_id)
    foot_pz_residuals.append(fp[2:3])

# ---------------------------------------------------------------------------
# Build stages
# ---------------------------------------------------------------------------
stages = []
for k in range(HORIZON):
    cost_stack = aligator.CostStack(space, nu)
    cost_stack.addCost(config_cost)
    cost_stack.addCost(vel_cost)
    cost_stack.addCost(control_cost)
    for fc in force_costs:
        cost_stack.addCost(fc)

    stage = aligator.StageModel(cost_stack, disc_dynamics)

    stage.addConstraint(config_err, config_bound)
    stage.addConstraint(vel_err, vel_bound)
    stage.addConstraint(torque_err, torque_bound)

    for i, foot in enumerate(FOOT_FRAMES):
        fv = foot_vel_residuals[i]
        stage.addConstraint(fv, foot_vel_bound)

        force_res = aligator.ContactForceResidual(
            ndx,
            model,
            B,
            constraint_models,
            prox_settings,
            np.zeros(3),
            foot,
        )
        cone_res = aligator.LinearFunctionComposition(
            force_res, A_cone, np.zeros(6)
        )
        stage.addConstraint(cone_res, friction_cone_bound)

        stage.addConstraint(foot_pz_residuals[i], foot_pz_bound)
    stages.append(stage)

# Terminal cost/constraint
term_cost_stack = aligator.CostStack(space, nu)
term_cost_stack.addCost(config_cost)

problem = aligator.TrajOptProblem(x0, stages, term_cost_stack)

problem.addTerminalConstraint(config_err, config_bound)
problem.addTerminalConstraint(vel_err, vel_bound)
problem.addTerminalConstraint(base_vel_residual, base_vel_bound)

# ---------------------------------------------------------------------------
# Initial guess: hold the standing configuration at every node, zero control
# ---------------------------------------------------------------------------
xs_init = [x0.copy() for _ in range(HORIZON + 1)]
us_init = [np.zeros(nu) for _ in range(HORIZON)]

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
tol = 1e-2
mu_init = 1e-2
solver = aligator.SolverProxDDP(
    tol, mu_init, verbose=aligator.VerboseLevel.VERBOSE
)
solver.max_iters = 100
solver.setup(problem)

np.set_printoptions(suppress=True)

start = time()
conv = solver.run(problem, xs_init, us_init)
end = time()

results = solver.results
xs_opt = results.xs
us_opt = results.us

print()
print("============================================================")
print("Aligator result")
print("============================================================")
print(f"Solve time: {(end - start) * 1000.0:.3f} ms")
print(f"Converged: {conv}")
print(f"Final base position: {xs_opt[-1][:3]}")
print(f"Max |u|: {np.max(np.abs(np.array(us_opt))):.4f}")
print("============================================================")

# ---------------------------------------------------------------------------
# Save solution
# ---------------------------------------------------------------------------
import json

constraint_datas = [cm.createData() for cm in constraint_models]

force_residuals_by_foot = {
    foot: aligator.ContactForceResidual(
        ndx, model, B, constraint_models, prox_settings, np.zeros(3), foot
    )
    for foot in FOOT_FRAMES
}

forces_traj = []

for k in range(HORIZON):
    x_k = xs_opt[k]
    u_k = us_opt[k]

    forces_k = []

    for foot in FOOT_FRAMES:
        res = force_residuals_by_foot[foot]
        res_data = res.createData()
        res.evaluate(x_k, u_k, res_data)
        forces_k.extend(res_data.value.copy().tolist())

    forces_traj.append(forces_k)

stage_data = []
for k in range(HORIZON + 1):
    stage = {
        "position": xs_opt[k][:3].tolist(),
        "orientation": xs_opt[k][3:7].tolist(),
        "linear_velocity": xs_opt[k][nq : nq + 3].tolist(),
        "angular_velocity": xs_opt[k][nq + 3 : nq + 6].tolist(),
        "joint_configuration": xs_opt[k][7:nq].tolist(),
        "joint_velocity": xs_opt[k][nq + 6 : nq + nv].tolist(),
    }
    if k < HORIZON:
        stage["control"] = [float(v) for v in us_opt[k]]
        stage["forces"] = forces_traj[k]
    stage_data.append(stage)

output = {
    "properties": {
        "horizon": HORIZON,
        "dt": dt,
        "solver_mode": "ForwardDynamics",
        "xml": True,
        "nq": nq,
        "nv": nv,
        "nu": nu,
        "njoints": model.njoints - 1,
        "total_joints": model.njoints,
        "floating_base": True,
        "state_joints": [model.names[i] for i in range(2, model.njoints)],
        "joint_types": ["revolute" for _ in range(model.njoints - 2)],
        "contact_frames": FOOT_FRAMES,
        "stages": [],
    },
    "stage_data": stage_data,
}

for k in range(HORIZON + 1):
    stage_info = {
        "size": ndx,
        "control": nu if k < HORIZON else 0,
        "nforces": len(FOOT_FRAMES),
        "ordered_force_frames": FOOT_FRAMES if k < HORIZON else None,
    }
    output["properties"]["stages"].append(stage_info)

with open("go2_standing.json", "w") as f:
    json.dump(output, f, indent=4)

print("Saved solution to go2_standing.json")
