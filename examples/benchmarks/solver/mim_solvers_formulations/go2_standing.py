import argparse
import json
import time

import numpy as np

np.set_printoptions(suppress=True)
import pinocchio as pin
import crocoddyl
import mim_solvers

# ============================================================================
# Problem parameters
# ============================================================================
HORIZON = 100
dt = 0.01

# Costs
A_WGT = 10.0
V_WGT = 1e-2
U_WGT = 1e-4
F_WGT = 1e-5

MAX_NORMAL_FORCE = 50.0
FOOT_EQ_TOL = 0.01

URDF_PATH = "urdf/go2/go2.urdf"

# ============================================================================
# Initial configuration
# ============================================================================
base_pos = np.array(
    [0.0, 0.0, 0.3879793820122],
    dtype=float,
)

base_config = np.array(
    [0.0, np.pi / 6.0, -np.pi / 3.0] * 4,
    dtype=float,
)

# ============================================================================
# Pinocchio model
# ============================================================================
model = pin.buildModelFromUrdf(
    URDF_PATH,
    pin.JointModelFreeFlyer(),
)

nq = model.nq
nv = model.nv

# Crocoddyl state.
state = crocoddyl.StateMultibody(model)

actuation = crocoddyl.ActuationModelFloatingBase(state)

nu = actuation.nu

print(f"nq = {nq}")
print(f"nv = {nv}")
print(f"nu = {nu}")
print(f"njoints = {model.njoints}")

# ============================================================================
# Initial state
# ============================================================================
q0 = pin.neutral(model)

q0[:3] = base_pos
q0[3:7] = np.array([0.0, 0.0, 0.0, 1.0])
q0[7:19] = base_config

v0 = np.zeros(nv)

x0 = np.concatenate([q0, v0])

# ============================================================================
# Find contact locations at the initial configuration
# ============================================================================
pin.forwardKinematics(model, model.createData(), q0)

data0 = model.createData()
pin.forwardKinematics(model, data0, q0)
pin.updateFramePlacements(model, data0)

FOOT_NAMES = [
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]

foot_ids = {}

for name in FOOT_NAMES:
    foot_ids[name] = model.getFrameId(name)

    if foot_ids[name] >= model.nframes:
        raise RuntimeError(f"Could not find frame '{name}' in the Go2 URDF")


# Store the initial world position of every foot.
#
# These are the contact references. The contact dynamics then keeps the
# feet at these positions throughout the standing phase.

foot_contact_positions = {
    name: data0.oMf[foot_ids[name]].translation.copy() for name in FOOT_NAMES
}

# ============================================================================
# Cost models
# ============================================================================
config_weights = np.concatenate(
    [
        np.zeros(6),
        A_WGT * np.ones(12),
        np.zeros(nv),
    ]
)

velocity_weights = np.concatenate(
    [
        np.zeros(nv),
        V_WGT * np.ones(nv),
    ]
)

activation_configuration = crocoddyl.ActivationModelWeightedQuad(
    config_weights
)

activation_velocity = crocoddyl.ActivationModelWeightedQuad(velocity_weights)

xref = x0.copy()

# Desired Go2 joint configuration
xref[7:19] = base_config

# Desired velocity is zero
xref[nq:] = 0.0


configuration_residual = crocoddyl.ResidualModelState(
    state,
    xref,
    nu,
)

configuration_cost = crocoddyl.CostModelResidual(
    state,
    activation_configuration,
    configuration_residual,
)


velocity_residual = crocoddyl.ResidualModelState(
    state,
    x0,
    nu,
)

velocity_cost = crocoddyl.CostModelResidual(
    state,
    activation_velocity,
    velocity_residual,
)


# Control cost
control_residual = crocoddyl.ResidualModelControl(
    state,
    nu,
)

control_cost = crocoddyl.CostModelResidual(
    state,
    control_residual,
)


# ============================================================================
# Constraints
# ============================================================================
# Configuration bounds
lb_q = np.asarray(model.lowerPositionLimit).reshape(-1)
ub_q = np.asarray(model.upperPositionLimit).reshape(-1)

# Velocity bounds (dimension state.nv)
lb_dq = -np.asarray(model.velocityLimit).reshape(-1)
ub_dq = np.asarray(model.velocityLimit).reshape(-1)

# Control bounds
lb_u = -np.asarray(model.effortLimit).reshape(-1)
ub_u = np.asarray(model.effortLimit).reshape(-1)

state_residual = crocoddyl.ResidualModelState(state, state.zero(), nu=12)

# ---------------------------------------------------------------------------
# Configuration constraint via ResidualModelState
# ---------------------------------------------------------------------------
# Pad bounds with +-inf for velocity
lb_q = np.concatenate([model.lowerPositionLimit[1:], -np.inf * np.ones(nv)])
ub_q = np.concatenate([model.upperPositionLimit[1:], np.inf * np.ones(nv)])

config_constraint = crocoddyl.ConstraintModelResidual(
    state, state_residual, lb_q, ub_q
)

# ---------------------------------------------------------------------------
# Velocity constraint via ResidualModelState
# ---------------------------------------------------------------------------
# Pad bounds with +-inf for position components so only velocities are constrained
nq_ndx = state.ndx - state.nv
lb_dq = np.concatenate([-np.inf * np.ones(nq_ndx), lb_dq])
ub_dq = np.concatenate([np.inf * np.ones(nq_ndx), ub_dq])

velocity_constraint = crocoddyl.ConstraintModelResidual(
    state,
    state_residual,
    lb_dq,
    ub_dq,
)

# ---------------------------------------------------------------------------
# Control constraint
# ---------------------------------------------------------------------------
control_residual = crocoddyl.ResidualModelControl(state, nu=12)

control_constraint = crocoddyl.ConstraintModelResidual(
    state,
    control_residual,
    lb_u,
    ub_u,
)

zero_motion = pin.Motion.Zero()

foot_velocity_constraints = {}

for name in FOOT_NAMES:
    residual = crocoddyl.ResidualModelFrameVelocity(
        state,
        foot_ids[name],
        zero_motion,
        pin.LOCAL_WORLD_ALIGNED,
        nu,
    )

    constraint = crocoddyl.ConstraintModelResidual(
        state,
        residual,
        -FOOT_EQ_TOL * np.ones(6),
        FOOT_EQ_TOL * np.ones(6),
    )

    foot_velocity_constraints[name] = constraint


# ---------------------------------------------------------------------------
# Contact force constraints
# ---------------------------------------------------------------------------
contact_force_constraints = {}

for name in FOOT_NAMES:
    force_residual = crocoddyl.ResidualModelContactForce(
        state,
        foot_ids[name],
        pin.Force.Zero(),
        3,
        nu,
        True,  # fwddyn=True
    )

    force_constraint = crocoddyl.ConstraintModelResidual(
        state,
        force_residual,
        np.array([-np.inf, -np.inf, 0.0]),
        np.array(
            [
                np.inf,
                np.inf,
                MAX_NORMAL_FORCE,
            ]
        ),
    )

    contact_force_constraints[name] = force_constraint


# ============================================================================
# Build standing contacts
# ============================================================================
def make_contacts():
    """
    Create the four rigid 3D contacts used during the standing phase.

    ContactModel3D enforces zero acceleration at the contact point and
    provides the contact force needed by the forward dynamics model.
    """
    contacts = crocoddyl.ContactModelMultiple(
        state,
        nu,
    )

    for name in FOOT_NAMES:
        contact = crocoddyl.ContactModel3D(
            state,
            foot_ids[name],
            foot_contact_positions[name],
            pin.LOCAL,
            nu,
        )

        contacts.addContact(
            name,
            contact,
        )

    return contacts


# ============================================================================
# Running models
# ============================================================================
running_models = []

for k in range(HORIZON):
    cost = crocoddyl.CostModelSum(
        state,
        nu,
    )

    # Configuration regularization
    cost.addCost(
        "configuration",
        configuration_cost,
        1.0,
    )

    # Velocity regularization
    cost.addCost(
        "velocity",
        velocity_cost,
        1.0,
    )

    # Control regularization
    cost.addCost(
        "control",
        control_cost,
        1.0,
    )

    # Contact-force regularization
    for name in FOOT_NAMES:
        force_residual = crocoddyl.ResidualModelContactForce(
            state,
            foot_ids[name],
            pin.Force.Zero(),
            3,
            nu,
            True,  # fwddyn=True
        )

        force_cost = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelQuad(3),
            force_residual,
        )

        cost.addCost(
            f"{name}_force",
            force_cost,
            F_WGT,
        )

    # ------------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------------
    constraints = crocoddyl.ConstraintModelManager(
        state,
        nu,
    )

    constraints.addConstraint("configuration_bounds", config_constraint)
    constraints.addConstraint("velocity_bounds", velocity_constraint)
    constraints.addConstraint("control_bounds", control_constraint)

    for name in FOOT_NAMES:
        constraints.addConstraint(
            f"{name}_velocity",
            foot_velocity_constraints[name],
        )

        constraints.addConstraint(
            f"{name}_force",
            contact_force_constraints[name],
        )

    # ------------------------------------------------------------------------
    # Contact forward dynamics
    # ------------------------------------------------------------------------
    differential_model = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        make_contacts(),
        cost,
        constraints,
        0.0,  # JMinvJt_damping
        True,  # enable_force
    )

    running_models.append(
        crocoddyl.IntegratedActionModelEuler(
            differential_model,
            dt,
        )
    )


# ============================================================================
# Terminal model
# ============================================================================
terminal_cost = crocoddyl.CostModelSum(
    state,
    nu,
)

terminal_cost.addCost(
    "configuration",
    configuration_cost,
    1.0,
)

terminal_cost.addCost(
    "velocity",
    velocity_cost,
    1.0,
)


# Terminal base linear velocity constraint.
#
# FrameVelocity residual ordering:
#
#     [angular velocity, linear velocity]
#
# We leave angular velocity unconstrained and force the three linear
# velocity components to zero
base_id = model.getFrameId("base")

if base_id >= model.nframes:
    raise RuntimeError("Could not find frame 'base' in the Go2 URDF")


base_velocity_residual = crocoddyl.ResidualModelFrameVelocity(
    state,
    base_id,
    pin.Motion.Zero(),
    pin.LOCAL_WORLD_ALIGNED,
    nu,
)

base_velocity_constraint = crocoddyl.ConstraintModelResidual(
    state,
    base_velocity_residual,
    np.array(
        [
            -np.inf,
            -np.inf,
            -np.inf,
            0.0,
            0.0,
            0.0,
        ]
    ),
    np.array(
        [
            np.inf,
            np.inf,
            np.inf,
            0.0,
            0.0,
            0.0,
        ]
    ),
)


terminal_constraints = crocoddyl.ConstraintModelManager(
    state,
    nu,
)

terminal_constraints.addConstraint("configuration_bounds", config_constraint)
terminal_constraints.addConstraint("velocity_bounds", velocity_constraint)

terminal_constraints.addConstraint(
    "base_linear_velocity",
    base_velocity_constraint,
)


terminal_differential_model = (
    crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        make_contacts(),
        terminal_cost,
        terminal_constraints,
        0.0,
        True,
    )
)

terminal_model = crocoddyl.IntegratedActionModelEuler(
    terminal_differential_model,
    0.0,
)

# ============================================================================
# Shooting problem
# ============================================================================
problem = crocoddyl.ShootingProblem(
    x0,
    running_models,
    terminal_model,
)

# ============================================================================
# mim_solvers / CSQP
# ============================================================================
solver = mim_solvers.SolverCSQP(problem)

solver.termination_tolerance = 1e-2
solver.max_qp_iters = 100

solver.eps_abs = 1e-3
solver.eps_rel = 1e-3
solver.adaptive_rho_tolerance = 1
solver.rho_update_interval = 100

solver.use_filter_line_search = True

solver.setCallbacks(
    [
        mim_solvers.CallbackVerbose(),
        mim_solvers.CallbackLogger(),
    ]
)

# ============================================================================
# Initial guess
# ============================================================================
xs_init = [x0.copy() for _ in range(HORIZON + 1)]
us_init = [np.zeros(nu) for _ in range(HORIZON)]

# ============================================================================
# Solve
# ============================================================================
start = time.time()

success = solver.solve(
    xs_init,
    us_init,
    1000,
)

end = time.time()

# ============================================================================
# Results
# ============================================================================
xs = solver.xs
us = solver.us

print()
print("============================================================")
print("mim_solvers result")
print("============================================================")
print(f"Solve time: {(end - start) * 1000.0:.3f} ms")
print(f"Final cost: {solver.cost}")
print(f"Iterations: {solver.iter}")
print(f"Success: {success}")
print("============================================================")

# ============================================================================
# Save solution
# ============================================================================
x_sol = np.asarray(xs)
u_sol = np.asarray(us)

running_datas = problem.runningDatas

forces_sol = []

for k in range(HORIZON):
    diff_data = running_datas[k].differential
    contact_data = diff_data.multibody.contacts

    stage_forces = []

    for name in FOOT_NAMES:
        foot_data = contact_data.contacts[name]

        force = np.asarray(
            foot_data.f.linear,
            dtype=float,
        ).copy()

        stage_forces.extend(force.tolist())

    forces_sol.append(stage_forces)

stage_data = []

for k in range(HORIZON):

    stage = {
        "position": x_sol[k][:3].tolist(),
        "orientation": x_sol[k][3:7].tolist(),
        "linear_velocity": x_sol[k][nq : nq + 3].tolist(),
        "angular_velocity": x_sol[k][nq + 3 : nq + 6].tolist(),
        "joint_configuration": x_sol[k][7:nq].tolist(),
        "joint_velocity": x_sol[k][nq + 6 : nq + nv].tolist(),
        "control": u_sol[k][:nv].tolist(),
        "forces": forces_sol[k],
    }

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
        "stages": [],
    },
    "stage_data": stage_data,
}

for k in range(HORIZON):

    output["properties"]["stages"].append(
        {
            "size": nq + nv,
            "control": nu,
            "nforces": 12,
            "ordered_force_frames": FOOT_NAMES,
        }
    )

with open("go2_standing.json", "w") as f:
    json.dump(
        output,
        f,
        indent=4,
    )

print("Saved solution to go2_standing.json")
