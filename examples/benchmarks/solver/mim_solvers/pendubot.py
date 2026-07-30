import pinocchio as pin
import crocoddyl
import mim_solvers
import numpy as np
import time
import json

H = 2500
dt = 0.001

np.set_printoptions(suppress=True)

urdf = "urdf/double_pendulum/pendubot.urdf"


# ============================================================================
# Robot model
# ============================================================================

model = pin.buildModelFromUrdf(urdf)

nq = model.nq
nv = model.nv

print(f"nq = {nq}, nv = {nv}")
print(f"damping = {model.damping}")
print(f"effort limits = {model.effortLimit}")

state = crocoddyl.StateMultibody(model)

# Initial state:
#
# x = [q1, q2, dq1, dq2]
x0 = np.array((0.0, 0.0, 0.0, 0.0))


# ============================================================================
# Pendubot actuation with viscous damping
# ============================================================================
class PendubotActuationModel(crocoddyl.ActuationModelAbstract):

    def __init__(self, state, model):
        super().__init__(state, nu=1)

        self.model = model

        # Joint 1 is actuated, joint 2 is passive.
        # tau_control = B @ u
        # B = [1]
        #     [0]
        self.B = np.array(
            [
                [1.0],
                [0.0],
            ]
        )

        # Viscous damping from the URDF.
        # tau_damping = damping * v
        self.damping = np.asarray(
            model.damping,
            dtype=float,
        ).reshape(-1)

        # Torque bounds on the actual control input u = tau1.
        self.u_lb = np.array(
            [
                -model.effortLimit[0],
            ]
        )

        self.u_ub = np.array(
            [
                model.effortLimit[0],
            ]
        )

    def calc(self, data, x, u):
        """
        Effective torque passed to Pinocchio:
            tau_eff = B @ u - damping * v
        i.e.
            tau_eff =
                [u0 - d1*dq1]
                [-d2*dq2]
        """
        v = x[self.model.nq :]
        data.tau[:] = self.B @ u - self.damping * v

    def calcDiff(self, data, x, u):
        """
        Derivatives of:
            tau = B @ u - D @ v
        with respect to the state and control:
            d tau / d x =
                [ 0       0 ]
                [ -D      ]
        more precisely, since x = [q, v]:
            d tau / d q = 0
            d tau / d v = -D
            d tau / d u = B
        """
        # Clear everything first
        data.dtau_dx[:] = 0.0
        data.dtau_du[:] = 0.0

        # x = [q, v], therefore velocity starts at nq
        # tau has dimension nv
        for i in range(self.model.nv):
            data.dtau_dx[i, self.model.nq + i] = -self.damping[i]

        # Control mapping:
        # tau1 = u
        # tau2 = 0
        data.dtau_du[:] = self.B[:, 0]


actuation = PendubotActuationModel(
    state,
    model,
)

# ============================================================================
# Costs
# ============================================================================
V_WGT = 1e-4
U_WGT = 1e-4
C_WGT = 1.0

activation_configuration = crocoddyl.ActivationModelWeightedQuad(
    np.concatenate(
        [
            np.ones(nv),
            np.zeros(nv),
        ]
    )
)

activation_velocities = crocoddyl.ActivationModelWeightedQuad(
    np.concatenate(
        [
            np.zeros(nv),
            np.ones(nv),
        ]
    )
)

# Velocity cost:
#     ||v||^2
velCost = crocoddyl.CostModelResidual(
    state,
    activation_velocities,
    crocoddyl.ResidualModelState(
        state,
        x0,
        nu=1,
    ),
)

# Control cost:
#     u^2
uCost = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ResidualModelControl(
        state,
        nu=1,
    ),
)

# Target configuration
qdes = np.array(
    [
        np.pi,
        0.0,
    ]
)

xref = np.concatenate(
    [
        qdes,
        np.zeros(nv),
    ]
)

configCost = crocoddyl.CostModelResidual(
    state,
    activation_configuration,
    crocoddyl.ResidualModelState(
        state,
        xref,
        nu=1,
    ),
)

# ============================================================================
# Constraints
# ============================================================================
# Velocity bounds (dimension state.nv)
lb_dq = -np.asarray(model.velocityLimit).reshape(-1)
ub_dq = np.asarray(model.velocityLimit).reshape(-1)

# Control bounds
tau1_max = float(model.effortLimit[0])

lb_u = np.array([-tau1_max])
ub_u = np.array([tau1_max])

# ---------------------------------------------------------------------------
# Velocity constraint via ResidualModelState
# ---------------------------------------------------------------------------
# State residual x ⊖ x0 has total dimension state.ndx (configuration + velocity)
velocity_residual_full = crocoddyl.ResidualModelState(
    state,
    state.zero(),  # x0 = 0 ensures the velocity residual equals v - 0 = v
    nu=1,
)

# Pad bounds with +-inf for position components so only velocities are constrained
nq_ndx = state.ndx - state.nv
lb_x = np.concatenate([-np.inf * np.ones(nq_ndx), lb_dq])
ub_x = np.concatenate([np.inf * np.ones(nq_ndx), ub_dq])

velocity_constraint = crocoddyl.ConstraintModelResidual(
    state,
    velocity_residual_full,
    lb_x,
    ub_x,
)

# ---------------------------------------------------------------------------
# Control constraint
# ---------------------------------------------------------------------------
control_residual = crocoddyl.ResidualModelControl(
    state,
    nu=1,
)

control_constraint = crocoddyl.ConstraintModelResidual(
    state,
    control_residual,
    lb_u,
    ub_u,
)

# ============================================================================
# Running models
# ============================================================================

runningModels = []

for k in range(H):
    cost = crocoddyl.CostModelSum(
        state,
        nu=1,
    )

    # Velocity cost
    cost.addCost(
        "vel",
        velCost,
        V_WGT,
    )

    # Control cost
    cost.addCost(
        "u",
        uCost,
        U_WGT,
    )

    # Goal cost during final part of horizon
    if k > H - 500:
        cost.addCost(
            "goal",
            configCost,
            C_WGT,
        )

    constraints = crocoddyl.ConstraintModelManager(
        state,
        nu=1,
    )

    # Velocity limits at every node
    constraints.addConstraint(
        "velocity",
        velocity_constraint,
    )

    # Torque limits at every controlled stage
    constraints.addConstraint(
        "control",
        control_constraint,
    )

    # Free forward dynamics:
    #     M(q) ddq + h(q,v) = tau_eff
    # where our actuation model supplies:
    #     tau_eff = [u, 0] - damping*v
    dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state,
        actuation,
        cost,
        constraints,
    )

    # Euler integration
    runningModels.append(
        crocoddyl.IntegratedActionModelEuler(
            dam,
            dt,
        )
    )

# ============================================================================
# Terminal model
# ============================================================================
terminalCost = crocoddyl.CostModelSum(
    state,
    nu=1,
)

terminalCost.addCost("vel", velCost, V_WGT)

terminalCost.addCost(
    "goal",
    configCost,
    C_WGT,
)

terminalConstraints = crocoddyl.ConstraintModelManager(
    state,
    nu=1,
)

terminalConstraints.addConstraint(
    "velocity",
    velocity_constraint,
)


terminalDAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state,
    actuation,
    terminalCost,
    terminalConstraints,
)

terminalModel = crocoddyl.IntegratedActionModelEuler(
    terminalDAM,
    0.0,
)

# ============================================================================
# Shooting problem
# ============================================================================
problem = crocoddyl.ShootingProblem(
    x0,
    runningModels,
    terminalModel,
)

# ============================================================================
# mim_solvers
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
xs_init = [x0.copy() for _ in range(H + 1)]
us_init = [np.zeros(1) for _ in range(H)]

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


print(f"Solve time: {(end - start) * 1000:.3f} ms")
print(f"Final cost: {solver.cost}")
print(f"Iterations: {solver.iter}")
print(f"Success: {success}")

# -------------------------------------------------------------------------
# Save solution in same visualizer format
# -------------------------------------------------------------------------
x_sol = np.asarray(xs)
u_sol = np.asarray(us)

stage_data = []

for k in range(H):
    stage = {
        "state": [
            float(x_sol[k, 0]),
            float(x_sol[k, 1]),
            float(x_sol[k, 2]),
            float(x_sol[k, 3]),
        ]
    }

    if k < H:
        # Only tau1 is part of the original problem
        stage["control"] = [float(u_sol[k, 0])]

    stage_data.append(stage)

output = {
    "properties": {
        "horizon": H,
        "dt": dt,
        "solver_mode": "ForwardDynamics",
        "xml": True,
        "nq": nq,
        "nv": nv,
        "nu": 1,
        "njoints": model.njoints - 1,
        "total_joints": model.njoints,
        "floating_base": False,
        "state_joints": [model.names[i] for i in range(1, model.njoints)],
        "joint_types": [
            "revolute",
            "revolute",
        ],
        "stages": [],
    },
    "stage_data": stage_data,
}

for k in range(H):
    stage_info = {
        "size": 4,
        "control": 1 if k < H else 0,
        "nforces": 0,
        "ordered_force_frames": [],
    }

    if k == H:
        stage_info["ordered_force_frames"] = None

    output["properties"]["stages"].append(stage_info)

with open("pendubot.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved solution to pendubot.json")
