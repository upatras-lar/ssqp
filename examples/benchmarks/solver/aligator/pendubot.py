"""
Pendubot swing-up trajectory optimization, translated to aligator.
"""

import numpy as np
import pinocchio as pin
import aligator
from aligator import manifolds, dynamics, constraints
from time import time

# ---------------------------------------------------------------------------
# Build the Pinocchio model
# ---------------------------------------------------------------------------
URDF_PATH = "urdf/double_pendulum/pendubot.urdf"

model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()

nq, nv = model.nq, model.nv
assert nq == 2 and nv == 2

tau1_max = float(model.effortLimit[0])
dq_max = np.asarray(model.velocityLimit).reshape(-1)
damping = np.asarray(model.damping).reshape(-1)

H = 2501
dt = 0.001
N = H - 1  # number of transitions / control intervals

nu = 1  # tau1 only

# ---------------------------------------------------------------------------
# State space: q, v both R^2 for revolute joints -> ndx = nq + nv = 4,
# and the manifold "difference" is plain vector subtraction
# ---------------------------------------------------------------------------
space = manifolds.MultibodyPhaseSpace(model)
ndx = space.ndx

x0 = np.array([0.0, 0.0, 0.0, 0.0])  # hang down
xf = np.array([np.pi, 0.0, 0.0, 0.0])  # balance up

B = np.array([[1.0], [0.0]])  # actuation matrix: joint1 actuated, joint2 not


# ---------------------------------------------------------------------------
# Custom continuous dynamics: ddq = ABA(q, v, B@u - damping*v)
# ---------------------------------------------------------------------------
class PendubotFwdDynamics(dynamics.ODEAbstract):
    def __init__(self, space, model, actuation_matrix, damping):
        super().__init__(space, actuation_matrix.shape[1])
        self.pin_model = model
        self.pin_data = model.createData()
        self.B = actuation_matrix
        self.damping = damping

    def __deepcopy__(self, memo):
        # Avoid the default deepcopy path, which tries to reconstruct this
        # pybind11-derived object by calling PendubotFwdDynamics() with no
        # arguments. All mutable state during forward()/dForward() lives on
        # the `data` object aligator passes in, not on self, so sharing the
        # same instance across copies is safe.
        return self

    def forward(self, x, u, data):
        try:
            nq_, nv_ = self.pin_model.nq, self.pin_model.nv
            q = x[:nq_]
            v = x[nq_:]
            tau_eff = self.B @ u - self.damping * v
            ddq = pin.aba(self.pin_model, self.pin_data, q, v, tau_eff)
            data.xdot[:nv_] = v
            data.xdot[nv_:] = ddq
        except Exception:
            # boost.python can swallow the real traceback and just abort;
            # print it here so it's visible, then re-raise.
            import traceback

            traceback.print_exc()
            raise

    def dForward(self, x, u, data):
        try:
            nq_, nv_ = self.pin_model.nq, self.pin_model.nv
            q = x[:nq_]
            v = x[nq_:]
            tau_eff = self.B @ u - self.damping * v

            pin.computeABADerivatives(
                self.pin_model, self.pin_data, q, v, tau_eff
            )
            ddq_dq = self.pin_data.ddq_dq
            ddq_dv_partial = self.pin_data.ddq_dv
            ddq_dtau = self.pin_data.Minv  # d(ddq)/d(tau)

            # chain rule: tau_eff depends on v through -damping*v
            ddq_dv = ddq_dv_partial - ddq_dtau @ np.diag(self.damping)
            ddq_du = ddq_dtau @ self.B

            data.Jx[:nv_, :nv_] = 0.0
            data.Jx[:nv_, nv_:] = np.eye(nv_)
            data.Jx[nv_:, :nv_] = ddq_dq
            data.Jx[nv_:, nv_:] = ddq_dv
            data.Ju[:nv_] = 0.0
            data.Ju[nv_:] = ddq_du[:, 0]
        except Exception:
            import traceback

            traceback.print_exc()
            raise


ode = PendubotFwdDynamics(space, model, B, damping)
disc_dynamics = dynamics.IntegratorEuler(ode, dt)

# ---------------------------------------------------------------------------
# Costs
#   - velocity cost:  V_WGT * |v|^2                       (every node)
#   - control cost:   U_WGT * tau1^2                      (every stage)
#   - config cost:    C_WGT * |q - xf_q|^2                (last 50 nodes)
# ---------------------------------------------------------------------------
V_WGT = 1e-4
U_WGT = 1e-4
C_WGT = 1.0

# Full-state error residual against xf
state_err_to_xf = aligator.StateErrorResidual(space, nu, xf)
config_err = state_err_to_xf[:nq]  # q - xf_q

state_err_zero = aligator.StateErrorResidual(space, nu, np.zeros(ndx))
vel_err = state_err_zero[nq : nq + nv]  # v - 0

control_err = aligator.ControlErrorResidual(ndx, nu)

vel_cost = aligator.QuadraticResidualCost(space, vel_err, V_WGT * np.eye(nv))
u_cost = aligator.QuadraticResidualCost(space, control_err, U_WGT * np.eye(nu))
config_cost = aligator.QuadraticResidualCost(
    space, config_err, C_WGT * np.eye(nq)
)

# ---------------------------------------------------------------------------
# Path constraints: velocity box (every stage), torque box (non-terminal stages)
# ---------------------------------------------------------------------------
lb_dq, ub_dq = -dq_max, dq_max
lb_u, ub_u = np.array([-tau1_max]), np.array([tau1_max])

vel_box = constraints.BoxConstraint(lb_dq, ub_dq)
u_box = constraints.BoxConstraint(lb_u, ub_u)

# ---------------------------------------------------------------------------
# Build stages
# ---------------------------------------------------------------------------
stages = []
for k in range(N):
    cost_stack = aligator.CostStack(space, nu)
    cost_stack.addCost(vel_cost)
    cost_stack.addCost(u_cost)
    if k > H - 500:
        cost_stack.addCost(config_cost)

    stage = aligator.StageModel(cost_stack, disc_dynamics)
    stage.addConstraint(vel_err, vel_box)
    stage.addConstraint(control_err, u_box)
    stages.append(stage)

# Terminal cost
term_cost_stack = aligator.CostStack(space, nu)
term_cost_stack.addCost(vel_cost)
term_cost_stack.addCost(config_cost)

# Terminal velocity limits
problem = aligator.TrajOptProblem(x0, stages, term_cost_stack)
problem.addTerminalConstraint(vel_err, vel_box)

# ---------------------------------------------------------------------------
# Initial guess: hang-down state at every node, zero torque
# ---------------------------------------------------------------------------
xs_init = [x0.copy() for _ in range(H)]
us_init = [np.zeros(nu) for _ in range(N)]

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
tol = 1e-2
mu_init = 1e-2
solver = aligator.SolverProxDDP(
    tol, mu_init, verbose=aligator.VerboseLevel.VERBOSE
)
solver.max_iters = 1000
solver.setup(problem)

start = time()
conv = solver.run(problem, xs_init, us_init)
print(f"Solve time: {time() - start:.2f} sec, converged: {conv}")

results = solver.results
xs_opt = results.xs
us_opt = results.us

q1_traj = np.array([x[0] for x in xs_opt])
q2_traj = np.array([x[1] for x in xs_opt])
tau1_traj = np.array([u[0] for u in us_opt])
times = np.linspace(0, H * dt, H)

print(f"Final state (q1, q2, dq1, dq2): {xs_opt[-1]}")
print(
    f"Max |tau1| used: {np.max(np.abs(tau1_traj)):.4f} Nm (limit {tau1_max})"
)

# ---------------------------------------------------------------------------
# Save solution in the JSON format required by the visualizer
# ---------------------------------------------------------------------------
import json

stage_data = []
for k in range(H):
    stage = {"state": [float(v) for v in xs_opt[k]]}
    if k < N:
        stage["control"] = [float(us_opt[k][0])]
    stage_data.append(stage)

output = {
    "properties": {
        "horizon": N,
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
        "joint_types": ["revolute", "revolute"],
        "stages": [],
    },
    "stage_data": stage_data,
}

for k in range(H):
    stage_info = {
        "size": ndx,
        "control": nu if k < N else 0,
        "nforces": 0,
        "ordered_force_frames": [] if k < N else None,
    }
    output["properties"]["stages"].append(stage_info)

with open("pendubot.json", "w") as f:
    json.dump(output, f, indent=4)

print("Saved solution to pendubot.json")
