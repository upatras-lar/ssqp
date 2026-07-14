import pinocchio as pin
import crocoddyl
import numpy as np
import time

H = 200
dt = 0.01

np.set_printoptions(suppress=True)

urdf = "../../../assets/urdf/double_pendulum/pendubot_cont.urdf"


x0 = np.array((1.0, 0.0, 1.0, 0.0, 0.0, 0.0))


class PendubotActuationModel(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        super().__init__(state, nu=1)
        self.u_lb = np.array((-model.effortLimit[0],))
        self.u_ub = np.array((model.effortLimit[0],))

    def calc(self, data, x, u):
        data.tau[0] = u[0]
        data.tau[1] = 0.0

    def calcDiff(self, data, x, u):
        data.dtau_du[0] = 1.0
        data.dtau_du[1] = 0.0


model = pin.buildModelFromUrdf(urdf)
state = crocoddyl.StateMultibody(model)

actuation = PendubotActuationModel(state)

activation_configuration = crocoddyl.ActivationModelWeightedQuad(
    np.concatenate([np.ones(model.nv), np.zeros(model.nv)])
)

activation_velocities = crocoddyl.ActivationModelWeightedQuad(
    np.concatenate([np.zeros(model.nv), np.ones(model.nv)])
)

velCost = crocoddyl.CostModelResidual(
    state, activation_velocities, crocoddyl.ResidualModelState(state, x0, nu=1)
)

uCost = crocoddyl.CostModelResidual(
    state, crocoddyl.ResidualModelControl(state, nu=1)
)

qdes = np.array([-1, 0, 1, 0])
xref = np.concatenate([qdes, np.zeros(model.nv)])

configCost = crocoddyl.CostModelResidual(
    state,
    activation_configuration,
    crocoddyl.ResidualModelState(state, xref, nu=1),
)

runningModels = []

for k in range(H):
    cost = crocoddyl.CostModelSum(state, nu=1)

    cost.addCost("vel", velCost, 1e-2)
    cost.addCost("u", uCost, 1e-2)

    if k > 150:
        cost.addCost("goal", configCost, 5.0)

    dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, cost
    )

    runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, dt))

terminalCost = crocoddyl.CostModelSum(state, nu=1)
terminalCost.addCost("goal", configCost, 5.0)

terminalDAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state, actuation, terminalCost
)

terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, 0.0)

problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

solver = crocoddyl.SolverKKT(problem)
solver.th_stop = 1e-2
solver.th_grad = 1e-2

solver.setCallbacks([crocoddyl.CallbackVerbose()])

start = time.time()
solver.solve([x0] * (H + 1), [np.zeros(1)] * H)
end = time.time()

xs = solver.xs
us = solver.us

for i, x in enumerate(xs):
    print(f"{i}. {x}")

for i, u in enumerate(us):
    print(f"{i}. {u}")

print(f"Solve time: {(end - start) * 1000}ms")

print(f"Final cost: {solver.cost}")
print(f"Iterations: {solver.iter}")
