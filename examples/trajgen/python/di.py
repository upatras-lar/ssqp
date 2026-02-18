import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import numpy as np
from math import pi


class DoubleIntegrator(ssqpy.model.dynamics.StageDynamics):
    def output_size(self):
        return 2

    def state_size(self):
        return 2

    def control_size(self):
        return 1

    def evaluate(self, x, u):
        p, v = x[:]
        vn = v + u[:] * 0.1
        pn = p + vn * 0.1

        return np.array([pn, vn])

    def gradient(self, x, u):
        dX = np.zeros((2, 2))
        dX[0, 0] = 1.0
        dX[0, 1] = 0.1
        dX[1, 1] = 1.0

        dU = np.zeros((2, 1))
        dU[1, 0] = 0.1

        return dX, dU

    def hessprod_xx(self, x, u):
        return np.zeros((2, 2))

    def hessprod_xu(self, x, u):
        return np.zeros((2, 1))

    def hessprod_uu(self, x, u):
        return np.zeros((1, 1))


class Cost(costs.StageCost):
    def evaluate(self, x, u):
        return 0.5 * (x[0] - 1.0) ** 2

    def gradient(self, x, u):
        dx = np.zeros(2)
        dx[0] = x[0] - 1.0

        du = np.zeros(1)

        return dx, du

    def hessian_xx(self, x, u):
        Hxx = np.zeros((2, 2))
        Hxx[0, 0] = 1.0

        return Hxx

    def hessian_xu(self, x, u):
        return np.zeros((2, 1))

    def hessian_uu(self, x, u):
        return np.zeros((1, 1))


stage_cost = Cost()

model = ssqpy.model.Model(
    10,
    0.01,
)

model.finalize(
    dynamics=[DoubleIntegrator()],
    stage_cost=stage_cost,
    terminal_cost=stage_cost,
)

# Create an SSQP solver for the model
solver = ssqpy.solvers.ssqp(model)

state, x, u = solver.solve()

print("x:")
x.print()
print("u:")
u.print()
