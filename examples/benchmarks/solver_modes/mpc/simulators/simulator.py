import numpy as np


class SSQPYSimulator:
    def __init__(self, model, dt, init_state):
        self.model = model
        self.nq = model.getnq()
        self.dt = dt
        self.state = np.array(init_state, dtype=float)

    def _dynamics(self, state, tau):
        q = state[: self.nq]
        v = state[self.nq :]

        a = self.model.forwardDynamics(q, v, tau)

        return np.concatenate([v, a])

    def step(self, controls):
        tau = np.array(controls, dtype=float)
        dt = self.dt

        x = self.state

        right_plus = self.model.getDynamics(0).state_right_plus

        k1 = self._dynamics(x, tau)
        k2 = self._dynamics(right_plus(x, 0.5 * dt * k1), tau)
        k3 = self._dynamics(right_plus(x, 0.5 * dt * k2), tau)
        k4 = self._dynamics(right_plus(x, dt * k3), tau)

        self.state = right_plus(x, (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

        return self.state
