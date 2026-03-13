import numpy as np


class SSQPYSimulator:
    def __init__(self, model, dt, init_state):
        self.model = model
        self.dt = dt
        self.state = np.array(init_state, dtype=float)

    def _dynamics(self, state, tau):
        nq = len(state) // 2
        q = state[:nq]
        v = state[nq:]

        a = self.model.forwardDynamics(q, v, tau)

        return np.concatenate([v, a])

    def step(self, controls):
        tau = np.array(controls, dtype=float)
        dt = self.dt

        x = self.state

        k1 = self._dynamics(x, tau)
        k2 = self._dynamics(x + 0.5 * dt * k1, tau)
        k3 = self._dynamics(x + 0.5 * dt * k2, tau)
        k4 = self._dynamics(x + dt * k3, tau)

        self.state = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return self.state
