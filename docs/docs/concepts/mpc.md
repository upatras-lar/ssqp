# MPC

`ssqpy.solvers.MPC` (`ssqp::solvers::MPC` in C++) is a thin helper around `ssqp` for real-time, receding-horizon
control: it repeatedly re-solves the OCP for the system's current state with a small iteration budget, applies the
first control of the resulting trajectory, and warm-starts the next solve from the previous solution plus fresh
state feedback.

```python
mpc = ssqpy.solvers.MPC(
    model,
    sqp_iters=5,
    qp_iters=50,
    keep_convergence_data=False,
)

mpc.reinitialize(initial_state)

while running:
    state = read_state_from_system()
    x, u = mpc.step(state)
    apply_controls(u.stage(0))
```

## Parameters

`MPC.Params` (or keyword arguments to the constructor) cap the iteration budget *independently* of whatever
`ssqpParams`/`admmParams`/`hpipmParams`/`piqpParams` are passed alongside them; the MPC-level `sqp_iters` and `qp_iters` are
what actually bound the per-step solve time:

* `sqp_iters`: maximum SSQP iterations per MPC step (default keeps this low, e.g. `5`–`10`, since MPC relies on
  warm-starting rather than solving to full convergence every step)
* `qp_iters`: maximum QP solver iterations per SSQP iteration
* `keep_convergence_data`: whether to retain per-step [`ssqpConvergenceData`](../reference/solvers.md), retrievable
  via `mpc.getConvergenceData()`, for logging/analysis

## Warm-starting

* `reinitialize(state)`: reset the warm start to a given state with zero controls.
* `reinitialize(states, controls)`: reset the warm start to full state/control trajectories.
* `step(state)`: the typical receding-horizon call, using `state` as feedback for stage 0, and the *previous*
  solution (shifted) as the warm start for the rest of the horizon.
* `step(states, controls)`: warm-start explicitly with the given trajectories instead of the previous solution.
* `step()`: advance using the current internal guess as-is.

Every `step` variant returns the full `(x, u)` [`FlatVec`](../reference/solvers.md#ssqpy.solvers.FlatVec)
trajectories; in a control loop you typically only apply `u.stage(0)` before the next `step` call.

See [`examples/mujoco_mpc`](../examples.md) and [`examples/real_mpc`](../examples.md) for complete simulated and
real-system MPC loops.
