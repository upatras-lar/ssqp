# Pendubot Swing-Up

A pendubot is a double pendulum actuated only at the shoulder joint; the elbow is free-swinging. Because there's
no direct control over the second link, the only way to bring it up is to swing the whole system through its
nonlinear dynamics and let the coupling between the two links do the work; no simple feedback law does this,
which is why it's a natural OCP: pick a cost that rewards being upright, hand SSQP the horizon, and let it
discover a trajectory that gets there.

[`examples/trajgen/python/pendubot.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/trajgen/python/pendubot.py)
generates a trajectory that swings the pendubot up from hanging and holds it stationary pointing against gravity.
It's a good first "real" example after the [Quickstart](../getting-started/quickstart.md): an XML-based model,
predefined costs, and a scheduling trick that's worth understanding before moving to constrained examples.

## Setup

```python title="examples/trajgen/python/pendubot.py"
--8<-- "examples/trajgen/python/pendubot.py:1:18"
```

The model is loaded straight from a URDF, so no custom dynamics are needed; see
[Defining a model](../concepts/model.md#defining-a-model) in Model.
[`pendubot_cont.urdf`](https://github.com/upatras-lar/ssqp/blob/main/examples/assets/urdf/double_pendulum/pendubot_cont.urdf)
declares both joints as `continuous` (unbounded revolute joints, as opposed to a joint with hard angle limits), which
Pinocchio represents on the circle manifold $S^1$ as a $(\cos\theta, \sin\theta)$ pair rather than a raw angle. That's
why `nq` is 4 for a 2-link system, one $(\cos, \sin)$ pair per joint, and it's a concrete instance of the
[manifold-valued states](../concepts/costs-constraints-dynamics.md#manifold-valued-states) SSQP supports: SSQP
optimizes over the joints' tangent (angular-velocity) space while keeping every intermediate guess a valid point on
the circle.

`DESIRED_CONFIG = (-1.0, 0.0, 1.0, 0.0)` decodes to shoulder angle $\pi$ (from $(\cos, \sin) = (-1, 0)$) and elbow
angle $0$ (from $(1, 0)$), i.e. both links extended straight up, opposing gravity. No `init_state` is passed to
`model.finalize`, so SSQP falls back to its own manifold-aware neutral configuration, which sets each continuous
joint's $\cos$ component to $1$ (angle $0$) rather than leaving the state at literal zero (which wouldn't even be a
valid point on the circle). Angle $0$ for both joints is exactly "hanging straight down", so the solver's initial
guess for every stage is the pendubot at rest, and the whole trajectory has to be discovered from there.

## Building the model

```python title="examples/trajgen/python/pendubot.py"
--8<-- "examples/trajgen/python/pendubot.py:20:30"
```

`ForwardDynamics` is the default [solver mode](../concepts/model.md#solver-mode), where controls are joint torques:
the most direct formulation, and a reasonable default when there's no particular reason to prefer accelerations
(as in [Panda Circle](panda-circle-trajgen.md)) or a reduced floating-base formulation (as in [Go2
Jump](go2-jump.md)).

## Costs, and why the tracking cost turns on late

```python title="examples/trajgen/python/pendubot.py"
--8<-- "examples/trajgen/python/pendubot.py:32:45"
```

Three [predefined costs](../concepts/costs-constraints-dynamics.md#predefined-costs) are used:
`SquaredJointVelocityCost` and `SquaredControlCost` are cheap regularization terms added at *every* stage, keeping
the motion and effort small by default (`V_WGT`/`U_WGT` are both small, so they barely influence the solution
except where nothing else does). `SquaredConfigurationErrorCost`, the one that actually pulls the pendulum toward
`DESIRED_CONFIG` (weighted much more heavily, `C_WGT = 5.0`), is only added for `k > 150`, i.e. only over the last
quarter of the horizon (`H = 200`), and separately set as the `terminal_cost` so the very last stage is held to it
too.

This is a deliberate swing-up trick, and the reason the example is worth reading rather than just running: SSQP is
a local method, it improves on whatever guess it's given rather than searching globally, and the initial guess here
is the pendulum at rest. If the tracking cost were active from stage 0, gradient descent on a strongly-weighted,
highly nonlinear configuration error from a resting start pulls the shoulder the "short way", directly toward the
target angle, which for a double pendulum is exactly the motion that *doesn't* generate the momentum needed to
bring the second link up; the solver settles into that local minimum and gets stuck well short of upright. Leaving
the early stages free to minimize only velocity/effort gives SSQP's line search room to explore a swinging motion
that costs some torque now in exchange for reaching the target cheaply later, and only the later stages commit to
actually holding it. This is a general pattern worth keeping in mind for other underactuated or highly nonlinear
swing-type problems: a cost that's "obviously correct" as a final objective can actively prevent convergence if
applied too early or too uniformly across the horizon.

## Solving

```python title="examples/trajgen/python/pendubot.py"
--8<-- "examples/trajgen/python/pendubot.py:47:67"
```

Solver and QP parameters are passed as keyword arguments: `hpipm_`/`admm_`-prefixed ones configure the
[QP solver](../concepts/qp-solvers.md) actually selected via `qp_solver_type`; the rest are ignored. This example
selects `HPIPM` explicitly (SSQP defaults to `ADMM`) and tunes several of its and ADMM's parameters away from
their defaults; swing-up is enough more nonlinear than plain regularization that the defaults, tuned for easier
problems, may need help. If you're adapting this example for your own underactuated system and convergence stalls
or `MaxIterReached` shows up, these are exactly the parameters worth revisiting first, alongside `tolerance` and
`filter_size` (see [The SSQP Solver](../concepts/solver.md)). `solver.write` saves the trajectory for later use,
e.g. with the [visualizer](../examples.md).

Run it (from `examples/trajgen/python/`, with the environment set up per [Installation](../getting-started/installation.md#using-the-python-bindings)):

```bash
python pendubot.py
```

The script itself prints nothing on success; it writes `../data/pendubot.json`. Visualize the result with the
[`visualizer` example](../examples.md) to see the actual swing-up motion, or load the JSON yourself and plot the
shoulder/elbow angles over time to see the "wait, then commit" pattern described above: little motion in the first
150 stages beyond light damping, then a swing into the upright configuration over the last quarter of the horizon.

## Things to try

* Move the threshold (`k > 150`) earlier or later and see how it trades off "the tracking cost has more time to
  act, so the final approach is gentler" against "the solver has less free room to discover the swing, so it's
  more likely to get stuck."
* Set the threshold to `k > 0` (tracking active from the very first stage) to reproduce the local-minimum failure
  described above, and compare the resulting (much worse) trajectory to the default.
* Increase `C_WGT` to see the final approach become more aggressive (and possibly jerkier); decrease it to see how
  loosely the terminal configuration is actually held.
* Remove the `qp_solver_type`/`hpipm_*` overrides to fall back to SSQP's default `ADMM` solver, and compare
  convergence (iteration count, `SolverState`) against the tuned HPIPM configuration used here.
