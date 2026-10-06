# Go2 Jump

[`examples/trajgen/python/go2/go2_jump.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/trajgen/python/go2/go2_jump.py)
generates a jumping trajectory for the Go2 quadruped: a stance phase, a flight phase with all four feet off the
ground, and a landing back into a specified footprint. Where the previous examples used a single, unchanging set of
costs and constraints across the whole horizon, a jump needs the problem itself to change over time: which
constraints even apply (a foot mid-air obviously can't be held in a friction cone) depends on which stage you're
looking at. This is the first example built around **contact scheduling**, per-stage enabling and disabling of
contact constraints, and it also demonstrates the object-based (as opposed to keyword-argument) style of
configuring solver parameters. It builds directly on [Panda Circle](panda-circle-trajgen.md).

## Setup

```python title="examples/trajgen/python/go2/go2_jump.py"
--8<-- "examples/trajgen/python/go2/go2_jump.py:1:72"
```

`ReducedInverseDynamics` [solver mode](../concepts/model.md#solver-mode) is used, with the `Go2`
[preset](../concepts/model.md#built-in-presets) (which fixes `free_flyer=True` for you) since the
Go2's base isn't bolted to anything: it's a free-floating rigid body (position + orientation on $SE(3)$, another
instance of the [manifold-valued states](../concepts/costs-constraints-dynamics.md#manifold-valued-states) SSQP
handles) whose motion is determined entirely by the forces the legs exert on the ground, not by any actuator
directly driving the base. `ReducedInverseDynamics` (rather than the plain `InverseDynamics` used for the fully
actuated Panda arm) accounts for exactly this: the base's 6 degrees of freedom are unactuated, so they're enforced
through the dynamics rather than treated as free control inputs.

`START_STEP`/`END_STEP` mark the flight window (stages 60–89 of a 150-stage horizon) during which no foot is in
contact with the ground. `MAX_NORMAL_FORCE` caps how hard each foot is allowed to push, which matters once contact
forces become part of the optimization below.

## Initial state and a static-equilibrium initial guess

```python title="examples/trajgen/python/go2/go2_jump.py"
--8<-- "examples/trajgen/python/go2/go2_jump.py:74:88"
```

`base_config` is a standing crouch pose, repeated for all four legs (each leg's three joints get the same
hip/thigh/calf angles, since the robot starts symmetric). `init_control` is not zero: each foot's vertical contact
force is initialized to roughly a quarter of the robot's weight ($3.75 \times 9.81\ \text{N}$), i.e. a static
standing equilibrium, a much better initial guess for the stance stages than zero. This matters more here than in
the earlier examples: a jump is a highly dynamic, contact-rich motion, and starting the solver's guess for the
*forces* at "the robot is already balanced on all four feet" rather than "no forces at all" avoids wasting early
iterations just discovering that a standing robot needs to push against the ground to not fall over.

## Costs and reusable constraint objects

```python title="examples/trajgen/python/go2/go2_jump.py"
--8<-- "examples/trajgen/python/go2/go2_jump.py:90:147"
```

Regularization costs (joint angle, velocity, control, force, and a configuration-tracking cost pulling toward the
crouch pose) are created once, to be reused across stages, exactly as in the earlier examples. What's new here is
that per-foot velocity and position bounds are also created upfront, *before* the per-stage loop, because the same
constraint objects will be conditionally attached to different stages depending on the contact schedule computed
below:

* The velocity bounds (zero lower and upper, an equality) are used during stance to keep planted feet from
  sliding: a foot that's actually in contact with the ground shouldn't be moving relative to it.
* The position bounds keep each foot within its own quadrant of the body frame (preventing legs from crossing,
  e.g. `FL_foot`'s y-coordinate is bounded to $y \geq 0.1$, keeping it on the robot's left side throughout).
* `head_constraint` keeps the head above a minimum ground clearance throughout, stance and flight alike, so the
  jump doesn't solve itself by having the robot faceplant.

## The per-stage loop: stance, flight, stance

```python title="examples/trajgen/python/go2/go2_jump.py"
--8<-- "examples/trajgen/python/go2/go2_jump.py:149:193"
```

Every stage gets the regularization costs and the position/head bounds unconditionally; those don't depend on
whether the robot is airborne. The contact scheduling itself happens in the `if k < START_STEP or k >= END_STEP`
branch, i.e. **outside** the flight window: for each foot, `Model.addContactConstraint` adds both a
`ContactConstraint` (friction cone, so the foot can only push, not pull, and only within a friction-limited cone of
directions, plus non-penetration, so the foot can't go through the ground) *and* the matching ground-reaction
force in one call, and the corresponding zero-velocity bound is added alongside it. These are exactly the
constraints that make "standing" or "walking" possible: without a friction cone, a solved trajectory could have
feet pushing sideways with unlimited force; without non-penetration, a foot could dip below the terrain to reduce
cost elsewhere.

Inside the flight window (the branch that `continue`s past this block), **no contact constraints are added at
all**: no friction cone, no non-penetration, no zero-velocity bound, for any foot. With nothing tying the feet to
the ground, the dynamics reduce to free-falling under gravity for those stages, exactly as a jump should be; the
solver isn't told "be airborne", it simply has no way to keep the feet in contact during that window, so the
physics does the rest.

Note that this particular script explicitly disables FL's contact and velocity constraints even during the
"stance" branch (`t1.set_enabled(False)`) while leaving FR/RL/RR enabled, a reminder that contacts are toggled
independently per foot per stage, not just per phase: nothing in the API requires all four feet to share the same
schedule, which is what makes asymmetric gaits (a limp, a three-legged stance, a diagonal trot) representable with
the same building blocks. `set_enabled(False)` also demonstrates the *other* way to keep a constraint out of the
problem for a given stage: instead of only calling `addConstraints` when a constraint should apply (as done for
the other three feet, conditionally, by virtue of being inside the `if` branch at all), a constraint can be added
unconditionally and toggled on/off afterward, which is convenient when a constraint's applicability depends on
something more complex than a single stage-index check.

## Terminal stage: settle into a landing footprint

```python title="examples/trajgen/python/go2/go2_jump.py"
--8<-- "examples/trajgen/python/go2/go2_jump.py:195:247"
```

At the terminal stage, a zero base-velocity constraint asks the robot to be at rest, and each foot gets a
`FrameTranslationBound` with **equal** lower and upper bounds, an equality constraint (see
[Constraints](../concepts/costs-constraints-dynamics.md#constraints) in Costs, Constraints & Dynamics) pinning each
foot to an exact landing position, symmetric front-to-back and left-to-right around the base. Without this, the
solver has no reason to prefer landing back in a stable, symmetric stance over any other configuration that happens
to be feasible at stage `H`; pinning the feet gives the last portion of the trajectory (after `END_STEP`, back
inside the stance branch) something concrete to converge back onto.

## Solving, with object-based parameter configuration

```python title="examples/trajgen/python/go2/go2_jump.py"
--8<-- "examples/trajgen/python/go2/go2_jump.py:249:281"
```

Rather than passing `admm_`/`hpipm_`-prefixed keyword arguments (as in [Pendubot](pendubot.md)), this example
constructs `ssqpParams`, `admmParams` and `hpipmParams` objects directly and assigns fields on them before passing
the top-level `ssqp_params` to the solver; this is equivalent, just a different style, useful when parameters are
computed or reused across multiple solver instances. `ssqp_params.loose_success = True` is worth calling out
specifically: for a problem this constrained and dynamic, waiting for the strict Lagrangian-norm tolerance to be
met on every single term can take far longer than reaching a solution that's already good enough for the jump to
look and behave correctly; loose success lets SSQP stop once progress has stalled rather than keep grinding on
diminishing returns.

Run it (from `examples/trajgen/python/go2/`, with the environment set up per [Installation](../getting-started/installation.md#using-the-python-bindings)):

```bash
python go2_jump.py
```

The script prints the final solver state and writes `../../data/go2_jump.json`. This is the example most worth visualizing to
actually understand: load the result with the [`visualizer` example](../examples.md) and watch the base crouch,
launch, tuck through flight, and land, with FL's foot behaving differently from the other three throughout stance
if you left the `set_enabled(False)` call in place.

## Things to try

* Comment out `t1.set_enabled(False)` so all four feet share the same stance/flight schedule, and compare the
  landing symmetry against the default (asymmetric FL) version.
* Shrink the flight window (bring `--start-step` and `--end-step` closer together) for a shorter hop, or widen it for a
  higher one, and see how far `MAX_NORMAL_FORCE` and the joint/velocity regularization weights can be pushed before
  the solver stops converging.
* Change the terminal foot-position targets to land in a different footprint (e.g. narrower or wider stance) than
  the one the robot started in.
* Try setting `admm_params.reset_rho`/`admm_params.reset_y` back to their defaults (`True`) instead of the `False`
  used here, and compare convergence; these control whether ADMM's internal state carries over between the
  different QP formulations solved across SSQP iterations.
