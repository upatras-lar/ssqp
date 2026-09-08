# Panda Circle (Trajectory Generation)

[`examples/trajgen/python/panda_circle.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/trajgen/python/panda_circle.py)
generates a trajectory that drives a Franka Panda arm's end effector around a circle on the yz-plane, optionally
constrained to stay within a square window inscribed in that circle. Where [Pendubot](pendubot.md) tracked a fixed
target *configuration*, this example tracks a moving target *pose in task space*: at every stage, the cost doesn't
care what the joint angles are, only where the end effector ends up and how it's oriented, which is a much more
common way to specify a manipulation task. It's the best next step after Pendubot: frame-relative pose tracking, a
real (non-XML-derived) box constraint, and `InverseDynamics` mode.

## Setup and the target trajectory

```python title="examples/trajgen/python/panda_circle.py"
--8<-- "examples/trajgen/python/panda_circle.py:1:55"
```

The circular path is precomputed as a plain Python list of 3D points, one per stage, parameterized by angular
velocity $\omega$ so the end effector completes exactly one full circle over the horizon ($H = 400$ stages at
$dt = 0.01\text{s}$, i.e. four seconds). This is the target the cost will track, not a constraint: nothing stops the
solver from cutting corners if doing so is cheaper, which matters once `CONSTRAIN` clips the circle into a square
later on. `Rd` is the desired end-effector orientation (constant throughout, on the yz-plane the arm reaches
"forward" with a fixed wrist rotation), expressed as a rotation matrix rather than e.g. Euler angles, since that's
what the frame-pose costs expect and it avoids the singularities/ambiguities of angle-based representations.

The two module-level flags at the top, `PLOT` and `CONSTRAIN`, are the two things worth toggling first if you want
to see this example do something different before reading further: `PLOT` shows the tracking error and the
resulting trajectory after solving; `CONSTRAIN` decides whether the box constraint (below) is actually applied.

## Building the model

```python title="examples/trajgen/python/panda_circle.py"
--8<-- "examples/trajgen/python/panda_circle.py:57:88"
```

This model uses `InverseDynamics` [solver mode](../concepts/model.md#solver-mode), so controls are joint
accelerations rather than torques, which the example's comments note can simplify convergence for this kind of
task: accelerations enter the dynamics linearly, so the QP the solver builds at each iteration is a better local
approximation of the true problem than it would be with torques, where the mapping to acceleration is nonlinear in
the joint configuration.

`init_state` is a single, fixed arm configuration (not the resting/neutral one) chosen to already be reasonably
close to the circle. As the comment in the source explains, this array does double duty: it's the fixed initial
state $\boldsymbol{x}_0$ (the arm has to actually start there), *and* the initial guess used to seed every other
stage before the solver improves on it. A better initial guess generally means fewer SSQP iterations to converge;
picking one near the expected solution, as here, is a common and effective way to help a local solver like SSQP.
See [`finalize` arguments](../concepts/model.md#finalize-arguments) in Model.

## A real constraint: keeping the end effector inside a box

```python title="examples/trajgen/python/panda_circle.py"
--8<-- "examples/trajgen/python/panda_circle.py:90:99"
```

`FrameTranslationBound` is a [predefined constraint](../concepts/costs-constraints-dynamics.md#predefined-constraints)
bounding a frame's position component-wise; `-np.inf`/`np.inf` leaves the x-coordinate free (the plane the circle
lives on is already fixed by the cost, so the constraint only needs to act on y/z) while the y/z coordinates are
boxed in around the circle. Since the box is inscribed *inside* the circle, this constraint is genuinely active:
whenever the tracked point on the circle falls outside the box (near the four "corners" of the circle relative to
the square), the solver has to trade off cost against constraint satisfaction, and the resulting path visibly
flattens against the box edges instead of following the circle exactly. This is a good example to compare
`CONSTRAIN = True` against `CONSTRAIN = False` and look at the `PLOT` output for; with it off, the trajectory
follows the circle everywhere, since nothing stops it from doing so.

The constraint object is created once and reused across every stage it's added to, since constraints don't need to
be re-instantiated per stage unless their bounds actually differ.

## Per-stage costs and constraints

```python title="examples/trajgen/python/panda_circle.py"
--8<-- "examples/trajgen/python/panda_circle.py:101:135"
```

`FrameSquaredPoseErrorCost` is added at every stage with that stage's own target point from `traj` and the fixed
orientation `Rd`, weighted separately for translation (`T_WGT`) and rotation (`R_WGT`); translation is weighted
three times as heavily here, so the solver prioritizes tracing the circle's shape over holding the wrist exactly
level, though both matter. Velocity and control regularization are added too, at every stage, for the same reason
as in Pendubot: without them, the solver has no reason to prefer a smooth, low-effort trajectory among the many
that satisfy the pose cost equally well. The box constraint is added too, if `CONSTRAIN` is set.

Note the terminal stage (`H`) only gets the velocity regularization, not the control one, since there are no
controls at the terminal stage (see
[Horizon, states and controls](../concepts/model.md#horizon-states-and-controls) in Model); it does still get its
own pose-tracking cost, added separately just below this block, so the very last point on the circle is tracked
too.

## Solving

```python title="examples/trajgen/python/panda_circle.py"
--8<-- "examples/trajgen/python/panda_circle.py:137:167"
```

The solver parameters here are mostly tightened tolerances (`admm_abs_tolerance`/`admm_rel_tolerance`) rather than
a different QP backend; this is a much less pathological problem than Pendubot's swing-up (there's no local-minimum
trap to route around), so the default `ADMM` solver, given a reasonably good initial guess, is enough. After
solving, `model.forwardKinematics(q, "panda_ee")` is used (in the plotting code that follows, not shown here) to
recover the actual end-effector path from the solved joint trajectory, for comparison against the target `traj`,
using the same [Pinocchio-backed algorithm](../cpp-reference/model.md#function-forwardkinematics) used internally
by the frame-based costs and constraints; this is exactly how you'd check tracking accuracy for any frame-based
task, not just in plotting code.

Run it (from `examples/trajgen/python/`, with the environment set up per [Installation](../getting-started/installation.md#using-the-python-bindings)):

```bash
python panda_circle.py
```

With `PLOT = True` (the default), two figures appear after solving: end-effector tracking error over the horizon
on the left (small and roughly constant with `CONSTRAIN = False`, spiking near the box edges with it `True`), and
the traced-out trajectory against the target circle on the right, with the box drawn in when constrained. The
solver also prints its final `SolverState` to the terminal; `Success` (or `LooseSuccess`) is what you should see
for the default configuration.

## Things to try

* Toggle `CONSTRAIN` and compare both plots: the tracking-error spikes and the flattened corners in the trajectory
  plot are a direct, visual consequence of the constraint being active.
* Loosen or tighten the box (`constraint_hlines`/`constraint_vlines`) and watch how much of the circle gets clipped.
* Try `solver_mode=ForwardDynamics` instead of `InverseDynamics` (you'll need to drop the `hpipm_max_iter` override
  or retune it) and compare convergence; this is a good way to build intuition for when the choice of solver mode
  actually matters in practice.
* Replace `init_state` with the zero/neutral configuration (drop `init_states=init_state` from `model.finalize`)
  and see how much slower or less reliable convergence becomes from a worse initial guess.
