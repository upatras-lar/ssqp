# Panda Circle (MPC)

[`examples/mujoco_mpc/panda/panda_circle.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/mujoco_mpc/panda/panda_circle.py)
runs the same circular end-effector motion as [Panda Circle (Trajectory Generation)](panda-circle-trajgen.md), but
as closed-loop [MPC](../concepts/mpc.md) against a live [MuJoCo](https://mujoco.org/) simulation instead of one
offline solve. The earlier examples all follow the same shape: build a model for the whole problem once, solve it
once, done. This one is different in kind, not just degree: SSQP only ever sees and solves a short window of the
motion, MuJoCo (not SSQP) is the source of truth for what the robot's state actually is at each moment, and the
two are wired together in a loop that re-solves every simulated timestep. It's the example to read once trajectory
generation makes sense and you want to see the receding-horizon pattern end to end: a short MPC horizon chasing a
long target trajectory, retargeted every step.

## Setup

```python title="examples/mujoco_mpc/panda/panda_circle.py"
--8<-- "examples/mujoco_mpc/panda/panda_circle.py:1:41"
```

Two different horizons are in play here, and the distinction matters: `H`/`dt` (400 steps at 0.01s) describe the
*full* circular trajectory the end effector should eventually trace over four seconds, exactly as in the
trajectory-generation version, while `mpc_H`/`mpc_dt` (10 steps at 0.05s) describe the *much shorter* horizon the
MPC model actually optimizes over at every control step. Note that `mpc_dt` is coarser than `dt`: the MPC model's
internal discretization doesn't have to match the simulator's, or the rate at which controls are actually applied;
it only has to be fine enough to represent the dynamics reasonably over its own, much shorter window. See
[Parameters](../concepts/mpc.md#parameters) in MPC for why this budget is kept small: an MPC step has to complete
fast enough to keep up with real time, so 10 stages at 5x the simulation timestep is a deliberate trade of horizon
length for solve speed, not an oversight.

`INVERSE_MODE` toggles between `ForwardDynamics` (the default here, unlike the trajectory-generation example's
`InverseDynamics`) and `InverseDynamics` for the *internal* SSQP model; when `True`, the control applied to MuJoCo
is computed by running the resulting acceleration back through `model.inverseDynamics` to recover torques (see the
simulation loop, below). This example defaults to `ForwardDynamics` so `mpc.step(...)` output can be sent to
MuJoCo directly, without that extra conversion step, at the cost of the smoother QP linearity `InverseDynamics`
would otherwise buy you (see [Panda Circle](panda-circle-trajgen.md#building-the-model)); with a short, frequently
re-solved MPC horizon, the two modes are worth comparing directly rather than assuming one is always better.

Note also that the internal model is built from a **URDF**
(`../../assets/urdf/panda/panda.urdf`) while MuJoCo, further down, loads a separate **MJCF**
(`../assets/mjcf/panda/panda.xml`). This mirrors a real deployment: the model SSQP optimizes against and the
"real" system it's controlling are two different descriptions of the same robot, not necessarily even the same
file format, and MuJoCo's simulated physics plays the role a real robot's sensors and actuators would.

## The target trajectory

```python title="examples/mujoco_mpc/panda/panda_circle.py"
--8<-- "examples/mujoco_mpc/panda/panda_circle.py:43:51"
```

Identical in spirit to the trajectory-generation example: a full circle, precomputed as `H + 1` points, though this
time it outlives the MPC model's own horizon, which only ever sees a 10-stage window of it at a time. This is the
trajectory MPC is chasing; the MPC model itself, built next, has no idea the full circle exists.

## Building the (short-horizon) model

```python title="examples/mujoco_mpc/panda/panda_circle.py"
--8<-- "examples/mujoco_mpc/panda/panda_circle.py:53:120"
```

The model itself has `mpc_H` stages, not `H`: as far as this `Model` object is concerned, the "problem" is only
ever 10 stages long, and it has no built-in notion of a 400-stage circle at all. A pose-tracking cost is created
per MPC stage and kept in the `pose_costs` list; this is what makes retargeting possible (see below), since a cost
that only knows its target at construction time couldn't be pointed at a different part of the circle later. A box
constraint again keeps the end effector inside a square window, same as the trajectory-generation version.

One addition not present in the offline version is `config_cost`, a `SquaredConfigurationErrorCost` pulling gently
(`C_WGT` is small relative to the tracking weight) toward the arm's starting configuration, added at every stage
alongside the pose-tracking cost. The Panda's 7-joint arm has one more degree of freedom than the 6-dimensional
pose (3 for position, 3 for orientation) the cost tracks, so multiple joint configurations can produce the same
end-effector pose; left unconstrained, that extra freedom is free for the solver to drift through arbitrarily from
one MPC step to the next, since the pose cost alone doesn't penalize it. `config_cost` resolves this, and,
incidentally, holds the two gripper finger joints near their starting position too, since it applies to the full
configuration, by mildly preferring joint configurations close to the nominal start pose: a standard way to keep a
redundant-arm controller's joint motion well-behaved over a long closed-loop run instead of just its end-effector
position.

## Retargeting the horizon every step

```python title="examples/mujoco_mpc/panda/panda_circle.py"
--8<-- "examples/mujoco_mpc/panda/panda_circle.py:123:126"
```

Since the MPC model only has stages for the *next* `mpc_H` steps, but the target trajectory `traj` describes a
full 400-step circle, `update_pose_costs` slides the window: given the current real-time stage, it pushes the
correct upcoming target point into each stage's pose cost via `set_desired_translation`, a mutator exposed on
[`FrameSquaredPoseErrorCost`](../reference/model-costs.md) for exactly this purpose (predefined costs' target
values aren't fixed at construction time). This is called once per simulation step, after `mpc.step`, so the model
is always tracking the correct absolute segment of the circle regardless of how short its own horizon is; without
this call, the MPC model would keep re-solving the same fixed 10-stage arc it was built with, forever, instead of
progressing around the circle.

## MPC and solver parameters

```python title="examples/mujoco_mpc/panda/panda_circle.py"
--8<-- "examples/mujoco_mpc/panda/panda_circle.py:128:141"
```

`ssqpy.solvers.MPC` wraps the model with its own iteration budget (`sqp_iters=4`, `qp_iters=100`), deliberately
low, since this now has to run once per real-time control step rather than once, offline (see
[Parameters](../concepts/mpc.md#parameters) in MPC). Four SSQP iterations is not enough to fully converge a fresh
problem from a poor guess; it's enough to make good progress from a guess that's already close to correct, which is
exactly what MPC provides at every step after the first, since consecutive re-solves only need to adjust a
trajectory that was already nearly right one timestep ago (see
[Warm-starting](../concepts/mpc.md#warm-starting) in MPC).

## The simulation loop

```python title="examples/mujoco_mpc/panda/panda_circle.py"
--8<-- "examples/mujoco_mpc/panda/panda_circle.py:143:199"
```

Each iteration: read the current MuJoCo state (`qpos`/`qvel`), which is treated as ground truth and fed to SSQP as
$\boldsymbol{x}_0$, call `mpc.step(state)` to get updated `(x, u)` trajectories, and apply `u.stage(0)`, the first
stage's control only, to the simulator (either directly as torques, or through `model.inverseDynamics` if
`INVERSE_MODE` is set). This is exactly the receding-horizon pattern described in [MPC](../concepts/mpc.md): only
the very first control of a freshly (re-)solved trajectory is ever actually used, and everything the solve computed
for the remaining 9 stages is discarded and recomputed from the new state next step, rather than played back open
loop. `mujoco.mj_step` then advances the physics under that control, `stage` is incremented, `update_pose_costs`
retargets the horizon for the next call, and the loop sleeps to hold real time; the control-frequency counter
printed to the terminal is a direct measure of whether the MPC solve is actually keeping up with `dt`.

Run it (from `examples/mujoco_mpc/panda/`, with the environment set up per [Installation](../getting-started/installation.md#using-the-python-bindings)
and MuJoCo's own virtual environment, per [Examples](../examples.md)):

```bash
python panda_circle.py
```

A MuJoCo viewer window opens and the Panda arm should trace the circle live, correcting continuously as the
simulation runs; the terminal prints the achieved control frequency. With `PLOT = True` (the default), the loop
exits after one full lap (`stage == H`) and the same tracking-error/trajectory plots as the offline example appear
for comparison; with `PLOT = False`, that check is skipped entirely (it's nested inside the `if PLOT` block), so
the simulation instead keeps looping the circle indefinitely, since `update_pose_costs` wraps the stage index with
`% H`, until you close the viewer window yourself. A control frequency well below $1/dt = 100\text{Hz}$ is a sign
that each MPC solve is taking too long relative to the simulation timestep; that's exactly the failure mode the low
`sqp_iters`/`qp_iters` budget above is there to avoid.

## Things to try

* Toggle `INVERSE_MODE` and compare tracking quality and the printed control frequency; inverse dynamics adds a
  conversion step per control application but may converge in fewer SSQP iterations per solve.
* Increase `mpc_H` (and/or decrease `mpc_dt`) to see the tracking improve as the MPC model looks further ahead, at
  the cost of a slower per-step solve and a lower achieved control frequency.
* Lower `sqp_iters`/`qp_iters` while keeping `CONSTRAIN = True` to see constraint satisfaction degrade gracefully while
  tracking defaults to following just the cost, or raise them to see how much headroom there is before the control frequency 
  starts dropping.
* Set `C_WGT` to `0` to remove the redundancy-resolving configuration cost, run a few laps, and watch the arm's
  posture drift over time even though the end-effector path stays correct.
