# Go2 Walking (MPC)

[`examples/mujoco_mpc/go2/go2_walking.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/mujoco_mpc/go2/go2_walking.py)
walks a Unitree Go2 quadruped around in a live [MuJoCo](https://mujoco.org/) simulation, with a trot driven by a
gamepad: the sticks command a forward, sideways and turning velocity, and further buttons change the base height,
tilt and step height, or make the robot jump in place. Everything is planned by a single whole-body
[MPC](../concepts/mpc.md) that solves for joint accelerations and contact forces, and the torques that come out of it
are applied to the simulator directly.

It combines two earlier examples. From [Go2 Jump](go2-jump.md) it takes the Go2 model, the
`ReducedInverseDynamics` [solver mode](../concepts/model.md#solver-mode) and the idea of scheduling contacts per
stage, but where the jump fixed its schedule once, here it is *regenerated at every plan stage* from the joystick.
From [Panda Circle (MPC)](panda-circle-mpc.md) it takes the receding-horizon loop and the pattern of keeping the
costs in lists, so that each step can point them at a different target. What is new is the part in the middle: the
code that turns "go forward at 0.4 m/s" into a base trajectory, a set of footholds and swing paths, and finally a
setting for each cost and constraint of the model.

The model itself knows nothing about gaits. This is worth keeping in mind while reading the script: the SSQP model
is built once, as a list of generic costs and contact constraints, and everything that makes it walk is in how they
are updated.

## Timing

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:20:33"
```

There are two clocks. MuJoCo runs at `dt` = 2 ms. The MPC has 20 stages of `mpc_dt` = 24 ms, so it plans 0.48 s
ahead. They are related by `STAGE_EVERY`, the 12 simulation steps in one plan stage, and `mpc_dt` is required to be a
multiple of `dt` for this to be a whole number. The script keeps a counter `stage` that advances once per plan stage,
not once per simulation step, and the whole planner is written in stages: the gait period, the jump phases and the
base references are all counted in them.

The MPC is also solved once per stage, which is every 24 ms of simulation time. Solving every simulation step would
be 12 times more work for little gain, since the plan only has a new first stage every `mpc_dt` anyway. The simulation
is synced with the controller, so if a solve takes longer than 24 ms of real time, the simulation simply runs slower
than real time.

## Weights

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:35:50"
```

`T_WGT` and `R_WGT` weigh position and rotation tracking, the base's and the feet's. `C_WGT` is a weak pull of the
joints toward a nominal standing configuration, `V_WGT` penalizes joint velocity, and `U_WGT` and `F_WGT` penalize the
control and the contact forces. The meaning of the *control* depends on the dynamics mode: in forward dynamics it is a
joint torque, but in `ReducedInverseDynamics` it is a joint acceleration, which has very different units, so the
weight is different for each (with `U_WGT` much smaller for accelerations). Changing the mode in this script without
checking the weights would silently retune the controller.

## Building the model

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:131:162"
```

The model is built from the robot's MJCF file with `free_flyer=True` (the [Go2 Jump](go2-jump.md) example uses the
`Go2` preset instead). MuJoCo simulates the same robot description, in a scene file that adds the floor and
obstacles. The initial configuration places the base at `BASE_HEIGHT` of
0.34 m and sets every leg with `leg_config`, the closed-form hip, thigh and calf angles that put the foot straight
below the hip at that height (a two-link inverse kinematics problem, using the thigh and calf lengths). The base
quaternion is the identity, so the robot starts facing along $+x$, away from the obstacles in the scene.

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:165:194"
```

Two sets of geometry are computed from the initial pose with forward kinematics. `foot_offsets` is each foot's
position relative to the base, flattened to the ground, which is where the foot would stand when the robot is
upright; the planner uses it as the foot's *nominal* position under the base. `hip_offsets` and `HIP_LENGTH` give the
hip geometry that the leg inverse kinematics (`leg_ik`, further down in the script) needs, since the Go2's thigh is
offset sideways from the hip joint.

## The gait

A trot moves diagonal pairs of feet together. `SWING_PAIRS` lists the two pairs, and they take turns: in even
gait periods FL and RR swing while FR and RL stand, and in odd periods the roles swap. `GAIT_PERIOD` is 10 stages
(0.24 s), so every foot swings for 0.24 s and then stands for 0.24 s, always with two feet on the ground.

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:295:301"
```

`foothold(name, period)` answers where a foot should land at the end of its swing in a given period. The answer is
the spot that puts the foot at its nominal position under the base *when the base passes over it* in the middle of
the following stance, using the base poses predicted from the current velocity command (`poses`, below). A robot
that wants to go forward will therefore put its feet ahead of where they are, proportionally to the command, and one
that wants to turn will land its feet on an arc. For periods that are already over (`pose_stage` has moved past
them), the foothold is where the foot actually landed, kept in `landed`.

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:354:372"
```

`foot_reference(name, k)` is what the model needs: given a foot and a plan stage $k$ (which may be anywhere in the
future), the position the foot should be at and whether it is swinging. A foot that is not in the swing pair for
that period stays at the foothold of its last swing. A swinging foot interpolates from its previous foothold to the
next with a cosine blend, which gives it zero velocity at both ends so that it lifts off and touches down without
sliding, and a raised-sine bump of `step_height` on top of that clears the ground in the middle of the swing.

## The reference trajectory

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:278:292"
```

`predict_poses` extends the base's pose $(x, y, \text{yaw})$ forward for `count` stages by integrating the current
velocity command, as if it were held constant. The command is in the base frame, so each step rotates it by the yaw.
These poses are used both as the base's reference, and, through `foothold`, to decide where the feet land. The
planner looks `mpc_H + 3 * GAIT_PERIOD` stages ahead rather than just `mpc_H`, since a foot swinging at the end of
the horizon needs the base pose from the *stance after* its swing to know where to land.

## Costs and constraints

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:375:423"
```

The shared costs (`config_cost`, `v_cost`, `u_cost`, `f_cost`) are single objects added to every stage, since they do
not change from stage to stage. The remaining ones are created *per stage* and kept in lists, so that
`update_model` can later retarget the one for stage `idx` independently:

* `base_costs[idx]` is a `FrameSquaredPoseErrorCost` on `base_link`, tracking the commanded base pose.
* `foot_funcs[name][idx]` holds, for each foot, a `FrameSquaredTranslationErrorCost` and the foot's contact
  constraint. The contact constraint has a friction coefficient of 0.6, a 150 N normal force limit, and
  `no_slip=False`, so that stance feet are not forced to a zero velocity.
* `flight_costs[idx]` is a configuration cost that is only used for the jump.

Every foot has *both* a contact constraint and a tracking cost at every stage; which of them is actually active
depends on whether the foot is swinging, and is decided at run time (below). This is the contact scheduling from
[Go2 Jump](go2-jump.md#the-per-stage-loop-stance-flight-stance), done with `set_enabled` on constraints and costs that
already exist, rather than by choosing which ones to add.

## Reading the joystick

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:448:490"
```

`update_command` is called once per stage and turns the gamepad state into the quantities the planner uses. The
stick command is first read, scaled to the maximum speeds `MAX_VEL_FRONT`, `MAX_VEL_SIDE` and so on, and then
`v_cmd` moves toward it by at most `MAX_ACC * mpc_dt` per stage, so the robot cannot be asked to accelerate faster
than it can follow. The remaining inputs are rate-limited in the same way, each one a value with a range and a rate:
the D-pad moves `base_height` and `step_height`, and pairs of buttons set `base_pitch`, `base_roll` and `base_yaw` to
a limit while held, and back to zero when released.

| Input | Effect |
| --- | --- |
| Sticks | Forward, sideways and yaw velocity command |
| D-pad up and down | Base height |
| D-pad left and right | Step height |
| `BTN_Z`, `BTN_TR` | Pitch the base up or down |
| `BTN_NORTH`, `BTN_EAST` | Roll the base to the left or right |
| `BTN_WEST`, `BTN_TL` | Turn the front to the left or right, with the feet staying in place |
| `BTN_C` | Jump in place, when the command is near rest |

The names are evdev codes, which map to different physical buttons on different gamepads.

## Retargeting the horizon every stage

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:493:538"
```

`update_model` is what makes the robot walk, and is called once per plan stage, after the joystick is read. It does
five things:

1. **Records landed feet.** On the first stage of each new gait period, the pair that just finished swinging has
   landed at its target foothold, and `landed` is updated, so from now on the foot references use where the feet
   *are*, not where they were planned to land.
2. **Rebuilds the base poses**, starting from the pose the plan has reached, with the latest command.
3. **Sets the nominal configuration.** `config_cost` is pointed at `stance_config(...)`, the leg angles that keep
   every foot on its nominal spot with the base at the commanded height and orientation. This is how the D-pad
   and the tilt buttons work: they do not command anything directly, they change what "standing" means.
4. **Sets the base cost** of every stage to the pose for that stage and the tilt commands. The stage $k$ of the model
   corresponds to the absolute stage `current_stage + idx` of the gait, which is what `foot_reference` takes.
5. **Sets each foot.** Swinging feet have their contact constraint disabled and their cost tracking the swing path.
   Stance feet have the contact constraint enabled and a cost holding them at their foothold. As the comment in
   the code says, the constraint only bounds the foot's height, so the cost is also what stops stance feet sliding.

Note that the schedule is entirely periodic: at every stage the same function is evaluated at a different $k$, and
the window slides along, with a foot's contact turning off as its swing enters the horizon and on again as it leaves
it. This is why no part of the model ever needs to be rebuilt after `model.finalize`.

## The jump

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:312:335"
```

Pressing the jump button when the command is near rest schedules a jump in place, starting at the next gait period.
It reuses the machinery above rather than adding a new one. During the jump (`jump_stage(k)` is not `None`):

* The base height reference follows `jump_base_height`: a crouch, a push-off with constant acceleration up to the
  takeoff speed, a ballistic flight, a landing that decelerates, and a recovery back to the normal height. The
  takeoff speed `JUMP_SPEED` is set from the flight time, so that the base lands at the height it left. The velocity
  command is held at zero throughout.
* All four feet stand at the same footholds, those of their last swing before the jump (`jump_foothold`).
* In flight (`in_flight(k)`), every foot is marked as swinging, so *all* contact constraints and foot costs are
  disabled. The base's translation weight is zeroed too, since its path can no longer be steered once the feet are off
  the ground, and a `flight_costs` configuration cost is enabled instead to hold the legs in their takeoff pose.

This is the same stance, flight, stance structure as [Go2 Jump](go2-jump.md), but built from the pieces that already
exist for walking, and decided at every stage instead of once.

## Solver parameters

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:548:567"
```

[HPIPM](../concepts/qp-solvers.md) is the QP solver, with tolerances of $10^{-4}$. The `MPC` wrapper is given just two
SQP iterations, since the problem is warm started from the previous stage's solution (see
[Warm-starting](../concepts/mpc.md#warm-starting)). The `admm_params` are set in the script but unused, as long as the
QP solver type is HPIPM.

## The simulation loop

```python title="examples/mujoco_mpc/go2/go2_walking.py"
--8<-- "examples/mujoco_mpc/go2/go2_walking.py:605:648"
```

Every simulation step, the state is read from MuJoCo and converted (quaternion order, base velocity frame and joint
ordering, by `utils_go2_mujoco.py`) and clipped to the joint limits that the model enforces, because MuJoCo's joint
limits are soft. Then:

* **Every `STAGE_EVERY` steps**, `mpc.step(state)` solves. From the result it keeps the next stage's state `xk`, as
  a joint target, and the torques that the first stage's accelerations and forces need at the *current* measured
  state, from `model.getActuatedTorqueVector`. In forward dynamics mode, the control is the torque already. If the
  solver fails with a regularization error, the loop exits.
* **Every step**, the command applied is those torques, held until the next solve, plus a small joint PD term
  (`Kp` = 0.01) toward the next stage's joint position and velocity, which corrects drift between solves. The
  `[6:18]` slice drops the base entries, which are not actuated, and `ssqp_to_mujoco_ctrl_kin` reorders the legs for
  MuJoCo's actuators.
* **Every `STAGE_EVERY` steps**, after the command is set, `stage` advances and `update_command` and `update_model`
  run, so the *next* solve sees the retargeted horizon. They run before `mj_step`, in the same iteration.

The torques are therefore computed once per solve, from the state at that moment, and held for the 12 simulation
steps of the stage. This matches the planner's assumption that the controls are constant over a stage.

## Running it

From `examples/mujoco_mpc/go2/`, with a gamepad connected, the environment set up per
[Installation](../getting-started/installation.md#using-the-python-bindings) and MuJoCo's own virtual environment
per [Examples](../examples.md):

```bash
python go2_walking.py
```

The MuJoCo viewer opens maximized and the robot stands. Use the sticks to walk, and the other controls to change the
posture. The script does not wait to match real time, so the simulation runs as fast as the solves allow, and
printing the frequency is available in the commented block at the top of the loop. The `inputs` package is needed to
read the gamepad.

## Things to try

* Change `GAIT_PERIOD` to make the trot faster or slower, and see how the speeds that work change with it.
* Raise `STEP_HEIGHT`, or the base height, and check how the robot handles it.
* Lower `mpc_H`, and notice that the footholds near the end of the horizon, and so the response to a command change,
  get less accurate.
* Set `Kp` to zero to see how much the small PD term between solves contributes.
* Change `dynamics_mode` to `ForwardDynamics` and retune `U_WGT` and `F_WGT`: the weights are different for the two
  modes for a reason.
* Press the jump button while walking: it is ignored unless the velocity command is below `JUMP_MAX_VEL`.
