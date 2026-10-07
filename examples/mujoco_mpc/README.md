# MuJoCo MPC examples

These examples run model predictive control (MPC) with SSQP in simulation, using [MuJoCo](https://mujoco.org/) as the simulator. The simulator plays the role of the real system: at every control step, the current state is read from MuJoCo, SSQP solves an optimal control problem over a short horizon starting from that state, and the first control of the solution is applied to the simulated system.

The simulation is synced with the controller. MuJoCo only advances after the controller has produced its control for the current step, so the simulation does not run in real time. If a solve takes longer than the simulation time step, the simulation simply runs slower than the wall clock. Some examples additionally wait between steps to avoid running faster than real time (see each example). The one exception is the Go2 kinematic walk, which solves the MPC in a separate process (see below).

## Running the examples

Make sure the SSQP Python extension is built and that the environment is set up, as described in the [examples README](../README.md):

```
source ../setup_env.sh
```

Besides SSQP, the examples need `mujoco` and `glfw`, and the Go2 examples driven by a gamepad also need [`inputs`](https://pypi.org/project/inputs/).

Run each script from its own directory, since asset paths are relative:

```
cd panda
python panda_circle.py --help
```

All scripts that take parameters use `argparse`, so `--help` lists them. A display is needed for the MuJoCo viewer.

## Directory structure

```
mujoco_mpc/
├── pendubot.py                  Pendubot swing-up and balance, with online identification
├── panda/
│   ├── panda_circle.py         Panda end effector tracking a circle
│   └── panda_move.py           Panda reaching random goals around an obstacle
└── go2/
    ├── go2_standing.py          Standing on four, three or two legs
    ├── go2_handstand.py         Handstand on one front foot
    ├── go2_walking.py           Gamepad-controlled trot, with a small jump
    ├── go2_walking_kinematic.py Gamepad-controlled trot using a kinematic model
    ├── utils_go2_mujoco.py      State and control conversions between SSQP and MuJoCo
    └── go2_controllers/
        └── joystick.py          Gamepad reader
```

The scripts use the models in [`../assets`](../assets) (URDF and MJCF files), and the Go2 kinematic walk also uses the planner in [`../models/quadruped_hlplanner`](../models/quadruped_hlplanner).

## Pendubot

### `pendubot.py`

A pendubot is a double pendulum with only its first (shoulder) joint actuated. The MPC swings the pendulum up and balances it upright, with a cost on the tip position, joint velocities and control, and a terminal cost on the upright configuration. The model is built from SSQP's double pendulum preset, while the simulated system is a MuJoCo model of the same pendulum.

The simulation sleeps to match real time. The horizon (45 stages of 0.01 s), cost weights and torque limit (0.15) are set at the top of the script.

Parameters:

| Parameter | Description |
|---|---|
| `-c`, `--control-delay` | Steps before a computed control is applied (default 1, no delay) |
| `-n`, `--noise` | Add random torque disturbances (default off) |
| `-i`, `--inverse-mode` | Use the inverse dynamics formulation instead of forward dynamics (default off) |
| `--identification` | Identify the second link's parameters online (default off) |
| `--identification-steps` | Steps of data to collect before identification (default 1000) |

With `--identification`, the script collects the joint torque regressor and the measured torques for the first steps. It then solves a least squares problem for the second link's mass, center of mass and inertia, and updates the model with the result.

## Panda

Both Panda examples keep the end effector pointing down, and both can use either dynamics formulation (`--inverse-mode`).

### `panda/panda_circle.py`

The end effector tracks a circle in the vertical plane in front of the robot, completing one lap every 4 seconds of simulated time and then repeating. The desired pose of every stage is shifted along the circle after each step. The simulation waits to match real time.

Parameters:

| Parameter | Description |
|---|---|
| `-p`, `--plot` | Run a single lap, then plot the end effector error and trajectory (default off) |
| `-i`, `--inverse-mode` | Use the inverse dynamics formulation (default off) |
| `-c`, `--constrain` | Constrain the end effector to the square inscribed in the circle (default on) |

### `panda/panda_move.py`

The end effector reaches a new random goal every few seconds. The goal is drawn as a red sphere in the viewer. With `--constrain`, a random pillar is placed on the floor between the end effector and the goal, tall enough to block the straight path, and drawn as a translucent box. The simulation waits to match real time.

Since a box-shaped bound is convex, the pillar is not excluded directly. Instead, any stage that the previous plan puts over the pillar must stay above its top face, so the end effector always passes over the pillar, never around it. This is not a hard guarantee. Since the MPC optimization runs for a limited number of SQP and inner QP iterations, and the initial state might result in a difficult or even infeasible problem, the end effector might occasionally clip the pillar's top.

Parameters:

| Parameter | Description |
|---|---|
| `-i`, `--inverse-mode` | Use the inverse dynamics formulation (default on) |
| `-c`, `--constrain` | Add the pillar and the constraint that keeps the end effector over it (default on) |
| `-f`, `--pd-change-freq` | Seconds between goals (default 2) |

## Go2

The Go2 examples use the Unitree Go2 quadruped, in the MJCF and URDF formats under `assets`. Most plan with the whole-body dynamics in the reduced inverse dynamics formulation, where the controls are joint accelerations and contact forces, which are converted to joint torques with inverse dynamics before being applied. These examples do not wait to match real time, and the standing and handstand examples print the loop frequency and the simulation time as they run.

The planner contacts the ground at the foot frames, while the Go2 MuJoCo model's feet are spheres whose contact point sits away from the frame. The examples that stand on a few feet on purpose (two or three legs, and the handstand) therefore change the simulated feet to point feet. `utils_go2_mujoco.py` converts states and controls between SSQP and MuJoCo, which differ in quaternion order, base velocity frame and actuator order.

### `go2/go2_standing.py`

The robot stands on a flat floor while holding its initial pose, with a contact constraint (friction cone and ground height) on each stance foot. The number of legs on the ground is chosen with `--legs`:

| Parameter | Description |
|---|---|
| `-l`, `--legs` | Number of legs to stand on: 4 (default), 3 or 2 |
| `--lifted-foot` | The foot to lift when standing on three legs: `FL_foot`, `FR_foot`, `RL_foot` or `RR_foot` (default `FR_foot`) |

- With 4 legs, the robot stands normally. The MPC is solved every few simulation steps, and the plan is held in between.
- With 3 legs, one foot is lifted. The base is first moved toward the centroid of the support triangle and the stance legs' joints are solved for, to keep the feet in place, so the robot starts in a balanced pose.
- With 2 legs, the diagonal pair FL and RR stays on the ground, and the other two are lifted.

You can push the robot with the MuJoCo viewer's perturbation tools (double click a body, then Ctrl and right drag) to see how (and whether) it recovers. Usually, if a contact is lost (due to slipping, or being lifted off the ground), the optimization for the next steps quickly fails, given that it assumes the contact is still there.

### `go2/go2_handstand.py`

The robot holds a handstand on its front left foot. The initial pose has the base pitched steeply nose down and rolled to the side, with the other three legs folded, and the center of mass is kept right above the foot with a tracking cost, with another cost to keep the foot from sliding, since the contact constraint only fixes the foot's height. The reference follows the foot's measured position at every MPC step.

### `go2/go2_walking_kinematic.py`

A gamepad-controlled trot based on a kinematic model. Unlike the other examples, the solver does not use torques. SSQP solves in the `Kinematics` mode, where the state is the robot's configuration and the controls are joint velocities, and a joint space PD controller tracks the planned joint positions and velocities of the next stage. The high-level references (CoM, base pose and feet) and the contact schedule come from the planner in [`models/quadruped_hlplanner`](../models/quadruped_hlplanner), with stance feet pinned by constraints, swing feet tracked with costs, and a cost that keeps the center of mass over the line between the two stance feet.

The MPC runs in a separate process, which reads the gamepad and uses the most recent state it has received. The simulation does not wait for each solve and keeps applying the last available plan. The sticks command the velocity, as in `go2_walking.py`.

### `go2/go2_walking.py`

A fully dynamic gamepad-controlled trot. The planner is a stage-based gait generator: the diagonal feet pairs swing in alternate gait periods, with footholds and swing trajectories computed from the commanded velocity, and the MPC tracks them with the whole-body dynamics. The MPC is solved once per plan stage (every 24 ms of simulation time), and its torques are held in between, with a small joint PD term on the plan's next stage.

Gamepad controls (button names are evdev codes, which map to different physical buttons on different gamepads):

| Input | Effect |
|---|---|
| Sticks | Forward, sideways and yaw velocity command |
| D-pad up and down | Change base height |
| D-pad left and right | Change step height |
| `BTN_Z`, `BTN_TR` | Pitch the base up or down |
| `BTN_NORTH`, `BTN_EAST` | Roll the base to the left or right |
| `BTN_WEST`, `BTN_TL` | Turn the front to the left or right, with the feet staying in place |
| `BTN_C` | Jump in place, when the robot is near rest |

The script stops if the solver fails with a regularization error.
