# Trajectory generation examples

These examples use SSQP to generate trajectories offline, as optimal control problems (OCPs) solved from scratch. Each one defines a model, costs and constraints over a finite horizon, often with next to no initialization beyond the starting state, and lets the solver find the whole trajectory. Nothing here runs against a simulator or a real system. The result is written to a JSON file, which can be visualized with the [`visualizer`](../visualizer) example or loaded for analysis.

There are two sets of examples, covering the same library from two sides:

- The **Python** examples use the `ssqpy` bindings, and most of them take their behavior from command line parameters.
- The **C++** examples use the library directly, and show how to define custom dynamics, costs and constraints in C++.

## Running the examples

### Python

Make sure the SSQP Python extension is built and the environment is set up, as described in the [examples README](../README.md):

```
source ../setup_env.sh
```

Run each script from its own directory, since asset paths are relative:

```
cd python/go2
python go2_gaits.py --gait trot
```

All the scripts that take parameters use `argparse`, so `--help` lists them. Every script that writes a trajectory takes `-o`/`--output`, which defaults to a file in `data`.

### C++

The C++ examples are only built when the library is configured with examples enabled, from the repository root:

```
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build
```

Each example becomes an executable under `build/bin`, in the same subfolder as its source, which should be run from the repository root since asset and output paths are relative to it:

```
./build/bin/urdf/panda/panda_circle
```

A few of the C++ examples have a constant marked `@script-parameter` near the top (such as which move to perform), which can be changed in the source before building.

### Output

Trajectories are written to `data/*.json`. This folder is not tracked by git. Visualize a result with the [`visualizer`](../visualizer), giving it the URDF of the model and the JSON file:

```
python visualize.py --urdf-path ../assets/urdf/go2/go2.urdf --input ../trajgen/data/go2_jump.json
```

## Directory structure

```
trajgen/
├── python/
│   ├── di.py                    Double integrator with custom dynamics
│   ├── ball_joint.py            Ball joint
│   ├── a1_jump.py               A1 jump
│   ├── double_pendulum/         Pendubot and acrobot
│   ├── panda/                   Panda circle and obstacle
│   ├── go2/                     Go2 (standing, jumps, gaits, ...)
│   │   └── utils/               Helper for building quadruped gaits
│   ├── g1/                      G1 humanoid
│   └── talos/                   Talos humanoid
├── c++/
│   ├── analytical/              Models with hand-written dynamics
│   └── urdf/                    Models loaded from URDF files
└── data/                        Generated trajectories (not tracked)
```

The scripts use the robot models in [`../assets`](../assets), either through SSQP's built in presets (`Preset.Go2`, `Preset.Panda`, and so on), or by loading an URDF or MJCF file. The Go2 scripts that use a single rigid body model, and the ones built on the kinematic planner, also use the models in [`../models`](../models).

## Python examples

### Simple systems

#### `python/di.py`

A double integrator, defined by custom dynamics written in Python (the `StageDynamics` interface), with a custom cost. It is the smallest example of defining a model without any robot description, and is the one used in the quickstart.

#### `python/ball_joint.py`

A ball joint, loaded from an MJCF file, driven to a target orientation. It shows a state living on a manifold, where the configuration is a quaternion.

| Parameter | Description |
|---|---|
| `-o`, `--output` | Output JSON file |

### Double pendulum

Both examples move a double pendulum to the upright position and keep it there, using costs that are only active for the last part of the horizon, and a terminal cost.

#### `python/double_pendulum/pendubot.py`

The pendubot, with only the shoulder joint actuated. The model has continuous joints, so the configuration holds the cosine and sine of each angle.

| Parameter | Description |
|---|---|
| `-o`, `--output` | Output JSON file |

#### `python/double_pendulum/acrobot.py`

The acrobot, with only the elbow joint actuated, using the double pendulum preset.

| Parameter | Description |
|---|---|
| `-a`, `--target-angle` | The angle of the shoulder joint to move to and remain at, in radians (default pi, upright) |
| `-o`, `--output` | Output JSON file |

### Panda

#### `python/panda/panda_circle.py`

The Panda's end effector traces a circle in the yz-plane, pointing down. Optionally, a box constraint keeps it inside a square inscribed in the circle. After solving, it can plot the end effector's error and trajectory.

| Parameter | Description |
|---|---|
| `-p`, `--plot` | Plot the tracking error and the trajectory (default on) |
| `-c`, `--constrain` | Add the box constraint (default on) |
| `-o`, `--output` | Output JSON file |

#### `python/panda/panda_obstacle.py`

The end effector moves from its starting position to a target, while a bound on the end effector's height, applied to the middle of the trajectory, acts as the obstacle.

| Parameter | Description |
|---|---|
| `-t`, `--target` | The target end effector position (default 0.3 0.3 0.1) |
| `--min-height` | Height the end effector may not go under in the middle of the trajectory (default 0.4) |
| `-o`, `--output` | Output JSON file |

### A1

#### `python/a1_jump.py`

A jump with the A1 quadruped. Contact constraints are added to all four feet for a number of steps at the start and at the end of the horizon, with no contacts in between.

| Parameter | Description |
|---|---|
| `--steps-before` | Steps with contacts at the start of the horizon (default 60) |
| `--steps-after` | Steps with contacts at the end of the horizon (default 50) |
| `-o`, `--output` | Output JSON file |

### Go2

All of these use the Go2 preset, with the reduced inverse dynamics formulation and contact constraints on the feet, unless noted.

#### `python/go2/go2_standing.py`

The Go2 stands in place, optionally on a slope that rises along the x axis.

| Parameter | Description |
|---|---|
| `-s`, `--slope` | Terrain slope along the x axis, rise over run (default 0, flat ground) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_handstand.py`

The Go2 balances upside down on its front left leg, with the center of mass directly above the foot.

| Parameter | Description |
|---|---|
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_jump.py`

A simple jump. The robot is in the air between two steps, where the contact constraints are left out. This is the example walked through in the documentation.

| Parameter | Description |
|---|---|
| `--start-step` | Step at which the robot leaves the ground (default 60) |
| `--end-step` | Step at which the robot lands (default 90) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_multi_jump.py`

A sequence of jumps, with a chosen set of feet to land on in between.

| Parameter | Description |
|---|---|
| `-j`, `--jump-steps` | The step ranges the robot is in the air for, as `START:END` pairs (default `50:70 100:120 150:180 210:240 270:300`) |
| `-l`, `--land-seq` | The feet to land on between consecutive jumps, one for each gap: `B` (back), `F` (front), `L` (left), `R` (right), or a single foot `FL`, `FR`, `RL`, `RR` (default `B R L F`) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_frog.py`

A series of jumps that moves the robot forward, with the final position fixed as a constraint.

| Parameter | Description |
|---|---|
| `-d`, `--distance` | Distance to travel along the x axis (default 2.0) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_gaits.py`

Gaits generated with the `GaitManager` helper, optionally on a slope that rises along the x axis, followed by a stretch of standing on all four feet.

| Parameter | Description |
|---|---|
| `-g`, `--gait` | `walk` (default), `amble`, `trot`, `pace`, `canter`, `gallop` or `rotatory-gallop` |
| `-c`, `--cycle-steps` | Steps per gait cycle (default depends on the gait) |
| `-f`, `--final-standing-steps` | Steps to stand on all four feet at the end (default 30) |
| `-s`, `--slope` | Terrain slope along the x axis, rise over run (default 0.2) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_stairs.py`

The Go2 walks up a staircase, which is defined as terrain.

| Parameter | Description |
|---|---|
| `-g`, `--gait` | `walk` (default), `amble` or `trot` |
| `-f`, `--final-standing-steps` | Steps to stand on all four feet at the end (default 40) |
| `-n`, `--num-steps` | Number of steps of the stairs (default 6) |
| `--step-height` | Height of each step (default 0.05) |
| `--step-length` | Length of each step along x (default 0.2) |
| `--step-x` | Position of the first step along x (default 0.2) |
| `--stand-height` | Standing height of the base above the terrain (default 0.3) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_circle.py`

The base traces a circle while all four feet stay on the ground.

| Parameter | Description |
|---|---|
| `-n`, `--circles` | Number of circles (default 1) |
| `-r`, `--radius` | Radius of the circle (default 0.15) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_heart.py`

The front feet trace a heart, while the rear feet and the base stay in place.

| Parameter | Description |
|---|---|
| `--standing-steps` | Steps to stand for at the start and end (default 20) |
| `--transitional-steps` | Steps to transition to and from the trace (default 20) |
| `--trace-x` | x-coordinate of the trace (default 0.2) |
| `-o`, `--output` | Output JSON file |

#### `python/go2/go2_srbd_walk.py` and `python/go2/go2_srbd_jump.py`

A walk and a jump using a single rigid body model, defined in Python in [`models/quadruped_srbd`](../models/quadruped_srbd), with the foot positions and the contact forces as the state and controls. The walk takes its references from the planner in [`models/quadruped_hlplanner`](../models/quadruped_hlplanner).

| Script | Parameter | Description |
|---|---|---|
| `go2_srbd_walk.py` | `-v`, `--velocity` | The commanded velocity, as forward, sideways and yaw (default 1 0 0) |
| `go2_srbd_jump.py` | `-t`, `--com-target` | The position of the center of mass to jump to (default 0.4 0 0.36) |
| both | `-o`, `--output` | Output JSON file |

### G1

#### `python/g1/g1_standing.py`

The G1 humanoid stands, with forces on its feet.

| Parameter | Description |
|---|---|
| `-o`, `--output` | Output JSON file |

#### `python/g1/g1_jump.py`

The G1 jumps in place.

| Parameter | Description |
|---|---|
| `--jump-start` | Step at which the robot leaves the ground (default 80) |
| `--jump-end` | Step at which the robot lands (default 120) |
| `-o`, `--output` | Output JSON file |

#### `python/g1/g1_walk.py`

The G1 walks forward, with alternating foot swings.

| Parameter | Description |
|---|---|
| `-d`, `--distance` | Distance to walk along the x axis (default 3.0) |
| `--foot-swing-steps` | Steps for each foot swing (default 25) |
| `--standing-steps` | Steps to stand for at the start and the end (default 15) |
| `-o`, `--output` | Output JSON file |

### Talos

#### `python/talos/talos_stand_1foot.py`

The Talos humanoid balances on one foot. Since its feet are flat, forces are applied at four points around each foot.

| Parameter | Description |
|---|---|
| `-o`, `--output` | Output JSON file |

#### `python/talos/talos_jump.py`

An in-place jump with Talos.

| Parameter | Description |
|---|---|
| `--steps-before` | Steps before the jump (default 45) |
| `--steps-after` | Steps after the jump (default 45) |
| `-o`, `--output` | Output JSON file |

#### `python/talos/talos_walk.py`

Talos walks forward, using a reduced model of the robot, and stands upright at the end.

| Parameter | Description |
|---|---|
| `-d`, `--distance` | Distance to walk along the x axis (default 2.0) |
| `--foot-swing-steps` | Steps for each foot swing (default 40) |
| `--final-standing-steps` | Steps to stand upright for at the end (default 20) |
| `-o`, `--output` | Output JSON file |

## C++ examples

The C++ examples have no command line parameters. Their settings are constants at the top of the source file.

### Analytical models

These define the dynamics by hand, as classes deriving from `StageDynamics`, together with their own costs and constraints.

| Example | Description |
|---|---|
| `analytical/di.cpp` | A double integrator |
| `analytical/point_mass.cpp` | A 1D point mass driven to a target position, with control bounds and a terminal constraint |
| `analytical/pendulum.cpp` | A damped pendulum, with its dynamics, costs and their derivatives written out |

### Models from URDF files

These load the model from an URDF file in `assets`, with SSQP computing the dynamics.

| Example | Description |
|---|---|
| `urdf/pendulum.cpp` | The same damped pendulum as above, but loaded from an URDF |
| `urdf/double_pendulum.cpp` | A double pendulum with only one joint actuated, and a custom cost defined with lambda functions |
| `urdf/iiwa.cpp` | A KUKA iiwa arm performing one of four moves (a small move, a mid-range move, and elbow-up and elbow-down poses) |
| `urdf/panda/panda_poses.cpp` | The Panda reaching one of five poses or sequences (a simple reach, a side sweep, an overhead lift, and two grasp motions) |
| `urdf/panda/panda_circle.cpp` | The Panda's end effector tracing a circle, with an optional box constraint inside it |
| `urdf/a1/a1_standing.cpp` | The A1 standing on four feet, or on two |
| `urdf/a1/a1_inplace.cpp` | The A1 raising its feet in place, in pairs |
| `urdf/a1/a1_circle.cpp` | The A1's trunk moving along a circular path |
| `urdf/a1/a1_walk.cpp` | The A1 walking |
| `urdf/a1/a1_jump.cpp` | The A1 jumping |
| `urdf/talos/talos_standing.cpp` | The Talos humanoid standing |

The A1 and Talos examples use contact constraints, and forces at the feet as part of the controls.
