# Real system MPC examples

These examples run model predictive control (MPC) with SSQP on real robots. Instead of a simulator, the state comes from the hardware, SSQP solves an optimal control problem over a short horizon from that state, and the result is sent back to the hardware as a command. They are the real world counterparts of the [`mujoco_mpc`](../mujoco_mpc) examples, so the formulations and the models are similar.

Each example has been tested on real hardware:

- The **double pendulum** code has been successfully tested on our own double pendulums, as well as on Chalmers University's cloud pendulum platform. The latter required some changes to the code, retuning, and adapting to its API, particularly due to the lower available computing resources requiring an easier problem formulation to solve in real time.
- The **Panda** code has been tested on our Franka Emika Panda robot.
- The **Go2** code has been tested on our Go2 Educational edition robot.

Various scripts use a server-client architecture, where an MPC server running SSQP exchanges the state and the controls with a separate hardware client. This is either because the computer controlling the hardware had to be a different one to the computer running the MPC, or because a hardware library was in some way incompatible with SSQPy, so the two could not live in the same process. The double pendulum scripts and the Panda `controller_server.py` and `hardware.py` pair work this way, while the Panda `direct.py` and the Go2 script run the MPC and the hardware control together.

> **Safety.** These scripts command real motors. Tuning is specific to our hardware and setups, so expect to retune weights, limits and gains for yours. Run them with the emergency stop within reach, start from a safe pose, read each script and apply any robot-specific safety nets you deem necessary before running it on your robot.

## Running the examples

Make sure the SSQP Python extension is built and the environment is set up, as described in the [examples README](../README.md):

```
source ../setup_env.sh
```

Each system needs its own hardware libraries, which are not part of SSQP:

| System | Requirements |
|---|---|
| Double pendulum | `pyCandle`, for the MD80 motor drivers (only for `hardware.py`), and `matplotlib` for `--plot` |
| Panda | `pylibfranka`, and `inputs` for the gamepad in `direct.py` |
| Go2 | `unitree_sdk2py`, and `inputs` for the gamepad |

Run each script from its own directory, since asset paths are relative. Several of these environments are heavy or hardware specific, so the repository does not include them. The `.gitignore` lists the virtual environment names we use (`candlevenv`, `libfrankavenv` and `unitree-sdk-venv`).

## Directory structure

```
real_mpc/
├── socket_utils.py            Length-prefixed pickle messages over a socket
├── double_pendulum/
│   ├── pendulum.py            MPC server for a fully actuated double pendulum
│   ├── pendubot.py            MPC server for the pendubot
│   ├── acrobot.py             MPC server for the acrobot
│   └── hardware.py            Motor driver client
├── panda/
│   ├── models.py              The MPC problems (reset, target, circle, sequence)
│   ├── controller_server.py   MPC server, for a separate hardware client
│   ├── hardware.py            Hardware client for the server
│   ├── direct.py              MPC and hardware in one script
│   └── joystick.py            Gamepad reader
├── go2/
│   ├── go2_joystick.py        The robot controller, driven with a gamepad
│   ├── dynamic_walk_model.py  Whole-body dynamics walking MPC
│   └── kinematic_walk_model.py  Kinematic walking MPC
└── plots/                     Where the double pendulum `--plot` option saves its plots
```

## Double pendulum

The double pendulum examples split the system into two processes that talk over a Unix socket (`/tmp/mpc_socket`, using the helpers in `socket_utils.py`):

- The **server** (`pendulum.py`, `pendubot.py` or `acrobot.py`) builds and solves the MPC. It creates the socket and waits for the hardware to connect.
- The **hardware client** (`hardware.py`) reads the joint positions, velocities and torques from the motors, sends them to the server, and applies the torques it receives back.

Start the server first, then the hardware:

```
cd double_pendulum
python pendubot.py
python hardware.py --pendubot
```

All three servers swing the pendulum up and balance it, and send back torques. `pendubot.py` and `acrobot.py` use an inverse dynamics formulation. `pendulum.py` uses forward dynamics by default, with an inverse dynamics option set by its `--inverse-mode` argument. The solver cost, weights and torque limits are tuned per system and set at the top of each script.

### `pendulum.py`

A double pendulum with both joints actuated, using the SSQP double pendulum preset.

| Parameter | Description |
|---|---|
| `-i`, `--inverse-mode` | Use the inverse dynamics formulation instead of forward dynamics (default off) |

### `pendubot.py`

The pendubot, where only the first (shoulder) joint is actuated. The measured positions and velocities are low-pass filtered before computing the torque, and the torque on the passive joint is zeroed.

| Parameter | Description |
|---|---|
| `-p`, `--plot` | Run for a fixed number of steps, then plot and save positions, velocities, torque and control frequency |
| `-s`, `--steps-to-plot` | Number of steps to run for when plotting (default 1000) |
| `--hpipm` | Use HPIPM as the QP backend, with its own tuning, instead of ADMM |

### `acrobot.py`

The acrobot, where only the second (elbow) joint is actuated. The cost also tracks the elbow's height, which helps the swing-up.

| Parameter | Description |
|---|---|
| `-p`, `--plot` | Run for a fixed number of steps, then plot and save the same data as the pendubot |
| `-s`, `--steps-to-plot` | Number of steps to run for when plotting |
| `--hpipm` | Use HPIPM as the QP backend, with its own tuning, instead of ADMM |
| `-f`, `--friction-compensation` | Also actuate the passive joint, with a maximum torque of 0.005, to compensate for friction |

The `--plot` option saves its figure in `real_mpc/plots`, which has to exist.

### `hardware.py`

Connects to the server, then in a loop reads the motors, sends the state, receives the torque and applies it to the motors, clipped to a limit.

| Parameter | Description |
|---|---|
| `-b`, `--pendubot` | Only actuate the shoulder joint |
| `-a`, `--acrobot` | Only actuate the elbow joint |
| `-t`, `--torque-limit` | Maximum allowed torque (default 0.15) |
| `-f`, `--friction-compensation` | Let the passive joint be actuated, with a maximum torque of 0.005 |

The hardware script and the servers must agree on the system, so run `hardware.py` with `--pendubot` alongside `pendubot.py`, `--acrobot` alongside `acrobot.py`, and neither alongside `pendulum.py`.

## Panda

The Panda examples move the end effector of a Franka Emika Panda, with the MPC producing joint position or torque references that are tracked by the robot using a compliant joint-space controller. There are four tasks, defined as MPC problems in `models.py`:

| Task | Description |
|---|---|
| `reset` | Move the robot to its initial configuration. The default, if no task is given |
| `target` | Move the end effector to a position (`--pos X Y Z`, default `0.46 0.35 0.0`), pointing down. The model bounds the end effector to `y <= 0.1` |
| `circle` | Move the end effector around a circle in the yz-plane. `--constrain VALUE` (any non-empty value enables it) also keeps it inside the inscribed square |
| `sequence` | Follow a list of points, interpolated in between (only in `direct.py`) |

There are two ways to run them.

### `controller_server.py` and `hardware.py`

The MPC runs in a server, which can be on a different machine to the one that talks to the robot, and the hardware client connects to it over TCP. Start the server first, then the client:

```
cd panda
python controller_server.py circle -i 0.0.0.0 -p 11880
python hardware.py -r ROBOT_IP -i SERVER_IP -p 11880
```

The server solves with the most recent state it received, and replies with the new control, so the hardware loop keeps running at its own rate with the client interpolating towards the latest control.

Server parameters (before the task, if the task has its own):

| Parameter | Description |
|---|---|
| `-i`, `--ip` | IP address to bind to (default `localhost`) |
| `-p`, `--port` | Port to bind to (default 11880) |
| `-t`, `--torque` | Send torques instead of joint positions |
| `-m`, `--dynamics-mode` | `forward` (default) or `inverse` |

Hardware parameters:

| Parameter | Description |
|---|---|
| `-r`, `--robot-ip` | The robot's IP (required) |
| `-i`, `--ip`, `-p`, `--port` | The server's address (required) |
| `-t`, `--torque` | Use torque control, with a joint-space impedance controller and Coriolis compensation, rather than joint position control. Match the server's `-t` |
| `-a`, `--interpolation-alpha` | Smoothing factor of the target control (default 0.99) |

### `direct.py`

The same, but in one script, without sockets. One process runs the MPC and the other runs the hardware loop with torque control, sharing the state and controls through shared memory. The robot must run on this machine.

The Panda requires its control loop to run at a high frequency (1 kHz), and the hardware loop here runs on the same machine as the MPC. You may therefore need a real-time kernel to run this script reliably, as the loop can miss its deadlines on a standard kernel, especially while the MPC is solving.

```
cd panda
python direct.py -r ROBOT_IP circle
```

| Parameter | Description |
|---|---|
| `-r`, `--robot-ip` | The robot's IP (required) |
| `-m`, `--dynamics-mode` | `forward` (default) or `inverse` |
| `-a`, `--interpolation-alpha` | Smoothing factor of the target control (default 0.999) |
| `target --joystick` | Move the target position with a gamepad |

With the gamepad (`joystick.py`), the stick moves the target in x and y, and two buttons (`BTN_SOUTH` and `BTN_C`) move it up and down. When the inputs are released the target returns to the end effector's current position.

## Go2

`go2/go2_joystick.py` controls a Unitree Go2 through the Unitree SDK, with a whole-body dynamics MPC that walks the robot, driven with a gamepad. It talks to the robot at the joint level, publishing joint commands at 100 Hz and reading the robot's low-level state, and it needs to be able to reach the robot over the network.

```
cd go2
python go2_joystick.py
```

The network interface connected to the robot is `enp5s0` in our setup, so pass yours with `--interface`. With `--simulation`, the script instead controls the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) simulator over the loopback interface. This also selects joint gains suited to the simulator. The simulator leaves the foot force sensors at zero, in which case the script goes without foot contacts, unless its bridge (`simulate_python/unitree_sdk2py_bridge.py`) is changed to fill `foot_force` with each foot's normal force in newtons.

The script releases the robot's own motion service when it starts, stands the robot up, then switches to the walk after a few seconds.

| Parameter | Description |
|---|---|
| `-k`, `--kinematic` | Run the kinematic walk instead of the dynamic walk (default off, see [The kinematic walk](#the-kinematic-walk)) |
| `-s`, `--simulation` | Control the unitree_mujoco simulator instead of the robot (default off) |
| `-i`, `--interface` | The network interface to use (default `lo` when simulating, `enp5s0` otherwise) |

The defaults of these parameters are the `SIMULATION`, `ROBOT_INTERFACE` and `SIMULATION_INTERFACE` constants at the top of the script.

### The walk

`dynamic_walk_model.py` builds the MPC. It is similar to `go2_walking.py` in the MuJoCo examples, with additions needed on a real robot:

- **Gaits.** A trot, a run (a trot with all feet in the air between stances, faster forward, and turning slower at its top speeds), and a slower walk lifting one foot at a time, with the gamepad switching between them. The robot stands on all four feet until it is commanded to move, and goes back to standing once the command has been at rest.
- **State estimation.** The robot only measures the base orientation, so the base position and velocity are estimated from the stance feet using leg odometry, fused with the accelerometer in a Kalman filter. The foot force sensors tell when a foot is on the ground.
- **Latency handling.** The solver runs in its own thread, pinned to the performance cores of hybrid CPUs, at a fixed rate. It solves from the state predicted for when the plan will start, and the plan is then played back at the command rate, with the joint torques recomputed from the measured state at every command.
- **A small jump in place, and a jump forward**, and adjustable base height, step height, pitch, roll and yaw.

### Gamepad controls

Button names are evdev codes, which map to different physical buttons on different gamepads.

| Input | Effect |
|---|---|
| Sticks | Forward, sideways and yaw velocity command |
| D-pad up and down | Base height |
| D-pad left and right | Step height |
| `BTN_Z`, `BTN_TR` | Pitch the base up or down |
| `BTN_NORTH`, `BTN_EAST` | Roll the base to the left or right |
| `BTN_WEST`, `BTN_TL` | Turn the front to the left or right, with the feet staying in place |
| `BTN_C` | Jump in place, while standing, or jump forward with `BTN_SELECT` held |
| `BTN_TL2` | Switch to the next gait, while standing |
| `BTN_SOUTH` | Fallback (see below) |

### Fallback

If anything looks wrong during the dynamic walk, the controller falls back. It moves the legs to a stance pose, lowers the robot to the ground and then damps the joints, and stays like that until the robot is stood up again. The fallback is started by:

- pressing `BTN_SOUTH`, at any time,
- the solver failing or stalling,
- the robot state no longer updating,
- non-finite commands,
- the joints stopping to follow the plan, which is a joint error above a limit for a few commands in a row,
- the base tilting too far from the planned tilt.

Once the robot has been lying down for a few seconds, pressing `BTN_SOUTH` again stands it back up and resumes the walk. The limits are constants at the top of `go2_joystick.py`. During the kinematic walk, only the button starts the fallback.

The script does whatever it can to predict that something is going wrong and to stop the robot gently when it does, but this is not guaranteed to work. Something can still go wrong without being caught in time, and the robot could still topple over. Exercise caution when using the controller and keep clear of the robot while it runs.

### The kinematic walk

With `--kinematic`, `kinematic_walk_model.py` is used in place of the dynamic walk. This is an MPC in the kinematic mode, like `go2_walking_kinematic.py` in the MuJoCo examples, where the plan is tracked by the joint controller. The sticks command the velocity, within the planner's velocity limits, and `BTN_SOUTH` starts the fallback, as for the dynamic walk. The other gamepad controls, the automatic fallback triggers and the state filter belong to the dynamic walk only.

Being kinematic, this walk is not as stable and does not perform as well as the dynamic walk. Its only advantage is the higher MPC frequency it can reach, thanks to the simplified model. Use extra caution when running it, since it has no automatic fallback and the robot is more likely to lose its balance.

### Other tasks

`go2_joystick.py` also has a drop task, which lowers the robot from its stance. It is not used by the script as it is, but it can be started with `set_task("drop")`.
