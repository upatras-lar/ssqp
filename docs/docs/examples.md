# Examples

The [`examples/`](https://github.com/upatras-lar/ssqp/tree/main/examples) directory contains robotics-related OCPs
built with SSQP, covering offline trajectory generation as well as real-time MPC in simulation and on real systems.

[**See the Gallery for videos of what SSQP produces →**](gallery.md){ .md-button .md-button--primary }

| Directory | Contents |
| --- | --- |
| [`trajgen/`](https://github.com/upatras-lar/ssqp/tree/main/examples/trajgen) | Offline trajectory generation, in both Python and C++: SSQP generates a full trajectory from scratch given costs/constraints and little to no initialization. Includes `di.py` (the [Quickstart](getting-started/quickstart.md) double integrator), pendulums, the Panda and iiwa arms, and legged-robot examples (Go2, G1, A1, Talos). Most Python examples take command line parameters, and each one prints the final solver state and writes its result to a JSON file. |
| [`mujoco_mpc/`](https://github.com/upatras-lar/ssqp/tree/main/examples/mujoco_mpc) | MPC examples using [MuJoCo](https://mujoco.org/) as the simulated "real" system: MuJoCo provides state feedback, and SSQP's [`MPC`](concepts/mpc.md) helper solves and re-solves the OCP each control step. Covers the pendubot (with optional online model identification), the Panda and the Go2. The simulation is synced with the controller, so it does not run in real time. |
| [`real_mpc/`](https://github.com/upatras-lar/ssqp/tree/main/examples/real_mpc) | MPC examples running against real hardware instead of a simulator: a double pendulum, a Franka Emika Panda and a Unitree Go2. Several of them use a server-client architecture, since the machine controlling the robot is not always the one running SSQP. |
| [`visualizer/`](https://github.com/upatras-lar/ssqp/tree/main/examples/visualizer) | `visualize.py` / `visualize_srbd.py`, using [viser](https://viser.studio/main/) to visualize trajectories written by `solver.write(...)`. |
| [`models/`](https://github.com/upatras-lar/ssqp/tree/main/examples/models) | Custom SSQP `StageDynamics`/etc. models usable in place of XML files (the single rigid body model of a quadruped, and the planner that generates its contact schedules and references), plus related utilities. |
| [`assets/`](https://github.com/upatras-lar/ssqp/tree/main/examples/assets) | URDF/MJCF assets used by the examples. The URDFs of SSQP's built in model presets (Go2, A1, Panda, G1, Talos and the double pendulum) come from here, and are embedded into the library when it is built. |

Each directory has its own `README` with the details of every example, including the parameters it takes.

Solver benchmarks, comparing SSQP against other OCP solvers on a shared set of problems, live separately in
[`benchmarks/`](https://github.com/upatras-lar/ssqp/tree/main/benchmarks) rather than under `examples/`; see
[Benchmarks](benchmarks.md).

## Walkthroughs

A few examples are covered in detail:

* [Pendubot Swing-Up](examples/pendubot.md): `trajgen`, XML-based model, predefined costs, and a swing-up
  scheduling trick, `ForwardDynamics` mode
* [Panda Circle (Trajectory Generation)](examples/panda-circle-trajgen.md): `trajgen`, frame-relative pose
  tracking, a real box constraint, `InverseDynamics` mode
* [Go2 Jump](examples/go2-jump.md): `trajgen`, per-stage contact scheduling for a jump (stance → flight → stance),
  `ReducedInverseDynamics` mode
* [Panda Circle (MPC)](examples/panda-circle-mpc.md): `mujoco_mpc`, the same circle tracked in closed-loop MPC
  against a live MuJoCo simulation, with per-step target retargeting, `ForwardDynamics` mode by default
  (`--inverse-mode` switches to `InverseDynamics`)

## Running the Python examples

The built `ssqpy` Python extension must be importable; see
[Using the Python bindings](getting-started/installation.md#using-the-python-bindings) in Installation. Once built,
source [`examples/setup_env.sh`](https://github.com/upatras-lar/ssqp/blob/main/examples/setup_env.sh):

```bash
cd examples
source setup_env.sh
python trajgen/python/di.py
```

Then run any example script from its directory the same way (for example, `cd trajgen/python/go2` and
`python go2_gaits.py --gait trot`). The scripts take their settings as command line parameters, so `--help` lists
them. Several example subdirectories (MuJoCo, real robot SDKs, the visualizer) use their own Python virtual
environments for extra dependencies, so check for a `*-venv`/`*venv` directory or a `README` alongside the specific
example before running it.

## Running the C++ examples

The C++ trajectory generation examples are only built when the library is configured with examples enabled:

```bash
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build
```

Each one becomes an executable under `build/bin`, in the same subfolder as its source, and should be run from the
repository root, since asset and output paths are relative to it (for example, `./build/bin/urdf/panda/panda_circle`).
