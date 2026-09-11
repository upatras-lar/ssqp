# Examples

The [`examples/`](https://github.com/upatras-lar/ssqp/tree/main/examples) directory contains robotics-related OCPs
built with SSQP, covering offline trajectory generation as well as real-time MPC in simulation and on real systems.

[**See the Gallery for videos of what SSQP produces →**](gallery.md){ .md-button .md-button--primary }

| Directory | Contents |
| --- | --- |
| [`trajgen/`](https://github.com/upatras-lar/ssqp/tree/main/examples/trajgen) | Offline trajectory generation: SSQP generates a full trajectory from scratch given costs/constraints and little to no initialization. Includes `di.py` (the [Quickstart](getting-started/quickstart.md) double integrator), `pendubot.py`, `acrobot.py`, and legged-robot examples (Go2, G1, A1, Talos, Panda, …). |
| [`mujoco_mpc/`](https://github.com/upatras-lar/ssqp/tree/main/examples/mujoco_mpc) | MPC examples using [MuJoCo](https://mujoco.org/) as the simulated "real" system: MuJoCo provides state feedback, and SSQP's [`MPC`](concepts/mpc.md) helper solves and re-solves the OCP each control step. |
| [`real_mpc/`](https://github.com/upatras-lar/ssqp/tree/main/examples/real_mpc) | MPC examples running against real hardware (e.g. Go2, Panda, a double pendulum) instead of a simulator. |
| [`cloud_pendulum/`](https://github.com/upatras-lar/ssqp/tree/main/examples/cloud_pendulum) | Adaptive-model pendulum/acrobot/pendubot examples that use [online dynamics parameter search](concepts/parameter-search.md) to adapt the model while running in MuJoCo. |
| [`visualizer/`](https://github.com/upatras-lar/ssqp/tree/main/examples/visualizer) | `visualize.py` / `visualize_srbd.py`, using [viser](https://viser.studio/main/) to visualize trajectories written by `solver.write(...)`. |
| [`models/`](https://github.com/upatras-lar/ssqp/tree/main/examples/models) | Custom SSQP `StageDynamics`/etc. models usable in place of XML files, plus related utilities. |
| [`assets/`](https://github.com/upatras-lar/ssqp/tree/main/examples/assets) | URDF/MJCF assets used by the examples. |

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
  (`INVERSE_MODE` switches to `InverseDynamics`)

## Running the Python examples

The built `ssqpy` Python extension must be importable; see
[Using the Python bindings](getting-started/installation.md#using-the-python-bindings) in Installation. Once built,
source [`examples/setup_env.sh`](https://github.com/upatras-lar/ssqp/blob/main/examples/setup_env.sh):

```bash
cd examples
source setup_env.sh
python trajgen/python/di.py
```

Then run any example script from its directory the same way. Several example subdirectories (MuJoCo, real robot
SDKs, the visualizer) use their own Python virtual environments for extra dependencies, so check for a
`*-venv`/`*venv` directory or a `README` alongside the specific example before running it.
